---
priority: high
---

# CV Scope Separation with Manual Synchronization (TLE DSA)

## Summary

Split a kernel body into `with tle.scope(core_mode="cube")` and `with tle.scope(core_mode="vector")` regions, and coordinate Cube↔Vector data handoffs with explicit `sync_block_set` / `sync_block_wait` cross-core semaphores. Intra-core pipe ordering uses `tile_set_flag` / `tile_wait_flag` / `tile_pipe_barrier`. This is the TLE DSA counterpart of the tilelang `T.Scope("C")/("V")` + `set_flag`/`set_cross_flag` pattern, and the direct analog of AscendC `SetFlag/WaitFlag` + cross-core notify.

## Use When

- A kernel mixes `tl.dot` (Cube) with element-wise math, reductions, or softmax (Vector) in one compute flow.
- Intermediate results must cross cores: the **only** legal Cube→Vector path is L0C →(FIX pipe)→ GM workspace →(MTE2)→ UB. There is no L0C→UB path.
- You are porting an AscendC kernel with explicit flag discipline, and the compiler's auto-injected sync is either wrong or too conservative.
- The kernel needs Cube and Vector work overlapped (see `workspace-pipeline` for the multi-task version).

## Avoid When

- The kernel is purely element-wise (no `tl.dot`).
- A plain Triton kernel with compiler-managed sync is already correct and fast enough.

## Pattern

### Step 0: Launch options replace tilelang's pass_configs

There is no Developer/Expert mode switch — TLE DSA is always explicit. What you must set at launch:

```python
kernel[grid](..., 
             enable_mixed_cv=True,                 # enable cube+vector mix mode
             sync_solver=True,
             disable_auto_inject_block_sync=True,  # REQUIRED when you hand-place sync_block_*
             unit_flag=True,
             multibuffer=False)                    # False when YOU manage buffer multiplicity
```

`disable_auto_inject_block_sync=True` is the analog of tilelang `TL_ASCEND_AUTO_CV_SYNC: False` — without it the compiler adds its own handshakes on top of yours. Tutorials that hand-place sync also pass `unit_flag=True` (seen in autotune configs of `05-lightning-indexer-v1.py` and `native_fa.py`).

### Step 1: Scope regions

```python
import triton.experimental.tle as tle
from triton.experimental.tle.language.dsa.ascend import PIPE, sync_block_set, sync_block_wait

@triton.jit
def kernel(...):
    with tle.scope(core_mode="cube"):
        ...   # tl.dot, GM->L1 copies, L0C->GM fixout
    with tle.scope(core_mode="vector"):
        ...   # UB element-wise, reduce, GM<->UB copies
```

Known-unverified subtlety: opening/closing a `tle.scope` pair **per iteration inside a runtime `for` loop** compiles and runs (that is the `fa_triton_arch.py` form) but has never passed a numeric gate. If you hit trouble, hoist scopes outside the loop as a structural fallback.

### Step 2: Cross-core semaphores

```python
sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
```

- `sender`/`receiver`: strings `"cube"` / `"vector"`, must differ.
- `event_id`: int, budget **0–15** total. Ring designs need `2 * RING <= 16` (READY/FREE pairs).
- Pipe pairs: the channel is keyed by `(event_id, sender_pipe, receiver_pipe)`; **set and wait must use the identical tuple**. Defaults when omitted:
  - cube→vector: `(PIPE.PIPE_FIX, PIPE.PIPE_MTE2)` — FIX retires L0C→GM writes
  - vector→cube: `(PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)` — MTE3 retires UB→GM writes
  - Tutorials also use vector→cube = `(PIPE.PIPE_MTE2, PIPE.PIPE_FIX)`; both appear in working code. Pick one convention and keep it identical on both sides.
- Producer sets, consumer waits. Direction is part of the contract.

### Step 3: Canonical double-buffered CV loop (compile-proven form)

From `python/tutorials/tle/dsa/04-tle-cv-mix-pipeline.py` (`disable_auto_inject_block_sync=True, multibuffer=False`, ~1.09× over single-buffer):

```python
pipe = PIPE
# Prologue: pre-arm both slots so Cube can run ahead twice
sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

for block_idx in range(num_blocks):
    buffer_id = block_idx % 2
    acc = tl.dot(a, b)                                                       # Cube compute
    sync_block_wait('vector', 'cube', buffer_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)  # slot free?
    tl.store(workspace_ptr + buffer_id * WS + offs, acc)                     # hand off via GM
    sync_block_set('cube', 'vector', buffer_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)   # data ready
    sync_block_wait('cube', 'vector', buffer_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)  # consumed?
    d = tl.load(workspace_ptr + buffer_id * WS + offs)                       # Vector consume
    ...
    sync_block_set('vector', 'cube', buffer_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)   # release slot

# Epilogue: drain the two pre-armed flags
sync_block_wait('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
sync_block_wait('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)
```

### Step 4: Prime / drain protocol

- Every "buffer empty" credit whose first producer is the consumer side must be **primed before the loop**, one `sync_block_set` per ring/buffer slot. Prime count = ring depth.
- Every primed credit must be **drained after the loop** with a matching `sync_block_wait`, or compilation fails with an extra-set-event error.
- First-iteration underflow: a `sync_block_wait` reachable on the first executed iteration whose credit is produced only under a guard is a structural hang (linter Tier-3). For every wait, trace its credit to either a prologue prime or an unguarded earlier set in program order.

### Step 5: Intra-core pipe events (finer control inside one scope)

```python
from triton.experimental.tle.language.dsa import tile_set_flag, tile_wait_flag, tile_pipe_barrier
tile_set_flag(producer_pipe, consumer_pipe, event_id)   # cross-ENGINE flag within one core
tile_wait_flag(producer_pipe, consumer_pipe, event_id)
tile_pipe_barrier(pipe)                                 # intra-engine barrier
```

`PIPE` enum: `PIPE_S` (scalar), `PIPE_V`, `PIPE_M`, `PIPE_MTE1` (L1→L0), `PIPE_MTE2` (GM→L1/UB), `PIPE_MTE3` (UB→GM), `PIPE_FIX` (L0C→UB/GM), `PIPE_ALL`. There is no `cc_sync` in the codebase — cross-core is `sync_block_*`, debug-only ordering is `tl.debug_barrier()`.

### Step 6: Vector sub-core split

910C has 2 Vector sub-cores per AI Core. Split rows explicitly per lane — never let both lanes write the same GM region (the tilelang `T.serial` race class applies verbatim):

```python
from triton.experimental.tle.language.dsa.ascend import sub_vec_id, sub_vec_num
vid = sub_vec_id()          # tl.tensor in {0, 1}
half = BLOCK_M // 2
rows = vid * half + tl.arange(0, half)
```

Cross-lane exchange must go through a GM tensor (MTE3 out / MTE2 in); an on-chip buffer shared across lanes is device error `507015` (linter Tier-4).

## What To Verify After Applying

- Run `python3 scripts/tle_sync_lint.py --tier1 --tier2 --tier3 --tier4 <kernel>.py` before any on-device test.
- Launch options: `disable_auto_inject_block_sync=True`, `unit_flag=True`, `enable_mixed_cv=True`; `multibuffer=False` if you manage multiplicity by hand.
- Every `sync_block_set` has a matching `sync_block_wait` with the **same (sender, receiver, event_id, sender_pipe, receiver_pipe)** tuple.
- Producer sets / consumer waits; FREE credits primed = ring depth before the loop and drained after.
- `event_id` budget: total distinct ids ≤ 16.
- Numeric gate on device: manual-sync errors most often manifest as NaN or silent corruption, not compile errors (`fa_triton_arch.py` precedent: compiles, runs, NaN).

## Related Patterns

- `explicit-memory`: allocate the GM workspace and UB/L1 buffers this pattern synchronizes.
- `double-buffer`: extends with L1 ping-pong for MTE/Cube overlap.
- `workspace-pipeline`: full RING-deep multi-task pipeline — read after mastering this pattern.
