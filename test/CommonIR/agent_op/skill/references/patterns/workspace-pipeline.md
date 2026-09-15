---
priority: high
---

# Cross-Core Workspace Pipeline / Task Ring (TLE DSA)

## Summary

Use GM workspace tensors as a ring buffer between Cube and Vector scopes, with READY/FREE `sync_block` semaphore pairs controlling a multi-task pipeline, and a **skewed** task loop so that MM1 of task `g` overlaps MM2 of task `g−1` (and Vec1(g) overlaps Vec2(g−1)). This is the AscendC FAInfer schedule and the structure of `fa_triton_arch.py` / `fa_tle_v2.py`.

> **Provenance warning.** The TLE DSA 3-task ring descends from `fa_triton_arch.py`, which compiles and runs on device but outputs **NaN** (`test/CommonIR/next_fix_error.md` — "同步问题待调试"). Every element below is *compile-proven*, not numerically proven. The minimal locked-step variant (skew=0, single slot) in `steps/fa_step01_cv_naive.py` exists precisely to validate numerics before this structure is trusted. Bring up in that order.

## Use When

- Each Q-tile task produces intermediates (S = Q·Kᵀ, P = softmax(S), P·V) flowing Cube → Vector → Cube.
- The Cube pipe must never drain between tasks.
- You are replicating an AscendC FAInfer-style overlapping schedule.

## Avoid When

- No cross-core dependency, or one `sync_block_all` between phases suffices.
- One KV tile per Q-tile — ring overhead exceeds benefit.
- The skew=0 single-slot variant has not passed numerics yet. Fix numerics first; add skew second.

## Pattern

### Step 1: Ring depth and workspace layout

```python
RING = 3            # pipeline depth: tasks in flight; 2 * RING <= 16 (event-id budget)
NUM_CORES = 20
BLOCK_M, BLOCK_N, DIM = 32, 32, 64
```

Workspace tensors are kernel pointer parameters, per-core sliced, ring-indexed:

```
ws_s  [NUM_CORES, RING, BLOCK_M, BLOCK_N]  fp32   # S = Q·Kᵀ   (Cube→Vector)
ws_p  [NUM_CORES, RING, BLOCK_M, BLOCK_N]  fp16   # P = softmax(S) (Vector→Cube)
ws_pv [NUM_CORES, RING, BLOCK_M, DIM]      fp32   # P·V partial  (Cube→Vector)
```

Budget check: `NUM_CORES × RING × tile_bytes × 3 workspaces` against HBM. Deeper rings (tilelang expert FA used 14 slots/core) let Cube run a whole batch ahead; grow RING only after the shallow version is numerically correct.

### Step 2: Semaphore banks

One READY/FREE pair per workspace; keep the two directions in disjoint id banks:

```python
# bank 0..RING-1: slot-ready flags; bank RING..2*RING-1: slot-free flags
SEM_S_READY  = 0            # cube->vector : ws_s  slot has data
SEM_P_READY  = ...          # vector->cube : ws_p  slot has data
SEM_PV_READY = ...          # cube->vector : ws_pv slot has data
SEM_S_FREE   = RING + 0     # vector->cube : ws_s  slot consumed
...
```

Or the flattened two-bank scheme from `fa_triton_arch.py`: `SEM_BANK_SP = 0`, `SEM_BANK_PV = RING`, slot flag = `bank + g % RING`, constraint `2 * RING <= 16`.

### Step 3: Prime FREE credits, then skewed loop

Consumer scope primes RING FREE credits per producer; producer waits FREE before writing a slot, sets READY after; consumer waits READY before reading, sets FREE after its **last** read of the slot. The skewed loop runs `GT + 1` iterations:

```python
with tle.scope(core_mode="cube"):
    for g in range(GT + 1):
        if g < GT:                       # MM1(g): S -> ws_s[g % RING]
            r = g % RING
            sync_block_wait('vector', 'cube', SEM_S_FREE + r, pipe.PIPE_MTE2, pipe.PIPE_FIX)
            ...  # tl.dot Q·Kᵀ, store S tile to ws_s slot r
            sync_block_set('cube', 'vector', SEM_S_READY + r, pipe.PIPE_FIX, pipe.PIPE_MTE2)
        if g >= 1:                       # MM2(g-1): P·V -> ws_pv[(g-1) % RING]
            r2 = (g - 1) % RING
            sync_block_wait('vector', 'cube', SEM_P_READY + r2, ...)   # P ready?
            ...  # tl.dot P·V, store partial to ws_pv slot r2
            sync_block_set('cube', 'vector', SEM_PV_READY + r2, ...)
            sync_block_set('cube', 'vector', SEM_P_FREE + r2, ...)     # release ws_p slot

with tle.scope(core_mode="vector"):
    # prime: all ws_s / ws_pv slots start FREE
    for r in range(RING):
        sync_block_set('vector', 'cube', SEM_S_FREE + r, ...)
        sync_block_set('vector', 'cube', SEM_PV_FREE + r, ...)
    for g in range(GT + 1):
        if g < GT:                       # Vec1(g): softmax ws_s -> ws_p
            ...
        if g >= 1:                       # Vec2(g-1): rescale + accumulate ws_pv into O
            ...
# epilogue: drain remaining primed credits on both sides
```

### Step 4: Task-skew timing (NR=1 simplified)

```
Task   g=0     g=1     g=2     g=3
C MM1: [=S0=]  [=S1=]  [=S2=]  [=S3=]
C MM2:         [=P0V=] [=P1V=] [=P2V=]
V V1:          [soft0] [soft1] [soft2]
V V2:                  [acc0]  [acc1]
```

The `g < GT` / `g >= 1` guards create the prologue/epilogue; three tasks stay in flight.

### Step 5 (optional, tilelang-proven): throttle cross-core signaling

The tilelang expert FA batches cross flags every `cross_interval=2` tiles (mandatory tail clause `(i+1)%2==0 or i==last`), worth ~8 points of AscendC-relative performance there. **Not yet validated on the TLE DSA side** — treat as a later optimization round, after the per-tile version passes numerics. Note `cross_interval` in TLE is a FlagOS-side concept; check FlagTree docs for current support before relying on it.

## Known-unverified structural points (audit first when debugging)

1. `tle.scope` pairs opened/closed **per loop iteration** — compiles and runs, numerically unproven. Fallback: hoist scopes outside the loop.
2. Channel keying: the pattern assumes a `sync_block` channel is keyed by `(event_id, sender_pipe, receiver_pipe)`; if flags interfere, switch to fully disjoint numeric ids per channel.
3. Drain-phase same-side waits (consuming the last primed credits after the loop) — verify against the compiler's set/wait balance check.
4. Softmax math: `fa_triton_arch.py`'s negated-max odd/even ping-pong rescale is **not** equivalent to a single global running max (counterexample: m0 > m1 inflates the denominator) and is the prime NaN suspect. Use one standard running-max online softmax.

## What To Verify After Applying

- `tle_sync_lint.py --tier1 --tier2 --tier3 --tier4`: `CROSS_DEADLOCK` (every wait has an opposite-scope set), `FLAG_IMBALANCE`, `PRIME_UNDERFLOW` (FREE primed = RING before loop; READY never waited before set), `LATE_RELEASE` (FREE set only after the slot's last read).
- `2 * RING <= 16`; pipe tuples identical between set and wait; `"FIX"` qualifies L0C→GM writes, `"MTE2"/"MTE3"` qualify UB→GM writes.
- Guards `g < GT` / `g >= 1` match the skew; loop trip count `GT + 1`.
- Workspace HBM budget; RING ≥ desired tasks-in-flight.
- **Numeric gate before any skew/depth increase.** If NaN appears: bisect by RING=1 → disjoint ids → skew=0 (the step01 form).

## Related Patterns

- `cv-sync`: semaphore mechanics, prime/drain, launch options — read first.
- `double-buffer`: per-core L1 ping-pong inside each task.
- `explicit-memory`: workspace tensors and capacity budgets.
- `layout-affinity`: operand orientation for the MM1/MM2 dots.
