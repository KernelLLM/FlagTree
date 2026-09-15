---
priority: high
---

# Explicit Hardware Memory Allocation (TLE DSA)

## Summary

Allocate on-chip buffers explicitly with `tle.dsa.alloc(shape, dtype, mem_addr_space)` into UB / L1 / L0A / L0B / L0C, and use GM workspace tensors (kernel parameters) for anything that crosses cores. Convert buffers to register tensors with `to_tensor` for compute, and slice without copying via `subview`.

## Use When

- The compiler's automatic placement is suboptimal (e.g. a Vector working buffer landing in L1) or blows a capacity budget.
- You need exact buffer sizes to fit hardware limits — UB pressure is the classic FA killer (see Hazards below).
- Double-buffering requires two buffer sets at a specific level.
- Cross-core (Cube↔Vector) exchange requires GM workspace tensors as kernel parameters.

## Avoid When

- You are still prototyping kernel structure — explicit memory slows iteration.
- The plain-Triton kernel already fits and performs.

## Pattern

### Memory-to-hardware mapping

| Purpose | TLE DSA API | Hardware | Notes |
|---|---|---|---|
| GEMM operand staging | `dsa.alloc([M, K], tl.float16, L1)` | L1 buffer | feed dot via `to_tensor(writable=False)` |
| Vector workspace | `dsa.alloc([M, N], tl.float32, UB)` | Unified Buffer | element-wise / softmax scratch |
| MMA left operand | `dsa.alloc([M, K], tl.float16, L0A)` | L0A | usually compiler-managed via `tl.dot` |
| MMA right operand | `dsa.alloc([K, N], tl.float16, L0B)` | L0B | usually compiler-managed via `tl.dot` |
| MMA accumulator | `dsa.alloc([M, N], tl.float32, L0C)` | L0C | fp32 only |
| Cross-core exchange | kernel parameter tensors | GM | the ONLY legal C↔V path |

```python
import triton.experimental.tle as tle
from triton.experimental.tle.language.dsa.ascend import UB, L1

q_l1  = tle.dsa.alloc([BLOCK_M, DIM], dtype=tl.float16, mem_addr_space=L1)
s_ub  = tle.dsa.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, mem_addr_space=UB)
```

### Buffer → tensor conversion for compute

`alloc` returns a `buffer`, **not** a `tl.tensor`. Convert at use:

```python
# dot operands: read-only view
q = tle.dsa.to_tensor(q_l1, writable=False)     # or q_l1.to_tensor(writable=False)
s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

# vector math: writable view, or compute on register tensors and store back
t = tle.dsa.to_tensor(s_ub)                     # writable=True default
```

Verified dot-operand forms (`native_fa.py`): whole-tile `tile_copy` GM→L1, then `tl.dot(tile_to_tensor(l1_buf, writable=False), ...)`; or plain register tensors from `tl.load` / `tl.make_block_ptr` (`fa_triton_arch.py` MM path).

### Zero-copy slicing

```python
view = tle.dsa.subview(src, offsets=[0, 0], sizes=[BLOCK_M, BLOCK_N], strides=[1, 1])
# or src.subview(offsets, sizes, strides)
```

`offsets` may be dynamic (loop indices); `sizes`/`strides` must be constexpr. Use this for ping-pong slots inside one 2×-sized buffer (see `double-buffer`).

### GM workspace for cross-core exchange

Workspace tensors are ordinary kernel pointer parameters; the launch option `set_workspace_multibuffer=2` exists for compiler-managed workspace multi-buffering. Per-core slicing is your responsibility (`ws_ptr + pid * SLOT_ELEMS + ...`).

## Hazards (all observed on device or in compiler source)

1. **Identical-shape `tile_alloc`s are merged.** `commonir_to_hivm` merges allocs of identical shape/dtype into one physical cbuf (`matmul_double_buffer.py` header comment: two same-shape ping-pong buffers silently alias → WRONG results). For multiplicity, allocate ONE buffer with a doubled leading dim and ping-pong via `subview`.
2. **UB budget interacts with `multibuffer`.** The `multibuffer=True` launch default doubles eligible local buffers. An earlier FA version allocated ~98 KB of explicit UB and blew the ~192 KB/core budget after doubling. When you size buffers by hand, either keep total explicit UB ≤ ~96 KB or launch with `multibuffer=False`.
3. **`dsa.to_buffer(bind_buffer=...)`** — the code accepts a shape-matched bind, but this form appeared in an on-device-failing kernel; prefer fresh `to_buffer(t, space)` or explicit `tile_copy`.
4. **`subview(L1) → to_tensor → tl.trans → tl.dot` chain** — known on-device failure (lowering issue around subviewed 4D-ish cbuf). Keep dot operands whole-tile.
5. **Dynamic leading-dim `dsa.copy`** (e.g. `copy(src, buf, [tail, DIM])` with dynamic first extent) appeared in a failing kernel; `native_fa.py`'s verified form uses constexpr extents `[tl.constexpr(BM), tl.constexpr(D)]`. Dynamic tails are verified for 1D/2D UB copies in `mhc_post.py` (`[4, tail_d]`) — the failing case combined it with L1 + dot.
6. **fp64 does not exist on Ascend**; accumulators are fp32. `reduce_sum`-style accumulation inherits buffer dtype — use fp32 buffers for softmax denominators.

## What To Verify After Applying

- Each `alloc` targets the right level: L1 for dot operand staging, UB for vector scratch, L0C for accumulation, GM for cross-core.
- Capacity: L0A+L0B+L0C ≤ 64 KB per core; explicit UB total × (2 if `multibuffer=True`) ≤ ~192 KB per core.
- No two same-shape/same-dtype allocs are intended to be distinct physical buffers.
- dot operands are whole-tile `to_tensor(writable=False)` or plain register tensors — never a subview→trans chain.
- Run `tle_sync_lint.py` if the kernel also uses manual sync.
- Numeric gate on device after any placement change — layout/placement mistakes corrupt silently.

## Related Patterns

- `cv-sync`: the sync discipline around these buffers.
- `double-buffer`: 2×-buffer + subview ping-pong built on this pattern.
- `workspace-pipeline`: GM workspace ring design.
- `layout-affinity`: what little layout control exists once placement is explicit.
