# TLE Memory — GPU Path (SMEM / TMEM)

> **Layer**: GPU path only. Requires NVIDIA Hopper or later.
> **Source**: `FlagTree/python/triton/experimental/tle/language/gpu/core.py`, `gpu/types.py`

This document covers on-chip buffer allocation, layout types, and data movement for the GPU (Hopper) TLE path.

## 1. Scope constants

```python
import triton.experimental.tle as tle

tle.smem   # shared memory (SRAM) — available on all CUDA GPUs
tle.tmem   # tensor memory (TMEM) — Hopper-only, used with MMA accumulators
```

Pass one of these as the `scope` argument to `tle.alloc`.

## 2. Layout types

### `tle.swizzled_shared_layout`

Controls the swizzle pattern used for shared memory to avoid bank conflicts.

```python
layout = tle.swizzled_shared_layout(
    vectorSize=8,   # elements per vector (granularity of swizzle)
    perPhase=1,     # rows sharing one phase
    maxPhase=8,     # total phases in the swizzle cycle
    order=[1, 0],   # dimension ordering (row-major = [1, 0])
)
# Convenience: non-swizzled default for rank-2 tensors
layout = tle.swizzled_shared_layout.make_default(rank=2)
```

### `tle.nv_mma_shared_layout`

MMA-compatible layout for SMEM operands fed into `tl.dot`. Handles swizzle + row-major automatically.

```python
layout = tle.nv_mma_shared_layout.make_default(shape=[128, 64], dtype=tl.float16)
```

Use `nv_mma_shared_layout` for any SMEM buffer that feeds `tl.dot`. Use `swizzled_shared_layout` for general-purpose SMEM that is consumed via `tle.local_ptr` + `tl.load`.

### `tle.tensor_memory_layout`

Layout for Hopper tensor memory (TMEM), used with MMA accumulator buffers.

```python
layout = tle.tensor_memory_layout.make_default(shape=[128, 128])
```

## 3. `tle.alloc` — allocate an on-chip buffer

```python
buf = tle.alloc(
    shape,                         # list[int | tl.constexpr] — tile dimensions
    dtype,                         # tl.float16 / tl.float32 / etc.
    layout=None,                   # tle.swizzled_shared_layout | tle.nv_mma_shared_layout | None
    scope=tle.smem,                # tle.smem | tle.tmem
    nv_mma_shared_layout=True,     # if True and layout is None, auto-selects MMA layout
)
# Returns: tle.buffered_tensor
```

- `nv_mma_shared_layout=True` is the default; set it to `False` when using `local_ptr` rather than MMA dot products, to avoid unnecessary swizzle overhead.
- `layout=None` with `nv_mma_shared_layout=False` allocates a plain linear SMEM buffer.

```python
# GEMM operand buffer — MMA-compatible layout
a_smem = tle.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, scope=tle.smem)

# General SMEM buffer for pointer-based access
tmp_smem = tle.alloc([BLOCK, BLOCK], dtype=tl.float32,
                     layout=None, scope=tle.smem, nv_mma_shared_layout=False)
```

## 4. `tle.copy` — move data between global memory and an on-chip buffer

```python
tle.copy(
    src,             # tl.tensor (global pointer) | tl.tensor_descriptor (TMA descriptor)
    dst,             # tle.buffered_tensor | tl.tensor (global pointer)
    shape,           # list[int] — number of elements to copy in each dimension
    offsets=None,    # list[int | tl.tensor] — required when src/dst is a tl.tensor_descriptor
)
```

Direction is auto-detected:
- `tl.tensor → buffered_tensor`: global memory → SMEM (load)
- `buffered_tensor → tl.tensor`: SMEM → global memory (store)
- `tl.tensor_descriptor → buffered_tensor`: TMA async copy (load)
- `buffered_tensor → tl.tensor_descriptor`: TMA async copy (store)

```python
# Standard global-to-SMEM copy
tle.copy(a_ptr + offsets, a_smem, [BLOCK_M, BLOCK_K])

# TMA copy using a pre-built descriptor
desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
tle.copy(desc, a_smem, [BLOCK_M, BLOCK_K], offsets=[m_off, k_off])
```

> **Note**: When `src` is a `tl.tensor_descriptor`, `offsets` is required and must match the descriptor's `block_shape` rank. Omitting offsets with a descriptor is a silent bug.

## 5. `tle.local_ptr` — materialise SMEM pointers for `tl.load` / `tl.store`

```python
ptrs = tle.local_ptr(
    buffer,    # tle.buffered_tensor
    indices,   # tuple of index tensors, one per buffer dimension
)
# Returns: tl.tensor of SMEM pointers
```

The returned pointer tensor has the same shape as the broadcast of all index tensors. Feed it directly to `tl.load`/`tl.store`.

```python
# 2-D SMEM access — load an entire tile
row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
col_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
a_ptrs  = tle.local_ptr(a_smem, (row_ids, col_ids))
a_tile  = tl.load(a_ptrs)

# Scalar SMEM read — all-scalar indices
val_ptr = tle.local_ptr(scalar_smem, (tl.constexpr(0),))
val     = tl.load(val_ptr)
```

> **Note**: `tle.local_ptr` with `nv_mma_shared_layout=True` buffers produces pointer values that incorporate the swizzle offset. Do not apply manual swizzle formulas on top.

## 6. `tle.load` — async global load

```python
result = tle.load(
    pointer,        # standard tl.tensor pointer
    mask=None,
    other=None,
    is_async=False, # True: emit async-copy IR consumed by LowerAsyncLoad pass
)
```

Use `is_async=True` inside a `tle.pipeline` loop body to enable hardware async copies on Hopper. Outside a pipeline context, `is_async=True` has no effect on semantics but does trigger the lowering pass.

## 7. Distributed: `tle.remote`

For NVIDIA thread-block cluster kernels, `tle.remote` marks a `buffered_tensor` as belonging to a remote CTA:

```python
remote_buf = tle.remote(local_buf, shard_id=remote_cta_id, scope=tle.smem)
remote_ptrs = tle.local_ptr(remote_buf, (row_ids, col_ids))
data = tl.load(remote_ptrs)  # loads from remote CTA's SMEM
```

**Next**: [tle-compute-gpu.md](tle-compute-gpu.md) for pipeline and dot-product patterns.
