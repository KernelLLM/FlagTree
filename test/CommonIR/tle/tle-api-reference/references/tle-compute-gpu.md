# TLE Compute — GPU Path (Pipeline, local_ptr, tl.dot)

> **Layer**: GPU path only. Requires NVIDIA Hopper or later.
> **Source**: `FlagTree/python/triton/experimental/tle/language/gpu/core.py`, `FlagTree/python/triton/experimental/tle/language/core.py`

This document covers software-pipeline loops, the `local_ptr` + `tl.dot` GEMM pattern, and async load on the GPU TLE path.

## 1. `tle.pipeline` — software-pipelined K-loop

```python
for k_off in tle.pipeline(start, stop, step, num_stages):
    ...
```

- Equivalent to `tl.range` but annotates the loop for the software-pipeline pass.
- `num_stages=2` gives one prefetch stage (double-buffer); higher values hide more latency at the cost of SMEM capacity.
- Use inside the k-loop of a GEMM or any loop where prefetch benefit is expected.

```python
a_smem = tle.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, scope=tle.smem)
b_smem = tle.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, scope=tle.smem)

acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

for k_off in tle.pipeline(0, K, BLOCK_K, num_stages=3):
    tle.copy(a_ptr + ..., a_smem, [BLOCK_M, BLOCK_K])
    tle.copy(b_ptr + ..., b_smem, [BLOCK_K, BLOCK_N])
    a_tile = tl.load(tle.local_ptr(a_smem, (row_ids, k_ids)))
    b_tile = tl.load(tle.local_ptr(b_smem, (k_ids, col_ids)))
    acc += tl.dot(a_tile, b_tile, input_precision="ieee")
```

> **Note**: `tle.pipeline` does not automatically issue async copies. To overlap DMA with compute, also use `tle.load(..., is_async=True)` for the global reads, or use TMA descriptors with `tle.copy`.

## 2. Standard GEMM kernel (SMEM tiling + `tl.dot`)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N

    # Index grids for local_ptr
    rm = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    rk_a = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    rk_b = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
    rn = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))

    # Allocate SMEM with MMA-compatible layout (default)
    a_smem = tle.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, scope=tle.smem)
    b_smem = tle.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, scope=tle.smem)

    a_ptrs_smem = tle.local_ptr(a_smem, (rm, rk_a))
    b_ptrs_smem = tle.local_ptr(b_smem, (rk_b, rn))

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_off in tle.pipeline(0, K, BLOCK_K, num_stages=2):
        a_gm_ptrs = a_ptr + (m_off + tl.arange(0, BLOCK_M))[:, None] * stride_am \
                           + (k_off + tl.arange(0, BLOCK_K))[None, :] * stride_ak
        b_gm_ptrs = b_ptr + (k_off + tl.arange(0, BLOCK_K))[:, None] * stride_bk \
                           + (n_off + tl.arange(0, BLOCK_N))[None, :] * stride_bn

        tle.copy(a_gm_ptrs, a_smem, [BLOCK_M, BLOCK_K])
        tle.copy(b_gm_ptrs, b_smem, [BLOCK_K, BLOCK_N])

        a_tile = tl.load(a_ptrs_smem)
        b_tile = tl.load(b_ptrs_smem)
        acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")

    # Write result to global memory
    c_ptrs = c_ptr + (m_off + tl.arange(0, BLOCK_M))[:, None] * stride_cm \
                   + (n_off + tl.arange(0, BLOCK_N))[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))
```

## 3. TMA copy pattern

When a pre-built `tl.tensor_descriptor` is available (Hopper), use it with `tle.copy` and supply `offsets`:

```python
a_desc = tl.make_tensor_descriptor(
    a_ptr, shape=[M, K], strides=[K, 1],
    block_shape=[BLOCK_M, BLOCK_K],
)

a_smem = tle.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, scope=tle.smem)

for k_idx in range(tl.cdiv(K, BLOCK_K)):
    # TMA async copy; offsets pick the tile
    tle.copy(a_desc, a_smem, [BLOCK_M, BLOCK_K], offsets=[m_off, k_idx * BLOCK_K])
    tl.debug_barrier()  # wait for TMA to complete before consuming SMEM
    a_tile = tl.load(tle.local_ptr(a_smem, (row_ids, col_ids)))
    ...
```

## 4. Distributed SMEM access (thread-block clusters)

To read a remote CTA's SMEM within a cluster, wrap the source buffer with `tle.remote` before calling `tle.local_ptr`:

```python
# local_buf is the buffered_tensor allocated in this CTA's SMEM
remote_buf = tle.remote(local_buf, shard_id=peer_cta_id, scope=tle.smem)
remote_ptrs = tle.local_ptr(remote_buf, (row_ids, col_ids))
data = tl.load(remote_ptrs)
```

`peer_cta_id` is an integer expression that selects which CTA in the cluster to read from.

## 5. Async load

```python
# Inside a tle.pipeline loop, mark global loads as async
for k_off in tle.pipeline(0, K, BLOCK_K, num_stages=2):
    a_val = tle.load(a_gm_ptrs, is_async=True)  # overlaps with prior-stage compute
    ...
```

`is_async=True` emits `tt.load {async=true}`, which the LowerAsyncLoad pass converts into async-copy intrinsics on supporting hardware.

**Next**: [tle-memory-dsa.md](tle-memory-dsa.md) for the DSA path, or [tle-compute-dsa.md](tle-compute-dsa.md) for Ascend compute.
