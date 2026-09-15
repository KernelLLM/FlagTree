---
priority: high
---

# Double-Buffered Matmul — L1 Ping-Pong + Cube MMA

## Summary

Two pairs of L1 buffers for the A and B tiles overlap DMA prefetch with Cube MMA. While the Cube engine processes tile pair 0, the DMA engine fills tile pair 1 into the alternate L1 buffers. Requires explicit `tl.debug_barrier()` between the prologue copy and the first MMA, and `tile_cube_wait()` after each MMA before the next DMA→L0 copy.

## Use When

- The kernel performs matrix multiplication (GEMM, batched matmul, attention QK/PV product).
- The K dimension is large enough that DMA latency is a bottleneck (K ≥ 128 for float16).
- The MMA result is written to L0C and then moved to UB or global memory without further Vector engine computation in the same kernel.

## Avoid When

- K is small (< 64); the pipeline overhead is not worth the complexity — use a single-buffer L1 copy + MMA.
- The MMA result must feed a Vector engine softmax, activation, or bias fuse in the same kernel — use Three-Task Pipeline instead.
- The working set does not fit in L1 (two copies of the A tile + two copies of the B tile must all reside there simultaneously).

## Pattern

### Step 1: Allocate double-buffered L1 and L0 operands

```python
import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as tle_dsa
from triton.experimental.tle.language.dsa import ascend

@triton.jit
def matmul_double_buffer_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Two L1 buffers per operand — ping-pong between them
    mat_a_l1_0 = tle_dsa.tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
    mat_a_l1_1 = tle_dsa.tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
    mat_b_l1_0 = tle_dsa.tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L1)
    mat_b_l1_1 = tle_dsa.tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L1)

    # L0 operand and accumulator buffers
    mat_a_l0a = tle_dsa.tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L0A)
    mat_b_l0b = tle_dsa.tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L0B)
    mat_c_l0c = tle_dsa.tile_alloc([BLOCK_M, BLOCK_N], tl.float32, ascend.L0C)
```

Name the two L1 buffers with `_0` and `_1` suffixes — the index makes the ping-pong logic readable.

### Step 2: Compute global memory pointers for each K tile

```python
    def a_tile_ptr(k_idx):
        return tle_dsa.tile_gm_offset(
            a_ptr,
            indices=[pid_m * BLOCK_M, k_idx * BLOCK_K],
            strides=[K, 1],
        )

    def b_tile_ptr(k_idx):
        return tle_dsa.tile_gm_offset(
            b_ptr,
            indices=[k_idx * BLOCK_K, pid_n * BLOCK_N],
            strides=[N, 1],
        )

    num_k = tl.cdiv(K, BLOCK_K)
```

### Step 3: Prologue — fill buffer 0 before entering the loop

```python
    # Fill first tile into buffer _0
    tle_dsa.tile_copy(a_tile_ptr(0), mat_a_l1_0, [BLOCK_M, BLOCK_K])
    tle_dsa.tile_copy(b_tile_ptr(0), mat_b_l1_0, [BLOCK_K, BLOCK_N])
    tl.debug_barrier()  # wait for prologue DMA before first L1→L0 copy
```

> **Critical**: `tl.debug_barrier()` is required here. Without it, the Cube engine may start reading L0A/L0B before the DMA into L1 completes.

### Step 4: Main loop — overlap DMA into _1 with MMA on _0, then swap

```python
    # Zero-initialize accumulator before first MMA
    mat_c_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(num_k):
        # Select which L1 buffer pair to read (current) and write (next)
        a_cur = mat_a_l1_0 if k % 2 == 0 else mat_a_l1_1
        b_cur = mat_b_l1_0 if k % 2 == 0 else mat_b_l1_1
        a_nxt = mat_a_l1_1 if k % 2 == 0 else mat_a_l1_0
        b_nxt = mat_b_l1_1 if k % 2 == 0 else mat_b_l1_0

        # Prefetch next tile into the alternate buffer (while Cube works on current)
        if k + 1 < num_k:
            tle_dsa.tile_copy(a_tile_ptr(k + 1), a_nxt, [BLOCK_M, BLOCK_K])
            tle_dsa.tile_copy(b_tile_ptr(k + 1), b_nxt, [BLOCK_K, BLOCK_N])

        # L1 → L0A / L0B for current tile
        tle_dsa.tile_copy(a_cur, mat_a_l0a, [BLOCK_M, BLOCK_K])
        tle_dsa.tile_copy(b_cur, mat_b_l0b, [BLOCK_K, BLOCK_N])

        # MMA on L0A × L0B → L0C; accumulate into mat_c_acc
        a_tensor = tle_dsa.tile_to_tensor(mat_a_l0a, writable=False)
        b_tensor = tle_dsa.tile_to_tensor(mat_b_l0b, writable=False)
        mat_c_acc = tl.dot(a_tensor, b_tensor, mat_c_acc, input_precision="ieee")

        tl.debug_barrier()  # ensure DMA into a_nxt/b_nxt completes before next iteration
```

### Step 5: Write result to global memory

```python
    # Store accumulator to global memory
    c_ptrs = tle_dsa.tile_gm_offset(
        c_ptr,
        indices=[pid_m * BLOCK_M, pid_n * BLOCK_N],
        strides=[N, 1],
    )
    # mat_c_acc is a tl.tensor; store directly
    tl.store(c_ptrs + tl.arange(0, BLOCK_M)[:, None] * N
                    + tl.arange(0, BLOCK_N)[None, :],
             mat_c_acc.to(tl.float16))
```

### Step 6: Launch

```python
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_double_buffer_kernel[grid](
        a, b, c, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

## What To Verify After Applying

- Compile to MLIR and confirm `tile.alloc` appears for both `_0` and `_1` L1 buffers.
- Run with K values that are not a multiple of BLOCK_K to confirm the `if k + 1 < num_k` guard works.
- Compare result to `torch.matmul` with atol=1e-2 for float16 accumulation.
- Check that `memref.alloc` and `bufferization.to_tensor` do not appear in the IR.

## Related Patterns

- **Vec Add** — no Cube needed; use for purely element-wise ops.
- **Three-Task Pipeline** — extend this pattern when Vector postprocessing (bias, activation, softmax) must fuse with the MMA in the same kernel at full throughput.
