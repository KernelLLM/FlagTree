---
priority: high
---

# Three-Task Pipeline — DMA + Cube + Vector Ring

## Summary

A producer-consumer ring that keeps all three Ascend 910B engines (DMA/MTE2, Cube, Vector) busy concurrently. MTE2 fills L1 buffers, Cube reads L0A/B and accumulates into L0C, Vector reads the completed L0C (via a UB copy) and writes the final result to UB or global memory. `tile_set_flag`/`tile_wait_flag` pairs at each engine boundary coordinate handshakes without unnecessary stalls.

## Use When

- The kernel requires Cube MMA followed by Vector postprocessing in the same kernel (matmul + bias + activation, attention QK product + softmax, etc.).
- Peak throughput requires all three engines to overlap execution across K-loop iterations.
- The K dimension has enough iterations (≥ 4) to amortize the pipeline fill and drain overhead.

## Avoid When

- The kernel is purely element-wise — use the Vec Add pattern instead.
- The kernel does GEMM with no Vector fusion — use Double-Buffered Matmul instead.
- K is very small (< 64); the flag overhead dominates.

## Pattern

### Step 1: Define pipe and event constants

```python
import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as tle_dsa
from triton.experimental.tle.language.dsa import ascend
from triton.experimental.tle.language.dsa.ascend import PIPE

# Event IDs — must be unique per handshake direction in the pipeline.
# Use distinct IDs for each producer→consumer pair.
EVT_MTE2_TO_CUBE = 0   # MTE2 signals that L1 is ready for Cube
EVT_CUBE_TO_VEC  = 1   # Cube signals that L0C is ready for Vector
EVT_VEC_TO_MTE2  = 2   # Vector signals that UB/L1 space is free for next DMA
```

The three event IDs must be distinct. Reusing an ID for two different handshakes silently produces incorrect synchronization.

### Step 2: Allocate buffers for each engine stage

```python
@triton.jit
def three_task_matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # DMA stage: L1 double buffers for A and B
    a_l1_0 = tle_dsa.tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
    a_l1_1 = tle_dsa.tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
    b_l1_0 = tle_dsa.tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L1)
    b_l1_1 = tle_dsa.tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L1)

    # Cube stage: L0 operands and accumulator
    a_l0a  = tle_dsa.tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L0A)
    b_l0b  = tle_dsa.tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L0B)
    c_l0c  = tle_dsa.tile_alloc([BLOCK_M, BLOCK_N], tl.float32, ascend.L0C)

    # Vector stage: UB buffer for postprocessing
    c_ub   = tle_dsa.alloc([BLOCK_M, BLOCK_N], tl.float32, ascend.UB)
    bias_ub = tle_dsa.alloc([BLOCK_N], tl.float32, ascend.UB)
```

### Step 3: Prologue — fill first DMA stage and signal readiness

```python
    num_k = tl.cdiv(K, BLOCK_K)

    # Helper to compute tile GM pointers
    def a_gptr(k): return tle_dsa.tile_gm_offset(a_ptr, [pid_m*BLOCK_M, k*BLOCK_K], [K, 1])
    def b_gptr(k): return tle_dsa.tile_gm_offset(b_ptr, [k*BLOCK_K, pid_n*BLOCK_N], [N, 1])

    # Prologue: DMA fills buffer _0, then tells Cube it is ready
    tle_dsa.tile_copy(a_gptr(0), a_l1_0, [BLOCK_M, BLOCK_K])
    tle_dsa.tile_copy(b_gptr(0), b_l1_0, [BLOCK_K, BLOCK_N])
    tle_dsa.tile_set_flag(PIPE.MTE2, PIPE.M, EVT_MTE2_TO_CUBE)

    # Load bias into UB (once, outside the K-loop)
    bias_gptr = tle_dsa.tile_gm_offset(bias_ptr, [pid_n*BLOCK_N], [1])
    tle_dsa.copy(bias_gptr, bias_ub, [BLOCK_N])
```

### Step 4: Main loop — three concurrent stages per iteration

```python
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(num_k):
        a_cur = a_l1_0 if k % 2 == 0 else a_l1_1
        b_cur = b_l1_0 if k % 2 == 0 else b_l1_1
        a_nxt = a_l1_1 if k % 2 == 0 else a_l1_0
        b_nxt = b_l1_1 if k % 2 == 0 else b_l1_0

        # ── DMA stage: prefetch next tile ──────────────────────────────────
        if k + 1 < num_k:
            # Wait until Vector has freed the L1 space (only needed after first iter)
            if k > 0:
                tle_dsa.tile_wait_flag(PIPE.V, PIPE.MTE2, EVT_VEC_TO_MTE2)
            tle_dsa.tile_copy(a_gptr(k + 1), a_nxt, [BLOCK_M, BLOCK_K])
            tle_dsa.tile_copy(b_gptr(k + 1), b_nxt, [BLOCK_K, BLOCK_N])
            tle_dsa.tile_set_flag(PIPE.MTE2, PIPE.M, EVT_MTE2_TO_CUBE)

        # ── Cube stage: wait for L1, run MMA, signal Vector ────────────────
        tle_dsa.tile_wait_flag(PIPE.MTE2, PIPE.M, EVT_MTE2_TO_CUBE)

        # L1 → L0A / L0B
        tle_dsa.tile_copy(a_cur, a_l0a, [BLOCK_M, BLOCK_K])
        tle_dsa.tile_copy(b_cur, b_l0b, [BLOCK_K, BLOCK_N])

        # MMA: accumulate into acc
        a_t = tle_dsa.tile_to_tensor(a_l0a, writable=False)
        b_t = tle_dsa.tile_to_tensor(b_l0b, writable=False)
        acc = tl.dot(a_t, b_t, acc, input_precision="ieee")

        # On last K tile, signal Vector that L0C accumulator is complete
        if k == num_k - 1:
            tle_dsa.tile_set_flag(PIPE.M, PIPE.V, EVT_CUBE_TO_VEC)
```

> **Note**: The `if k == num_k - 1` guard ensures the Cube-to-Vector signal fires only once, after the final accumulation. Firing it every iteration would consume an event slot and cause incorrect synchronization.

### Step 5: Vector stage — wait for Cube, postprocess, write output

```python
    # ── Vector stage: wait for Cube, apply bias + activation ──────────────
    tle_dsa.tile_wait_flag(PIPE.M, PIPE.V, EVT_CUBE_TO_VEC)

    # Move accumulator result to UB for Vector ops
    tle_dsa.copy(acc, c_ub, [BLOCK_M, BLOCK_N])   # L0C tensor → UB buffer

    # Broadcast bias across rows and add (illustrative — real kernels may use to_tensor)
    # For a direct Vector-engine op, use tile_to_tensor + standard tl.* arithmetic:
    acc_t   = tle_dsa.to_tensor(c_ub, writable=True)
    bias_t  = tle_dsa.to_tensor(bias_ub, writable=False)
    result  = acc_t + bias_t[None, :]              # broadcast bias over M rows
    result  = tl.maximum(result, 0.0)              # ReLU

    # Signal DMA that this iteration's L1 buffers are now free
    tle_dsa.tile_set_flag(PIPE.V, PIPE.MTE2, EVT_VEC_TO_MTE2)

    # Store to global memory
    c_ptrs = tle_dsa.tile_gm_offset(c_ptr, [pid_m*BLOCK_M, pid_n*BLOCK_N], [N, 1])
    tl.store(c_ptrs + tl.arange(0, BLOCK_M)[:, None] * N
                    + tl.arange(0, BLOCK_N)[None, :],
             result.to(tl.float16))
```

### Step 6: Summary of flag pairs

| set_flag call | wait_flag call | Purpose |
|---------------|----------------|---------|
| `MTE2 → M, EVT_MTE2_TO_CUBE` | `MTE2 → M, EVT_MTE2_TO_CUBE` | L1 filled; Cube may read L0A/B |
| `M → V, EVT_CUBE_TO_VEC` | `M → V, EVT_CUBE_TO_VEC` | Accumulator complete; Vector may postprocess |
| `V → MTE2, EVT_VEC_TO_MTE2` | `V → MTE2, EVT_VEC_TO_MTE2` | UB/L1 consumed; DMA may overwrite with next tile |

Every `set_flag` has exactly one matching `wait_flag` with the same `(producer_pipe, consumer_pipe, event_id)` triple. Adding extra waits or mismatching the IDs produces silent hangs or incorrect results.

## What To Verify After Applying

- Each `tile_set_flag` has a matching `tile_wait_flag` with identical pipe and event arguments.
- Event IDs 0, 1, 2 are each used by exactly one set/wait pair.
- Compile to MLIR; check `tile.set_flag` and `tile.wait_flag` appear, and that `memref.alloc` does not appear.
- Run with a K value that is not a multiple of BLOCK_K to confirm the `k + 1 < num_k` guard is correct.
- Compare output to `torch.addmm` + ReLU reference within atol=1e-2.

## Related Patterns

- **Double-Buffered Matmul** — simpler version without the Vector stage; use when no post-MMA fusion is needed.
- **Vec Add** — purely element-wise; no Cube engine involved.
