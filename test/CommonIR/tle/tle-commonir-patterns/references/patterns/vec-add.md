---
priority: medium
---

# Vec Add — Single-Engine Vector Kernel (UB Only)

## Summary

The baseline TLE DSA kernel. Allocates Unified Buffer (UB) space for each operand and result, copies input tiles from global memory into UB, performs element-wise computation on UB buffers using `tle_dsa.add` (or other vector ops), then copies the result back to global memory. No Cube engine, no L1, no synchronization flags required.

## Use When

- The operation is purely element-wise: add, sub, mul, div, max, min, or a chain of them.
- No matrix multiplication is needed in this kernel.
- The per-program-id working set fits in UB.
- You want the simplest possible TLE DSA kernel as a starting point.

## Avoid When

- The kernel includes matrix multiply — use Double-Buffered Matmul instead.
- The output of a Cube MMA must feed into element-wise ops — use Three-Task Pipeline instead.
- The input tiles are too large for UB; tile the problem and loop instead.

## Pattern

### Step 1: Set up program-id and offsets

```python
import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as tle_dsa
from triton.experimental.tle.language.dsa import ascend

@triton.jit
def vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tail = tl.minimum(n_elements - pid * BLOCK, BLOCK)
```

`tail` limits the DMA copy to the actual valid elements in the last block.

### Step 2: Allocate UB buffers

```python
    x_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)
    y_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)
    z_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)
```

All three allocations must use `ascend.UB`. The size must be a `tl.constexpr` — `BLOCK` satisfies this.

### Step 3: DMA from global memory into UB

```python
    tle_dsa.copy(x_ptr + offsets, x_ub, [tail])
    tle_dsa.copy(y_ptr + offsets, y_ub, [tail])
```

`tail` (not `BLOCK`) is passed as the shape so the DMA does not read out-of-bounds on the last tile.

### Step 4: Compute on UB

```python
    tle_dsa.add(x_ub, y_ub, z_ub)
```

For a chain of ops, intermediate results need their own UB buffer:

```python
    tmp_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)
    tle_dsa.mul(x_ub, y_ub, tmp_ub)   # tmp = x * y
    tle_dsa.add(tmp_ub, z_ub, z_ub)   # z  += tmp
```

### Step 5: DMA result back to global memory

```python
    tle_dsa.copy(z_ub, out_ptr + offsets, [tail])
```

### Step 6: Launch the kernel

```python
def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 2048
    grid = (triton.cdiv(n, BLOCK),)
    vec_add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out
```

## What To Verify After Applying

- Compile the kernel to MLIR and check that `tile.alloc` is present and `memref.alloc` is absent.
- Run with `n_elements` that is not a multiple of `BLOCK` to confirm the tail-block path is correct.
- Compare output to a reference `torch.add` result to within float32 precision.

## Related Patterns

- **Double-Buffered Matmul** — add a Cube MMA stage before the Vector result.
- **Three-Task Pipeline** — when Cube + Vector must run concurrently with DMA.
