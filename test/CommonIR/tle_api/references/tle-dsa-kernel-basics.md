# TLE DSA Kernel Basics

> **Layer**: Shared — required reading before all other TLE DSA docs.
> **Source**: `FlagTree/documents/tle/tle.dsa.ops.md`, `FlagTree/documents/tle/ascend/tle.dsa.ascend.ops.md`

## Imports

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle

# Address spaces — always imported from this path
from triton.experimental.tle.language.dsa import ascend
# Equivalent: tle.dsa.ascend.UB, tle.dsa.ascend.L1, etc.
```

TLE DSA builtins live under `tle.dsa.*`. Every kernel is a standard `@triton.jit` function — there is no separate entry point or compilation step.

## Namespace mapping from community docs

The triton-ascend community documentation uses `triton.language.extra.cann.extension as al`. In TLE DSA the same interfaces live in `tle.dsa.ascend`. The APIs are identical; only the import path differs.

```python
# Community triton-ascend style
import triton.language.extra.cann.extension as al
al.sync_block_all("all", 0)
al.ascend_address_space.UB

# TLE DSA style
import triton.experimental.tle as tle
tle.dsa.ascend.sync_block_all("all", 0)
tle.dsa.ascend.UB
```

When reading community examples, mentally replace `al.` with `tle.dsa.ascend.` and `bl.alloc(...)` with `tle.dsa.alloc(...)`.

## Address spaces

Ascend 910B has five named on-chip address spaces. Pass one as `mem_addr_space` to `tle.dsa.alloc`.

| Constant | Hardware region | Who reads it | Who writes it |
|----------|----------------|--------------|---------------|
| `tle.dsa.ascend.UB` | Unified Buffer | Vector engine | DMA, Vector |
| `tle.dsa.ascend.L1` | L1 cache | Cube engine (via L0 fill) | DMA |
| `tle.dsa.ascend.L0A` | Cube left operand | Cube engine | DMA from L1 |
| `tle.dsa.ascend.L0B` | Cube right operand | Cube engine | DMA from L1 |
| `tle.dsa.ascend.L0C` | Cube accumulator | DMA to UB | Cube engine |

Typical data flow: `GM → L1 → L0A/L0B → (Cube MMA) → L0C → UB → (Vector) → GM`.

In IR, these map to: `#hivm.address_space<ub>`, `<cbuf>`, `<ca>`, `<cb>`, `<cc>`.

## Minimal runnable skeleton

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle

@triton.jit
def vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)

    # Allocate on-chip UB buffers
    a_ub = tle.dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    # Tail-safe copy extent
    tail = tl.minimum(n_elements - block_start, BLOCK)

    # GM → UB
    tle.dsa.copy(x_ptr + offsets, a_ub, [tail])
    tle.dsa.copy(y_ptr + offsets, b_ub, [tail])

    # Compute on UB
    tle.dsa.add(a_ub, b_ub, c_ub)

    # UB → GM
    tle.dsa.copy(c_ub, out_ptr + offsets, [tail])


# Launch
BLOCK = 2048
grid = (triton.cdiv(n, BLOCK),)
vec_add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
```

## Key rules

- All TLE DSA builtins can only be used inside `@triton.jit` functions.
- `tle.dsa.alloc` returns a `buffer`, not a `tl.tensor`. They are not interchangeable. Use `tle.dsa.to_tensor(buf)` to get a tensor view for standard Triton arithmetic.
- `mem_addr_space` is required and cannot be `None`.
- All shape arguments to `tle.dsa.alloc` must be `tl.constexpr` (compile-time constants). Runtime values are not accepted.

## OP table

| OP | Description |
|----|-------------|
| `tle.dsa.alloc` | Allocate a buffer in the given address space |
| `tle.dsa.copy` | Copy between GM pointer and buffer (bidirectional) |
| `tle.dsa.subview` | Create a zero-copy subview of an existing buffer |
| `tle.dsa.to_tensor` | Convert a buffer to a `tl.tensor` |
| `tle.dsa.to_buffer` | Convert a `tl.tensor` to a buffer |
| `tle.dsa.hint` | Pass `inter_no_alias` hint to `copy` calls in a scope |
| `tle.dsa.extract_slice` | Extract a subtensor from a `tl.tensor` |
| `tle.dsa.insert_slice` | Insert a subtensor into a `tl.tensor`, return new tensor |
| `tle.dsa.extract_element` | Extract a scalar element from a `tl.tensor` |
| `tle.dsa.parallel` | Loop iterator expressing independent iterations |
| `tle.dsa.ascend.UB/L1/L0A/L0B/L0C` | Address space constants |
| `tle.dsa.ascend.sub_vec_id` | Current vector sub-core ID (AIC/AIV mixed mode only) |
| `tle.dsa.ascend.sync_block_set` | Inter-core sync: set event flag |
| `tle.dsa.ascend.sync_block_wait` | Inter-core sync: wait for event flag |
| `tle.dsa.ascend.sync_block_all` | Inter-core global barrier |
| `tle.dsa.ascend.compile_hint` | Attach optimization hint to a tensor |

**Next**: [tle-dsa-memory.md](tle-dsa-memory.md)
