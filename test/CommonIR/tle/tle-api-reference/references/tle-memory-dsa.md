# TLE Memory — DSA Path (Ascend UB / L1 / L0)

> **Layer**: DSA path only. Targets Ascend 910B and compatible hardware.
> **Source**: `FlagTree/python/triton/experimental/tle/language/dsa/core.py`, `dsa/types.py`, `dsa/ascend/core.py`

This document covers on-chip buffer allocation, address spaces, and data movement for the DSA (Ascend) TLE path.

## 1. Address space constants

The Ascend memory hierarchy has five named on-chip spaces, all under `tle.language.dsa.ascend`:

```python
from triton.experimental.tle.language.dsa import ascend

ascend.UB    # Unified Buffer — vector engine scratchpad (read/write by vector ops)
ascend.L1    # L1 cache — shared between Cube and Vector; DMA target for input tiles
ascend.L0A   # Cube left operand registers (A matrix)
ascend.L0B   # Cube right operand registers (B matrix)
ascend.L0C   # Cube accumulator registers (C / output matrix)
```

Data flows: `GM → L1 → L0A/L0B → (Cube) → L0C → (move) → UB → (Vector) → UB → GM`.

| Space | Who reads it | Who writes it | Typical use |
|-------|-------------|---------------|-------------|
| UB | Vector engine | DMA, Vector | Activation, elementwise ops, output staging |
| L1 | Cube engine (via L0 fill) | DMA | Input tile double-buffer |
| L0A | Cube engine | DMA from L1 | Left matmul operand |
| L0B | Cube engine | DMA from L1 | Right matmul operand |
| L0C | DMA to UB | Cube engine | MMA accumulator |

## 2. `tle_dsa.alloc` — allocate an on-chip buffer

```python
buf = tle_dsa.alloc(
    shape,             # list[int | tl.constexpr]
    dtype,             # tl.float16 / tl.float32 / tl.bfloat16 / etc.
    mem_addr_space,    # ascend.UB | ascend.L1 | ascend.L0A | ascend.L0B | ascend.L0C
)
# Returns: tle.language.dsa.buffer
```

Allocation is static (compile-time shape only). Do not use Python variables that are not `tl.constexpr` as shape arguments.

```python
# UB buffer for vector result
z_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)

# L1 buffers for double-buffered matmul operands
a_l1_0 = tle_dsa.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, mem_addr_space=ascend.L1)
a_l1_1 = tle_dsa.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, mem_addr_space=ascend.L1)

# L0A / L0B / L0C for Cube MMA
a_l0a = tle_dsa.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, mem_addr_space=ascend.L0A)
b_l0b = tle_dsa.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, mem_addr_space=ascend.L0B)
c_l0c = tle_dsa.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, mem_addr_space=ascend.L0C)
```

## 3. `tle_dsa.copy` — bidirectional DMA

```python
tle_dsa.copy(
    src,                  # global pointer (tl.tensor) | tle.buffer
    dst,                  # tle.buffer | global pointer (tl.tensor)
    shape,                # list[int] — number of elements per dimension to transfer
    inter_no_alias=False, # set True if src and dst cannot alias (enables DMA coalescing)
)
```

Direction is auto-detected from the type of `src` and `dst`:
- `tl.tensor → buffer`: global memory → on-chip (load)
- `buffer → tl.tensor`: on-chip → global memory (store)
- `buffer → buffer`: on-chip to on-chip (e.g., L1 → L0A)

```python
tail = tl.minimum(n - pid * BLOCK, BLOCK)

# GM → UB
tle_dsa.copy(x_ptr + offsets, x_ub, [tail])

# GM → L1 with a strided global pointer
a_gm_ptr = tle_dsa.tile_gm_offset(a_ptr, indices=[m_off, k_off], strides=[K, 1])
tle_dsa.copy(a_gm_ptr, a_l1, [BLOCK_M, BLOCK_K])

# UB → GM
tle_dsa.copy(z_ub, out_ptr + offsets, [tail])
```

## 4. `tle_dsa.subview` — strided subview into a buffer

```python
sub = tle_dsa.subview(
    src,       # tle.buffer
    offsets,   # list[int | tl.tensor] — one per dimension
    sizes,     # list[int] — tile size in each dimension
    strides,   # list[int] — step in each dimension (1 = contiguous)
)
# Returns: tle.buffer (a view, not a copy)
```

Use `subview` to tile a larger on-chip buffer without re-allocating:

```python
# Split a BLOCK_M×(2*BLOCK_K) L1 buffer into two halves
left  = tle_dsa.subview(ab_l1, offsets=[0, 0],       sizes=[BLOCK_M, BLOCK_K], strides=[1, 1])
right = tle_dsa.subview(ab_l1, offsets=[0, BLOCK_K], sizes=[BLOCK_M, BLOCK_K], strides=[1, 1])
```

## 5. `tle_dsa.to_tensor` — read buffer as `tl.tensor`

```python
tensor = tle_dsa.to_tensor(
    buf,           # tle.buffer
    writable,      # bool — False for read-only (Cube inputs); True for writeable (accumulator)
    target_shape,  # list[int] | None — if None, uses buf.shape
)
# Returns: tl.tensor
```

Required whenever a buffer must be passed to `tl.dot` or standard Triton arithmetic:

```python
a_tensor = tle_dsa.to_tensor(a_l0a, writable=False)
b_tensor = tle_dsa.to_tensor(b_l0b, writable=False)
acc      = tle_dsa.to_tensor(c_l0c, writable=True)
result   = tl.dot(a_tensor, b_tensor, acc)
```

## 6. `tle_dsa.to_buffer` — write tensor result into a new buffer

```python
buf = tle_dsa.to_buffer(
    tensor,       # tl.tensor
    space,        # ascend.UB | ascend.L1 | etc.
    bind_buffer,  # tle.buffer — pre-allocated target (avoids re-allocation)
)
```

## 7. `tle_dsa.tile_gm_offset` — multi-dimensional global memory pointer

```python
ptr = tle_dsa.tile_gm_offset(
    base,     # tl.tensor — global base pointer
    indices,  # list[int | tl.tensor] — per-dimension tile indices
    strides,  # list[int] — per-dimension strides (in elements)
)
# Returns: tl.tensor — computed GM pointer
```

```python
a_ptr = tle_dsa.tile_gm_offset(a_base, indices=[m_off, k_off], strides=[stride_m, stride_k])
tle_dsa.copy(a_ptr, a_l1, [BLOCK_M, BLOCK_K])
```

## 8. Lower-level tile_* ops

When working at the CommonIR level (see `tle-commonir-patterns` skill), use `tile_alloc`, `tile_copy`, `tile_subview`, and `tile_to_tensor` directly. These are identical in semantics to `alloc`, `copy`, `subview`, and `to_tensor` but generate `tile.*` MLIR ops unconditionally, bypassing any high-level path selection.

**Next**: [tle-compute-dsa.md](tle-compute-dsa.md) for vector ops, Cube launch, and synchronization.
