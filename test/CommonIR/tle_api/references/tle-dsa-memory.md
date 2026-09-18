# TLE DSA Memory

> **Layer**: Core — used by every TLE DSA kernel.
> **Source**: `tle.dsa.alloc.md`, `tle.dsa.copy.md`, `tle.dsa.subview.md`, `tle.dsa.to_tensor.md`, `tle.dsa.to_buffer.md`, `tle.dsa.hint.md`

## 1. `tle.dsa.alloc` — allocate an on-chip buffer

```python
tle.dsa.alloc(
    shape,             # list[tl.constexpr] or tuple — compile-time only
    dtype,             # tl.float32 / tl.float16 / tl.bfloat16 / etc.
    mem_addr_space,    # tle.dsa.ascend.UB | L1 | L0A | L0B | L0C
) -> buffer
```

Returns a `buffer` object. This is not a `tl.tensor` and cannot be used in standard Triton arithmetic directly. The buffer carries element type, shape, and address space.

```python
# 1D UB buffer
a_ub = tle.dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

# 2D L1 buffer for a matmul tile
a_l1 = tle.dsa.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.L1)
```

> L0A, L0B, L0C cannot be manually allocated via `tle.dsa.alloc`. L0 memory is managed by the compiler.

> `mem_addr_space` is required and cannot be `None`. `shape` must be compile-time constants — runtime tensor values are not accepted.

## 2. `tle.dsa.copy` — bidirectional DMA

```python
tle.dsa.copy(
    src,                  # tl.tensor (GM pointer) | buffer
    dst,                  # buffer | tl.tensor (GM pointer)
    shape,                # list[int | tl.constexpr | tl.tensor] — explicit copy extent
    inter_no_alias=False, # True: iterations don't alias, enables coalescing
) -> None
```

Direction is auto-detected from the types of `src` and `dst`:
- `tl.tensor → buffer`: GM load into on-chip buffer
- `buffer → tl.tensor`: on-chip buffer store to GM
- `to_buffer() result → tl.tensor`: tensor computation result to GM

`shape` is always required — the copy extent is never inferred from buffer shapes.

```python
tail = tl.minimum(n_elements - pid * BLOCK, BLOCK)

# GM → UB (1D)
tle.dsa.copy(x_ptr + offsets, a_ub, [tail])

# UB → GM (1D)
tle.dsa.copy(c_ub, out_ptr + offsets, [tail])

# Write tensor result back to GM via to_buffer
result = c_val - b_val                           # tl.tensor arithmetic
d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
tle.dsa.copy(d_ub, out_ptr + offsets, [tail])
```

> `tle.dsa.copy` does not support masks. Use it only for full tiles. For boundary tiles with partial data, fall back to `tl.store` with a mask.

## 3. `tle.dsa.subview` — zero-copy subview

```python
tle.dsa.subview(
    src,      # buffer
    offsets,  # list[int | tl.constexpr | tl.tensor] — one per dimension
    sizes,    # list[int | tl.constexpr] — subview shape
    strides,  # list[int | tl.constexpr] — stride per dimension (1 = contiguous)
) -> buffer
```

Returns a new buffer pointing into `src`'s memory without copying. The returned buffer has dtype and address space from `src`; its shape equals `sizes`.

`sizes` and `strides` accept only plain integers or `tl.constexpr`. Tensor values are not accepted. `offsets` may be tensors.

```python
# Split a [2*BLOCK_M, BLOCK_K] UB buffer into two slots for ping-pong
a_ub = tle.dsa.alloc([2*BLOCK_M, BLOCK_K], tl.float16, tle.dsa.ascend.UB)

a_ub_0 = tle.dsa.subview(a_ub, offsets=[0,       0], sizes=[BLOCK_M, BLOCK_K], strides=[1, 1])
a_ub_1 = tle.dsa.subview(a_ub, offsets=[BLOCK_M, 0], sizes=[BLOCK_M, BLOCK_K], strides=[1, 1])
```

> `subview` on L1 buffers is not currently supported — only UB buffers support subview. Do not allocate two separate `tle.dsa.alloc` calls with identical shape and dtype for double-buffering; the compiler (`commonir_to_hivm`) merges them into a single physical buffer, causing silent wrong results. Use `subview` on a single larger UB allocation instead.

## 4. `tle.dsa.to_tensor` — buffer to `tl.tensor`

```python
tle.dsa.to_tensor(
    memref,        # buffer
    writable=True, # False for Cube inputs; True when the result will be modified
    target_shape=None,  # list[int] — if set, triggers a layout conversion
) -> tl.tensor
```

Required whenever a buffer must be passed to `tl.dot`, standard Triton arithmetic, or `tl.store`.

```python
a_tensor = tle.dsa.to_tensor(a_l0a, writable=False)   # Cube left input
b_tensor = tle.dsa.to_tensor(b_l0b, writable=False)   # Cube right input
acc      = tle.dsa.to_tensor(c_l0c, writable=True)    # L0C accumulator

result = tl.dot(a_tensor, b_tensor, acc, input_precision="ieee")
```

`target_shape`, when given, must differ from `memref.shape`. Whether a specific transformation is valid depends on the underlying `create_convert_layout` implementation.

## 5. `tle.dsa.to_buffer` — `tl.tensor` to buffer

```python
tle.dsa.to_buffer(
    tensor,       # tl.tensor — must be non-scalar (rank ≥ 1)
    space,        # tle.dsa.ascend.UB | L1 | etc.
    bind_buffer=None,  # must be None (binding not currently supported)
) -> buffer
```

Converts a tensor computation result into a buffer so it can be passed to `tle.dsa.copy` for writing to GM. Does not copy data — it wraps the tensor value as a buffer representation.

```python
c_val = tle.dsa.to_tensor(c_ub)
b_val = tle.dsa.to_tensor(b_ub)
result = c_val - b_val              # standard tl.tensor arithmetic

d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
tle.dsa.copy(d_ub, out_ptr + offsets, [tail])
```

The typical round-trip: `buffer → to_tensor → tl.tensor ops → to_buffer → buffer → copy → GM`.

## 6. `tle.dsa.hint` — per-scope `inter_no_alias`

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src, dst, [size])   # inter_no_alias applies here
```

A context manager that passes `inter_no_alias=True` to all `tle.dsa.copy` calls in the scope. Tells the compiler that copies from different loop iterations do not alias, enabling more aggressive DMA coalescing. Equivalent to passing `inter_no_alias=True` directly to `copy`.

If `tle.dsa.copy` is called with an explicit `inter_no_alias` argument inside the `with` block, that explicit value takes precedence.

```python
with tle.dsa.hint(inter_no_alias=True):
    for i in range(BLOCK_SIZE_TOKEN):
        offset_i = i
        if offset_i < token_num:
            k_cache_offset = tle.dsa.extract_element(index_value, (i,)) * HEAD_DIM
            res_buf = tle.dsa.to_buffer(reload_result, tle.dsa.ascend.UB)
            tle.dsa.copy(res_buf, k_cache_ptr + k_cache_offset + row_ids, [HEAD_DIM])
```

`hint` is a compile-time AST construct, not a runtime call. Keyword argument values must be Python constants (`True`/`False`), not runtime variables. Only `inter_no_alias` is currently applied.

**Next**: [tle-dsa-compute.md](tle-dsa-compute.md)
