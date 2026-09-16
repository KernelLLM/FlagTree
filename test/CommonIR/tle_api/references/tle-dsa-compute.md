# TLE DSA Compute

> **Layer**: Tensor operations.
> **Source**: `tle.dsa.extract_slice.md`, `tle.dsa.insert_slice.md`, `tle.dsa.extract_element.md`

## 1. `tle.dsa.extract_slice` — extract a subtensor

```python
tle.dsa.extract_slice(
    ful,      # tl.tensor — source, must be ranked (rank ≥ 1)
    offsets,  # tuple — start offset per dimension
    sizes,    # tuple[int] — size per dimension (determines return shape)
    strides,  # tuple[int] — stride per dimension (≥ 0)
) -> tl.tensor
```

Returns a new tensor with shape equal to `sizes`. The source `ful` is a `tl.tensor`, not a buffer. `tl.constexpr` values in `offsets` are converted to tensors automatically.

```python
# Extract first 32 elements from a 1D tensor
out_sub = tle.dsa.extract_slice(out, (0,), (32,), (1,))
tl.store(out_ptr + tl.arange(0, 32), out_sub)

# Extract a single row from a 2D tensor
row = tle.dsa.extract_slice(x, (i, 0), (1, BLOCK_N), (1, 1))
row = tl.reshape(row, (BLOCK_N,))  # drop the leading size-1 dim

# Extract weight slice for tap j in a depthwise conv kernel
w_tile = tl.reshape(w_tile_2d, (KERNEL_WIDTH * BLOCK_N,))  # flatten first
w_j = tle.dsa.extract_slice(w_tile, (j * BLOCK_N,), (BLOCK_N,), (1,))
```

## 2. `tle.dsa.insert_slice` — insert a subtensor

```python
tle.dsa.insert_slice(
    ful,      # tl.tensor — destination tensor (ranked, rank ≥ 1)
    sub,      # tl.tensor — subtensor to insert (same rank as ful)
    offsets,  # tuple — insertion offset per dimension
    sizes,    # tuple[int] — size of the insertion region
    strides,  # tuple[int] — stride per dimension (≥ 0)
) -> tl.tensor
```

Returns a new tensor with the same shape as `ful`, not `sub`. Does not mutate `ful` in place — the return value must be captured.

```python
# Build a result tensor by inserting computed slices row by row
out = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
out = tle.dsa.insert_slice(out, out_sub, (SLICE_OFFSET,), (SLICE_SIZE,), (1,))
tl.store(out_ptr + offsets, out, mask=mask)

# Gather pattern: insert per-row values into a 2D accumulator
tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
for i in range(g_block_sub):
    val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
    tmp_buf = tle.dsa.insert_slice(
        tmp_buf, val[None, :],
        offsets=(i, 0), sizes=(1, other_block), strides=(1, 1),
    )
```

> `ful` and `sub` must have the same rank. `insert_slice` does not mutate `ful` — always assign the return value back.

## 3. `tle.dsa.extract_element` — extract a scalar

```python
tle.dsa.extract_element(
    src,     # tl.tensor — source, must be ranked
    indice,  # tuple — one index per dimension
) -> tl.tensor  # scalar (shape=None)
```

Returns a scalar tensor. The length of `indice` must match the rank of `src`. Raises `ValueError: Indice's rank must be equal to src tensor's rank` otherwise.

```python
# Use an index tensor to compute a gather offset
gather_offset = tle.dsa.extract_element(indices, (i,)) * g_stride
val = tl.load(in_ptr + gather_offset + other_idx, other_mask)

# Read a single element from a 2D tensor
value = tle.dsa.extract_element(x_2d, (0, 0))

# Use in CV inter-core pipeline: extract per-iteration index
k_cache_offset = tle.dsa.extract_element(index_value, (i,)) * HEAD_DIM
```

Use `extract_slice` when the result should be a ranked tensor; use `extract_element` only when a scalar is needed.

**Next**: [tle-dsa-ascend-advanced.md](tle-dsa-ascend-advanced.md) for compile hints, sub-vector ID, and inter-core CV pipeline sync.
