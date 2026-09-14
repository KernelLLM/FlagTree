---
priority: medium
---

# compile_hint Usage — Suppressing Compiler Transformations

## Summary

Five `compile_hint` annotations suppress specific Ascend compiler transformations that can silently inflate UB usage, hoist loop-body ops incorrectly, or generate inefficient instruction sequences. Apply them only when you observe the specific symptom each one addresses — they add coupling to compiler internals and should not be applied preemptively.

## Use When

- UB usage is unexpectedly large for a kernel with a scalar access pattern (K=1 or BLOCK_K=1).
- The compiler hoists Vector ops outside a loop (bubble-up), corrupting results in a scatter-write pattern.
- A `tl.where` condition uses an i1 tensor and UB usage or instruction count is higher than expected.
- A multi-matmul kernel (FlashAttention QKᵀ + PV) overflows L1 at moderate tile sizes.
- tl.dot padding is expanding to M and N dimensions unnecessarily when only K needs padding.

## Avoid When

- The kernel has no measurable performance problem — hints add fragile compiler coupling.
- The symptom is a correctness bug — hints do not fix logic errors.
- The symptom is a regular (non-scalar) memory access pattern — try tile size reduction first.

## Pattern

### `disable_bubble_up` — prevent loop-body hoisting

```python
from triton.experimental.tle.language.dsa.ascend import compile_hint

# Conditions: heavy computation inside loop, large N, simple store after loop,
# extract_slice with a reference defined outside the loop.
# Without the hint, the compiler may float the Vector ops above the loop.
for i in range(N):
    result = tle_dsa.extract_slice(big_tensor, offsets=[i*D], sizes=[D], strides=[1])
    compile_hint(result, "disable_bubble_up")   # attach to the slice, before compute
    processed = some_heavy_op(result)
    tle_dsa.copy(processed, out_ptr + i*D, [D])
```

### `mayDiscretememaccess` — prevent 32B alignment expansion for scalar accesses

```python
# Symptom: BLOCK_K=1 or K=1 causes UB to balloon by 8×.
# Cause: compiler expands <Nx1xf32> to 32B-aligned row access (8 float32 per row).
# Hint tells compiler to use discrete (non-aligned) access instead.

k_slice = tle_dsa.extract_slice(tensor, offsets=[0, k], sizes=[M, 1], strides=[1, 1])
compile_hint(k_slice, "mayDiscretememaccess")
val = tle_dsa.to_tensor(k_slice, writable=False)
```

### `dot_pad_only_k` — restrict tl.dot padding to K dimension

```python
# Symptom: tl.dot pads both K and M/N dimensions, causing L1 overflow
# when M and N are already aligned to hardware requirements.
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
compile_hint(acc, "dot_pad_only_k")
acc = tl.dot(a_t, b_t, acc, input_precision="ieee")
```

### `hivm.tile_mix_cube_num` — sub-tile L1 for multi-matmul kernels

```python
# For kernels with two matmuls (e.g., attention QKᵀ and PV):
# hint_val controls sub-tiling factor; reduces effective L1 per matmul.
# Use when L1 overflows despite reasonable BLOCK_M/N/K choices.
o_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
compile_hint(o_acc, "hivm.tile_mix_cube_num", 4)  # sub-tile by factor 4
o_acc = tl.dot(p_t, v_t, o_acc)
```

### `bitwise_mask` — skip vcast+vcmp+vnot chain for i1 conditions

```python
# Symptom: tl.where with an i1 mask generates vcast(i8→f16) + vcmp + vnot,
# which is inefficient. This hint tells the compiler to use a direct bitwise mask.
mask = offsets < n_elements   # produces i1 tensor
compile_hint(mask, "bitwise_mask")
result = tl.where(mask, x_t, 0.0)
```

## What To Verify After Applying

- The specific symptom (UB overflow, bubble-up, padding) is reduced or eliminated.
- Output is still numerically correct after adding the hint — compare to PyTorch reference.
- The hint is attached to the right tensor (the slice, the accumulator, or the condition, depending on the hint type).

## Related Patterns

- **Tile Size Selection** — reduce tile sizes first before reaching for hints.
- **Autotune** — after hints stabilize UB/L1, run autotune to find the best tile sizes within the new budget.
