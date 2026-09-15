# TLE Repair Experience — Ascend DSA

Real bugs extracted from FlagTree `success_case/`, `failed_case/`, and `next_fix_error.md`. Each entry maps a symptom to the smallest fix.

---

## 1. Two `tile_alloc` calls with identical shape/dtype get merged into one physical buffer

**Symptom**: Double-buffered kernel produces wrong results silently. No crash, no compile error. The double-buffer loop structure looks correct in Python but outputs are numerically wrong. IR contains only one `tile.alloc` where two were expected.

**Diagnosis**: `commonir_to_hivm` merges two `tile.alloc` calls that have the same shape and dtype into a single physical `cbuf` allocation. Both ping-pong variables (`mat_a_l1_0` and `mat_a_l1_1`) refer to the same memory region, so writing the "next" tile immediately corrupts the "current" tile.

**Fix**: Use `tile_subview` on a single larger buffer instead of two separate `tile_alloc` calls. Allocate `[2*BLOCK_M, BLOCK_K]` once, then derive the two slots as subviews:

```python
# Wrong — compiler merges these into one physical cbuf
mat_a_l1_0 = tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
mat_a_l1_1 = tile_alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)

# Correct — one allocation, two statically-offset subviews
mat_a_l1 = tle_dsa.tile_alloc([2*BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
mat_a_l1_0 = tle_dsa.tile_subview(mat_a_l1, [0,       0], [BLOCK_M, BLOCK_K], [1, 1])
mat_a_l1_1 = tle_dsa.tile_subview(mat_a_l1, [BLOCK_M, 0], [BLOCK_M, BLOCK_K], [1, 1])
```

**Cross-ref**: `tle-commonir-patterns/references/patterns/double-buffer-matmul.md`

**Real case**: `success_case/matmul_double_buffer.py` has a warning comment: "commonir_to_hivm merges tile.alloc of identical shape/dtype into a single physical cbuf allocation... This kernel produces WRONG results." `success_case/native_matmul_dsa_slice.py` is the correct fix — uses a doubled allocation with static offsets.

---

## 2. `tile_subview` with runtime (non-constexpr) offsets fails or produces wrong results

**Symptom**: Kernel with a rotating 3-slot buffer fails to compile, produces an error about dynamic offsets, or silently accesses the wrong memory region. The slot index is computed as `k_idx % NUM_SLOT` inside a regular `range()` loop.

**Diagnosis**: `tile_subview` requires offsets to be compile-time constants (`tl.constexpr`). When the offset is a runtime value (e.g., `s * BLOCK_M` where `s = k_idx % 3`), the compiler cannot lower it to a static cbuf address and either rejects it or selects slot 0 for every access.

**Fix**: Use `tl.static_range` instead of `range()` so the loop variable is unrolled and folds to a compile-time constant:

```python
# Wrong — s is a runtime value, tile_subview offset is dynamic
for k_idx in range(NUM_K):
    s = k_idx % 2
    cur = tle_dsa.tile_subview(mat_a_l1, [s * BLOCK_M, 0], [BLOCK_M, BLOCK_K], [1, 1])

# Correct — static_range unrolls so s_cur is constexpr
for k_idx in tl.static_range(NUM_K):
    s_cur = k_idx % NUM_SLOT   # folds to a compile-time constant after unrolling
    cur = tle_dsa.tile_subview(mat_a_l1, [s_cur * BLOCK_M, 0], [BLOCK_M, BLOCK_K], [1, 1])
```

**Cross-ref**: `tle-commonir-patterns/references/patterns/double-buffer-matmul.md`

**Real case**: `failed_case/matmul_prefetch.py` and `failed_case/matmul_triton.py` both fail because they use `range()` with runtime slot offsets. `success_case/native_matmul_dsa_slice.py` fixes this by using `tl.static_range`.

---

## 3. `tile_copy` + `tile_to_tensor` on a buffer that feeds the second matmul produces a 4D cbuf

**Symptom**: bishengir-compile sees a `4D cbuf memref` and fails. Error appears during the `commonir_to_hivm` pass, not during Python compilation. The kernel involved has two matmuls (e.g., FlashAttention QKᵀ and PV).

**Diagnosis**: When a buffer is filled via `tile_copy` and then read via `tile_to_tensor` and this pattern appears twice in the same kernel (once per matmul), the compiler's cbuf inference sometimes produces a 4D memref shape for the second operand.

**Fix**: For the second matmul's V operand, use plain `tl.load` on a global pointer instead of `tile_copy` + `tile_to_tensor`:

```python
# Causes 4D cbuf issue on the PV matmul
v_l1 = tile_alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L1)
tile_copy(v_ptr, v_l1, [BLOCK_K, BLOCK_N])
v_t = tile_to_tensor(v_l1, writable=False)
p_v_acc = tl.dot(p_t, v_t, p_v_acc)

# Correct: use tl.load directly for the second operand
v_t = tl.load(v_ptr + offsets)
p_v_acc = tl.dot(p_t, v_t, p_v_acc)
```

**Real case**: `failed_case/fa_cv.py` has the comment: "Using tile_copy+tile_to_tensor for either operand causes bishengir-compile to see a 4D cbuf memref." `failed_case/fa_serial.py` uses plain `tl.load` for `_mm2_pv` as the fix.

---

## 4. Wrong sync namespace: `tl.sync_block_set` vs `tle.dsa.ascend.sync_block_set`

**Symptom**: Cross-core synchronization produces no errors at compile time but the kernel hangs at runtime, or different AI cores race and produce wrong results. The sync ops appear to have no effect.

**Diagnosis**: `tl.sync_block_set` / `tl.sync_block_wait` (called directly on `tl.*`) do not resolve to the correct HIVM IR sync ops. The correct namespace is `tle.dsa.ascend.sync_block_set` / `tle.dsa.ascend.sync_block_wait`.

Note: `sync_block_set/wait` signature is `(sender, receiver, event_id, sender_pipe, receiver_pipe)` — 5 arguments. This is different from the intra-core `tile_set_flag(producer_pipe, consumer_pipe, event_id)` — 3 arguments.

```python
# Wrong
tl.sync_block_set(cube_core, vec_core, 0)

# Correct — cross-core sync
from triton.experimental.tle.language.dsa.ascend import sync_block_set, sync_block_wait, PIPE
sync_block_set(sender=cube_core, receiver=vec_core, event_id=0,
               sender_pipe=PIPE.M, receiver_pipe=PIPE.V)
sync_block_wait(sender=cube_core, receiver=vec_core, event_id=0,
                sender_pipe=PIPE.M, receiver_pipe=PIPE.V)
```

**Cross-ref**: `tle-api-reference/references/tle-compute-dsa.md §4`

**Real case**: `failed_case/fa_4func.py` uses `tl.sync_block_set/wait` directly — listed as a failed case. `success_case/matmul_add_residual_cv.py` uses `tle.dsa.ascend.sync_block_set` via scope wrappers — listed as a success case.

---

## 5. UB overflow for large tile sizes

**Symptom**: Kernel crashes with a UB overflow error, or the compiler rejects the kernel with a message about on-chip buffer capacity exceeded. Tile sizes that work at BLOCK=32 fail at BLOCK=64 or BLOCK=128.

**Diagnosis**: The Ascend 910B UB (Unified Buffer) is 192 KB. All UB allocations across the kernel must fit simultaneously. For float16, BLOCK_M=128, BLOCK_N=128 requires ~128×128×2 bytes ≈ 32 KB per buffer. Three such buffers (Q/K/V tiles) plus accumulators can easily exceed 192 KB.

**Calculation formula**:
```
UB bytes = sum(BLOCK_M × BLOCK_N × element_bytes) for each tle_dsa.alloc with ascend.UB
L1 bytes = sum(shape × element_bytes) for each tile_alloc with ascend.L1
```

- UB limit: 192 KB (196,608 bytes)
- L1 limit: ~512 KB (depends on hardware configuration)

**Fix**: Reduce tile sizes, reduce the number of simultaneously live UB buffers, or move intermediate results to L1 instead of UB:

```python
# Check before writing: does this fit?
ub_usage = BLOCK_M * BLOCK_K * 2  # a_ub (float16)
ub_usage += BLOCK_K * BLOCK_N * 2  # b_ub
ub_usage += BLOCK_M * BLOCK_N * 4  # c_ub (float32 accumulator)
assert ub_usage <= 192 * 1024, f"UB overflow: {ub_usage} bytes"
```

**Real case**: `next_fix_error.md` documents that FA at BLOCK=64 requires 5.3 Mbit and BLOCK=128 requires 21 Mbit — both far exceed the 192 KB UB limit. Only BLOCK=32 fits.

---

## 6. `dump_commonir` uses wrong dialect load call

**Symptom**: `dump_commonir()` fails with `AttributeError: module 'tle_ir' has no attribute 'load_tile_dialects'`, or the emitted IR contains `memref.alloc` instead of `tile.alloc`, suggesting the tile dialect was not registered.

**Diagnosis**: The correct call is `tle_ir.dsa_ir.load_tile_dialects(context)`. The older `tle_ir.load_tile_dialects(context)` does not register the `tile.*` dialect ops.

**Fix**:
```python
# Wrong
import tle_ir
ctx = mlir.MLIRContext()
tle_ir.load_tile_dialects(ctx)

# Correct
import tle_ir.dsa_ir
ctx = mlir.MLIRContext()
tle_ir.dsa_ir.load_tile_dialects(ctx)
```

**Real case**: `failed_case/fa_4func.py` uses the old `tle_ir.load_tile_dialects`. All success case files (`native_matmul.py`, `matmul_add_residual_cv.py`, etc.) use `tle_ir.dsa_ir.load_tile_dialects`.

---

## 7. Python runtime conditionals inside `@triton.jit` cause unexpected behavior

**Symptom**: A loop guard like `if k == num_k - 1: tile_set_flag(...)` fires on every iteration instead of only the last, or never fires at all. Or a `if k + 1 < num_k` guard that appears to work in Python silently behaves differently in the JIT.

**Diagnosis**: In `@triton.jit`, Python `if` statements on values that are not `tl.constexpr` are evaluated at trace time, not at runtime. If `k` is a loop variable from `range()` (not `tl.static_range`), the condition `k == num_k - 1` is traced with a symbolic value and may not fold correctly.

**Fix options**:
1. Use `tl.static_range` so loop variables are constexpr and conditions fold at compile time.
2. Structure the loop so the last-iteration logic is in an explicit epilogue outside the loop, not guarded inside it.
3. Use `tl.where` for runtime conditional selection on tensor values (not for control flow).

```python
# Risky: Python if on loop variable
for k in range(num_k):
    if k == num_k - 1:
        tile_set_flag(...)  # may not behave as expected

# Safe: epilogue outside loop
for k in range(num_k - 1):
    ...
# epilogue
tile_set_flag(...)
```

---

## 8. `compile_hint` required for UB expansion and bubble-up issues

**Symptom**: A kernel with `tl.where` on an i1 condition tensor uses unexpectedly large UB. Or a `T.extract_slice` scatter-write pattern inside a loop has Vector ops incorrectly hoisted outside the loop by the compiler ("bubble-up"), corrupting results.

**Diagnosis**: The Ascend compiler applies aggressive code motion. Specific patterns need `compile_hint` annotations to suppress it.

**Available hints**:

```python
from triton.experimental.tle.language.dsa.ascend import compile_hint

# Prevent compiler from hoisting Vector ops out of a loop (scatter-write pattern)
compile_hint(tensor, "disable_bubble_up")

# Avoid 32B alignment expansion for K=1 / BLOCK_K=1 scalar access pattern
compile_hint(tensor, "mayDiscretememaccess")

# Tell compiler to pad only K dim in tl.dot (M/N already aligned)
compile_hint(tensor, "dot_pad_only_k")

# Sub-tile L1 usage in multi-matmul kernels (e.g. FA with two GEMMs)
# hint_val = number of sub-tiles; prevents L1 overflow
compile_hint(o_acc, "hivm.tile_mix_cube_num", 4)

# Skip vcast+vcmp+vnot chain for i1 condition in tl.where
compile_hint(cond_tensor, "bitwise_mask")
```

**Real case**: `documents/tle/ascend/tle.dsa.ascend.compile_hint.md` documents all five hints with the exact conditions that trigger each one.

---

## 9. `tle.dsa.insert_slice` return value must be captured

**Symptom**: After calling `insert_slice`, the destination tensor appears unchanged. The inserted data is silently discarded.

**Diagnosis**: `tle.dsa.insert_slice` does not mutate `ful` in place — it returns a new tensor with the inserted values. If the return value is not captured, the operation has no effect.

**Fix**:
```python
# Wrong — return value discarded
tle_dsa.insert_slice(sub_tile, full_tensor, offsets=[0, 0], sizes=[BLOCK, BLOCK], strides=[1, 1])

# Correct — capture the return value
full_tensor = tle_dsa.insert_slice(sub_tile, full_tensor, offsets=[0, 0], sizes=[BLOCK, BLOCK], strides=[1, 1])
```

**Cross-ref**: `FlagTree/documents/tle/tle.dsa.insert_slice.md`

---

## 10. `tle.dsa.extract_element` rank mismatch

**Symptom**: `ValueError: Indice's rank must be equal to src tensor's rank` at trace time.

**Diagnosis**: `extract_element` requires the `indice` tuple to have exactly one index per dimension of the source tensor. Passing a scalar or a tuple with the wrong length raises this error.

**Fix**:
```python
# Wrong — source is 2D but indice has only one element
val = tle_dsa.extract_element(tensor_2d, indice=(i,))

# Correct — one index per dimension
val = tle_dsa.extract_element(tensor_2d, indice=(i, j))
```

**Cross-ref**: `FlagTree/documents/tle/tle.dsa.extract_element.md`
