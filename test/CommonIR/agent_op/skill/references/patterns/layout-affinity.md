---
priority: medium
---

# Layout & Affinity (TLE DSA)

## Summary

TileLang Expert mode exposes `T.annotate_layout` with `make_zn_layout` / `make_nz_layout` to control L1 buffer data layout for MMA throughput. **TLE DSA has no equivalent API** — buffer data layout is compiler-managed, and the only `swizzle*` symbols in the TLE surface (`tle.swizzle2d_Nz`, `tle.gemm_swizzle2d_Nz`, in `dsa/ascend/distributed.py`) are **block-scheduling** swizzles (which core computes which output tile), not data-layout annotations. Do not invent one. This pattern collects what you *can* control, and how to verify what the compiler chose.

## Use When

- Profiling shows MTE copy bursts or MMA operand staging as the bottleneck.
- A buffer stays resident across many MMAs (Q per Q-tile) and you want to confirm the layout the compiler picked.
- A multi-tile grid thrashes L2/L1 — apply a scheduling swizzle.

## Avoid When

- You are looking for a direct `make_zn_layout` port — it does not exist in TLE DSA.
- The kernel is purely element-wise (no MMA).

## What you control

### 1. Operand orientation (the actionable part of zN/nZ discipline)

The tilelang rules — left operands (Q, P) prefer zN (inner-dim contiguous), K prefers nZ because it feeds L0B through a transpose copy, V stays zN as the untransposed right operand of P·V — translate in TLE DSA to **where you put the `tl.trans`** and how GM tensors are stored:

```python
# MM1: S = Q · Kᵀ — keep K stored as (N, D) in GM, transpose the register tile:
s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

# MM2: O = P · V — V consumed untransposed as right operand:
o_partial = tl.dot(p, v, out_dtype=tl.float32)
```

Keep GM layouts so that exactly one operand needs an in-kernel transpose, and that transpose happens on a **plain register tensor**, never on a `subview → to_tensor` chain (known on-device failure — see `explicit-memory` hazards).

### 2. Copy granularity for L1 staging

Whole-tile `tile_copy` with constexpr extents is the verified form (`native_fa.py`):

```python
tle.dsa.tile_copy(k_block_ptr, k_l1, [tl.constexpr(BLOCK_N), tl.constexpr(DIM)])
k = tle.dsa.tile_to_tensor(k_l1, writable=False)
```

This gives the compiler the cleanest signal to pick MMA-friendly fractal layouts itself.

### 3. `fixpipe` DMA layout modes (910_95 only)

L0C→UB/GM drains can request layout conversion:

```python
import triton.language.extra.cann.extension as extension
extension.fixpipe(src_on_l0c, dst_ub, dma_mode=extension.FixpipeDMAMode.NZ2ND, ...)
```

Alignment rules: fp32 last dim % 8, fp16 last dim % 16; NZ2DN additionally constrains M. 910_95-only — not available on 910B.

### 4. `compile_hint` value annotations

```python
extension.compile_hint(ptr, "hint_name", hint_val)
```

Used in `performance_case/13-matrix-multiplication-optimized.py`. Sparsely documented — treat as experiment-only, verify each hint against `dump-commonir` output.

### 5. Scheduling swizzle for cache affinity (multi-tile grids)

When each core loops over many output tiles, the *visit order* is controllable:

```python
block_id_m, block_id_n = tle.gemm_swizzle2d_Nz(pid, ...)
```

(From `tutorials/tle/dsa/09-ascend-allgather-gemm.py`; the benchmark there compares `both_swizzle` vs `no_swizzle`.) This is L2/L1-reuse affinity across tiles — the scheduling-level analog of layout affinity.

## How to verify what the compiler chose

Since layout is compiler-managed, inspect rather than annotate:

```bash
python fa_tle_v2.py --dump-commonir=tmp/commonir.mlir --S 1024     # tile.* level
python fa_tle_v2.py --dump-hivm=tmp/hivm.mlir                       # post commonir_to_hivm
python fa_tle_v2.py --dump-linalg=tmp/linalg.mlir                   # final lowering
```

Look for the `tile.alloc` layout attributes and the MTE copy forms around your resident buffers. The pass chain is documented in `native_matmul_dsa.py::dump_linalg`.

## What To Verify After Applying

- Exactly one in-kernel transpose per dot; it sits on a register tensor.
- L1 staging copies are whole-tile with constexpr extents.
- Any `fixpipe` use is gated to 910_95 and meets the alignment rules.
- Numeric gate after any orientation change — wrong orientation compiles fine and corrupts MMA results silently.
- If a layout hypothesis matters, confirm via dumped IR before building on it.

## Related Patterns

- `explicit-memory`: the buffers whose layout is at stake.
- `double-buffer`: ping-pong buffers benefit most from confirmed layouts.
- `workspace-pipeline`: the resident-Q discipline across a task ring.
