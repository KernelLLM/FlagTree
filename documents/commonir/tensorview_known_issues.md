# TensorView lowering — known issues

Open problems and resolved regressions in the `commonir-to-hivm` TensorView
path (`third_party/ascend/lib/CommonIRToHIVM/TensorViewLowering.cpp` and
`TensorViewStrategy.cpp`). Probe measurements below are historical unless
explicitly dated; they are not results of the latest functional test run.

Measured on Ascend910B4-1 / CANN 9.0.0. Probes live in
`third_party/ascend/unittest/pytest_ut/tensorview_probes/`; each is a standalone
script. These 25 probes still use the removed `tl.make_*_view` API and need
migration before they can be rerun with `python <probe>.py`; their measurements
below are historical. After migration, use `TRITON_ALWAYS_COMPILE=1
TRITON_DUMP_DIR=<dir> TRITON_KERNEL_DUMP=1` to inspect the emitted `.ttadapter`.
The current lowering has no `commonir-tv-access` debug logging; setting
`TRITON_LLVM_DEBUG_ONLY=commonir-tv-access` does not print strategy decisions.

Two companion suites guard what already works, and are the first thing to run
after touching this path:

* `test/CommonIR/Ascend/dsl_extension/test_gather_scatter_view.py` — 18 native
  pytest instances (14 functions, two with three parameter sets), covering dim
  arrangements, payload bounds, leading-dense partial/fully OOB tiles, negative
  payload blocks and scatter memory safety. Any mismatch fails the test.
* `test/CommonIR/Ascend/performance_case/tensorview/bench_gather_scatter_view.py`
  — four groups producing seven comparison rows. Gather pairs have equal output
  element counts but different source accesses/layouts; scatter compares one
  tile from one program against masked `tl.store`, using unique valid indices.
  Each timed side is checked against its own CPU reference before profiling.
  A ratio below `floor * (1 - tolerance)` fails. Invalid timings and correctness
  failures also fail; `--update-floors` only prints suggestions and cannot excuse
  them or an existing-floor regression. `None` marks a changed workload needing
  recalibration and fails normal checking, but is allowed by `--update-floors`.

### Expanded dense validation (2026-10-13; known interface limitation)

After removing two unused helpers from `CommonIRToHIVM.cpp`, the locally built
compiler passed all 18 formal GSV tests. An expanded dense run on
Ascend910B4-1 / CANN 9.0.0 (NPU 1) reported **4 failed, 10 passed**. The ten
passing tests use `tile.load/store`; these four TensorView tests use
`tl.load/tl.store` and fail during compilation:

* `third_party/ascend/unittest/pytest_ut/test_tensor_view_partition.py::test_tensor_view_partition_load_store`
* `third_party/ascend/unittest/pytest_ut/test_tensor_view_partition.py::test_tensor_view_partition_padding`
* `third_party/ascend/unittest/pytest_ut/test_tensor_view_strided.py::test_tensor_view_strided_load_store`
* `third_party/ascend/unittest/pytest_ut/test_tensor_view_strided.py::test_tensor_view_strided_padding`

All four fail at `LinalgIRToBytecodeByTritonMLIROpt`, when `triton-mlir-opt`
converts `.ttadapter.mlir` to bytecode. The common diagnostic is:

```text
!"tv"<"ptr<f32>"> type created with unregistered dialect
```

The diagnostic points to a `builtin.unrealized_conversion_cast` from a GM
memref to `!tv.ptr<f32>`. These are compilation failures, not numerical
mismatches. In this historical run, TensorView accesses through
`tile.load/tile.store` passed, while these `tl.load/tl.store` cases failed. The
four failures are retained as a known interface limitation; extending support
is outside this submission. The tests were not rerun against the pre-deletion compiler, so no
claim is made about their status before that deletion. The expanded suite is
not fully passing.

Loaded compiler extension SHA256:
`c621b91422b6274209548bd70b4e86e59747024b093a0441f5cf20b45b65c956`.

### Post-cleanup validation

After include, namespace and comment cleanup, the rebuilt local compiler passed
18 formal GSV tests and 10 dense `tile.load/store` tests. The same four
`tl.load/tl.store` tests above still failed with the unregistered TensorView
pointer diagnostic; this does not establish their status before helper deletion.

Separate temporary probes passed all 48 payload and 90 runtime-leading cases,
including the six block=-2 leading variants previously only syntax-checked.
These probes are not additional committed pytest regressions.

The default performance benchmark passed all seven ratio checks with unchanged
floors and tolerance, using warmup 25, active 50 and five repeats per side.
The ratios were 12.09 / 26.17 / 43.39 (multi-sparse), 23.06 (non-prefix),
14.35 (strided vs transposed), and 106.55 / 203.56 (scatter). These compare the
benchmark paths, not compiler before/after speedups. The overall test matrix
is not fully passing because of the four compilation failures.

Compiler extension SHA256 for this validation:
`293b233799c16e79453a6fd3b397faf0c6c358ca727237087f4f3c6bc84904cb`.

### Python view API migration (2026-10-14; direct-import snapshot)

The three constructors moved from `tl.make_*_view` to direct imports from
`triton.experimental.tle.language.dsa`; the old `tl` exports were removed:

```python
from triton.experimental.tle.language.dsa import (
    make_partition_view,
    make_strided_view,
    make_gather_scatter_view,
)
```

On NPU 1, the final direct-import snapshot passed 18 formal GSV tests and 15
partition/strided/gather-scatter `dst_space` tests: **33 passed, 1 warning**.
DSA exports, builtin markers, argument forwarding, old API removal and the
non-JIT guard also passed their checks. Compiler extension SHA256:
`c163f92f628a22ad569a53ada3cb9a9efc985b07d94a513025d251f4afb32a59`.

An earlier alias-form snapshot failed eight TensorView `tl.load/tl.store`
tests at `LinalgIRToBytecodeByTritonMLIROpt` with the unregistered `tv.ptr`
diagnostic: the four partition/strided cases above and four gather/scatter
cases without a pre-migration comparison. These eight tests were **not rerun**
on the final direct-import snapshot; their failures are not proof that the
migration introduced or did not introduce a regression.

A separate `native_matmul` TTIR attempt stopped at
`ASTSource(matmul_kernel.fn, ...)` with an `arg_names` AttributeError; neither
successful TTIR generation nor device correctness was established. Neither
the performance benchmark nor the 25 standalone probes was rerun on the final
direct-import snapshot. The probes and two exploratory attention scripts
still use the old API and remain outside the migration scope. The historical
compiler hashes and measurements in the other sections do not validate this
snapshot. The full end-to-end validation matrix is not complete.

### Public TLE constructors (2026-10-14; current API snapshot)

The direct-import snapshot above is historical. The current constructor
implementations live in `triton.experimental.tle.language.core` and are exported
through both `tle.language` and the top-level `tle` namespace:

```python
import triton.experimental.tle as tle

# Inside a Triton kernel:
# view = tle.make_partition_view(base, shape, strides, tile)
# The other constructors are tle.make_strided_view and
# tle.make_gather_scatter_view.
```

The old DSA constructor exports are not retained. Nine formal test, example and
benchmark files now call `tle.make_*_view`. This migration does not move
`tile_load/tile_store`, and does not implement GPU TensorView support.

Using a fresh compilation cache on Ascend910B4-1 (NPU 1), the final snapshot
passed 18 formal GSV cases and 15 `dst_space` cases: **33 passed, 1 warning in
155.82s**. Public export identity, builtin recognition, signatures, forwarding
and the non-JIT guard also passed. The local compiler extension SHA256 was
`78415b3fe5634923acaeda19e37a6898c3e130dd2e6a33ac083fe588b49fc7a1`.

After correcting its ASTSource input and dialect registration, `native_matmul`
generated and verified TTIR containing TensorView operations. It was not run on
the device. Eight `tl.load/tl.store` cases were collected, but were not device-run
in this snapshot; the known failures above remain unresolved. GPU tests and
performance benchmarks were not rerun, and the final constructor-only changes
did not include a full C++ rebuild. These results are not a fully passing
end-to-end matrix or a new performance measurement.

### Corrected benchmark baseline (2026-10-13)

Ascend910B4-1 / CANN 9.0.0; seed 0, warmup 25, active 50, five profiler repeats
per side. Latency is the median of five profiler means in ms; reported SD is
sample SD across those five means. Gather ratios compare layouts/access paths,
not the same operator with only a different lowering, and are not compiler
before/after speedups.

The transposed baseline now moves 4096 elements, matching the strided side
(previously 2048). Scatter now launches one program rather than multiple
programs writing the same tile, and its masked baseline checks both bounds.
Old floors 6 / 180 / 195 therefore do not apply to the corrected workloads.

| Changed workload | Ratios from three independent runs | New floor |
|---|---|---|
| strided vs transposed, tile `[8,8,64]`, pitch 72 | 14.2859 / 14.3003 / 14.2995 | 12 |
| scatter `[1024,128]`, tile `[32,128]` | 108.8854 / 108.9231 / 107.1530 | 91 |
| scatter `[4096,256]`, tile `[32,256]` | 205.3483 / 205.8899 / 208.0889 | 174 |

New floors are `floor(0.85 * minimum ratio across the three runs)`; the unchanged
multi-sparse floors 10 / 22 / 35 and non-prefix floor 19 are retained. Default
`tolerance=0.15` applies another factor of 0.85 to the floor. Duplicate concurrent
calibration runs were excluded. These short-term measurements do not establish
cross-machine or long-term stability.

After applying the floors, a separate default-mode NPU run passed all 14
pre-timing correctness checks and all seven floor checks (exit 0). Its changed
workload timings are below; each cell is median of means +/- sample SD, in ms.

| Workload | Fast | Baseline | Ratio |
|---|---|---|---|
| strided vs transposed | 0.00318924 +/- 0.00001316 | 0.04568332 +/- 0.00005272 | 14.3242 |
| scatter `[1024,128]` | 0.00487646 +/- 0.00001507 | 0.53034740 +/- 0.00202423 | 108.7566 |
| scatter `[4096,256]` | 0.00515332 +/- 0.00001648 | 1.04423850 +/- 0.00269378 | 202.6341 |

Compiler extension SHA256 for these measurements:
`5358f7b47d61fb9105cdf5aeb6700ef48eb387e3ced06ad4e6b8160866f78ae8`.
Calibrated benchmark SHA256:
`6590adaaf77144ab7c16b6de7f30a17f63265a5fc8dc2eb11226f7a7d9ec963a`.

Background on how the path is structured: the sparse dims of a
`gather_scatter_view` plus any dense dims before the payload form a *leading
segment*, one loop level each; the payload is the trailing dense suffix after
the last sparse dim (`payload.firstDenseDim == lastSparseDim + 1`). When the
payload is statically strideable and the index patterns pass the strategy gate,
`X...X D...D` takes `PerIndexPayloadDMA` with one payload DMA per valid leading
tuple. Leading dims need not be a sparse prefix. Segmented / General patterns
remain on `ScalarTransfer`.

---

## Issue 1 — `SingleAffineDMA` is unreachable; an affine index run still emits one DMA per row

**Severity:** performance only. Results are correct.

### What happens

`analyzeGatherScatterAccess` recognises a compile-time affine index run:

```
dim0: contiguous step=1 base=0
```

but `chooseTensorViewPlan` unconditionally returns `PerIndexPayloadDMA`, so the
run is lowered as `tile[0]` separate row DMAs instead of being collapsed into
one strided whole-tile DMA. `emitSingleAffineGather` / `emitSingleAffineScatter`
are `return failure()` stubs in `TensorViewLowering.cpp`, and the
`IndexPatternKind` switch that used to route to them was removed — deliberately,
because routing to an unimplemented emitter would make a currently-working
kernel fail to lower. `TensorViewPlan::indexStep` is consequently never written.

### Probe

`tensorview_probes/affine_cost.py` — the same `[64,128]` tile fetched two ways:
a `gather_scatter_view` whose index is `tl.arange(0, BL)` (compile-time
contiguous) versus a `partition_view`, which already lowers to the single DMA
that `SingleAffineDMA` should produce.

```
GSV contiguous idx (64 row DMAs): 0.0081 ± 0.0001 ms
partition view (1 tile DMA)     : 0.0011 ± 0.0000 ms
headroom                        : 7.48x
```

`tensorview_probes/affine_probe.py` is the minimal correctness case: contiguous
index, `maxdiff 0.0`, and the dump shows `hir.load=1 scf.for=1` — i.e. a per-row
DMA inside a loop, not a whole-tile transfer.

### Picture

```
index = [0,1,2,...,63],  tile=[64,128],  base strides=[128,1]

now (PerIndexPayloadDMA):
  scf.for row = 0..63
    extract idx[row]  -> 0,1,2,...
    DMA src[idx*128 .. +128] -> ub[row]
  = 64 DMAs x 128 elements

possible (SingleAffineDMA):
  one DMA over memref<64x128xf32, strided<[128,1], offset: 0>>
  = 1 DMA x 8192 elements

  GM layout — a contiguous index run means the rows are already adjacent:
  |-row0-||-row1-||-row2-| ... |-row63-|
  \_______________ one block _____________/
```

`AffineStrided` (step > 1) is the same shape with the sparse dim's stride
multiplied by the step, so the rows are evenly spaced rather than adjacent.

### To implement

* Implement the two stubs. `TensorViewLowering.inc` (now dead code, see Issue 3)
  still holds a working reference: `emitAffineGmTile` multiplies the sparse
  dim's stride by `indexStep`, and `emitAffineRunInBounds` range-checks both
  ends of the run once per tile.
* Restore the `IndexPatternKind` dispatch in `chooseTensorViewPlan` — **only
  together with the emitters**, never before.
* Degrade to `PerIndexPayloadDMA` (not to `ScalarTransfer`) when the run leaves
  the extent: a whole-tile transfer cannot pad or drop an individual row.
* This does **not** require the sparse dim to be dim 0. Each affine sparse dim
  multiplies its own stride by its own step, so `D S D` and even several affine
  sparse dims work. An earlier note in this file claiming otherwise was wrong.

### Is it worth doing

Unclear, and this is the main open question. If the index is a compile-time
`tl.arange`, a `partition_view` already gives the single DMA at 0.0011 ms — so
why reach for a `gather_scatter_view` at all? GSV earns its keep when the index
comes from memory, and that is `IndexPatternKind::Unknown`, which correctly
belongs on the per-index path.

The case that would really pay is "the index happens to be contiguous at run
time but the compiler cannot see it" — that is `Unknown`, needs a runtime
contiguity test, and is a much larger piece of work than filling in these stubs.

---

## Issue 2 — a dynamic payload stride forces `ScalarTransfer`

**Severity:** performance only. Results are correct.

### What happens

The payload contiguity/strideability analysis is skipped outright when any
payload stride is a runtime value:

```cpp
// TensorViewAccessAnalysis.cpp
if (!payload.hasDynamicStride && !payload.staticStrides.empty()) {
  // contiguity + strideability checks
}
```

so `staticallyContiguous` and `staticallyStrideable` both stay `false`, the gate
in `chooseTensorViewPlan` rejects the block DMA, and the access degrades to the
element-wise path.

### Probe

`tensorview_probes/dynstride_cost.py` — a payload spanning **two** dense dims
(`[8,64]`), once with `constexpr` extents and once with `tl.int64` extents:

| | `hasDynamicStride` | `staticallyStrideable` | time |
|---|---|---|---|
| static extents | false | true | 0.0019 ± 0.0002 ms |
| dynamic extents | **true** | **false** | 0.0506 ± 0.0001 ms |

**26.6x.**

### Picture

```
payload spans two dims; the test is  stride[1] >= tile[2] * stride[2]

static :  stride=[64, 1],  tile=[8, 64]
          64 >= 64*1 = 64   OK, decided at compile time

dynamic:  stride=[d2, 1],  tile=[8, 64],  d2 runtime
          d2 >= 64*1 = 64   unknown at compile time
                                |
          today:  give up, go element-wise (26.6x)
          could:  emit a runtime predicate
                  scf.if (d2 >= 64) { block DMA } else { scalar }
```

### Scope is narrower than it looks

A **single** payload dim is unaffected, which covers most real views. The
payload is the *trailing* segment, and its innermost stride is almost always the
literal `1` of a row-major layout:

```
view strides=[m_cols, 1] with m_cols a runtime tl.int64,  payload dims=[1]
  -> payload.staticStrides = [1]
  -> hasDynamicStride = false
  -> staticallyStrideable = true      block DMA, unaffected
```

Confirmed with `tensorview_probes/dynsize_probe.py`, whose extent is `tl.int64`
and which still takes the DMA path. So the issue needs *all* of: a payload
spanning two or more dims, **and** a non-innermost payload stride that is a
runtime expression (e.g. `d1*d2`).

```
BAD   shape=[D0, d1, d2],  strides=[d1*d2, d2, 1],  payload=[1,2]   d2 dynamic
OK    shape=[D0, d1],      strides=[d1, 1],         payload=[1]     stride is literal 1
OK    shape=[D0,D1,D2],    strides=[D1*D2,D2,1],    payload=[1,2]   all constexpr
```

### To implement

Structurally identical to the payload-OOB clamp that already shipped: compute
the comparison in IR instead of in the analysis, guard the block DMA with it,
and keep `emitScalarGather` as the else branch. The analysis needs to export
which dims require a runtime check (or the lowering can derive it from
`info.strides` itself).

---

## Issue 3 — `TensorViewLowering.inc` was dead code but still compiled (RESOLVED)

**Severity:** maintenance. **Fixed:** the `#include` is gone; the file stays.

`lowerViewLoad` / `lowerViewStore` call `emitGatherLoad` / `emitScatterStore` in
`TensorViewLowering.cpp`. The older `.inc` was still `#include`d in
`CommonIRToHIVM.cpp` and still compiled, though nothing reached it:

```
emitPerIndexPayloadGather  references in CommonIRToHIVM.cpp: 0
emitSingleAffineGatherDMA  references in CommonIRToHIVM.cpp: 0
emitDensePayloadInBounds   references in CommonIRToHIVM.cpp: 0
```

The real cost was not the wasted compile time but that it carried a **second,
divergent copy of ten helpers sharing names with the live ones** — editing one
silently did not affect the other, which cost time during this work.

Removing the `#include` compiles clean: no errors and no `unused-function`
warnings, since everything in the file is `static` and self-contained. The file
itself is **kept, not deleted** — it holds the only working implementation of
the affine fast path (`emitAffineGmTile`, which folds the index step into the
sparse dim's stride, and `emitAffineRunInBounds`, which range-checks both ends
of a run once per tile), which Issue 1 needs. Both the `.inc` header and the
site where the `#include` used to be now say so.

Verified after removal: `test_gather_scatter_dst_space.py` 5/5,
`test_triton_unified_attention_gsv_load.py` 16/16, all 16 probes clean,
FlagGems `ALL EXACT`.

---

## Issue 4 — the `tile_store` side has no padding mechanism of its own

**Severity:** by design, documented so it is not mistaken for a bug.

`hivm::LoadOp` has native `pad_mode` / `pad_value` / `right_padding_num`;
`hivm::StoreOp` has none. A store also cannot be redirected the way a read can —
clamping an out-of-range coordinate would corrupt whatever block it clamps onto.
So the two directions are necessarily asymmetric:

* **gather**: pre-fill the tile with one `hivm.hir.vbrc`, then leave
  out-of-range blocks unwritten — they keep the pad. One Vector op regardless of
  how many sparse dims there are, no else region.
* **scatter**: drop out-of-range blocks with the `scf.if` itself.
  `padding_value` does not apply to a sparse store lane; there is nothing to
  pad.

Related asymmetry, already bitten once: a scalar GM write must go through
`tensor.insert` + `bufferization.materialize_in_destination`. A plain
`memref.store` into the same single-element GM view compiles cleanly and then
silently drops ~74% of lanes (759/1024 mismatched on
`test_gather_scatter_dst_space.py -k scatter_store`). The gather side reads that
same view with a plain `memref.load` and is fine. There is a comment at the site
marking the three-op form as load-bearing.

---

## Issue 5 — a `tl.dot` and a scatter `tile_store` in one kernel cost 10x the two split apart

**Severity:** performance. Results are correct. **Cause not identified** —
investigation paused deliberately, see the end of this section.

### What happens

A GatherScatterView `tile_store` fed by a `tl.dot` is ~24x slower than the same
scatter fed by a plain `tl.load`. Neither half is slow on its own, so it is the
fusion that costs. Splitting the GEMM and the scatter into two kernels recovers
all of it (`tensorview_probes/via_ub.py`):

| | ms |
|---|---|
| dot kernel alone, plain store | 0.0359 |
| scatter kernel alone, plain load | 0.0191 |
| **sum** | **0.0550** |
| **both in one kernel** | **0.5505** |

The 2x2 that isolates it (`tensorview_probes/dotstore.py`, M=4096 K=N=128 BM=32
BK=64 fp16):

| upstream | store | mix_mode | ms |
|---|---|---|---|
| load | contiguous | aiv | 0.0119 |
| load | gsv scatter | aiv | 0.0226 |
| dot | contiguous | mix | 0.0362 |
| **dot** | **gsv scatter** | **mix** | **0.5514** |

### It is a per-row cost, not fixed setup

`tensorview_probes/rows.py` sweeps BM with the total row count held at M=4096:

| BM | grid | extra us/program | us per row |
|---|---|---|---|
| 8 | 512 | 1.048 | 0.1310 |
| 16 | 256 | 2.096 | 0.1310 |
| 32 | 128 | 4.136 | 0.1292 |
| 64 | 64 | 8.341 | 0.1303 |

Constant to three decimals across an 8x change in BM, so each row DMA itself
runs ~24x slower (a load-fed row is 0.0054 us). Total time also scales linearly
with grid (M=4096/1024/256 -> 0.551/0.143/0.039 ms), confirming it is per-row
work rather than a constant penalty.

### Ruled out

* **The scatter lowering.** The emitted store loop is byte-identical between the
  load-fed and dot-fed kernels — same `memref.reinterpret_cast`, same
  `memref.subview`, same `hivm.hir.store`, same guard.
* **`mix_mode`.** `mix` tracks `tl.dot` alone: `dot` + contiguous store is also
  `mix` and still fast. (An earlier revision of this file blamed `mix`; that was
  drawn from 3 of the 4 cells, with the disconfirming one missing.)
* **The fp32 accumulator.** `fp16 load -> fp32 -> *2.0 -> scatter` is 0.0224 ms,
  the same as staying in fp16 (`tensorview_probes/acctype.py`).
* **Dynamic vs static DMA extent.** The payload extent was being marked
  `kDynamic` unconditionally even when the valid length folds to a constant.
  Fixing that (keep it static when `m_ConstantInt` matches) left the timing at
  0.5517 ms — unchanged. The fix is kept because the type should reflect what is
  known, but it is not the cause.
* **`parallel_loop` / `ExtractedLoadOrStore` markers.** Neither kernel has
  either; both are `parallel_mode = "simd"`.
* **The extra `memref.copy`** (3 vs 2, the L0C accumulator staged to UB) — one
  32x128 copy cannot be 24x.

### Why it is paused

Everything visible at the `.ttadapter` level is identical between the fast and
slow kernels, so the cause is below this pass — plausibly MTE3 contending with
the cube pipeline, or multi-buffering being disabled when both coexist. That
cannot be confirmed from the IR, and `msprof` yields nothing for these kernels
(`No profiling data dumped`, with both `msprof` and `msprof op`). Continuing
would mean reading `bishengir-compile` intermediates (`.bcmlir`, `.npubin`),
which is harder to interpret and not guaranteed to localise it.

### Workaround

Split the GEMM and the scatter into two kernels. At this shape that is
0.055 vs 0.5505 ms; the cost is one round trip of the intermediate through GM,
far cheaper than the fusion penalty.

---

## fused_moe: measured end to end, ~1.01x

Recorded here because the earlier screening report's 27.9-76.9x is a probe
number and does not carry to the operator.

The op did not compile at all on this box until now — the default tiling
overflows UB:

```
CONFIG: BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_stages=3
ub overflow, requires 2103296 bits (257 KB) while 1572864 bits (192 KB) available
```

The accumulator alone is 128x128x4 = 64 KB. `BLOCK_SIZE_N=64` with
`num_stages=2` fits; seven configurations were swept and all compile with
`rel-err 0.0005` against a float32 per-token reference. At
T=512 E=8 K=1024 N=1408 topk=2:

| config | end-to-end |
|---|---|
| BM=128 BN=64 BK=32 | **565 ms** |
| BM=64 BN=64 BK=64 | 589 ms |
| BM=64 BN=64 BK=32 | 625 ms |
| BM=64 BN=128 BK=32 | 687 ms |
| BM=32 BN=64 BK=32 | 788 ms |
| BM=32 BN=128 BK=32 | 944 ms |

This was done by patching `try_get_optimal_moe_config` from the test, not by
changing FlagGems. Making it the default needs an NPU branch in
`get_default_config`.

With the op finally measurable, the epilogue store's share is small
(`tensorview_probes/realshape.py` uses the real geometry EM=1024 N=1408 BM=128
BN=64):

| | |
|---|---|
| whole op | 565 ms |
| epilogue store (masked scatter) | 5.32 ms — **0.94%** |
| same store as a gsv scatter | 0.086 ms (**61.7x** on the store) |
| **end-to-end speedup** | **1.01x** |

The GEMM dominates: K=1024, N=1408, two GEMMs, topk=2. Probe shapes with
K=N=128 overstate the store's share by an order of magnitude. Re-running the
original probe script today gives 16.6-43.0x rather than the recorded
27.9-76.9x, with the gsv side at 0.571 ms versus 0.308 ms; why it moved is
unresolved, and the store IR is identical, so it is downstream of this pass.

**Conclusion: do not rewrite the fused_moe epilogue.** 1.01x end to end, and the
dot-fed form carries Issue 5's unexplained 24x. The valuable finding here is the
UB-overflow fix, which turns the operator from uncompilable into usable.

---

## Resolved regressions and historical probe results

Formal pytest regressions and standalone probes are distinguished below; the
historical timings in the probe table are not the calibrated benchmark above.

### Formal boundary regressions

The formal suite passed 18 instances after the boundary fixes. The eight added
instances cover leading-dense partial/fully OOB tiles (three parameter sets for
each of gather/scatter) and negative payload blocks (one each). Dense valid
lengths use `index >= 0 ? max(0, min(extent - index * traversal, tile)) : 0`;
leading lengths bound loops and payload lengths bound DMA sizes. Gather retains
padding outside those lengths; scatter leaves those destinations untouched.
Sparse extent/index handling remains i32.

Separate temporary validation passed 48 payload probes and 84 runtime-leading
probes; these are not additional formal pytest instances. Six later block=-2
leading variants were syntax-checked only, not device-validated in that run.

### Historical standalone probes

| | probe | was | now |
|---|---|---|---|
| payload dim OOB read/write | `dense.py`, `dense_diag.py`, `dense_tail.py` | read the next row; store clobbered 6 elements past the buffer | clamped per dim via dynamic-size DMA; `maxdiff 0.0`, canary clean |
| `X...X D...D` arrangements | `xdd.py` (`D S D`, `S D S D`) | gated to `ScalarTransfer` | block DMA; `D S D` 0.0517 -> 0.0035 ms (14.8x) |
| multi-sparse-dim payload | `two_sparse.py`, `sc3d.py` | `ScalarTransfer` | block DMA, 12.0-47.6x |
| multi-dim payload | `multidense.py` (`S D D`, `D S D D`) | untested | one block per tuple; `S D D` 8 DMAs x 512 elem, 0.0016 ms |
| strided (non-dense) payload | `padstride.py` | `ScalarTransfer` (`staticallyContiguous=false`) | strided DMA; 0.0553 -> 0.0031 ms (17.8x) |
| transposed `dim_map` | `transposed.py` | — | correctly still rejected (`staticallyStrideable=false`) |
| zero-length transfer | `zerolen.py` | — | safe; whole block takes the pad |
| `padding_value=zero` | any | emitted a NaN constant (switch fallthrough in `Utils.h`) | fixed |

---

## Priority

The optimization opportunities in Issues 1 and 2 are **low priority**: each is
a measured gap (7.5x and 26.6x) but neither has a confirmed caller. Issue 5 is a
separate, paused investigation.

| | headroom | how often it fires | cost to implement | risk |
|---|---|---|---|---|
| Issue 1 affine | 7.5x | medium — needs a compile-time affine index, where `partition_view` is already the better tool | medium — two emitters, gate dispatch, OOB degrade | medium |
| Issue 2 dynamic stride | 26.6x | low — needs a 2+ dim payload **and** a dynamic non-innermost stride | medium — runtime predicate plus scalar fallback | low |
| ~~Issue 3 dead `.inc`~~ | — | — | **resolved** — `#include` removed, file kept for Issue 1 | — |
| Issue 5 dot + scatter | 10x | **high** — any GEMM epilogue rewrite | unknown; cause not localised | — |

Issue 5 is the one with a real caller (every GEMM-epilogue rewrite hits it), but
it has a cheap workaround — split into two kernels — and no identified cause, so
the useful next step is below-IR profiling rather than a code change.

Recommendation: leave Issues 1 and 2 as recorded gaps until a real operator hits one, then
implement against that operator's actual shape. None of the operators screened
so far (`index_copy_`, `index_select`, `fused_moe`, the unified-attention
variants) hits either — they all have a single payload dim and a runtime-loaded
index.

If one must be picked without a driving operator, Issue 2 is the cheaper and
safer build (a pure runtime predicate, no new DMA shape), though it is also the
rarer trigger.
