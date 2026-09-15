---
name: tle-commonir-patterns
description: >-
  Pattern cards for writing multi-engine Ascend kernels with the CommonIR tile.* dialect
  via TLE DSA. Read this when writing any kernel that uses L1/L0/UB buffers across the
  DMA, Cube, and Vector engines, or that requires double-buffering, pipeline barriers,
  or cross-engine flag synchronization.
---

# TLE CommonIR Patterns — Ascend Multi-Engine Kernels

On Ascend 910B, three hardware engines run concurrently: the DMA engine (MTE1/MTE2/MTE3), the Cube engine (matrix multiply), and the Vector engine (element-wise). TLE's `tile.*` ops directly represent this hardware structure. Writing a high-performance kernel means pipelining these three engines so none of them stalls waiting for the others.

This skill is for kernels that use `tle_dsa.*` (especially the `tile_*`-prefixed ops) to target the Ascend hardware explicitly. It is not needed for kernels that only use `tle_dsa.alloc` + `tle_dsa.copy` + `tle_dsa.add/sub/mul` on UB (those are single-engine vector kernels covered by `tle-api-reference`).

## Reference Documents

| File | Contents |
|------|----------|
| [pattern_index.md](pattern_index.md) | Quick index of all pattern cards with Use When / Avoid When summaries |
| [patterns/vec-add.md](patterns/vec-add.md) | Baseline: single-engine vector kernel (UB only, no Cube) |
| [patterns/double-buffer-matmul.md](patterns/double-buffer-matmul.md) | Double-buffered GEMM: L1 ping-pong + Cube MMA + Vector postprocess |
| [patterns/three-task-pipeline.md](patterns/three-task-pipeline.md) | Full 3-task ring pipeline: DMA + Cube + Vector with tile_set_flag/tile_wait_flag |

## How to use

1. Read `pattern_index.md` to identify which pattern applies.
2. Read the full pattern card for that pattern.
3. Follow the numbered steps in the pattern card exactly — the flag event IDs, pipe constants, and barrier placement are not decorative; wrong placement silently produces incorrect results on hardware.
4. After writing the kernel, run the CommonIR unit test check to verify the IR uses `tile.*` ops (see `tle-api-reference/tle-kernel-basics.md` §5).
5. For performance tuning after correctness, read the `tle-npu-optimize-knowledge` skill.

## Scope

This skill covers correctness patterns for CommonIR multi-engine kernel structure. It does not define autotuning, profiling, or performance optimization workflows — those belong to `tle-npu-optimize`. It does not cover GPU (Hopper) patterns — those are in `tle-api-reference/tle-compute-gpu.md`.
