---
name: tle-api-reference
description: >-
  Reference for the TLE (Triton Language Extension) Python API. Read this before writing
  any TLE kernel — it covers both the GPU (Hopper) path and the DSA (Ascend) path, with
  memory allocation, data movement, compute, and pipeline control for each target.
---

# TLE API Reference

TLE is a hardware-explicit extension on top of Triton that exposes on-chip memory, data movement, and pipeline control directly to the programmer. Standard Triton manages shared memory implicitly; TLE makes buffer allocation, DMA copies, and synchronization explicit so kernels can saturate hardware bandwidth and compute units.

## Two independent sub-APIs

TLE exposes two target paths under `triton.experimental.tle.language`:

| Path | Import | Target hardware | On-chip scopes |
|------|--------|-----------------|----------------|
| `gpu` | `from triton.experimental.tle.language import gpu as tle_gpu` | NVIDIA Hopper+ | SMEM, TMEM |
| `dsa` | `from triton.experimental.tle.language import dsa as tle_dsa` | Ascend 910B | UB, L1, L0A/B/C |

Both paths use the standard `@triton.jit` decorator and `tl.program_id` for kernel dispatch. Neither replaces standard Triton ops — they extend them by adding explicit on-chip buffers and DMA copy primitives.

## Reference Documents

| File | Contents |
|------|----------|
| [tle-kernel-basics.md](tle-kernel-basics.md) | Imports, decorator pattern, kernel structure, and the difference between GPU and DSA paths |
| [tle-memory-gpu.md](tle-memory-gpu.md) | GPU path: `tle.alloc`, `tle.copy`, `tle.local_ptr`, layout types, scope constants |
| [tle-memory-dsa.md](tle-memory-dsa.md) | DSA path: `tle.dsa.alloc`, `tle.dsa.copy`, `buffer` type, address spaces (UB/L1/L0) |
| [tle-compute-gpu.md](tle-compute-gpu.md) | GPU compute: `tle.pipeline`, async load, `local_ptr` + `tl.dot` pattern, TMA descriptor |
| [tle-compute-dsa.md](tle-compute-dsa.md) | DSA compute: vector ops, `tile_cube_launch`, `tile_set_flag`/`tile_wait_flag`, `parallel` range |

## How to use

1. Read `tle-kernel-basics.md` first to understand the kernel skeleton.
2. Choose the path (GPU or DSA) and read the corresponding memory document.
3. Read the compute document for the same path.
4. For multi-engine Ascend kernels (Cube + Vector + DMA), also read the `tle-commonir-patterns` skill.
5. For NVIDIA TMA copies or software-pipelined prefetch, also read the `tle-gpu-patterns` skill.
