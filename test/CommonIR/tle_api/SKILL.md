---
name: tle-dsa-api-reference
description: >-
  TLE DSA Ascend NPU API reference. Read this before writing any TLE DSA kernel —
  covers buffer allocation and data movement, tensor slicing and loop control,
  and Ascend-specific advanced ops including compile hints, sub-vector ID,
  and inter-core CV pipeline synchronization.
---

# TLE DSA API Reference

TLE (Triton Language Extension) DSA is the Ascend-backend extension for Triton. It makes on-chip memory allocation and data movement explicit: instead of relying on the compiler to manage shared memory, you allocate named buffers in specific address spaces (UB, L1, L0A/B/C), copy data into them manually, operate on them, and copy results back.

TLE DSA is a lower-level path than TileLang. TileLang abstracts the memory hierarchy (`T.alloc_shared`) and auto-inserts sync; TLE DSA exposes the hierarchy directly (`tle.dsa.alloc` with an explicit address space) and lets you control sync yourself when needed. Use TLE DSA when you need explicit control over which on-chip buffer a value lives in, when you need to pipeline DMA and compute manually, or when you are writing a kernel that TileLang's automatic passes cannot express.

## Reference Documents

| File | Contents |
|------|----------|
| [tle-dsa-kernel-basics.md](references/tle-dsa-kernel-basics.md) | Imports, `@triton.jit` skeleton, address space constants (UB / L1), namespace mapping from `triton-ascend` community docs |
| [tle-dsa-memory.md](references/tle-dsa-memory.md) | `tle.dsa.alloc`, `tle.dsa.copy`, `tle.dsa.subview` (UB only), `tle.dsa.to_tensor`, `tle.dsa.to_buffer`, `tle.dsa.hint` |
| [tle-dsa-compute.md](references/tle-dsa-compute.md) | `tle.dsa.extract_slice`, `tle.dsa.insert_slice`, `tle.dsa.extract_element` |
| [tle-dsa-ascend-advanced.md](references/tle-dsa-ascend-advanced.md) | `compile_hint` (5 hint types with IR examples), `sub_vec_id`, `sync_block_set/wait/all`, CV inter-core double-buffer pipeline pattern |

## Reading order

Start with `tle-dsa-kernel-basics.md` for the skeleton and address-space overview. Then read `tle-dsa-memory.md` — every TLE DSA kernel uses alloc and copy. Read `tle-dsa-compute.md` if the kernel needs tensor slicing. Read `tle-dsa-ascend-advanced.md` only when the kernel requires manual CV inter-core sync or compiler hints for specific optimization problems.
