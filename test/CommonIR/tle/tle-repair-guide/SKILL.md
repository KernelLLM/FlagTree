---
name: tle-repair-guide
description: >-
  Repair guide for TLE DSA kernel bugs on Ascend. Read this when a kernel compiles but
  produces wrong results, hangs at runtime, overflows UB/L1, or fails in bishengir-compile.
  Contains real cases extracted from FlagTree success_case/ and failed_case/ tests.
---

# TLE Repair Guide — Ascend DSA

When a kernel compiles without Python errors but produces wrong results, crashes, or fails in a downstream pass (`commonir_to_hivm`, bishengir-compile), the cause is almost always one of the patterns in this guide.

## Reference Documents

| File | Contents |
|------|----------|
| [repair-experience.md](repair-experience.md) | Numbered symptom → diagnosis → fix entries from real kernels |

## How to use

1. Match your symptom to an entry in `repair-experience.md`.
2. Apply the fix exactly as shown — most of these bugs have no compile-time error, so the symptom is the only signal.
3. After fixing, rerun `dump_commonir()` and check the IR for the asserts in `tle-api-reference/tle-kernel-basics.md §5`.
4. If the symptom is silent wrong output (not a crash), compare against a reference PyTorch result with `atol=1e-2`.

## Scope

This skill covers bugs in `tle.dsa.*` DSA kernels targeting Ascend. It does not cover GPU path bugs, compilation environment setup, or performance optimization — those belong to `tle-commonir-patterns` and `tle-optimize-knowledge` respectively.
