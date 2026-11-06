# Minimal frozen 8328 layout model

This packages the HCU model-only pass on KernelLLM/FlagTree `commonir-tle-layout-autotune` at `663f84b53`. The weights and selected search behavior are preserved from the original Gluon comparison. Training code and datasets are excluded.

## Enable

Build/install this worktree with the HCU toolchain using the repository instructions, then run from the repository root:

```bash
source layout_model/activate.sh
python your_program.py
```

The pass is opt-in. To disable it, set `FLAGTREE_LAYOUT_TUNING=0`. Activation selects the frozen 183-feature linear model, beam width 8, 32 candidates per anchor, balanced-coherent proposals, pool/trial cap 512 and the historical gfx936 MFMA warp domain. Runtime shape/stride/scalar/grid binding and cache/launch guards are retained.

Weights SHA256: `8328f1f9a2d6f94ade3b03b8a8f8d67cb5833f48b37906a7b9508e5358a9428e`. Keep the complete weights file including normalization and the input contract.

## Scope

The native implementation supports the frozen model beam-search path. NiceTree DP, ProgramCost, local/DP search, oracle/probe/replay sampling, older Joint/Candidate/Ordinal/Typed models, operator-family workload adapters and research-only Python bindings are removed. Shared memory-access geometry is retained in LayoutTuningMemory without the old pricing/DP dependencies.

The Gluon HCU aliases, redundant-register mask correction and atomic-result propagation remain explicit correctness dependencies. Target FlagPrism/distributed validation and volatile-load handling are preserved. The existing MetaX CommonIR tuner is unchanged.

## Reproduction check

A fresh CMake native build used the original HCU LLVM toolchain. All four configurations were recompiled, validated and timed with three processes and ten samples per process, after five warmups. Target compilation was forbidden in timed processes. Numerical checks, model scores, selected warps and all 16 executable identities matched the previous frozen implementation.

| Configuration | Compiler (us) | Model (us) | Gluon A (us) | Gluon B (us) |
| --- | ---: | ---: | ---: | ---: |
| add_rms_norm-9 | 23.520 | 22.880 | 22.800 | 23.120 |
| dequantize_and_gather_k_cache-0 | 6.560 | 6.720 | 6.560 | 6.480 |
| mhc_post-0 | 84.016 | 84.314 | 84.071 | 69.150 |
| router_gemm-0 | 72.191 | 62.514 | 129.881 | 90.621 |

Gluon A/B denote s4/s8 for Add RMSNorm, and repository baseline/optimized for other operators. Timings are this run’s pooled medians. Compiler controls passed in all processes for 2/4 configurations; Add RMSNorm and cache gather remained noisy. Short-kernel variability and per-process control results are retained in [minimal-reproduction.json](validation/minimal-reproduction.json). This four-configuration check does not replace the historical 228-configuration evaluation.

Native SHA256: `ea00edb367de81a412bd95a8ca2bd5e4c3008b70f6584d3fb565875935acb066`.

## Direct Gluon numerical quick check

The same admitted binaries were run with freshly prepared inputs on GPU 0. One new process per configuration used five warmups and ten samples per method. Model outputs were compared directly with compiler/Gluon outputs outside timing, and each method was checked against independent PyTorch semantics before and after timing.

| Configuration | Model (us) | Gluon A (us) | Gluon B (us) | Model vs Gluon A/B max absolute error |
| --- | ---: | ---: | ---: | --- |
| add_rms_norm-9 | 18.400 | 18.400 | 20.960 | 0 / 0.000244141 |
| dequantize_and_gather_k_cache-0 | 10.720 | 10.480 | 10.560 | 0 / 0 |
| mhc_post-0 | 84.442 | 84.080 | 69.170 | 0 / 0 |
| router_gemm-0 | 62.431 | 130.012 | 90.674 | 0 / 0 |

All direct numerical comparisons passed the existing dtype tolerances: FP16 rtol=0.01/atol=0.001, BF16 rtol=0.02/atol=0.01. Bitwise equality is recorded separately per pair. Single-process timings are a quick comparison; this check does not establish stable speedups. Details are in the `gluon_direct_numerics_quick_check` field of [minimal-reproduction.json](validation/minimal-reproduction.json).
