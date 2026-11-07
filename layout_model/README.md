# Frozen 8328 layout model

Opt-in HCU layout optimization on KernelLLM/FlagTree `commonir-tle-layout-autotune` at `663f84b53`. The frozen 183-feature linear model and beam search are preserved from the Gluon comparison. Training code and datasets are excluded.

## Enable

Build/install this worktree with the repository's HCU toolchain, then run from the repository root:

```bash
source layout_model/activate.sh
python your_program.py
```

Set `FLAGTREE_LAYOUT_TUNING=0` to disable model selection. Activation uses beam width 8, 32 candidates per anchor, balanced-coherent proposals, pool/trial cap 512, total-warps search and the gfx936 MFMA warp domain. Runtime argument/grid binding, cache keys and launch guards are retained.

Weights SHA256: `8328f1f9a2d6f94ade3b03b8a8f8d67cb5833f48b37906a7b9508e5358a9428e`. Keep the complete file, including normalization and its input contract.

## Upstream source scope

The search, feature extraction, scoring and candidate checks live in new HCU files. The only existing C++ transformation implementation changed is `AccelerateHCUMatmul.cpp`: it consumes the selected MFMA warp distribution and calls model-owned hint validation. Other existing-source edits register/build the pass, connect compilation/runtime guards, or provide the two HCU Gluon layout aliases needed by the comparison.

`RemoveLayoutConversions.cpp`, `LoadStoreOpToLLVM.cpp` and the GPU-to-LLVM `Utility.cpp` match the target baseline byte for byte. This PR excludes the earlier layout-pin behavior change and redundant-register/atomic lowering fixes. NiceTree DP, ProgramCost, older models and operator-family workload adapters are excluded. Target FlagPrism/distributed checks, volatile-load handling and the existing MetaX tuner are preserved.

## Validation

Five changed translation units were rebuilt and linked against the prior full native build. All four configurations were freshly compiled into separate caches. Each of the 16 method/configuration combinations passed independent PyTorch checks before and after timing; model outputs also passed direct comparisons with the compiler and both Gluon variants.

| Configuration | Compiler (us) | Model (us) | Gluon A (us) | Gluon B (us) |
| --- | ---: | ---: | ---: | ---: |
| add_rms_norm-9 | 19.040 | 18.480 | 18.560 | 18.720 |
| dequantize_and_gather_k_cache-0 | 10.880 | 10.720 | 8.480 | 9.040 |
| mhc_post-0 | 83.906 | 84.137 | 83.949 | 69.152 |
| router_gemm-0 | 72.126 | 62.499 | 130.533 | 90.591 |

Gluon A/B mean s4/s8 for RMSNorm, baseline/optimized for the other operators. This is one fresh timing process per configuration, with five warmups and ten samples per method. Short-kernel controls can fluctuate; these timings do not establish stable speedups. Numerical tolerances are FP16 rtol=0.01/atol=0.001 and BF16 rtol=0.02/atol=0.01. Bitwise equality and executable/score comparisons with the preceding package are recorded separately in [minimal-reproduction.json](validation/minimal-reproduction.json).

This check covers these four configurations; it does not establish general atomic correctness or replace the historical 228-configuration evaluation. The preceding package and its full-build/measurement records remain available in commit `9d9c608b1`.
