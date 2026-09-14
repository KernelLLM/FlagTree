---
priority: high
---

# Autotune — Searching Block Sizes and Pipeline Stages

## Summary

Use TLE kernel autotune to search `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and `num_stages` when the kernel structure is already correct and the remaining question is parameter choice. Start with a direct config list for small search spaces; generate configs programmatically for larger ones.

## Use When

- The kernel is correct and the open question is block size or pipeline stage count.
- Manual tuning has explored 3-4 configurations without clear improvement.
- The factory function already exposes the tuning parameters as arguments.
- The kernel is compute-bound (GEMM-heavy) or a mixed compute/memory pattern where tiling trade-offs matter.

## Avoid When

- The kernel has a structural problem — wrong memory hierarchy, missing barriers, incorrect algorithm.
- All relevant parameters are fixed at module level (no free factory arguments).
- A semantic constraint ties one parameter to another (e.g., `BLOCK_N` must equal the head dimension).
- The kernel is not yet correct — autotune on a wrong kernel wastes time and may select a config that hides the bug.

## Pattern

### Step 1: Structure the factory function for autotuning

The factory function must accept the tuning parameters as arguments, not have them hard-coded:

```python
import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as tle_dsa
from triton.experimental.tle.language.dsa import ascend

def make_matmul_kernel(BLOCK_M, BLOCK_N, BLOCK_K):
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        ...
    return matmul_kernel
```

### Step 2: Define the config list

Start with a static list for small spaces (≤ 12 configs):

```python
CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K":  64},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K":  64},
    {"BLOCK_M":  64, "BLOCK_N": 128, "BLOCK_K": 128},
]
```

For larger spaces, generate configs programmatically:

```python
def get_configs(M, N, K, dtype_bytes=2):
    configs = []
    for bm in [64, 128, 256]:
        for bn in [64, 128, 256]:
            for bk in [32, 64, 128]:
                # L1 budget check: double-buffered A + B
                l1 = 2 * bm * bk * dtype_bytes + 2 * bk * bn * dtype_bytes
                if l1 > 512 * 1024:
                    continue
                # UB budget check: float32 accumulator
                ub = bm * bn * 4
                if ub > 192 * 1024:
                    continue
                # Divisibility check
                if M % bm or N % bn or K % bk:
                    continue
                configs.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
    return configs
```

### Step 3: Run the search

```python
import time

best_time = float("inf")
best_config = None

for cfg in CONFIGS:
    kernel = make_matmul_kernel(**cfg)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]), triton.cdiv(N, cfg["BLOCK_N"]))

    # Warmup
    for _ in range(3):
        out = kernel[grid](a, b, c, M, N, K, **cfg)

    # Timing
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = kernel[grid](a, b, c, M, N, K, **cfg)
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - t0) / 10

    # Correctness check (run once)
    ref = torch.matmul(a.float(), b.float()).half()
    max_diff = (out - ref).abs().max().item()
    if max_diff > 0.1:
        print(f"  SKIP {cfg}: max_diff={max_diff:.4f} (wrong)")
        continue

    print(f"  {cfg}: {elapsed*1000:.3f} ms, max_diff={max_diff:.4f}")
    if elapsed < best_time:
        best_time = elapsed
        best_config = cfg

print(f"Best: {best_config} — {best_time*1000:.3f} ms")
```

### Step 4: Tune pipeline stages (for pipelined loops)

When using `tle_dsa.pipeline` with `num_stages`, include it in the search:

```python
PIPELINE_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K":  64, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K":  64, "num_stages": 3},
]
```

Higher `num_stages` prefetches more tiles ahead of time but requires more L1 space (each additional stage adds one tile pair to the in-flight L1 footprint):

```python
# L1 with num_stages double-buffer:
l1 = num_stages * BLOCK_M * BLOCK_K * dtype_bytes + \
     num_stages * BLOCK_K * BLOCK_N * dtype_bytes
assert l1 <= 512 * 1024
```

## What To Verify After Applying

- Every config in the search list passes the L1 and UB budget checks before running.
- Each timed config includes a correctness check — skip configs that produce wrong output.
- The best config is validated against the PyTorch reference at the actual target shape, not just at the tuning shape.
- If the tuning shape is different from the production shape, re-run with the production shape before committing.

## Ascend-Specific Notes

- Dominant tuning knobs on 910B: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`. These control L1 tile sizes and K-loop trip count.
- `BLOCK_K` affects both L1 pressure and loop overhead. Typical optimum: 64–128 for float16.
- `num_stages=2` (double-buffer) is the default starting point; `num_stages=3` is worth trying when BLOCK_K is small.
- The Cube engine has a minimum effective tile size — very small BLOCK_M × BLOCK_N × BLOCK_K combinations may underutilize it. Avoid configs where `BLOCK_M * BLOCK_N < 8192`.

## Related Patterns

- **Tile Size Selection** — use to pick initial candidates that satisfy budget constraints before running autotune.
- **compile_hint Usage** — after tile sizes are settled, hints can address remaining UB/L1 pressure issues.
