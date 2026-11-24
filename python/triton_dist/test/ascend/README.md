# Ascend Operator Verification and Performance Tests

These scripts reuse kernels and golden references from [`tutorials/ascend/`](../../../../tutorials/ascend/README.md) for correctness checks, NPU Event timing, or both. They require an Ascend multi-device environment with CANN, torch_npu, Triton, triton_dist, and ACLSHMEM installed.

## Launch and run modes

Run the sweep from this directory. Defaults are `--mode=both --case=full`:

```bash
bash run.sh --mode=verify --case=full
bash run.sh --mode=perf --case=random
bash run.sh --mode=both --case=full
```

| Mode | Behavior |
| --- | --- |
| `--mode=verify` | Check correctness only; each `@prof` call executes once without warmup or timing Events |
| `--mode=perf` | Warm up and time the kernel and golden reference, report performance, and skip output comparisons |
| `--mode=both` | Run performance timing and correctness checks |

Case selection rules:

- `--case=full`: Run all shape and parameter combinations in the script's current configuration.
- `--case=random`: Sample up to 10 combinations without replacement and run them in their original order.
- If fewer than 10 combinations are available, run all of them. A fixed-shape script still runs its single shape.
- Sampling applies to parameter combinations within each script, not to the list of scripts.
- Environment-based shape configuration is applied before sampling.

Reproducing a sample:

- All ranks use the same sampling seed.
- In random mode, `run.sh` generates and prints `ASCEND_CASE_SEED`. Set it explicitly to reproduce a sample.
- For direct `torchrun` launches, `ASCEND_CASE_SEED` takes precedence. If unset, the fallback is the shared `MASTER_PORT`, then `0`.

Launch an individual script from the repository root:

```bash
ASCEND_CASE_SEED=42 ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  python/triton_dist/test/ascend/qkv-all2all.py --mode=both --case=random
```

Runner behavior:

- Run this directory's `.py` files containing a `__main__` entry in filename order, skipping helper modules. Do not recurse into `api/` or `unit/`.
- Run UDMA cases only when Ascend 950PR&950DT products are detected through `npu-smi info` or `ASCEND_SOC_VERSION`.
- Continue with the remaining scripts after a script fails, then summarize PASS/FAIL/SKIP.
- Exit with `1` if any script fails, or `2` for invalid arguments.

| Runner environment variable | Default | Purpose |
| --- | --- | --- |
| `NPROC_PER_NODE` | `2` | Processes per script |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,...,NPROC_PER_NODE-1` | Visible devices |
| `SKIP_UDMA` | `0` | Set to `1` to skip all UDMA cases, including MoE and fused Reverse A2A |
| `RUN_AUTOTUNE` | `0` | Set to `1` to include entries whose names contain `autotune`; none currently exist here |
| `ASCEND_CASE_SEED` | Generated | Reproducible random-case sampling seed |

Communication addresses and ports:

- The runner assigns a fresh `ASH_MASTER_PORT` per script, starting at port 31007 and incrementing by 7.
- Direct launches use each script's address and port defaults. Scripts that use shmem also support `ASH_MASTER_ADDR`.
- The runner does not set `HCCL_NPU_SOCKET_PORT_RANGE`.

## The @prof Decorator (`prof.py`)

`prof.py` hosts the shared timing decorator:

```python
from prof import check_all_ranks_close, load_tutorial, prof, prof_report

tut = load_tutorial("allgather-gemm")   # import tutorials/ascend/<name>.py by path

@prof(label="allgather_gemm")                     # warmup and timed rounds around each call
def run_kernel():
    return tut.allgather_gemm(A, B, C, peer_mem, ...)

@prof(label="torch_allgather_gemm")
def run_golden():
    return tut.torch_allgather_gemm(A, B, world_size)

C_golden = run_golden()
run_kernel()
prof_report(run_golden, run_kernel)               # prints the EVENT_TIMING line on rank 0
check_all_ranks_close(C, C_golden, rtol=1e-3, atol=1e-3, context="C")
```

Every round is `prepare() -> barrier -> sync -> invoke -> sync -> barrier`, and timed rounds additionally record NPU events around the invoke, so state resets stay outside the measured window:

- `prepare=` — optional hook run at the start of every round, before the barrier and outside the event window. It is the hook for resets that must precede each launch, such as `signal_mem.fill_(0)` for wait/notify kernels or `C.zero_()` before accumulate-style kernels.
- `barrier=` — optional custom rank barrier. Defaults to `dist.barrier()`.
- `label=` — name used in the `EVENT_TIMING` output.
- `warmup=` / `iters=` — round counts. Resolution order: `ASCEND_PROFILE_WARMUP` / `ASCEND_PROFILE_ITERS` environment variables (when set) > decorator arguments > defaults (`5` / `50`).

Timings land on the wrapper's `.prof` attribute (`run_kernel.prof.times_ms`); the record is overwritten on every call, so shape-loop scripts can report per shape.

Behavior by mode:

- In `verify` mode, `.prof` is `None` and `prof_report()` emits no timing report.
- In `perf` mode, `check_all_ranks_close()` skips comparisons.
- Custom verification and reporting paths follow the same mode selection.

`load_tutorial("allgather-gemm")` loads the matching tutorial from the source checkout, so these scripts require the tutorial directory.

## Timing and shape controls

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `ASCEND_PROFILE_WARMUP` | `5` (decorator default) | Warmup rounds per function; must be non-negative |
| `ASCEND_PROFILE_ITERS` | `50` (decorator default) | Timed rounds per function; must be positive |
| `MOE_PROFILE_WARMUP` / `MOE_PROFILE_ITERS` | `5` / `50` | Counts passed by MoE scripts; explicit `ASCEND_PROFILE_*` settings still take precedence in the decorator |

Timing and verification settings by script:

- AllToAllV UDMA overrides the decorator defaults with 3 warmup rounds and 10 timed rounds.
- AllToAllV MTE uses the shared `@prof` defaults and separately measures the first warmup when its warmup setting (default 3) is nonzero.
- Both AllToAllV scripts support `TRITON_VALIDATION_ITERS` (default 1). The verify/both modes perform at least one validation, while perf skips these validations.
- Verify mode skips first-warmup timing.

Shape sweep configuration:

- QKV: Configure the sweep with `QKV_BENCH_S` and `QKV_BENCH_H`.
- Transpose UDMA: Configure the sweep with `A2A_BENCH_S` and `A2A_BENCH_H`.
- Fused Reverse A2A: Run one shape by default; set `FUSED_RUN_BENCH=1` to sweep the built-in combinations.
- MoE: Configure combinations with `MOE_N_LIST`, `MOE_TOPK_LIST`, `MOE_NLE_LIST`, and `MOE_ROUTE_MODES`.

Random mode samples the combinations produced by these settings.

Quick performance run from the repository root:

```bash
ASCEND_PROFILE_WARMUP=1 ASCEND_PROFILE_ITERS=2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  python/triton_dist/test/ascend/allgather-gemm.py --mode=perf --case=full
```

Dispatch GMM UDMA (`dispatch-gmm-udma.py`) loads the standalone tutorial with
one example and tests 36 random-routing combinations, including empty buckets,
tail dimensions and offsets beyond INT32. Only the GMM output uses
`rtol=atol=1e-3`; public layouts and count lists must match exactly.
`verify` executes both recipes once. `perf` and `both` default to 10 warmup
and 30 timed rounds for **each** recipe; override with
`ASCEND_PROFILE_WARMUP` / `ASCEND_PROFILE_ITERS`. Each timed recipe includes
routing inversion, count/offset exchange, pack, UDMA, layout restoration,
GMM and host count-list conversion. Reported times use the maximum rank
for each round. Input generation and capacity allocation are outside timing.
The tutorial runs one correctness invocation; repeated changed-input tests
belong to the test diagnostics rather than its command-line interface.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASH_MASTER_PORT=9598 \
torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29798 \
  python/triton_dist/test/ascend/dispatch-gmm-udma.py --mode=both --case=full
```

## Script Overview

The table lists the current operator entry points. `prof.py` is a shared helper and not an operator entry point.

| Script | Operator | Notes |
| --- | --- | --- |
| `all-to-all-single-2d.py` | UDMA All-to-All-Single 2D | 288 matrix cases cover (M, N, dtype, split_pattern) combinations |
| `allgather.py` | AllGather using UDMA and notify/wait synchronization | 108 matrix cases plus 9 boundary cases |
| `allgather-gemm.py` | AllGather fused with GEMM using MTE | Single fixed shape |
| `allgather-gemm-udma.py` | AllGather fused with GEMM using UDMA | Reads and iterates over M/K/N combinations from  `allgather-gemm-case.csv`. |
| `alltoallv-gmm-udma.py` | UDMA-based fused AllToAllV and GEMM operator | 300 fixed cases in `gmm-alltoallv-cases.csv`; FP16, topk=1/2/8, experts per rank=8/40/160; indexed-reorder reference |
| `alltoallv-gmm.py` | MTE AllToAllV-GMM | One shape from `TRITON_M/K/N`; separate first-warmup timing |
| `gemm-allreduce-oneshot-notify.py` | GEMM fused with AllReduce using MTE and notify/wait synchronization | Single fixed shape; resets `C` and `signal_mem` |
| `gemm-allreduce-oneshot.py` | GEMM fused with one-shot AllReduce using MTE | Single fixed shape; `prepare=C.zero_` |
| `gemm-allreduce-udma.py` | GEMM fused with AllReduce using UDMA | BF16; reads M/K/N from `allgather-gemm-case.csv` and tuned overrides from `gemm-allreduce-udma-config.csv`; selects one-shot or Owner-RSAG by shape |
| `gemm-reduce-scatter-udma.py` | GEMM fused with ReduceScatter using UDMA | Single fixed shape; normalized-error verification |
| `gemm-reduce-scatter.py` | GEMM fused with ReduceScatter using MTE | Single fixed shape; `prepare=C.zero_` |
| `moe-combine-udma.py` | UDMA MoE Combine | Full-recipe timing; per-iteration maximum across ranks; MoE parameter sweep |
| `moe-dispatch-udma.py` | UDMA MoE Dispatch | Full-recipe timing; per-iteration maximum across ranks; MoE parameter sweep |
| `qkv-all2all-udma.py` | UDMA QKV All-to-All | `QKV_BENCH_S` / `QKV_BENCH_H` |
| `qkv-all2all.py` | MTE QKV All-to-All | `QKV_BENCH_S` / `QKV_BENCH_H` |
| `reverse-a2a-norm-matmul-2.py` | Fused Reverse A2A, RMSNorm, and MatMul using UDMA | Dual-input variant; `FUSED_RUN_BENCH=1` enables the shape sweep |
| `reverse-a2a-norm-matmul.py` | Fused Reverse A2A, RMSNorm, and MatMul using UDMA | Single-input variant; `FUSED_RUN_BENCH=1` enables the shape sweep |
| `reverse-all2all-udma.py` | UDMA Reverse All-to-All | 45 × 11 shape combinations before sampling |
| `reverse-all2all.py` | MTE Reverse All-to-All | Built-in sequence-length/head-count sweep |
| `sp_ulysses_qkv_gemm_all2all.py` | Ulysses QKV GEMM fused with All2All | BF16/FP16; independent `--hidden-dim K`, `--out-features N`, global `--seq-len S_total`, `--head-dim D`, `--gqa G` (batch size 1) |
| `transpose-all2all-udma.py` | UDMA Transpose All-to-All | `A2A_BENCH_S` / `A2A_BENCH_H`; retains `SKIP (OOM)` |
| `transpose-all2all.py` | MTE Transpose All-to-All | Built-in sequence-length/head-count sweep |
| `dispatch-gmm-udma.py` | UDMA Dispatch GMM | 36 random cases; full-recipe max-rank Event timing; BF16 output tolerance 1e-3 |
