# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Full-recipe performance test for the fused Ascend UDMA Dispatch GMM.

Run with torchrun, e.g. --nproc_per_node=8, from the repository root.
Default: all 36 random cases, correctness + timing, warmup 10 / iterations 30.
Shared options: --mode=verify|perf|both and --case=full|random (up to ten cases).
ASCEND_PROFILE_WARMUP / ASCEND_PROFILE_ITERS override the defaults.
Timings include routing inversion, counts/offset exchange,
packing, UDMA, public-layout restoration, GMM and host count-list conversion.
Only public inputs and data-independent workspace allocation stay outside timing.
"""

import os

import torch
import torch.distributed as dist

from prof import (
    distributed_max_times,
    load_tutorial,
    perf_enabled,
    prof,
    run_options,
    select_cases,
    timing_stats,
    verify_enabled,
)

WARMUP = int(os.environ.get("ASCEND_PROFILE_WARMUP", "10"))
ITERS = int(os.environ.get("ASCEND_PROFILE_ITERS", "30"))
if WARMUP < 0 or ITERS <= 0:
    raise ValueError("warmup must be nonnegative and iterations must be positive")


# (M, K, N): tiny/empty buckets, aligned tiles, tails, and larger random cases.
CASE_SHAPES = (
    (7, 128, 128),
    (127, 129, 255),
    (128, 256, 256),
    (129, 257, 257),
    (511, 511, 511),
    (512, 512, 512),
    (513, 513, 960),
    (2047, 1536, 960),
    (2048, 1536, 960),
    (4097, 1536, 960),
    (8192, 1536, 960),
    (32768, 1536, 960),
)
# (top_k, local experts); never exceeds the tested compiler scale.
CASE_ROUTING = ((1, 2), (2, 4), (8, 8))
TEST_CASES = [
    (m, k, n, top_k, experts)
    for m, k, n in CASE_SHAPES
    for top_k, experts in CASE_ROUTING
]


def _make_case_executor(speedups, median_speedups):
    def execute(golden_recipe, candidate_recipe, case):
        timed_golden = prof(
            golden_recipe, label="moe_dispatch_gmm_golden", warmup=WARMUP, iters=ITERS
        )
        timed_candidate = prof(
            candidate_recipe, label="moe_dispatch_gmm_udma", warmup=WARMUP, iters=ITERS
        )
        golden = timed_golden()
        candidate = timed_candidate()
        if not perf_enabled():
            return golden, candidate
        golden_stats = timing_stats(distributed_max_times(timed_golden.prof.times_ms))
        candidate_stats = timing_stats(
            distributed_max_times(timed_candidate.prof.times_ms)
        )
        speedup = golden_stats["mean"] / candidate_stats["mean"]
        median_speedup = golden_stats["median"] / candidate_stats["median"]
        if dist.get_rank() == 0:
            speedups.append(speedup)
            median_speedups.append(median_speedup)
            print(
                f"  [PERF recipe] metric=npu_event, max-rank mean: "
                f"golden={golden_stats['mean']:.4f} ms, "
                f"triton={candidate_stats['mean']:.4f} ms, speedup={speedup:.3f}x; "
                f"median: golden={golden_stats['median']:.4f} ms, "
                f"triton={candidate_stats['median']:.4f} ms, "
                f"speedup={median_speedup:.3f}x; "
                f"range: golden=[{golden_stats['min']:.4f}, {golden_stats['max']:.4f}] ms, "
                f"triton=[{candidate_stats['min']:.4f}, {candidate_stats['max']:.4f}] ms",
                flush=True,
            )
            print(
                "EVENT_TIMING "
                f"op=moe_dispatch_gmm_udma case={case['case']} route={case['route']} "
                f"m={case['m']} k={case['k']} n={case['n']} topk={case['topk']} "
                f"local_experts={case['local_experts']} ranks={case['ranks']} "
                f"golden_ms={golden_stats['mean']:.4f} "
                f"candidate_ms={candidate_stats['mean']:.4f} speedup={speedup:.3f}",
                flush=True,
            )
        return golden, candidate

    return execute


def main():
    tutorial = load_tutorial("dispatch-gmm-udma")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        if dist.get_rank() == 0 and perf_enabled():
            print(
                f"[PERF CONFIG] warmup={WARMUP}, iters={ITERS}, "
                "boundary=public_api_complete_recipe",
                flush=True,
            )
        speedups, median_speedups = [], []
        tutorial.run_test_distributed(
            select_cases(list(enumerate(TEST_CASES, start=1))),
            iterations=1,
            case_executor=_make_case_executor(speedups, median_speedups),
            verify=verify_enabled(),
        )
        if dist.get_rank() == 0 and speedups:
            print(
                f"\n[PERF SUMMARY] metric=npu_event, cases={len(speedups)}, "
                f"arithmetic_mean_speedup={sum(speedups)/len(speedups):.3f}x, "
                "arithmetic_mean_median_speedup="
                f"{sum(median_speedups)/len(median_speedups):.3f}x",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    main()
