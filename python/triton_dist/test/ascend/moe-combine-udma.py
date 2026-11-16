# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Full-recipe performance harness for the Ascend UDMA MoE Combine operator."""

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

WARMUP = int(
    os.environ.get("MOE_PROFILE_WARMUP", os.environ.get("ASCEND_PROFILE_WARMUP", "5"))
)
ITERS = int(
    os.environ.get("MOE_PROFILE_ITERS", os.environ.get("ASCEND_PROFILE_ITERS", "50"))
)

if WARMUP < 0:
    raise ValueError(f"MOE_PROFILE_WARMUP must be >= 0, got {WARMUP}")
if ITERS <= 0:
    raise ValueError(f"MOE_PROFILE_ITERS must be > 0, got {ITERS}")


def _make_case_executor(speedups, median_speedups):
    def execute(golden_recipe, candidate_recipe, case):
        timed_golden = prof(
            golden_recipe, label="moe_combine_golden", warmup=WARMUP, iters=ITERS
        )
        timed_candidate = prof(
            candidate_recipe, label="moe_combine_udma", warmup=WARMUP, iters=ITERS
        )
        golden = timed_golden()
        timed_candidate()
        if not perf_enabled():
            return golden
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
                f"udma={candidate_stats['mean']:.4f} ms, "
                f"speedup={speedup:.3f}x; "
                f"median: golden={golden_stats['median']:.4f} ms, "
                f"triton={candidate_stats['median']:.4f} ms, "
                f"speedup={median_speedup:.3f}x; "
                f"range: golden=[{golden_stats['min']:.4f}, "
                f"{golden_stats['max']:.4f}] ms, "
                f"triton=[{candidate_stats['min']:.4f}, "
                f"{candidate_stats['max']:.4f}] ms",
                flush=True,
            )
            print(
                "EVENT_TIMING "
                f"op=moe_combine_udma route={case['route']} n={case['n']} "
                f"d={case['d']} topk={case['topk']} "
                f"local_experts={case['local_experts']} ranks={case['ranks']} "
                f"golden_ms={golden_stats['mean']:.4f} "
                f"candidate_ms={candidate_stats['mean']:.4f} speedup={speedup:.3f}",
                flush=True,
            )
        return golden

    return execute


def main():
    tutorial = load_tutorial("moe-combine-udma")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank)
    print(f"[INFO] rank {local_rank}/{dist.get_world_size()} initialized", flush=True)
    dist.barrier()
    if local_rank == 0 and perf_enabled():
        print(f"[PERF CONFIG] warmup={WARMUP}, iters={ITERS}", flush=True)
    speedups = []
    median_speedups = []
    tutorial.run(
        case_executor=_make_case_executor(speedups, median_speedups),
        case_selector=select_cases,
        verify=verify_enabled(),
    )
    if local_rank == 0 and speedups:
        print(
            f"[PERF SUMMARY] metric=npu_event, cases={len(speedups)}, "
            f"arithmetic_mean_speedup={sum(speedups) / len(speedups):.3f}x, "
            "arithmetic_mean_median_speedup="
            f"{sum(median_speedups) / len(median_speedups):.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    run_options()
    main()
