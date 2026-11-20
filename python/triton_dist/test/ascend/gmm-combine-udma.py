# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Verify or benchmark Ascend GMM Combine UDMA cases.

Examples (default benchmark configuration; set --nproc-per-node to choose the NPU count):
    torchrun --standalone --nproc-per-node=4 python/triton_dist/test/ascend/gmm-combine-udma.py --mode=verify --case=random
    torchrun --standalone --nproc-per-node=4 python/triton_dist/test/ascend/gmm-combine-udma.py --mode=perf --case=full
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist

from prof import (
    distributed_max_times,
    load_tutorial,
    prof,
    run_options,
    select_cases,
    timing_stats,
    verify_enabled,
)


WARMUP = int(os.environ.get("ASCEND_PROFILE_WARMUP", "3"))
ITERS = int(os.environ.get("ASCEND_PROFILE_ITERS", "10"))
CASE_FILE = Path(__file__).with_name("gmm-combine-udma-cases.csv")


def _parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "perf", "both"), default="both")
    parser.add_argument("--case", choices=("full", "random"), default="full")
    parser.add_argument("--case-file", type=Path, default=CASE_FILE,
                        help="CSV case list; defaults to the checked-in cases")
    return parser.parse_args()


def _load_cases(tutorial, case_file):
    cases = []
    with case_file.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            cases.append(
                tutorial.GmmCombineShape(
                    row["name"],
                    int(row["tokens_per_rank"]),
                    int(row["hidden_size"]),
                    int(row["ffn_size"]),
                    int(row["topk"]),
                    int(row["num_experts"]),
                )
            )
    if not cases:
        raise ValueError(f"no benchmark cases found in {case_file}")
    return tuple(cases)


def _make_case_executor(speedups, mode):
    def execute(golden_recipe, candidate_recipe, case):
        if mode == "verify":
            golden = golden_recipe()
            candidate_recipe()
            return golden
        timed_golden = prof(
            golden_recipe,
            label="gmm_combine_golden",
            warmup=WARMUP,
            iters=ITERS,
        )
        timed_candidate = prof(
            candidate_recipe,
            label="gmm_combine_udma",
            warmup=WARMUP,
            iters=ITERS,
        )
        golden = timed_golden()
        timed_candidate()
        golden_stats = timing_stats(
            distributed_max_times(timed_golden.prof.times_ms)
        )
        candidate_stats = timing_stats(
            distributed_max_times(timed_candidate.prof.times_ms)
        )
        speedup = golden_stats["mean"] / candidate_stats["mean"]

        if dist.get_rank() == 0:
            speedups.append(speedup)
            print(
                "EVENT_TIMING "
                f"op=gmm_combine_udma case={case['name']} "
                f"tokens={case['tokens_per_rank']} "
                f"hidden={case['hidden_size']} ffn={case['ffn_size']} "
                f"topk={case['topk']} experts={case['num_experts']} "
                f"ranks={case['rank_count']} "
                f"golden_ms={golden_stats['mean']:.6f} "
                f"candidate_ms={candidate_stats['mean']:.6f} "
                f"speedup={speedup:.3f}x",
                flush=True,
            )
        return golden

    return execute


def main():
    options = _parse_options()
    run_options()
    if options.mode != "verify" and (WARMUP <= 0 or ITERS <= 0):
        raise ValueError("performance modes require positive warmup and iterations")

    tutorial = load_tutorial("gmm-combine-udma")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        rank = dist.get_rank()
        rank_count = dist.get_world_size()
        benchmark_cases = _load_cases(tutorial, options.case_file)
        case_name = os.environ.get("GMM_COMBINE_CASE")
        filtered_cases = tuple(
            case
            for case in benchmark_cases
            if case_name is None or case.name == case_name
        )
        if not filtered_cases:
            raise ValueError(f"unknown GMM_COMBINE_CASE: {case_name}")
        cases = select_cases(filtered_cases)
        if rank == 0:
            print(
                f"[PERF CONFIG] mode={options.mode} selection={options.case} "
                f"cases={len(cases)} "
                f"warmup={0 if options.mode == 'verify' else WARMUP} "
                f"iters={1 if options.mode == 'verify' else ITERS}",
                flush=True,
            )

        speedups = []
        heap_shape = max(
            cases,
            key=lambda shape: tutorial.required_ash_size_mb(shape, rank_count),
        )
        ash_size_mb = tutorial.initialize_shmem(heap_shape)
        try:
            if rank == 0:
                print(f"[PERF CONFIG] ash_heap={ash_size_mb}MB", flush=True)
            executor = _make_case_executor(speedups, options.mode)
            for case_id, case in enumerate(cases):
                if rank == 0:
                    print(
                        f"CASE {case_id + 1}/{len(cases)}: "
                        f"name={case.name} experts={case.num_experts} "
                        f"tokens={case.tokens_per_rank} topk={case.topk} "
                        f"hidden={case.hidden_size} ffn={case.ffn_size}",
                        flush=True,
                    )
                tutorial.run_case(case, case_executor=executor,
                                  verify=verify_enabled())
        finally:
            tutorial.finalize_shmem()
        if rank == 0 and speedups:
            print(
                f"[PERF SUMMARY] cases={len(speedups)} "
                f"mean_speedup={sum(speedups) / len(speedups):.3f}x",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
