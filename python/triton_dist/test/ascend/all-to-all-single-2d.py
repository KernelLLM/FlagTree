# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Correctness and performance sweep for the All-to-All-Single 2D kernel.

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/all-to-all-single-2d.py \
      --mode=both --case=random
"""

import gc
import os

import numpy as np
import shmem as ash
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # Register the torch.npu backend.

from prof import (
    check_all_ranks_close,
    load_kernel,
    perf_enabled,
    prof,
    prof_report,
    run_options,
    select_cases,
)
from utils import init_aclshmem

kernel_module = load_kernel("all_to_all_single_2d")


SHMEM_BYTES = 8 * 1024 * 1024 * 1024
DTYPES = (torch.bfloat16, torch.float16, torch.int8, torch.int32)


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, SHMEM_BYTES, ash.OpEngineType.UDMA)


def _make_random_splits(rank, world_size, max_m, seed):
    """Generate deterministic unequal splits that sum exactly to max_m."""
    remaining = max_m - world_size
    if remaining < 0:
        raise ValueError(f"max_m={max_m} is too small for world_size={world_size}")
    rng = np.random.default_rng(seed + rank)
    extra = rng.multinomial(remaining, np.ones(world_size) / world_size)
    return [1 + int(value) for value in extra]


def _make_hotspot_splits(world_size, max_m):
    if max_m < world_size:
        raise ValueError(f"max_m={max_m} is too small for hotspot splits")
    return [max_m - (world_size - 1)] + [1] * (world_size - 1)


def _resolve_input_splits(rank, world_size, max_m, pattern, seed):
    if pattern == "random":
        return _make_random_splits(rank, world_size, max_m, seed)
    if pattern == "hotspot":
        return _make_hotspot_splits(world_size, max_m)
    raise ValueError(f"unknown split pattern: {pattern}")


def _make_test_input(input_rows, hidden_dim, rank, dtype, device, seed):
    torch.manual_seed(seed + rank)
    shape = (input_rows, hidden_dim)
    if dtype.is_floating_point:
        return (-2 * torch.rand(shape, dtype=dtype, device=device) + 1) * (rank + 1)
    return torch.randint(-8, 9, shape, dtype=dtype, device=device) * (rank + 1)


def _cases():
    m_shapes = sorted({base + offset for base in (2**12, 2**13, 2**14, 2**15) for offset in (-1, 0, 1)})
    return [
        (m, n, split_pattern, dtype)
        for m in m_shapes
        for n in (36 * 128, 40 * 128, 44 * 128)
        for split_pattern in ("random", "hotspot")
        for dtype in DTYPES
    ]


def run_case(max_m, hidden_dim, split_pattern, dtype, rank, world_size, device):
    seed = 2026 + max_m * hidden_dim
    input_splits_cpu = _resolve_input_splits(rank, world_size, max_m, split_pattern, seed)
    input_splits = torch.tensor(input_splits_cpu, dtype=torch.int32, device=device)
    output_splits = torch.empty_like(input_splits)
    dist.all_to_all_single(output_splits, input_splits)
    output_splits_cpu = output_splits.cpu().tolist()
    input_tensor = _make_test_input(
        sum(input_splits_cpu), hidden_dim, rank, dtype, device, seed
    )
    output = torch.empty((sum(output_splits_cpu), hidden_dim), dtype=dtype, device=device)
    expected = torch.empty_like(output)
    context = kernel_module.create_all_to_all_single_2d_context(
        max_m, hidden_dim, rank, world_size, dtype
    )

    @prof(label="torch_all_to_all_single")
    @torch.no_grad()
    def run_golden():
        dist.all_to_all_single(
            expected,
            input_tensor,
            output_splits_cpu,
            input_splits_cpu,
        )
        return expected

    @prof(label="all_to_all_single_2d", prepare=context.reset)
    @torch.no_grad()
    def run_kernel():
        return kernel_module.all_to_all_single_2d(
            context,
            input_tensor,
            output,
            input_splits,
            output_splits,
        )

    case = f"M={max_m} N={hidden_dim} pattern={split_pattern} dtype={dtype}"
    try:
        if perf_enabled():
            run_golden.__wrapped__()
            run_kernel.__wrapped__()
            torch.npu.synchronize()
            dist.barrier()

        expected_result = run_golden()
        actual = run_kernel()
        prof_report(run_golden, run_kernel, context=case)
        check_all_ranks_close(
            actual,
            expected_result,
            rtol=0.0,
            atol=0.0,
            context=case,
        )
    finally:
        context.finalize()
        dist.barrier()


def main():
    options = run_options()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    shmem_initialized = False
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"npu:{local_rank}")
        cases = select_cases(_cases())
        dist.barrier()
        _init_aclshmem(rank, world_size)
        shmem_initialized = True
        torch.manual_seed(2026 + rank)
        if rank == 0:
            print(
                f"[CONFIG] ranks={world_size} mode={options.mode} "
                f"case={options.case} selected={len(cases)}",
                flush=True,
            )
        for index, (max_m, hidden_dim, split_pattern, dtype) in enumerate(cases, start=1):
            if rank == 0:
                print(
                    f"[CASE {index}/{len(cases)}] M={max_m} N={hidden_dim} "
                    f"pattern={split_pattern} dtype={dtype}",
                    flush=True,
                )
            run_case(max_m, hidden_dim, split_pattern, dtype, rank, world_size, device)
            gc.collect()
            torch.npu.empty_cache()
        if rank == 0:
            print(f"[DONE] Completed {len(cases)} cases in {options.mode} mode.", flush=True)
    finally:
        try:
            if shmem_initialized:
                torch.npu.synchronize()
                result = ash.aclshmem_finalize()
                if result not in (None, 0):
                    raise RuntimeError(f"aclshmem_finalize failed with code {result}")
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
