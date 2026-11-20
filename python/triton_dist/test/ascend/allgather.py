# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Correctness and performance sweep for the Ascend AllGather kernel.

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/allgather.py \
      --mode=both --case=random
"""

import gc
import os

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

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

kernel_module = load_kernel("allgather")

SHMEM_BYTES = 8 * 1024 * 1024 * 1024
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
M_SHAPES = tuple(sorted({base + offset for base in (4096, 8192, 16384, 32768) for offset in (-1, 0, 1)}))
HIDDEN_DIMS = tuple(value * 128 for value in (36, 40, 44))

# The kernel copies the local shard in 8192-element tiles. UDMA transfers are
# split at 256 MB, which corresponds to M=16384 for 8192-wide 2-byte tensors
# and M=8192 for 8192-wide 4-byte tensors.
BOUNDARY_CASES = (
    (1, 8191, torch.bfloat16),
    (1, 8192, torch.bfloat16),
    (1, 8193, torch.bfloat16),
    (16383, 8192, torch.bfloat16),
    (16384, 8192, torch.bfloat16),
    (16385, 8192, torch.bfloat16),
    (8191, 8192, torch.float32),
    (8192, 8192, torch.float32),
    (8193, 8192, torch.float32),
)


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, SHMEM_BYTES, ash.OpEngineType.UDMA)


def _cases():
    matrix_cases = [
        (m, hidden_dim, dtype)
        for m in M_SHAPES
        for hidden_dim in HIDDEN_DIMS
        for dtype in DTYPES
    ]
    return matrix_cases + list(BOUNDARY_CASES)


def _torch_all_gather_into(input_tensor, output_tensor):
    if hasattr(dist, "all_gather_into_tensor"):
        dist.all_gather_into_tensor(output_tensor, input_tensor)
    else:
        dist.all_gather(list(output_tensor.chunk(dist.get_world_size(), dim=0)), input_tensor)
    return output_tensor


def run_case(m, hidden_dim, dtype, case_idx, rank, world_size, device, store):
    torch.manual_seed(2026 + rank + case_idx * world_size)
    input_tensor = torch.randn((m, hidden_dim), dtype=dtype, device=device)
    expected = torch.empty((world_size * m, hidden_dim), dtype=dtype, device=device)
    output = torch.empty_like(expected)
    context = kernel_module.create_allgather_context(
        m, hidden_dim, rank, world_size, dtype
    )

    @prof(label="torch_all_gather")
    @torch.no_grad()
    def run_golden():
        return _torch_all_gather_into(input_tensor, expected)

    @prof(label="allgather", prepare=context.reset)
    @torch.no_grad()
    def run_kernel():
        return kernel_module.allgather(context, input_tensor, output)

    case = f"M={m} N={hidden_dim} dtype={dtype}"
    try:
        if perf_enabled():
            run_golden.__wrapped__()
            run_kernel.__wrapped__()
            torch.npu.synchronize()
            dist.barrier()

        golden = run_golden()
        actual = run_kernel()
        prof_report(run_golden, run_kernel, context=case)
        check_all_ranks_close(
            actual,
            golden,
            rtol=0.0,
            atol=0.0,
            context=case,
        )
    finally:
        torch.npu.synchronize()
        dist.barrier()
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
        from torch.distributed.distributed_c10d import _get_default_store

        store = _get_default_store()
        device = torch.device(f"npu:{local_rank}")
        cases = select_cases(_cases())
        dist.barrier()
        _init_aclshmem(rank, world_size)
        shmem_initialized = True
        if rank == 0:
            print(
                f"[CONFIG] ranks={world_size} mode={options.mode} "
                f"case={options.case} selected={len(cases)}",
                flush=True,
            )
        for case_idx, (m, hidden_dim, dtype) in enumerate(cases):
            if rank == 0:
                print(
                    f"[CASE {case_idx + 1}/{len(cases)}] M={m} "
                    f"N={hidden_dim} dtype={dtype}",
                    flush=True,
                )
            run_case(m, hidden_dim, dtype, case_idx, rank, world_size, device, store)
            # Some NPU/HCCL/Triton runtime objects participate in Python
            # reference cycles and keep the case tensors alive after
            # run_case() returns.  Their device storage can be several GB,
            # while the cyclic GC only sees a small number of Python objects,
            # so it may not run before the next large allocation.
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
