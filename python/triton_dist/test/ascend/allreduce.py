# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Correctness and performance sweep for the Ascend AllReduce kernel.

Sweeps 1-D BF16/FP16/FP32 tensors around power-of-two, reduction-tile,
and algorithm-switch boundaries.

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/allreduce.py \
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
    verify_enabled,
)
from utils import init_aclshmem

kernel_module = load_kernel("allreduce")

SHMEM_BYTES = 8 * 1024 * 1024 * 1024
DTYPES = (torch.bfloat16, torch.float16, torch.float32)

SWEEP = [2**exp + offset for exp in range(16, 30) for offset in (-1, 0, 1)]

TOLERANCES = {
    torch.bfloat16: (5e-3, 5e-3),
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-5, 1e-5),
}

def cpu_allreduce_reference(local_input, rank, world_size):
    """Gather rank inputs one at a time and accumulate them in FP32 on CPU."""
    received = torch.empty_like(local_input)
    expected = torch.zeros(local_input.numel(), dtype=torch.float32)
    for peer_rank in range(world_size):
        peer_input = local_input if rank == peer_rank else received
        dist.broadcast(peer_input, src=peer_rank)
        expected.add_(peer_input.cpu())
    return expected.to(device=local_input.device, dtype=local_input.dtype)


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, SHMEM_BYTES, ash.OpEngineType.UDMA)


def _cases():
    cases = []
    for dtype in DTYPES:
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        borders = [
            kernel_module.COMM_BLOCK_SIZE // elem_bytes,
            kernel_module.ONESHOT_TWOSHOT_BORDER_BYTE // elem_bytes,
        ]
        numels = SWEEP + [
            border + offset for border in borders for offset in (-1, 0, 1)
        ]
        cases.extend((numel, dtype) for numel in numels)
    return cases


def run_case(numel, dtype, case_idx, rank, world_size, device):
    torch.manual_seed(2026 + rank + case_idx * world_size)
    input_tensor = torch.randn(numel, dtype=dtype, device=device)
    expected = torch.empty_like(input_tensor)
    actual = torch.empty_like(input_tensor)
    context = kernel_module.AllReduceContext(
        numel,
        rank,
        world_size,
        dtype=dtype,
        device=device,
    )

    def prepare_golden():
        expected.copy_(input_tensor)

    def prepare_kernel():
        actual.copy_(input_tensor)

    @prof(label="torch_all_reduce", prepare=prepare_golden)
    @torch.no_grad()
    def run_golden():
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        return expected

    @prof(label="allreduce", prepare=prepare_kernel)
    @torch.no_grad()
    def run_kernel():
        return kernel_module.allreduce(context, actual)

    case = f"numel={numel} dtype={dtype} path={context.algorithm}"
    try:
        if perf_enabled():
            prepare_golden()
            run_golden.__wrapped__()
            prepare_kernel()
            run_kernel.__wrapped__()
            torch.npu.synchronize()
            dist.barrier()

        expected = run_golden()
        result = run_kernel()
        prof_report(run_golden, run_kernel, context=case)
        atol, rtol = TOLERANCES[dtype]
        if verify_enabled():
            try:
                check_all_ranks_close(
                    result,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    context=f"{case} vs Torch eager",
                )
            except AssertionError:
                cpu_expected = cpu_allreduce_reference(input_tensor, rank, world_size)
                check_all_ranks_close(
                    result,
                    cpu_expected,
                    rtol=rtol,
                    atol=atol,
                    context=f"{case} vs CPU reference",
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
        device = torch.device(f"npu:{local_rank}")
        cases = select_cases(_cases())
        dist.barrier()
        _init_aclshmem(rank, world_size)
        shmem_initialized = True
        if rank == 0:
            print(
                f"[CONFIG] ranks={world_size} mode={options.mode} case={options.case} selected={len(cases)}",
                flush=True,
            )
        for case_idx, (numel, dtype) in enumerate(cases):
            if rank == 0:
                print(
                    f"[CASE {case_idx + 1}/{len(cases)}] numel={numel} dtype={dtype}",
                    flush=True,
                )
            run_case(
                numel,
                dtype,
                case_idx,
                rank,
                world_size,
                device,
            )
            gc.collect()
            torch.npu.empty_cache()
        if rank == 0:
            print(
                f"[DONE] Completed {len(cases)} cases in {options.mode} mode.",
                flush=True,
            )
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
