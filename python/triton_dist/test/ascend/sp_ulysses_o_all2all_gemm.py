# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Correctness and event timing for the fused Ulysses output All2All and GEMM.

Run from the repository root:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/sp_ulysses_o_all2all_gemm.py \
      --mode=both --case=random

ULYSSES_BENCH_S selects comma-separated local sequence lengths;
ULYSSES_BENCH_H selects comma-separated total head counts. Total heads must
be divisible by world_size. Incompatible default head counts are skipped.
All cases run sequentially within one HCCL and aclshmem session.
"""

import os

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from prof import (
    load_kernel,
    perf_enabled,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
from utils import init_aclshmem

kernel = load_kernel("sp_ulysses_o_all2all_gemm")


DTYPE = torch.bfloat16
HEAD_DIM = 128
OUT_FEATURES = 4096
SHMEM_BYTES = 8 * 1024 * 1024 * 1024
SEQUENCE_LENGTHS = tuple(
    list(range(2040, 2049))
    + list(range(4088, 4097))
    + list(range(8184, 8193))
    + list(range(32760, 32769))
    + list(range(131064, 131073))
)
TOTAL_HEADS = (8, 16, 32, 56)


def _parse_int_list(name, default):
    value = os.environ.get(name)
    if value is None:
        return tuple(default)
    values = tuple(int(item.strip()) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(dict.fromkeys(values))


def make_cases(world_size):
    """Build compatible shape pairs before applying the shared case selector."""
    kernel.SpUlyssesOAll2AllGemmKernel.workspace_size(world_size, 1, HEAD_DIM)
    sequence_lengths = _parse_int_list("ULYSSES_BENCH_S", SEQUENCE_LENGTHS)
    heads = _parse_int_list("ULYSSES_BENCH_H", TOTAL_HEADS)
    if "ULYSSES_BENCH_H" in os.environ and any(h % world_size for h in heads):
        raise ValueError("ULYSSES_BENCH_H values must be divisible by world_size")
    heads = tuple(h for h in heads if h % world_size == 0)
    if not heads:
        raise ValueError(
            "No compatible total head counts; set ULYSSES_BENCH_H to "
            f"positive multiples of world_size={world_size}"
        )
    return [(s, h) for s in sequence_lengths for h in heads]


def input_factory(local_sequence_length, total_heads, rank, world_size, device):
    """Create the original rank-distinct BF16 inputs."""
    attn_out = torch.rand(
        (local_sequence_length * world_size, total_heads // world_size, HEAD_DIM),
        dtype=DTYPE,
        device=device,
    )
    attn_out.mul_(-2).add_(1).mul_(rank + 1)
    weight = torch.rand(
        (OUT_FEATURES, total_heads * HEAD_DIM), dtype=DTYPE, device=device
    )
    weight.mul_(-2).add_(1).mul_(rank + 1)
    return attn_out, weight


def golden(attn_out, weight):
    """Reference HCCL All2All, layout conversion, and GEMM."""
    return torch.matmul(kernel.reference_gemm_input(attn_out, weight), weight.transpose(0, 1))


def run_case(local_sequence_length, total_heads, rank, world_size, local_rank):
    """Allocate once per case and reuse buffers throughout all measured calls."""
    device = torch.device(f"npu:{local_rank}")
    attn_out, weight = input_factory(
        local_sequence_length, total_heads, rank, world_size, device
    )
    output = torch.empty(
        (local_sequence_length, OUT_FEATURES), dtype=DTYPE, device=device
    )
    staging_elems = kernel.SpUlyssesOAll2AllGemmKernel.workspace_size(
        world_size, total_heads // world_size, HEAD_DIM
    )
    if staging_elems * attn_out.element_size() > SHMEM_BYTES:
        raise ValueError("symmetric workspace exceeds the configured aclshmem heap")
    staging = ash.aclshmem_create_tensor(
        [staging_elems], dtype=DTYPE, device_id=local_rank
    )
    try:
        output.fill_(float("nan"))
        staging.fill_(float("nan"))
        operator = kernel.SpUlyssesOAll2AllGemmKernel(rank, world_size, staging)
        torch.npu.synchronize()
        dist.barrier()

        @prof(label="torch_ulysses_o_all2all_gemm")
        def run_golden():
            return golden(attn_out, weight)

        @prof(label="ulysses_o_all2all_gemm")
        def run_kernel():
            return operator.forward(attn_out, weight, output)

        # Compile and initialize this case before timing, even with zero warmup.
        if perf_enabled():
            run_golden.__wrapped__()
            run_kernel.__wrapped__()
            torch.npu.synchronize()
            dist.barrier()

        expected = run_golden()
        actual = run_kernel()
        context = (
            f"S_local={local_sequence_length} H_total={total_heads} "
            f"D={HEAD_DIM} N={OUT_FEATURES} ranks={world_size}"
        )
        prof_report(run_golden, run_kernel, context=context)
        if verify_enabled():
            kernel.check_accuracy(
                actual, expected, attn_out, weight,
                rtol=1e-3, atol=1e-3, context=context,
            )
    finally:
        torch.npu.synchronize()
        dist.barrier()
        ash.aclshmem_free_tensor(staging)


def main():
    options = run_options()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    shmem_initialized = False
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        cases = select_cases(make_cases(world_size))
        init_aclshmem(rank, world_size, SHMEM_BYTES, ash.OpEngineType.UDMA)
        shmem_initialized = True
        dist.barrier()
        torch.manual_seed(42 + rank)
        if rank == 0:
            print(
                f"[CONFIG] ranks={world_size} mode={options.mode} "
                f"case={options.case} selected={len(cases)}",
                flush=True,
            )
        for index, (sequence_length, total_heads) in enumerate(cases, start=1):
            if rank == 0:
                print(
                    f"[CASE {index}/{len(cases)}] S_local={sequence_length} "
                    f"H_total={total_heads}",
                    flush=True,
                )
            run_case(sequence_length, total_heads, rank, world_size, local_rank)
        if rank == 0:
            print(
                f"[DONE] Completed {len(cases)} cases in {options.mode} mode.",
                flush=True,
            )
    finally:
        try:
            if shmem_initialized:
                torch.npu.synchronize()
                ash.aclshmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
