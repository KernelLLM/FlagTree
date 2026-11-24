# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for qkv-all2all-udma.py (UDMA QKV all-to-all).

Correctness path lives in tutorials/ascend/qkv-all2all-udma.py; this
script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    QKV_BENCH_S=2048,4096 QKV_BENCH_H=1,2 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/qkv-all2all-udma.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist
from triton.runtime import driver

from prof import (
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
from utils import init_aclshmem

tut = load_tutorial("qkv-all2all-udma")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    device = torch.npu.current_device()

    COMM_BLOCK_S = tut.COMM_BLOCK_S
    COMM_BLOCK_D = tut.COMM_BLOCK_D

    default_sequence_lengths = (
        list(range(2040, 2049))
        + list(range(4088, 4097))
        + list(range(8184, 8193))
        + list(range(32760, 32769))
        + list(range(131064, 131073))
    )
    sequence_lengths = tut._parse_int_list("QKV_BENCH_S", default_sequence_lengths)
    local_heads = tut._parse_int_list("QKV_BENCH_H", range(1, 8))
    expected_cases = len(sequence_lengths) * len(local_heads)

    if rank == 0:
        print(
            "CONFIG "
            f"engine=UDMA-event-s32k-chunk ranks={rank_size} "
            f"aicore_num={driver.active.utils.get_aicore_num()} "
            f"vector_cores={driver.active.utils.get_aivector_core_num()} "
            f"block_s={COMM_BLOCK_S} "
            f"head_group_override={tut.COMM_HEAD_GROUP_OVERRIDE} "
            f"cases={expected_cases}",
            flush=True,
        )

    dist.barrier()
    init_aclshmem(
        rank,
        rank_size,
        tut.ash_pool_bytes(rank_size, sequence_lengths, local_heads),
        ash.OpEngineType.UDMA,
    )

    try:
        for sequence_length, heads_per_rank in select_cases(
            (s, h) for s in sequence_lengths for h in local_heads
        ):
            n_head = heads_per_rank * rank_size
            case_block_s = COMM_BLOCK_S

            torch.manual_seed(42 + rank)
            q = torch.randn(
                sequence_length,
                n_head,
                COMM_BLOCK_D,
                dtype=torch.bfloat16,
                device=device,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Keep the same contiguous QKV output contract as the UDMA
            # kernel requires, while exposing Golden-compatible views.
            qkv_out = torch.empty(
                sequence_length * rank_size,
                3 * heads_per_rank,
                COMM_BLOCK_D,
                dtype=q.dtype,
                device=device,
            )
            q_out, k_out, v_out = qkv_out.split([heads_per_rank] * 3, dim=1)

            @prof(label="qkv_all2all_udma")
            def run_golden():
                return tut.qkv_fuse_a2a_golden(q, k, v, rank_size)

            peer_elements = tut.workspace_size(
                sequence_length,
                n_head,
                rank_size,
                comm_block_s=case_block_s,
            )
            peer_mem = ash.aclshmem_create_tensor(
                [peer_elements], dtype=q.dtype, device_id=local_rank
            )
            torch.npu.synchronize()
            dist.barrier()

            # Validate once and retain the low-overhead launcher.
            udma_call = tut._prepare_qkv_fuse_a2a(
                q,
                k,
                v,
                q_out,
                k_out,
                v_out,
                peer_mem,
                rank,
                rank_size,
                comm_block_s=case_block_s,
            )
            try:

                @prof(label="qkv_all2all_udma")
                def run_kernel():
                    udma_call()

                q_golden, k_golden, v_golden = run_golden()
                run_kernel()
                prof_report(
                    run_golden,
                    run_kernel,
                    context=(
                        f"S={sequence_length} H={heads_per_rank} "
                        f"block_s={udma_call.block_s} "
                        f"hgroup={udma_call.heads_per_group} "
                        f"cores={udma_call.launch_ai_cores}"
                    ),
                )

                if verify_enabled():
                    tut._check_accuracy(
                        q,
                        q_golden,
                        k_golden,
                        v_golden,
                        q_out,
                        k_out,
                        v_out,
                        rank,
                        rank_size,
                    )
            finally:
                ash.aclshmem_free_tensor(peer_mem)
                dist.barrier()
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    main()
