# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for qkv-all2all.py (MTE QKV all-to-all).

Correctness path lives in tutorials/ascend/qkv-all2all.py; this
script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    QKV_BENCH_S=2048,4096 QKV_BENCH_H=1,2 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/qkv-all2all.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist

from prof import (
    GREEN,
    RED,
    RESET,
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
from utils import init_aclshmem

tut = load_tutorial("qkv-all2all")

COMM_BLOCK_S = tut.COMM_BLOCK_S
COMM_BLOCK_D = tut.COMM_BLOCK_D


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    device = torch.npu.current_device()

    default_sequence_lengths = (
        list(range(2040, 2049)) + list(range(4088, 4097)) + list(range(8184, 8193))
    )
    sequence_lengths = [
        int(value)
        for value in os.environ.get(
            "QKV_BENCH_S",
            ",".join(str(value) for value in default_sequence_lengths),
        ).split(",")
    ]
    local_heads = [
        int(value)
        for value in os.environ.get("QKV_BENCH_H", "1,2,3,4,5,6,7").split(",")
    ]

    dist.barrier()
    init_aclshmem(
        rank,
        rank_size,
        tut.ash_pool_bytes(rank_size, sequence_lengths, local_heads),
        ash.OpEngineType.MTE,
    )

    try:
        for sequence_length, heads_per_rank in select_cases(
            (s, h) for s in sequence_lengths for h in local_heads
        ):
            n_head = heads_per_rank * rank_size

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
            output_shape = (
                sequence_length * rank_size,
                heads_per_rank,
                COMM_BLOCK_D,
            )
            q_out = torch.empty(output_shape, dtype=q.dtype, device=device)
            k_out = torch.empty_like(q_out)
            v_out = torch.empty_like(q_out)

            @prof(label="qkv_all2all")
            def run_golden():
                return tut.qkv_fuse_a2a_golden(q, k, v, rank_size)

            peer_elements, signal_elements, buffer_num = tut.workspace_sizes(
                sequence_length, n_head, rank_size
            )
            peer_mem = ash.aclshmem_create_tensor(
                [peer_elements], dtype=q.dtype, device_id=local_rank
            )
            signal_mem = ash.aclshmem_create_tensor(
                [signal_elements], dtype=torch.int32, device_id=local_rank
            )
            peer_mem.zero_()
            signal_mem.zero_()
            dist.barrier()

            @prof(label="qkv_all2all", prepare=signal_mem.zero_)
            def run_kernel():
                tut.qkv_fuse_a2a(
                    q,
                    k,
                    v,
                    q_out,
                    k_out,
                    v_out,
                    peer_mem,
                    signal_mem,
                    rank,
                    rank_size,
                    buffer_num,
                )

            q_golden, k_golden, v_golden = run_golden()
            run_kernel()
            prof_report(
                run_golden, run_kernel, context=f"S={sequence_length} H={n_head}"
            )

            if verify_enabled():
                # Accuracy check
                passed = torch.tensor([1], dtype=torch.int32, device=device)
                error_msg = ""
                try:
                    torch.testing.assert_close(q_out, q_golden, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(k_out, k_golden, rtol=1e-3, atol=1e-3)
                    torch.testing.assert_close(v_out, v_golden, rtol=1e-3, atol=1e-3)
                except AssertionError as e:
                    passed[0] = 0
                    error_msg = str(e)
                all_passed = [
                    torch.zeros(1, dtype=torch.int32, device=device)
                    for _ in range(rank_size)
                ]
                dist.all_gather(all_passed, passed)
                dist.barrier()
                for rank_id in range(rank_size):
                    if rank == rank_id:
                        if all_passed[rank_id].item() == 1:
                            print(
                                f"  {GREEN}[PASS]{RESET} Rank {rank}: "
                                f"S={sequence_length}, H={n_head} matches golden.",
                                flush=True,
                            )
                        else:
                            print(
                                f"  {RED}[FAIL]{RESET} Rank {rank}: "
                                f"S={sequence_length}, H={n_head} failed. "
                                f"Details:\n{error_msg}",
                                flush=True,
                            )
                    dist.barrier()

            ash.aclshmem_free_tensor(peer_mem)
            ash.aclshmem_free_tensor(signal_mem)
            dist.barrier()

            if verify_enabled() and any(r.item() == 0 for r in all_passed):
                raise AssertionError(
                    f"Parameter sweep failed at case: "
                    f"S={sequence_length}, H={n_head}"
                )
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    main()
