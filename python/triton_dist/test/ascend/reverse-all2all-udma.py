# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for reverse-all2all-udma.py (UDMA reverse A2A).

Correctness path lives in tutorials/ascend/reverse-all2all-udma.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/reverse-all2all-udma.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist
from triton.runtime import driver

from prof import (
    check_all_ranks_close,
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
)
from utils import init_aclshmem

tut = load_tutorial("reverse-all2all-udma")


def main():
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # Default sweeps match the tutorial.
    S_ranges = (
        list(range(2040, 2049))  # 2040 to 2048 (9 values)
        + list(range(4088, 4097))  # 4088 to 4096 (9 values)
        + list(range(8184, 8193))  # 8184 to 8192 (9 values)
        + list(range(32760, 32769))  # 32760 to 32769 (9 values)
        + list(range(131064, 131073))  # 131064 to 131073 (9 values)
    )
    H_list = [3, 6, 8, 12, 16, 24, 28, 32, 40, 48, 56]
    D = 128
    COMM_BLOCK_S = 128
    COMM_BLOCK_D = D
    buffer_num = 2

    if pe == 0:
        print(
            f"CONFIG engine=reverse-a2a-event ranks={world_size} "
            f"aicore_num={driver.active.utils.get_aicore_num()} "
            f"vector_cores={driver.active.utils.get_aivector_core_num()} "
            f"block_s={COMM_BLOCK_S} block_d={COMM_BLOCK_D} "
            f"buffer_num={buffer_num} cases={len(S_ranges) * len(H_list)}",
            flush=True,
        )

    dist.barrier()
    init_aclshmem(
        pe,
        world_size,
        tut.ash_pool_bytes(world_size, S_ranges, H_list),
        ash.OpEngineType.UDMA,
    )

    try:
        for S, H in select_cases(((S, H) for S in S_ranges for H in H_list)):
            S_total = S * world_size
            peer_mem_size = world_size * S * H * D
            peer_mem = ash.aclshmem_create_tensor(
                [peer_mem_size], dtype=torch.bfloat16, device_id=local_pe
            )
            num_blocks_s = -(-S // COMM_BLOCK_S)
            num_blocks_d = -(-D // COMM_BLOCK_D)
            signal_mem_size = (
                num_blocks_s * world_size + buffer_num * num_blocks_d * H
            ) * 8
            signal_mem = ash.aclshmem_create_tensor(
                [signal_mem_size], dtype=torch.int64, device_id=local_pe
            )
            torch.npu.synchronize()
            dist.barrier()

            torch.manual_seed(42 + pe)
            A_local = torch.randn(
                [S_total, H, D], dtype=torch.bfloat16, device="npu"
            )
            C_local = torch.zeros(
                [S, H * world_size, D], dtype=torch.bfloat16, device="npu"
            )

            launch = tut._prepare_reverse_launch(
                A_local,
                C_local,
                peer_mem,
                signal_mem,
                pe,
                world_size,
                buffer_num,
                S,
                H,
                D,
                COMM_BLOCK_S,
                COMM_BLOCK_D,
            )

            @prof(label="reverse_all2all_udma")
            def run_golden():
                return tut.torch_reverse_a2a(A_local, world_size)

            @prof(label="reverse_all2all_udma", prepare=lambda: signal_mem.fill_(0))
            def run_kernel():
                launch()

            try:
                C_golden = run_golden()
                run_kernel()
                prof_report(run_golden, run_kernel, context=f"S={S} H={H}")
                check_all_ranks_close(
                    C_local,
                    C_golden,
                    rtol=1e-3,
                    atol=1e-3,
                    context=f"S={S}, H={H}",
                )
            finally:
                ash.aclshmem_free_tensor(peer_mem)
                ash.aclshmem_free_tensor(signal_mem)
                dist.barrier()
    finally:
        ash.aclshmem_finalize()


if __name__ == "__main__":
    run_options()
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl")

    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialized")
    dist.barrier()
    main()
    if local_pe == 0:
        print(f"[INFO] Perf run passed successfully for {dist.get_world_size()} ranks!")
