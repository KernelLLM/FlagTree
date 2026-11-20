# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for reverse-all2all.py (MTE wait/notify reverse A2A).

Correctness path lives in tutorials/ascend/reverse-all2all.py; this
script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/reverse-all2all.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist

from prof import (
    check_all_ranks_close,
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
)
from utils import init_aclshmem

tut = load_tutorial("reverse-all2all")

SIGNAL_SLOT_STRIDE = 16


def main():
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # Generate contiguous S and H dimension test suites
    S_ranges = (
        list(range(2040, 2049))  # 2040 to 2048 (9 values)
        + list(range(4088, 4097))  # 4088 to 4096 (9 values)
        + list(range(8184, 8193))  # 8184 to 8192 (9 values)
    )
    H_list = [8, 12, 24, 28, 32, 40, 48, 56]
    D = 128

    init_aclshmem(
        pe, world_size, tut.ash_pool_bytes(world_size, D, H_list), ash.OpEngineType.MTE
    )

    test_cases = [(s_val, h_val) for s_val in S_ranges for h_val in H_list]
    test_cases = select_cases(test_cases)
    total_cases = len(test_cases)
    if pe == 0:
        print(
            f"[START] Sweeping {total_cases} cases on world_size={world_size}...",
            flush=True,
        )

    try:
        for case_idx, (S, H) in enumerate(test_cases):
            dist.barrier()
            dtype = torch.bfloat16
            S_total = S * world_size

            COMM_BLOCK_S = 64
            COMM_BLOCK_D = 128
            buffer_num = 2

            peer_mem_size = COMM_BLOCK_S * (H * world_size) * D * buffer_num
            peer_mem = ash.aclshmem_create_tensor(
                [peer_mem_size],
                dtype=dtype,
                device_id=local_pe,
            )
            signal_mem = ash.aclshmem_create_tensor(
                [H * world_size * buffer_num * SIGNAL_SLOT_STRIDE],
                dtype=torch.int32,
                device_id=local_pe,
            )
            signal_mem.fill_(0)

            A_local = torch.ones([S_total, H, D], dtype=dtype).npu()
            C_local = torch.zeros([S, H * world_size, D], dtype=dtype).npu()

            @prof(label="reverse_all2all")
            def run_golden():
                return tut.torch_reverse_a2a(A_local, world_size)

            @prof(label="reverse_all2all", prepare=lambda: signal_mem.fill_(0))
            def run_kernel():
                tut.hccl_reverse_a2a_kernel_launcher(
                    A_local,
                    C_local,
                    peer_mem,
                    signal_mem,
                    pe,
                    world_size,
                    buffer_num,
                    COMM_BLOCK_S,
                    COMM_BLOCK_D,
                )

            C_golden = run_golden()
            run_kernel()
            prof_report(run_golden, run_kernel, context=f"S={S} H={H}")
            check_all_ranks_close(
                C_local, C_golden, rtol=1e-3, atol=1e-3, context=f"S={S}, H={H}"
            )

            ash.aclshmem_free_tensor(peer_mem)
            ash.aclshmem_free_tensor(signal_mem)
    finally:
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    run_options()
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)

    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialized")
    dist.barrier()
    main()
    if local_pe == 0:
        print(f"[INFO] Perf run passed successfully for {dist.get_world_size()} ranks!")
