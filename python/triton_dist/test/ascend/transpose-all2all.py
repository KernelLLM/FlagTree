# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for transpose-all2all.py (MTE transpose all-to-all).

Correctness path lives in tutorials/ascend/transpose-all2all.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    ASCEND_PROFILE_WARMUP=1 ASCEND_PROFILE_ITERS=2 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/transpose-all2all.py
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

tut = load_tutorial("transpose-all2all")


def main():
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    init_aclshmem(pe, world_size, tut.ash_pool_bytes(world_size), ash.OpEngineType.MTE)

    # Build test cases: S_RANGES x H_LIST
    test_cases = [(s_val, h_val) for s_val in tut.S_RANGES for h_val in tut.H_LIST]
    HEAD_DIM = tut.HEAD_DIM
    SIGNAL_SLOT_STRIDE = tut.SIGNAL_SLOT_STRIDE

    test_cases = select_cases(test_cases)
    total_cases = len(test_cases)
    if pe == 0:
        print(
            f"[START] Starting transpose All2All sweep tests "
            f"({total_cases} cases) on world_size={world_size}...",
            flush=True,
        )

    try:
        for case_idx, (S, H) in enumerate(test_cases):
            dist.barrier()

            dtype = torch.bfloat16
            n_head = H * world_size
            D = HEAD_DIM

            COMM_BLOCK_S = 128
            COMM_BLOCK_D = 128
            num_blocks_s = (S + COMM_BLOCK_S - 1) // COMM_BLOCK_S
            num_blocks_d = (D + COMM_BLOCK_D - 1) // COMM_BLOCK_D
            buffer_num = num_blocks_s

            # Allocate symmetric memory
            peer_mem_size = COMM_BLOCK_S * (H * world_size) * D * buffer_num
            peer_mem = ash.aclshmem_create_tensor(
                [peer_mem_size],
                dtype=dtype,
                device_id=local_pe,
            )
            signal_mem = ash.aclshmem_create_tensor(
                [buffer_num * world_size * num_blocks_d * H * SIGNAL_SLOT_STRIDE],
                dtype=torch.int32,
                device_id=local_pe,
            )
            signal_mem.fill_(0)

            # Create input/output tensors with seeded data
            torch.manual_seed(42 + case_idx * world_size + pe)
            A_local = torch.randn([S, n_head, D], dtype=dtype).npu()
            C_local = torch.zeros([S * world_size, H, D], dtype=dtype).npu()

            @prof(label="transpose_all2all")
            def run_golden():
                return tut.torch_transpose_a2a(A_local, world_size)

            @prof(label="transpose_all2all", prepare=lambda: signal_mem.fill_(0))
            def run_kernel():
                tut.hccl_transpose_a2a_kernel_launcher(
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

            # Release symmetric memory
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

    dist.barrier()
    main()
    if local_pe == 0:
        print("[INFO] Perf run finished successfully")
