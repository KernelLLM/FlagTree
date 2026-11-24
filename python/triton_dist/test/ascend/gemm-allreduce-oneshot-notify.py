# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for gemm-allreduce-oneshot-notify.py (notify/wait AllReduce).

Correctness path lives in tutorials/ascend/gemm-allreduce-oneshot-notify.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/gemm-allreduce-oneshot-notify.py
"""

import math
import os

import torch
import shmem as ash
import torch.distributed as dist
from triton.runtime import driver

from prof import run_options, check_all_ranks_close, load_tutorial, prof, prof_report
from utils import init_aclshmem

tut = load_tutorial("gemm-allreduce-oneshot-notify")


def main():
    dtype, M, N, K = torch.float16, 4096, 4096, 4096
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256
    COMM_BLOCK_SIZE_M = 8
    COMM_BLOCK_SIZE_N = 256
    signal_dtype = torch.int32
    ## 64 byte alignment on aclshemem wait
    SIGNAL_STRIDE = 64 // (torch.iinfo(signal_dtype).bits // 8)
    buffer_num = 2
    pvalue = 4
    ncore = driver.active.utils.get_aicore_num()
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    num_loops_m = math.ceil(M / BLOCK_SIZE_M)
    num_loops_n = math.ceil(N / BLOCK_SIZE_N)
    total_loops = num_loops_m * num_loops_n
    # World-sized fp32 C staging plus the int32 signal arena.
    required = (
        world_size * M * N * torch.float32.itemsize
        + world_size * total_loops * SIGNAL_STRIDE * signal_dtype.itemsize
    )

    init_aclshmem(pe, world_size, tut._ash_pool_bytes(required), ash.OpEngineType.MTE)

    ## allocate full result x rank on global shared memory to reduce sync points (producer doesnt have to wait)
    peer_mem = ash.aclshmem_create_tensor(
        [world_size * M * N],
        dtype=torch.float32,
        device_id=local_pe,
    )
    signal_mem = ash.aclshmem_create_tensor(
        [world_size * total_loops * SIGNAL_STRIDE],
        dtype=signal_dtype,
        device_id=local_pe,
    )
    signal_mem.fill_(0)

    A_r, B_r = tut.generate_matrices_for_rank(pe, M, N, K, dtype)
    A_rank = A_r.npu()
    B_rank = B_r.npu()

    C = torch.zeros([M, N], dtype=torch.float32).npu()

    def reset_kernel_state():
        C.zero_()
        signal_mem.fill_(0)

    @prof(label="gemm_allreduce_oneshot_notify", prepare=reset_kernel_state)
    def run_kernel():
        tut.gemm_allreduce(
            A_rank,
            B_rank,
            C,
            peer_mem,
            signal_mem,
            pe,
            world_size,
            buffer_num,
            pvalue,
            ncore,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            COMM_BLOCK_SIZE_M,
            COMM_BLOCK_SIZE_N,
            SIGNAL_STRIDE,
        )

    @prof(label="torch_gemm_allreduce")
    def run_golden():
        return tut.torch_gemm_allreduce(A_rank, B_rank)

    C_golden = run_golden()
    run_kernel()
    prof_report(run_golden, run_kernel)
    check_all_ranks_close(
        C.to(torch.float16), C_golden, rtol=1e-3, atol=1e-3, context="C"
    )

    ash.aclshmem_free_tensor(peer_mem)
    ash.aclshmem_free_tensor(signal_mem)
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
