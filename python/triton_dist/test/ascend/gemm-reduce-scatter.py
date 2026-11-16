# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for gemm-reduce-scatter.py (MTE GEMM + ReduceScatter).

Correctness path lives in tutorials/ascend/gemm-reduce-scatter.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/gemm-reduce-scatter.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist

from prof import run_options, check_all_ranks_close, load_tutorial, prof, prof_report
from utils import init_aclshmem

tut = load_tutorial("gemm-reduce-scatter")


def main():
    dtype, M, N, K = torch.float16, 4096, 4096, 4096
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256
    COMM_BLOCK_SIZE_M = 8
    COMM_BLOCK_SIZE_N = 256
    buffer_num = 2
    pvalue = 4
    ncore = 20
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # fp32 peer staging elements; the heap is sized before aclshmem_init.
    peer_mem_size = BLOCK_SIZE_M * pvalue * ncore * buffer_num * BLOCK_SIZE_N

    init_aclshmem(
        pe,
        world_size,
        tut._ash_pool_bytes(peer_mem_size * torch.float32.itemsize),
        ash.OpEngineType.MTE,
    )

    peer_mem = ash.aclshmem_create_tensor(
        [peer_mem_size],
        dtype=torch.float32,
        device_id=local_pe,
    )

    A_local = torch.randn([M, K], dtype=dtype).npu()
    B_local = torch.randn([K, N], dtype=dtype).npu()
    C = torch.zeros([M // world_size, N], dtype=torch.float32).npu()

    # Gather all A and B tensors from all ranks
    A_list = [torch.empty_like(A_local) for _ in range(world_size)]
    B_list = [torch.empty_like(B_local) for _ in range(world_size)]
    dist.all_gather(A_list, A_local)
    dist.all_gather(B_list, B_local)

    @prof(label="gemm_reduce_scatter", prepare=C.zero_)
    def run_kernel():
        tut.gemm_reduce_scatter(
            A_local,
            B_local,
            C,
            peer_mem,
            pe,
            world_size,
            buffer_num,
            pvalue,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            COMM_BLOCK_SIZE_M,
            COMM_BLOCK_SIZE_N,
        )

    @prof(label="torch_gemm_reduce_scatter")
    def run_golden():
        return tut.torch_gemm_reduce_scatter(A_list, B_list, pe, world_size)

    C_golden = run_golden()
    run_kernel()
    prof_report(run_golden, run_kernel)
    check_all_ranks_close(
        C.to(torch.float16), C_golden, rtol=1e-3, atol=1e-3, context="C"
    )

    ash.aclshmem_free_tensor(peer_mem)
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
