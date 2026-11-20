# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for gemm-reduce-scatter-udma.py (UDMA GEMM + ReduceScatter).

Correctness path lives in tutorials/ascend/gemm-reduce-scatter-udma.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/gemm-reduce-scatter-udma.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist

from prof import run_options, verify_enabled, load_tutorial, prof, prof_report
from utils import init_aclshmem

tut = load_tutorial("gemm-reduce-scatter-udma")


def main():
    dtype, M, N, K = torch.float16, 4096, 4096, 4096
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256
    pvalue=2
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    m_per_rank = M // world_size
    num_loops_m = ((m_per_rank + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * world_size
    num_loops_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_loops = num_loops_m * num_loops_n
    # Two fp16 staging tensors (recv + C local); the heap is sized at init.
    staging_bytes = 2 * total_loops * BLOCK_SIZE_M * BLOCK_SIZE_N * dtype.itemsize

    init_aclshmem(
        pe, world_size, tut._ash_pool_bytes(staging_bytes), ash.OpEngineType.UDMA
    )

    dist.barrier()

    recv_mem_size = total_loops * BLOCK_SIZE_M * BLOCK_SIZE_N
    recv_mem = ash.aclshmem_create_tensor(
        [recv_mem_size],
        dtype=dtype,
        device_id=local_pe,
    )
    C_local_size = total_loops * BLOCK_SIZE_M * BLOCK_SIZE_N
    C_local = ash.aclshmem_create_tensor(
        [C_local_size],
        dtype=dtype,
        device_id=local_pe,
    )
    A_local = torch.randn([M, K], dtype=dtype).npu().contiguous()*2-1
    B_local = torch.randn([K, N], dtype=dtype).npu().contiguous()*2-1
    C = torch.zeros([M // world_size, N], dtype=torch.float32).npu()

    def reset_outputs():
        C.zero_()
        C_local.zero_()

    @prof(label="gemm_reduce_scatter_udma", prepare=reset_outputs)
    def run_kernel():
        tut.gemm_reduce_scatter(
            A_local,
            B_local,
            C,
            C_local,
            recv_mem,
            pe,
            world_size,
            pvalue,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K
        )

    @prof(label="torch_gemm_reduce_scatter")
    def run_golden():
        return tut.torch_gemm_reduce_scatter(A_local, B_local, world_size)

    C_golden = run_golden()
    run_kernel()
    prof_report(run_golden, run_kernel)

    if verify_enabled():
        # Same normalized-error check as the tutorial.
        rtol = 1e-2
        diff = torch.abs(C.to(dtype) - C_golden)
        scale = torch.clamp(torch.abs(C_golden), min=1.0)
        normalized_error = diff / scale
        bad_count = int((normalized_error > rtol).sum().item())
        if bad_count:
            raise AssertionError(
                f"{bad_count}/{C_golden.numel()} elements exceed "
                f"{bad_count / C_golden.numel() * 100}% the normalized error threshold; "
                f"max_normalized_error={float(normalized_error.max().item())}, "
                f"rtol={rtol}."
            )
        if pe == 0:
            print(
                f"[PASS] Rank {pe}: C and C_golden match within the normalized "
                f"threshold (rtol={rtol}).",
                flush=True,
            )

    ash.aclshmem_free_tensor(recv_mem)
    ash.aclshmem_free_tensor(C_local)
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
