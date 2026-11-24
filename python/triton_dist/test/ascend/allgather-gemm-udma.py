# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""UDMA AllGather + GEMM verification and performance sweep over 2p.csv.

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/allgather-gemm-udma.py --mode=both --case=random
"""

import csv
import os

import torch
import torch.distributed as dist
from prof import (
    check_all_ranks_close,
    load_tutorial,
    perf_enabled,
    prof,
    prof_report,
    run_options,
    select_cases,
)
from triton.runtime import driver

import shmem as ash
from utils import init_aclshmem

tut = load_tutorial("allgather-gemm-udma")


RESET = "\033[0m"
CYAN = "\033[96m"


def read_test_cases(csv_path):
    """Read M, K, N test cases from CSV file."""
    test_cases = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            M = int(row["M"].strip())
            K = int(row["K"].strip())
            N = int(row["N"].strip())
            test_cases.append((M, K, N))
    return test_cases


def adjust_configuration(M, K, N):
    BLOCK_SIZE_M = min(128, M)
    BLOCK_SIZE_N = min(256, N)
    BLOCK_SIZE_K = min(256, K)
    COMM_BLOCK_SIZE_M = 20
    COMM_BLOCK_SIZE_K = BLOCK_SIZE_K
    pvalue = 4
    if K <= 128:
        pvalue = 64
    return (
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        COMM_BLOCK_SIZE_M,
        COMM_BLOCK_SIZE_K,
        pvalue,
    )


def peer_staging_elems(M, K, N, world_size, buffer_num):
    """Return the fp16 peer staging elements one (M, K, N) case needs."""
    BLOCK_SIZE_M, BLOCK_SIZE_K, _, _, _, pvalue = adjust_configuration(M, K, N)
    return BLOCK_SIZE_M * pvalue * world_size * buffer_num * max(K, BLOCK_SIZE_K)


def shmem_init(pe, world_size, pool_bytes):
    """Initialize SHMEM once."""
    init_aclshmem(pe, world_size, pool_bytes, ash.OpEngineType.UDMA)


def run_test_case(
    M,
    N,
    K,
    pe,
    world_size,
    dtype,
    buffer_num,
    pvalue,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    COMM_BLOCK_SIZE_M,
    COMM_BLOCK_SIZE_K,
):
    local_pe = int(os.environ["LOCAL_RANK"])
    # Calculate peer_mem size based on world_size
    peer_mem_size = peer_staging_elems(M, K, N, world_size, buffer_num)
    peer_mem = ash.aclshmem_create_tensor(
        [peer_mem_size],
        dtype=dtype,
        device_id=local_pe,
    )
    # Each rank creates its own local A and B tensors
    torch.manual_seed(42 + pe)
    A_local = torch.randn([M, K], dtype=dtype).npu()
    torch.manual_seed(1234)
    B = torch.randn([K, N], dtype=dtype).npu()
    C = torch.zeros([M * world_size, N], dtype=dtype).npu()

    # Compute golden reference using torch.distributed.all_gather
    @prof(label="allgather_gemm_udma")
    def run_kernel():
        tut.allgather_gemm(
            A_local,
            B,
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
            COMM_BLOCK_SIZE_K,
        )

    @prof(label="torch_allgather_gemm")
    def run_golden():
        return tut.torch_allgather_gemm(A_local, B, world_size)

    context = f"M={M} K={K} N={N}"
    try:
        C_golden = run_golden()
        run_kernel()
        prof_report(run_golden, run_kernel, context=context)
        check_all_ranks_close(C, C_golden, rtol=1e-3, atol=1e-3, context=f"C {context}")
    finally:
        ash.aclshmem_free_tensor(peer_mem)


if __name__ == "__main__":
    options = run_options()
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)
    dtype = torch.float16
    world_size = dist.get_world_size()
    buffer_num = 2
    # Read test cases from CSV
    current_path = os.path.dirname(os.path.abspath(__file__))
    test_cases = read_test_cases(f"{current_path}/allgather-gemm-case.csv")
    selected_cases = select_cases(test_cases)
    if local_pe == 0:
        print(f"[INFO] Rank {local_pe} of {world_size} initialized")
        print(f"[INFO] Selected test cases: {len(selected_cases)}/{len(test_cases)}")
        print(
            "CONFIG "
            f"engine=allgather-gemm-udma ranks={world_size} "
            f"mode={options.mode} case={options.case} "
            f"aicore_num={driver.active.utils.get_aicore_num()} "
            f"vector_cores={driver.active.utils.get_aivector_core_num()} ",
            flush=True,
        )
        if perf_enabled():
            print(
                f"{CYAN}event timing: torch.npu.Event per iteration, "
                f"speedup = golden / kernel{RESET}",
                flush=True,
            )
    dist.barrier()

    # One heap for the whole CSV sweep: the largest case's peer staging.
    max_required = max(
        peer_staging_elems(M, K, N, world_size, buffer_num) * dtype.itemsize
        for M, K, N in selected_cases
    )
    shmem_init(pe, world_size, tut._ash_pool_bytes(max_required))
    try:
        for M, K, N in selected_cases:
            (
                BLOCK_SIZE_M,
                BLOCK_SIZE_K,
                BLOCK_SIZE_N,
                COMM_BLOCK_SIZE_M,
                COMM_BLOCK_SIZE_K,
                pvalue,
            ) = adjust_configuration(M, K, N)
            run_test_case(
                M,
                N,
                K,
                pe,
                world_size,
                dtype,
                buffer_num,
                pvalue,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                COMM_BLOCK_SIZE_M,
                COMM_BLOCK_SIZE_K,
            )
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()
