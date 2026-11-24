# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import torch_npu
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend.algorithm import (
    gemm_swizzle2d_Nz,
)

import shmem as ash
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"


@triton.jit
def kernel_allgather_gemm(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    peer_mem_ptr,
    # Distributed parameters
    rank,
    rank_size,
    buffer_num,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    pvalue: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    COMM_BLOCK_SIZE_M: tl.constexpr,
    COMM_BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    dtype = tl.float16
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    num_loops_m = tl.cdiv(M, BLOCK_SIZE_M * pvalue)
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    buffer_row_size = BLOCK_SIZE_M * pvalue * rank_size
    for global_id_m in range(num_loops_m):
        buffer_id = global_id_m % buffer_num
        actual_block_size_m = BLOCK_SIZE_M * pvalue
        # process tail block
        if global_id_m == num_loops_m - 1:
            actual_block_size_m = M - global_id_m * BLOCK_SIZE_M * pvalue
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        comm_num_m_blocks = tl.cdiv(actual_block_size_m, COMM_BLOCK_SIZE_M)
        comm_num_k_blocks = tl.cdiv(K, COMM_BLOCK_SIZE_K)
        if subblock_idx == 1 and pid < rank_size:
            max_put_bytes = 8192 * 12544 * dtype.primitive_bitwidth // 8
            rows_per_put = max_put_bytes // (K * dtype.primitive_bitwidth // 8)
            rows_per_put = max(rows_per_put, 1)
            for target_rank in range(pid, rank_size, ncore):
                if target_rank != rank:
                    dst_base = (
                        buffer_id * buffer_row_size + rank * BLOCK_SIZE_M * pvalue
                    ) * K
                    src_base = global_id_m * BLOCK_SIZE_M * pvalue * K
                    for m_off in range(0, actual_block_size_m, rows_per_put):
                        chunk_rows = tl.minimum(
                            rows_per_put, actual_block_size_m - m_off
                        )
                        libshmem_device.putmem(
                            peer_mem_ptr + dst_base + m_off * K,
                            a_ptr + src_base + m_off * K,
                            chunk_rows * K * dtype.primitive_bitwidth // 8,
                            target_rank,
                        )
        if subblock_idx == 0 or (subblock_idx == 1 and pid >= rank_size):
            total_core = ncore + max(ncore - rank_size, 0)
            local_pid = pid
            if subblock_idx == 1:
                local_pid = ncore + pid - rank_size
            for k in range(
                local_pid, comm_num_m_blocks * comm_num_k_blocks, total_core
            ):
                block_id_m = k // comm_num_k_blocks
                block_id_k = k % comm_num_k_blocks
                comm_row_shape = tl.minimum(
                    actual_block_size_m - block_id_m * COMM_BLOCK_SIZE_M,
                    COMM_BLOCK_SIZE_M,
                )
                remote_ptr = dl.symm_at(peer_mem_ptr, rank)
                comm_offs_m = (
                    tl.arange(0, COMM_BLOCK_SIZE_M)
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + global_id_m * BLOCK_SIZE_M * pvalue
                ).to(tl.int64)
                comm_offs_k = (
                    tl.arange(0, COMM_BLOCK_SIZE_K) + block_id_k * COMM_BLOCK_SIZE_K
                )
                a_ptrs = a_ptr + (
                    comm_offs_m[:, None] * stride_am + comm_offs_k[None, :] * stride_ak
                )
                peermem_comm_offs_m = (
                    buffer_id * buffer_row_size
                    + rank * BLOCK_SIZE_M * pvalue
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + tl.arange(0, COMM_BLOCK_SIZE_M)
                ).to(tl.int64)
                remote_ptrs = remote_ptr + (
                    peermem_comm_offs_m[:, None] * stride_am
                    + comm_offs_k[None, :] * stride_ak
                )
                comm_msk_m = comm_offs_m[:, None] < M
                peermem_comm_msk_m = (
                    peermem_comm_offs_m[:, None]
                    < buffer_id * buffer_row_size
                    + BLOCK_SIZE_M * rank * pvalue
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + comm_row_shape
                )
                a = tl.load(
                    a_ptrs, mask=(comm_offs_k[None, :] < K) & comm_msk_m, other=0.0
                )
                tl.store(
                    remote_ptrs, a, mask=(comm_offs_k[None, :] < K) & peermem_comm_msk_m
                )
        libshmem_device.barrier_all()
        num_tiles_m = tl.cdiv(actual_block_size_m, BLOCK_SIZE_M)
        for block_id in range(pid, num_tiles_m * num_loops_n * rank_size, ncore):
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                block_id,
                rank_size * BLOCK_SIZE_M * num_tiles_m,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            rank_idx = block_id_m // num_tiles_m
            block_id_m = rank_idx * pvalue + (block_id_m % num_tiles_m)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = (
                buffer_id * buffer_row_size
                + block_id_m * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            ).to(tl.int64)
            matmul_msk_am = matmul_offs_am[:, None] < (
                buffer_id * buffer_row_size
                + BLOCK_SIZE_M * rank_idx * pvalue
                + actual_block_size_m
            )
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(num_k_blocks):
                offs_k = (tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K).to(
                    tl.int64
                )
                a_ptrs = peer_mem_ptr + (
                    matmul_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
                )
                b_ptrs = b_ptr + (
                    offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
                )
                a = tl.load(
                    a_ptrs, mask=(offs_k[None, :] < K) & matmul_msk_am, other=0.0
                )
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & msk_n, other=0.0)
                # We accumulate along the K dimension.
                accumulator += tl.dot(a, b)
            tl.extra.cann.extension.compile_hint(accumulator, "matmul_at_least_once")
            c = accumulator.to(dtype)
            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = (
                block_id_m // pvalue * M
                + global_id_m * BLOCK_SIZE_M * pvalue
                + (block_id_m % pvalue) * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            ).to(tl.int64)
            offs_cn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M * (block_id_m // pvalue + 1)) & (
                offs_cn[None, :] < N
            )
            tl.store(c_ptrs, c, mask=c_mask)


def allgather_gemm(
    A,
    B,
    C,
    peer_mem,
    rank,
    rank_size,
    buffer_num,
    pvalue,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    COMM_BLOCK_SIZE_M,
    COMM_BLOCK_SIZE_K,
):
    """consume gemm"""
    M, K = A.shape
    _, N = B.shape
    ncore = driver.active.utils.get_aicore_num()
    kernel_allgather_gemm[ncore, 1, 1](
        A,
        B,
        C,
        peer_mem,
        rank,
        rank_size,
        buffer_num,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        pvalue,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        COMM_BLOCK_SIZE_M,
        COMM_BLOCK_SIZE_K,
    )


def torch_allgather_gemm(A_local, B, world_size):
    """
    torch allgather gemm using torch.distributed.all_gather
    - A_local: local A tensor on this rank
    - B: B tensor (same across all ranks)
    - world_size: number of ranks
    Returns: C_golden computed from all-gathered A tensors
    """
    # Create a list to hold gathered A tensors from all ranks
    A_list = [torch.empty_like(A_local) for _ in range(world_size)]

    # Gather all A tensors from all ranks
    dist.all_gather(A_list, A_local)

    # Concatenate all A tensors along the first dimension
    A_golden = torch.cat(A_list, dim=0)

    # Compute golden reference: C = A_golden @ B

    C_golden = torch.matmul(A_golden, B)
    return C_golden


def allgather_gemm_check_accuracy(C_golden, C_local, rank, rank_size, M, K, N):
    """Compare the final benchmark outputs collectively on every rank."""
    passed = torch.tensor([1], dtype=torch.int32, device=C_local.device)
    error_message = ""
    try:
        torch.testing.assert_close(
            C_golden, C_local.to(torch.float16), rtol=1e-3, atol=1e-3
        )
    except AssertionError as error:
        passed[0] = 0
        error_message = str(error)
    all_passed = [torch.zeros_like(passed) for _ in range(rank_size)]
    dist.all_gather(all_passed, passed)
    dist.barrier()
    for rank_id in range(rank_size):
        if rank == rank_id:
            if passed.item() == 1:
                print(
                    f"  {GREEN}[PASS]{RESET} Rank {rank}: M={M}, K={K}, N={N} "
                    "matches golden.",
                    flush=True,
                )
            else:
                print(
                    f"  {RED}[FAIL]{RESET} Rank {rank}: M={M}, K={K}, N={N} failed. "
                    f"Details:\n{error_message}",
                    flush=True,
                )
        dist.barrier()
    if any(value.item() == 0 for value in all_passed):
        raise AssertionError(
            f"Allgather gemm kernel output does not match Torch Golden "
            f"at M={M}, K={K}, N={N}"
        )


def _peer_mem_elems(M, K, world_size):
    """Return the fp16 peer staging elements one (M, N, K) case needs."""
    return min(128, M) * 4 * world_size * 2 * max(K, min(256, K))


def symmetric_memory_size(world_size, M, K) -> int:
    """Return the aligned ACLSHMEM heap capacity one GEMM case needs."""
    required = _peer_mem_elems(M, K, world_size) * 2
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def _ash_pool_bytes(required: int) -> int:
    """Symmetric heap: ALLGATHER_GEMM_ASH_SIZE_GB override, else the requirement, page-aligned."""
    override = os.environ.get("ALLGATHER_GEMM_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the ALLGATHER_GEMM_ASH_SIZE_GB override"
        )
    return pool_bytes


def ash_pool_bytes(world_size, M, K) -> int:
    """Symmetric heap: ALLGATHER_GEMM_ASH_SIZE_GB override, else the case requirement."""
    return _ash_pool_bytes(symmetric_memory_size(world_size, M, K))


def shmem_init(pe, world_size, pool_bytes):
    """Initialize SHMEM once."""
    init_aclshmem(pe, world_size, pool_bytes, ash.OpEngineType.UDMA)


def run_test_case(M, N, K, pe, world_size, dtype):
    local_pe = int(os.environ["LOCAL_RANK"])
    buffer_num = 2
    BLOCK_SIZE_M = min(128, M)
    BLOCK_SIZE_N = min(256, N)
    BLOCK_SIZE_K = min(256, K)
    COMM_BLOCK_SIZE_M = 20
    COMM_BLOCK_SIZE_K = BLOCK_SIZE_K
    pvalue = 4
    # Calculate peer_mem size based on world_size
    peer_mem_size = _peer_mem_elems(M, K, world_size)
    peer_mem = ash.aclshmem_create_tensor(
        [peer_mem_size],
        dtype=dtype,
        device_id=local_pe,
    )

    # Each rank creates its own local A and B tensors
    A_local = torch.randn([M, K], dtype=dtype).npu()
    B = torch.randn([K, N], dtype=dtype).npu()
    C = torch.zeros([M * world_size, N], dtype=dtype).npu()

    # Compute golden reference using torch.distributed.all_gather
    C_golden = torch_allgather_gemm(A_local, B, world_size)
    allgather_gemm(
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
    # Run the distributed kernel with local A
    allgather_gemm_check_accuracy(C_golden, C, pe, world_size, M, K, N)
    ash.aclshmem_free_tensor(peer_mem)
    dist.barrier()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)
    dtype = torch.float16
    world_size = dist.get_world_size()
    buffer_num = 2
    # Read test cases from CSV
    M, N, K = [4096, 4096, 4096]
    shmem_init(pe, world_size, ash_pool_bytes(world_size, M, K))
    ncore = driver.active.utils.get_aicore_num()
    run_test_case(M, N, K, pe, world_size, dtype)

    _ = ash.aclshmem_finalize()
    if local_pe == 0:
        print(f"[INFO] Test passed successfully for {world_size} ranks!")
