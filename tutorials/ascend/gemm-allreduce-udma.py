# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""UDMA GEMM-AllReduce with one-shot and owner ReduceScatter/AllGather paths.

This tutorial provides the kernels and a single fixed-shape correctness
example. NPU event timing is provided by the matching script under
python/triton_dist/test/ascend/ through the shared prof.py helper.
"""

import math
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend.algorithm import (
    gemm_swizzle2d_Nz,
)

import shmem as ash
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


def _ash_pool_bytes(required: int) -> int:
    """Symmetric heap: GEMM_ALLREDUCE_ASH_SIZE_GB override, else the requirement, page-aligned."""
    override = os.environ.get("GEMM_ALLREDUCE_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the GEMM_ALLREDUCE_ASH_SIZE_GB override"
        )
    return pool_bytes

MAX_UDMA_PUT_BYTES = 256 * 1024 * 1024
BITWISE_COMPARE_CHUNK_BYTES = 64 * 1024 * 1024
HYBRID_RSAG_MAX_K = 4096
HYBRID_RSAG_MIN_OUTPUT_BYTES = 2 * 1024 * 1024
# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Two-level BF16 accuracy check. Direct mismatches trigger a complete-domain
# CPU FP64 comparison for the normal-value and small-value rules.
ACCURACY_RTOL = 5e-3
ACCURACY_ATOL = 5e-3
MARE_RATIO_LIMIT = 2.0
MERE_RATIO_LIMIT = 1.2
RMSE_RATIO_LIMIT = 1.2
RELATIVE_ERROR_EPSILON = 1e-7
SMALL_VALUE_THRESHOLD = 2 ** (-8)
SMALL_VALUE_ERROR = 2 ** (-16)
SMALL_VALUE_COUNT_RATIO_LIMIT = 2.0
CPU_REFERENCE_ROW_CHUNK = 64

def use_owner_rsag(M, K, N):
    """Select the collective without shape-specific exceptions."""
    output_bytes = M * N * 2
    return K <= HYBRID_RSAG_MAX_K and output_bytes >= HYBRID_RSAG_MIN_OUTPUT_BYTES


def collective_name(M, K, N):
    return "owner-rsag" if use_owner_rsag(M, K, N) else "one-shot"


@triton.jit
def kernel_gemm_allreduce_oneshot(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    peer_mem_ptr,
    # Distributed parameters
    rank,
    rank_size,
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
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    elem_bytes = 2  # The communication buffer stores BF16 GEMM results.
    loop_num_per_comm = ncore * pvalue
    num_loops_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_loops = num_loops_m * num_loops_n
    # Promote packed-buffer address arithmetic before multiplying by rank or tile size.
    total_loops_i64 = total_loops.to(tl.int64)
    num_loops_comm = tl.cdiv(total_loops, loop_num_per_comm)
    for global_id in range(num_loops_comm):
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_id_offset = global_id * loop_num_per_comm
        # process tail block
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        for block_id in range(pid, actual_loop_num_per_comm, ncore):
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                output_block_id_offset + block_id,
                M,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            matmul_msk_am = matmul_offs_am[:, None] < M
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(num_k_blocks):
                offs_k = tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K
                a_ptrs = a_ptr + (
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
            c = accumulator
            tl.extra.cann.extension.compile_hint(c, "matmul_at_least_once")
            # -----------------------------------------------------------
            # Write the block of the output matrix C with masks.
            offs_peer_mem_m = (
                rank * total_loops_i64 + output_block_id_offset + block_id
            ) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_peer_mem_n = tl.arange(0, BLOCK_SIZE_N)
            peer_mem_ptrs = (
                peer_mem_ptr
                + BLOCK_SIZE_N * offs_peer_mem_m[:, None]
                + offs_peer_mem_n[None, :]
            )
            tl.store(peer_mem_ptrs, c)
        libshmem_device.barrier_all()
        if subblock_idx == 1 and pid < rank_size:
            for target_rank in range(pid, rank_size, ncore):
                if target_rank != rank:
                    libshmem_device.putmem(
                        # dst: symmetric memory of target_rank, slot from rank
                        peer_mem_ptr
                        + (rank * total_loops_i64 + output_block_id_offset)
                        * BLOCK_SIZE_M
                        * BLOCK_SIZE_N,
                        # src: entire data segment of target_rank in local A
                        peer_mem_ptr
                        + (rank * total_loops_i64 + output_block_id_offset)
                        * BLOCK_SIZE_M
                        * BLOCK_SIZE_N,
                        BLOCK_SIZE_N
                        * actual_loop_num_per_comm
                        * BLOCK_SIZE_M
                        * elem_bytes,
                        # pe: the target rank of putmem
                        target_rank,
                    )
    libshmem_device.barrier_all()
    for global_id in range(num_loops_comm):
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_id_offset = global_id * loop_num_per_comm
        # process tail block
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        total_core = ncore * 2
        local_pid = pid
        if subblock_idx == 1:
            local_pid = ncore + pid
        # Both vector subblocks provide a total of 2 * ncore reduction workers.
        # Each worker owns an output tile and stores it after summing all ranks.
        # Reduction starts after the final communication barrier above.
        reduce_split_m = BLOCK_SIZE_M // REDUCE_BLOCK_SIZE_M
        reduce_split_n = BLOCK_SIZE_N // REDUCE_BLOCK_SIZE_N
        reduce_parts_per_tile = reduce_split_m * reduce_split_n
        reduce_work_items = actual_loop_num_per_comm * reduce_parts_per_tile
        for reduce_idx in range(local_pid, reduce_work_items, total_core):
            block_id = reduce_idx // reduce_parts_per_tile
            reduce_part = reduce_idx % reduce_parts_per_tile
            reduce_part_m = reduce_part // reduce_split_n
            reduce_part_n = reduce_part % reduce_split_n
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                output_block_id_offset + block_id,
                M,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            reduce_m_offset = reduce_part_m * REDUCE_BLOCK_SIZE_M
            offs_cm = (
                block_id_m * BLOCK_SIZE_M
                + reduce_m_offset
                + tl.arange(0, REDUCE_BLOCK_SIZE_M)
            )
            reduce_n_offset = reduce_part_n * REDUCE_BLOCK_SIZE_N
            offs_cn = (
                block_id_n * BLOCK_SIZE_N
                + reduce_n_offset
                + tl.arange(0, REDUCE_BLOCK_SIZE_N)
            )
            c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

            reduce_accumulator = tl.zeros(
                (REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N), dtype=tl.float32
            )
            # Use the same source-rank order on every rank so all one-shot
            # outputs are bitwise identical, matching the Owner-RSAG path.
            for source_rank in range(rank_size):
                comm_offs_m = (
                    reduce_m_offset
                    + tl.arange(0, REDUCE_BLOCK_SIZE_M)
                    + (source_rank * total_loops_i64 + output_block_id_offset + block_id)
                    * BLOCK_SIZE_M
                )
                comm_offs_n = reduce_n_offset + tl.arange(0, REDUCE_BLOCK_SIZE_N)
                local_ptrs = peer_mem_ptr + (
                    comm_offs_m[:, None] * BLOCK_SIZE_N + comm_offs_n[None, :]
                )
                # Packed tiles include padding; only output stores need a mask.
                c_temp = tl.load(local_ptrs)
                reduce_accumulator = reduce_accumulator + c_temp
            # Reduce in FP32, then round the final
            # AllReduce result back to the operator output dtype.
            tl.store(
                c_ptr + c_offs,
                reduce_accumulator.to(tl.bfloat16),
                mask=c_mask,
            )


@triton.jit
def kernel_gemm_allreduce_owner_rsag(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    peer_mem_ptr,
    # Distributed parameters
    rank,
    rank_size,
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
    REDUCE_BLOCK_SIZE_M: tl.constexpr,
    REDUCE_BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    elem_bytes = 2  # The communication buffer stores BF16 GEMM results.
    loop_num_per_comm = ncore * pvalue
    num_loops_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_loops = num_loops_m * num_loops_n
    # Promote packed-buffer address arithmetic before multiplying by rank or tile size.
    total_loops_i64 = total_loops.to(tl.int64)
    num_loops_comm = tl.cdiv(total_loops, loop_num_per_comm)
    for global_id in range(num_loops_comm):
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_id_offset = global_id * loop_num_per_comm
        # process tail block
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        for block_id in range(pid, actual_loop_num_per_comm, ncore):
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                output_block_id_offset + block_id,
                M,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            matmul_msk_am = matmul_offs_am[:, None] < M
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(num_k_blocks):
                offs_k = tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K
                a_ptrs = a_ptr + (
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
            c = accumulator
            tl.extra.cann.extension.compile_hint(c, "matmul_at_least_once")
            # -----------------------------------------------------------
            # Write the block of the output matrix C with masks.
            offs_peer_mem_m = (
                rank * total_loops_i64 + output_block_id_offset + block_id
            ) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_peer_mem_n = tl.arange(0, BLOCK_SIZE_N)
            peer_mem_ptrs = (
                peer_mem_ptr
                + BLOCK_SIZE_N * offs_peer_mem_m[:, None]
                + offs_peer_mem_n[None, :]
            )
            tl.store(peer_mem_ptrs, c)
        libshmem_device.barrier_all()

        # Reduce-Scatter transport.  Split this batch into rank_size contiguous
        # tile ranges.  The source sends range r only to owner rank r, instead
        # of replicating the entire batch to every peer.  The strided
        # pid-to-owner mapping preserves one UDMA issuer per peer and also
        # remains correct if rank_size ever exceeds ncore.
        if subblock_idx == 1:
            for owner_rank in range(pid, rank_size, ncore):
                if owner_rank != rank:
                    owner_begin = actual_loop_num_per_comm * owner_rank // rank_size
                    owner_end = actual_loop_num_per_comm * (owner_rank + 1) // rank_size
                    owner_tile_count = owner_end - owner_begin
                    if owner_tile_count > 0:
                        owner_global_begin = output_block_id_offset + owner_begin
                        owner_offset = (
                            (rank * total_loops_i64 + owner_global_begin)
                            * BLOCK_SIZE_M
                            * BLOCK_SIZE_N
                        )
                        libshmem_device.putmem(
                            # dst: source-rank slot on the owner rank
                            peer_mem_ptr + owner_offset,
                            # src: the same packed segment in local symmetric GM
                            peer_mem_ptr + owner_offset,
                            owner_tile_count * BLOCK_SIZE_M * BLOCK_SIZE_N * elem_bytes,
                            owner_rank,
                        )

    # All source contributions have landed in their owner ranks.
    libshmem_device.barrier_all()

    # Owner reduction.  Rank r reduces only range r from every batch.  The
    # reduced BF16 result overwrites source slot 0; all source values are loaded
    # before that store, so the alias is safe.  Different owners operate on
    # disjoint global tile ranges.
    for global_id in range(num_loops_comm):
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_id_offset = global_id * loop_num_per_comm
        # process tail block
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        total_core = ncore * 2
        local_pid = pid
        if subblock_idx == 1:
            local_pid = ncore + pid
        owner_begin = actual_loop_num_per_comm * rank // rank_size
        owner_end = actual_loop_num_per_comm * (rank + 1) // rank_size
        owner_tile_count = owner_end - owner_begin

        # Both Vector subblocks provide 2*ncore independent reduction workers.
        # A worker owns one <=128x128 subtile and stores it exactly once.
        reduce_split_m = BLOCK_SIZE_M // REDUCE_BLOCK_SIZE_M
        reduce_split_n = BLOCK_SIZE_N // REDUCE_BLOCK_SIZE_N
        reduce_parts_per_tile = reduce_split_m * reduce_split_n
        reduce_work_items = owner_tile_count * reduce_parts_per_tile
        for reduce_idx in range(local_pid, reduce_work_items, total_core):
            owner_block_id = reduce_idx // reduce_parts_per_tile
            block_id = owner_begin + owner_block_id
            reduce_part = reduce_idx % reduce_parts_per_tile
            reduce_part_m = reduce_part // reduce_split_n
            reduce_part_n = reduce_part % reduce_split_n
            reduce_m_offset = reduce_part_m * REDUCE_BLOCK_SIZE_M
            reduce_n_offset = reduce_part_n * REDUCE_BLOCK_SIZE_N

            reduce_accumulator = tl.zeros(
                (REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N), dtype=tl.float32
            )
            # Every owner uses the same source-rank order.  This makes the FP32
            # addition order independent of the owner assigned to a tile.
            for source_rank in range(rank_size):
                comm_offs_m = (
                    reduce_m_offset
                    + tl.arange(0, REDUCE_BLOCK_SIZE_M)
                    + (source_rank * total_loops_i64 + output_block_id_offset + block_id)
                    * BLOCK_SIZE_M
                )
                comm_offs_n = reduce_n_offset + tl.arange(0, REDUCE_BLOCK_SIZE_N)
                local_ptrs = peer_mem_ptr + (
                    comm_offs_m[:, None] * BLOCK_SIZE_N + comm_offs_n[None, :]
                )
                # Packed tiles include padding; only output stores need a mask.
                c_temp = tl.load(local_ptrs)
                reduce_accumulator = reduce_accumulator + c_temp

            # Slot 0 is the packed final-result buffer after owner reduction.
            final_packed_m = (
                (output_block_id_offset.to(tl.int64) + block_id) * BLOCK_SIZE_M
                + reduce_m_offset
                + tl.arange(0, REDUCE_BLOCK_SIZE_M)
            )
            final_packed_n = reduce_n_offset + tl.arange(0, REDUCE_BLOCK_SIZE_N)
            final_packed_ptrs = peer_mem_ptr + (
                final_packed_m[:, None] * BLOCK_SIZE_N + final_packed_n[None, :]
            )
            tl.store(
                final_packed_ptrs,
                reduce_accumulator.to(tl.bfloat16),
            )

    # Every owner must finish reading source slot 0 before another owner may
    # overwrite a disjoint slot-0 range on that rank during AllGather.
    libshmem_device.barrier_all()

    # AllGather the already reduced owner segments.  Each owner rank broadcasts
    # only its 1/rank_size shard, preserving one UDMA issuer per target peer.
    if subblock_idx == 1:
        for target_rank in range(pid, rank_size, ncore):
            if target_rank != rank:
                for global_id in range(num_loops_comm):
                    actual_loop_num_per_comm = loop_num_per_comm
                    output_block_id_offset = global_id * loop_num_per_comm
                    if global_id == num_loops_comm - 1:
                        actual_loop_num_per_comm = (
                            total_loops - global_id * loop_num_per_comm
                        )
                    owner_begin = actual_loop_num_per_comm * rank // rank_size
                    owner_end = actual_loop_num_per_comm * (rank + 1) // rank_size
                    owner_tile_count = owner_end - owner_begin
                    if owner_tile_count > 0:
                        owner_global_begin = output_block_id_offset + owner_begin
                        owner_offset = (
                            owner_global_begin.to(tl.int64) * BLOCK_SIZE_M * BLOCK_SIZE_N
                        )
                        libshmem_device.putmem(
                            peer_mem_ptr + owner_offset,
                            peer_mem_ptr + owner_offset,
                            owner_tile_count * BLOCK_SIZE_M * BLOCK_SIZE_N * elem_bytes,
                            target_rank,
                        )

    # Every rank now has the complete reduced packed output in source slot 0.
    libshmem_device.barrier_all()

    # Unpack tile-major slot 0 to the row-major output tensor.  This local MTE
    # phase is deliberately separate because UDMA cannot copy local shmem to
    # local GM.
    for global_id in range(num_loops_comm):
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_id_offset = global_id * loop_num_per_comm
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        total_core = ncore * 2
        local_pid = pid
        if subblock_idx == 1:
            local_pid = ncore + pid
        reduce_split_m = BLOCK_SIZE_M // REDUCE_BLOCK_SIZE_M
        reduce_split_n = BLOCK_SIZE_N // REDUCE_BLOCK_SIZE_N
        parts_per_tile = reduce_split_m * reduce_split_n
        total_copy_work = actual_loop_num_per_comm * parts_per_tile
        for copy_idx in range(local_pid, total_copy_work, total_core):
            block_id = copy_idx // parts_per_tile
            copy_part = copy_idx % parts_per_tile
            copy_part_m = copy_part // reduce_split_n
            copy_part_n = copy_part % reduce_split_n
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                output_block_id_offset + block_id,
                M,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            copy_m_offset = copy_part_m * REDUCE_BLOCK_SIZE_M
            copy_n_offset = copy_part_n * REDUCE_BLOCK_SIZE_N
            offs_cm = (
                block_id_m * BLOCK_SIZE_M
                + copy_m_offset
                + tl.arange(0, REDUCE_BLOCK_SIZE_M)
            )
            offs_cn = (
                block_id_n * BLOCK_SIZE_N
                + copy_n_offset
                + tl.arange(0, REDUCE_BLOCK_SIZE_N)
            )
            c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            packed_m = (
                (output_block_id_offset.to(tl.int64) + block_id) * BLOCK_SIZE_M
                + copy_m_offset
                + tl.arange(0, REDUCE_BLOCK_SIZE_M)
            )
            packed_n = copy_n_offset + tl.arange(0, REDUCE_BLOCK_SIZE_N)
            packed_ptrs = peer_mem_ptr + (
                packed_m[:, None] * BLOCK_SIZE_N + packed_n[None, :]
            )
            reduced = tl.load(packed_ptrs)
            tl.store(c_ptr + c_offs, reduced, mask=c_mask)


def gemm_allreduce(
    A,
    B,
    C,
    peer_mem,
    rank,
    rank_size,
    pvalue,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    REDUCE_BLOCK_SIZE_M,
    REDUCE_BLOCK_SIZE_N,
):
    """Launch the shape-selected BF16 GEMM-AllReduce kernel."""
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert peer_mem.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
    M, K = A.shape
    _, N = B.shape
    ncore = driver.active.utils.get_aicore_num()
    kernel = (
        kernel_gemm_allreduce_owner_rsag
        if use_owner_rsag(M, K, N)
        else kernel_gemm_allreduce_oneshot
    )
    kernel[ncore, 1, 1](
        A,
        B,
        C,
        peer_mem,
        rank,
        rank_size,
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
        REDUCE_BLOCK_SIZE_M,
        REDUCE_BLOCK_SIZE_N,
    )


def torch_gemm_allreduce(A_rank, B_rank):
    """Compute a local BF16 GEMM followed by an HCCL AllReduce."""
    C_golden = torch.matmul(A_rank, B_rank)
    dist.all_reduce(C_golden, op=dist.ReduceOp.SUM)
    return C_golden


def _error_metrics(result, truth):
    """Return MARE, MERE and RMSE against a CPU FP64 truth tensor."""
    result = result.detach().cpu().to(torch.float64)
    truth = truth.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("result contains NaN or Inf")
    if not bool(torch.isfinite(truth).all().item()):
        raise ValueError("CPU FP64 reference contains NaN or Inf")
    difference = result - truth
    relative_error = difference.abs() / (truth.abs() + RELATIVE_ERROR_EPSILON)
    return {
        "mare": relative_error.max().item(),
        "mere": relative_error.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def _metric_ratio(custom_value, reference_value):
    if reference_value == 0.0:
        return 1.0 if custom_value == 0.0 else float("inf")
    return custom_value / reference_value


def _cpu_fp64_reference_full(M, N, K, rank_size):
    """Regenerate deterministic rank inputs on CPU and sum their FP64 GEMMs.

    Row chunks limit temporary allocations while retaining the complete truth
    tensor for the normal-value and small-value accuracy checks.
    """
    truth = torch.zeros((M, N), dtype=torch.float64)
    for source_rank in range(rank_size):
        A_cpu, B_cpu = generate_matrices_for_rank(source_rank, M, N, K, torch.bfloat16)
        B_cpu_fp64 = B_cpu.to(torch.float64)
        for row_start in range(0, M, CPU_REFERENCE_ROW_CHUNK):
            row_end = min(row_start + CPU_REFERENCE_ROW_CHUNK, M)
            truth[row_start:row_end] += torch.matmul(
                A_cpu[row_start:row_end].to(torch.float64), B_cpu_fp64
            )
        del A_cpu, B_cpu, B_cpu_fp64
    return truth


def gemm_allreduce_check_accuracy(C_golden, C_local, rank, rank_size, M, K, N):
    """Use complete normal/small domains for CPU-FP64 third-party checking."""
    close_mask = torch.isclose(
        C_local, C_golden, rtol=ACCURACY_RTOL, atol=ACCURACY_ATOL
    )
    bad_count = int((~close_mask).sum().item())
    max_abs_error = float((C_local - C_golden).abs().max().item())

    count_tensor = torch.tensor([bad_count], dtype=torch.int32, device=C_local.device)
    max_abs_tensor = torch.tensor(
        [max_abs_error], dtype=torch.float32, device=C_local.device
    )
    gathered_counts = [torch.zeros_like(count_tensor) for _ in range(rank_size)]
    gathered_max_abs = [torch.zeros_like(max_abs_tensor) for _ in range(rank_size)]
    dist.all_gather(gathered_counts, count_tensor)
    dist.all_gather(gathered_max_abs, max_abs_tensor)
    counts = [int(value.item()) for value in gathered_counts]
    max_abs_values = [float(value.item()) for value in gathered_max_abs]

    decisions = torch.zeros(rank_size, dtype=torch.int32, device=C_local.device)
    metric_summary = torch.full(
        (rank_size, 9), float("nan"), dtype=torch.float32, device=C_local.device
    )
    # normal_count, small_count, NPU small errors, Torch small errors
    domain_summary = torch.zeros(
        (rank_size, 4), dtype=torch.int32, device=C_local.device
    )
    domain_pass = torch.zeros((rank_size, 2), dtype=torch.int32, device=C_local.device)
    cpu_reference_failed = torch.zeros(1, dtype=torch.int32, device=C_local.device)

    if rank == 0:
        decisions_cpu = torch.tensor(
            [1 if value == 0 else 0 for value in counts], dtype=torch.int32
        )
        metrics_cpu = torch.full((rank_size, 9), float("nan"), dtype=torch.float32)
        domains_cpu = torch.zeros((rank_size, 4), dtype=torch.int32)
        domain_pass_cpu = torch.zeros((rank_size, 2), dtype=torch.int32)
        try:
            if any(value > 0 for value in counts):
                cpu_truth = _cpu_fp64_reference_full(M, N, K, rank_size)
                kernel_result = C_local.detach().cpu().to(torch.float64)
                torch_result = C_golden.detach().cpu().to(torch.float64)

                small_mask = cpu_truth.abs() < SMALL_VALUE_THRESHOLD
                normal_mask = ~small_mask
                normal_count = int(normal_mask.sum().item())
                small_count = int(small_mask.sum().item())

                metric_names = ("mare", "mere", "rmse")
                kernel_metrics = {name: float("nan") for name in metric_names}
                torch_metrics = {name: float("nan") for name in metric_names}
                ratios = {name: float("nan") for name in metric_names}
                normal_pass = True
                if normal_count > 0:
                    kernel_metrics = _error_metrics(
                        kernel_result[normal_mask], cpu_truth[normal_mask]
                    )
                    torch_metrics = _error_metrics(
                        torch_result[normal_mask], cpu_truth[normal_mask]
                    )
                    ratios = {
                        name: _metric_ratio(kernel_metrics[name], torch_metrics[name])
                        for name in metric_names
                    }
                    limits = {
                        "mare": MARE_RATIO_LIMIT,
                        "mere": MERE_RATIO_LIMIT,
                        "rmse": RMSE_RATIO_LIMIT,
                    }
                    normal_pass = all(
                        ratios[name] <= limits[name]
                        or math.isclose(
                            ratios[name],
                            limits[name],
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        for name in metric_names
                    )

                npu_small_error_count = 0
                torch_small_error_count = 0
                small_count_ratio = 0.0
                small_pass = True
                if small_count > 0:
                    npu_small_error_count = int(
                        (
                            (kernel_result[small_mask] - cpu_truth[small_mask]).abs()
                            > SMALL_VALUE_ERROR
                        )
                        .sum()
                        .item()
                    )
                    torch_small_error_count = int(
                        (
                            (torch_result[small_mask] - cpu_truth[small_mask]).abs()
                            > SMALL_VALUE_ERROR
                        )
                        .sum()
                        .item()
                    )
                    small_count_ratio = npu_small_error_count / max(
                        torch_small_error_count, 1
                    )
                    small_pass = (
                        small_count_ratio <= SMALL_VALUE_COUNT_RATIO_LIMIT
                        or math.isclose(
                            small_count_ratio,
                            SMALL_VALUE_COUNT_RATIO_LIMIT,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )

                metric_row = torch.tensor(
                    [kernel_metrics[name] for name in metric_names]
                    + [torch_metrics[name] for name in metric_names]
                    + [ratios[name] for name in metric_names],
                    dtype=torch.float32,
                )
                domain_row = torch.tensor(
                    [
                        normal_count,
                        small_count,
                        npu_small_error_count,
                        torch_small_error_count,
                    ],
                    dtype=torch.int32,
                )
                pass_row = torch.tensor(
                    [int(normal_pass), int(small_pass)], dtype=torch.int32
                )
                for rank_id in range(rank_size):
                    metrics_cpu[rank_id] = metric_row
                    domains_cpu[rank_id] = domain_row
                    domain_pass_cpu[rank_id] = pass_row
                    if counts[rank_id] == 0 or (normal_pass and small_pass):
                        decisions_cpu[rank_id] = 1
        except Exception as error:  # noqa: BLE001
            # Broadcast any rank-0 reference failure so other ranks do not hang.
            cpu_reference_failed[0] = 1
            print(
                f"{RED}[CPU-FP64 ERROR]{RESET} M={M}, K={K}, N={N}: {error}",
                flush=True,
            )

        decisions.copy_(decisions_cpu.to(C_local.device))
        metric_summary.copy_(metrics_cpu.to(C_local.device))
        domain_summary.copy_(domains_cpu.to(C_local.device))
        domain_pass.copy_(domain_pass_cpu.to(C_local.device))

    dist.broadcast(decisions, src=0)
    dist.broadcast(metric_summary, src=0)
    dist.broadcast(domain_summary, src=0)
    dist.broadcast(domain_pass, src=0)
    dist.broadcast(cpu_reference_failed, src=0)

    metric_values = metric_summary[rank].cpu().tolist()
    domain_values = domain_summary[rank].cpu().tolist()
    pass_values = domain_pass[rank].cpu().tolist()
    failure_summaries = []
    all_passed = [decisions[index : index + 1] for index in range(rank_size)]

    dist.barrier()
    for rank_id in range(rank_size):
        if rank == rank_id:
            if decisions[rank].item() == 1:
                if bad_count == 0:
                    print(
                        f"  {GREEN}[PASS]{RESET} Rank {rank}: M={M}, K={K}, N={N} "
                        f"matches Torch golden (atol={ACCURACY_ATOL}, "
                        f"rtol={ACCURACY_RTOL}).",
                        flush=True,
                    )
                else:
                    print(
                        f"  {GREEN}[PASS]{RESET} Rank {rank}: M={M}, K={K}, N={N} "
                        "passed by complete-domain CPU FP64 comparison. "
                        f"direct_bad={bad_count}/{C_golden.numel()} "
                        f"max_abs={max_abs_error:.9g} "
                        f"normal_points={domain_values[0]} "
                        f"normal_ratio(MARE/MERE/RMSE)="
                        f"({metric_values[6]:.9g},{metric_values[7]:.9g},"
                        f"{metric_values[8]:.9g}) "
                        f"small_points={domain_values[1]} "
                        f"small_errorcount_npu={domain_values[2]} "
                        f"small_errorcount_torch={domain_values[3]}.",
                        flush=True,
                    )
            else:
                if cpu_reference_failed.item():
                    reason = "cpu_fp64=ERROR"
                else:
                    reason = (
                        f"normal_points={domain_values[0]} "
                        f"normal_kernel(MARE/MERE/RMSE)="
                        f"({metric_values[0]:.9g},{metric_values[1]:.9g},"
                        f"{metric_values[2]:.9g}) "
                        f"normal_torch(MARE/MERE/RMSE)="
                        f"({metric_values[3]:.9g},{metric_values[4]:.9g},"
                        f"{metric_values[5]:.9g}) "
                        f"normal_ratio(MARE/MERE/RMSE)="
                        f"({metric_values[6]:.9g},{metric_values[7]:.9g},"
                        f"{metric_values[8]:.9g}) "
                        f"normal_pass={bool(pass_values[0])} "
                        f"small_points={domain_values[1]} "
                        f"small_errorcount_npu={domain_values[2]} "
                        f"small_errorcount_torch={domain_values[3]} "
                        f"small_count_ratio="
                        f"{domain_values[2] / max(domain_values[3], 1):.9g} "
                        f"small_pass={bool(pass_values[1])}"
                    )
                print(
                    f"  {RED}[FAIL]{RESET} Rank {rank}: M={M}, K={K}, N={N} "
                    f"direct_bad={bad_count}/{C_golden.numel()} "
                    f"max_abs={max_abs_error:.9g} {reason}",
                    flush=True,
                )
        dist.barrier()

    if rank == 0:
        for rank_id in range(rank_size):
            if decisions[rank_id].item() == 1:
                continue
            values = metric_summary[rank_id].cpu().tolist()
            domains = domain_summary[rank_id].cpu().tolist()
            passes = domain_pass[rank_id].cpu().tolist()
            reason = (
                "cpu_fp64=ERROR"
                if cpu_reference_failed.item()
                else (
                    f"normal_points={domains[0]} "
                    f"normal_kernel(MARE/MERE/RMSE)="
                    f"({values[0]:.9g},{values[1]:.9g},{values[2]:.9g}) "
                    f"normal_torch(MARE/MERE/RMSE)="
                    f"({values[3]:.9g},{values[4]:.9g},{values[5]:.9g}) "
                    f"normal_ratio(MARE/MERE/RMSE)="
                    f"({values[6]:.9g},{values[7]:.9g},{values[8]:.9g}) "
                    f"normal_pass={bool(passes[0])} "
                    f"small_points={domains[1]} "
                    f"small_errorcount_npu={domains[2]} "
                    f"small_errorcount_torch={domains[3]} "
                    f"small_count_ratio={domains[2] / max(domains[3], 1):.9g} "
                    f"small_pass={bool(passes[1])}"
                )
            )
            failure_summaries.append(
                f"M={M} K={K} N={N} rank={rank_id} "
                f"direct_bad={counts[rank_id]}/{M * N} "
                f"max_abs={max_abs_values[rank_id]:.9g} {reason}"
            )

    return all(value.item() == 1 for value in all_passed), failure_summaries


def check_rank_outputs_bitwise(C_local, rank, rank_size):
    """Verify every rank received exactly the same final BF16 output.

    The comparison is chunked so the validation does not allocate another
    complete MxN tensor.  It is outside all timed regions.
    """
    row_bytes = C_local.shape[1] * C_local.element_size()
    row_chunk = max(1, BITWISE_COMPARE_CHUNK_BYTES // row_bytes)

    local_mismatches = 0
    for row_start in range(0, C_local.shape[0], row_chunk):
        row_end = min(row_start + row_chunk, C_local.shape[0])
        local_chunk = C_local[row_start:row_end]
        if rank == 0:
            rank0_chunk = local_chunk.contiguous()
        else:
            rank0_chunk = torch.empty_like(local_chunk)
        dist.broadcast(rank0_chunk, src=0)
        local_bits = local_chunk.view(torch.int16)
        rank0_bits = rank0_chunk.view(torch.int16)
        local_mismatches += int(torch.count_nonzero(local_bits != rank0_bits).item())

    mismatch_tensor = torch.tensor(
        [local_mismatches], dtype=torch.int64, device=C_local.device
    )
    gathered_mismatches = [torch.zeros_like(mismatch_tensor) for _ in range(rank_size)]
    dist.all_gather(gathered_mismatches, mismatch_tensor)
    mismatch_counts = [int(value.item()) for value in gathered_mismatches]
    passed = all(value == 0 for value in mismatch_counts)
    if rank == 0:
        if passed:
            print(
                f"  {GREEN}[PASS]{RESET} All rank outputs are bitwise identical.",
                flush=True,
            )
        else:
            print(
                f"  {RED}[FAIL]{RESET} Rank outputs differ bitwise from rank 0: "
                f"mismatch_counts={mismatch_counts}",
                flush=True,
            )
    return passed, mismatch_counts


def generate_matrices_for_rank(rank, M, N, K, dtype, seed_base=42):
    """
    Generate consistent A and B matrices for a given rank using a seed.

    Args:
        rank: Rank ID
        M, N, K: Matrix dimensions
        dtype: Data type
        seed_base: Base seed for reproducibility

    Returns:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
    """
    # Use rank-specific seed for reproducibility
    torch.manual_seed(seed_base + rank)
    # Match triton_dist.utils.rand_tensor for floating-point inputs:
    # sample uniformly from [-1, 1].
    A = torch.rand([M, K], dtype=dtype) * 2 - 1
    B = torch.rand([K, N], dtype=dtype) * 2 - 1
    return A, B


def peer_memory_size(M, K, N, world_size, ncore, config):
    """Return the packed BF16 buffer size after checking the UDMA transfer limit."""
    block_m, block_n, _, _, _, pvalue = config
    total_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    batch_tiles = min(total_tiles, ncore * pvalue)
    transfer_tiles = (
        triton.cdiv(batch_tiles, world_size) if use_owner_rsag(M, K, N) else batch_tiles
    )
    transfer_bytes = transfer_tiles * block_m * block_n * 2
    if transfer_bytes > MAX_UDMA_PUT_BYTES:
        raise ValueError(
            f"{collective_name(M, K, N)} transfer exceeds the 256 MB "
            f"UDMA putmem limit: {transfer_bytes} bytes"
        )
    return total_tiles * block_m * block_n * world_size


def check_accuracy(C_golden, C, rank, world_size, M, K, N):
    """Check rank agreement before using rank 0 for the FP64 fallback.

    The inputs must come from generate_matrices_for_rank with its default seed.
    Both checks run outside the profiled region and fail on every rank.
    """
    bitwise_passed, mismatch_counts = check_rank_outputs_bitwise(C, rank, world_size)
    if not bitwise_passed:
        raise AssertionError(f"Rank outputs differ: {mismatch_counts}")
    passed, failures = gemm_allreduce_check_accuracy(
        C_golden, C, rank, world_size, M, K, N
    )
    if not passed:
        raise AssertionError(
            "; ".join(failures) or "gemm_allreduce_udma correctness check failed"
        )


def shmem_init(pe, world_size, pool_bytes):
    """Initialize SHMEM once."""
    init_aclshmem(pe, world_size, pool_bytes, ash.OpEngineType.UDMA)


def run_test_distributed():
    dtype, M, N, K = torch.bfloat16, 4096, 4096, 4096
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    local_pe = int(os.environ["LOCAL_RANK"])
    ncore = driver.active.utils.get_aicore_num()
    # Fixed configuration for the 4096 x 4096 x 4096 BF16 example.
    config = (128, 256, 256, 128, 128, 32)
    block_m, block_n, block_k, reduce_m, reduce_n, pvalue = config
    peer_mem_size = peer_memory_size(M, K, N, world_size, ncore, config)
    # The bf16 peer arena sized above is the only symmetric allocation.
    pool_bytes = _ash_pool_bytes(peer_mem_size * 2)
    shmem_init(pe, world_size, pool_bytes)
    peer_mem = None
    try:
        peer_mem = ash.aclshmem_create_tensor(
            [peer_mem_size], dtype=dtype, device_id=local_pe
        )
        A_cpu, B_cpu = generate_matrices_for_rank(pe, M, N, K, dtype)
        A_rank, B_rank = A_cpu.npu(), B_cpu.npu()
        C = torch.zeros([M, N], dtype=dtype).npu()
        C_golden = torch_gemm_allreduce(A_rank, B_rank)
        gemm_allreduce(
            A_rank,
            B_rank,
            C,
            peer_mem,
            pe,
            world_size,
            pvalue,
            block_m,
            block_n,
            block_k,
            reduce_m,
            reduce_n,
        )
        torch.npu.synchronize()
        check_accuracy(C_golden, C, pe, world_size, M, K, N)
    finally:
        if peer_mem is not None:
            ash.aclshmem_free_tensor(peer_mem)
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl")
    pe = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[INFO] Rank {pe} of {world_size} initialized")
    dist.barrier()
    try:
        run_test_distributed()
        if pe == 0:
            print(f"[INFO] Test passed successfully for {world_size} ranks!")
    finally:
        dist.destroy_process_group()
