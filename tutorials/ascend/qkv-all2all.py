# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""QKV All2All with contiguous head-major peer tiles on every path.

For one to four local heads, all ``(target rank, S block)`` tasks are flattened
over half of the available AIV cores.  One task moves Q, K, and V together,
writes the local rank directly to the ordinary outputs, and uses exactly one
remote fence/notify for a remote rank.  Remote staging stores each B112xD
source/fused-head tile contiguously before the matching consumer reads it.
Five to seven local heads retain V0's B112/W4 head-grouped task schedule while
using the same contiguous source/fused-head staging layout.  The public
launcher, output layout, workspace API,
and 189-shape HCCL benchmark contract are unchanged.
"""

import math
import os

import shmem as ash
import torch
import torch_npu
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.runtime import driver
from triton_dist.language.extra import libshmem_device
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


COMM_BLOCK_S = 112
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
COMM_BLOCK_D = 128
FLAT_HEAD_LIMIT = 4

# dl.notify requires an int32/uint32 signal pointer on Ascend.
# aclshmem_wait walks barriers 64 bytes apart: 64 / sizeof(int32) = 16 elements.
SIGNAL_SLOT_STRIDE = 16


def _aiv_role_counts(total_aiv_cores):
    """Split every available AIV between concurrent producers and consumers."""
    if total_aiv_cores < 2:
        raise RuntimeError(
            "QKV All2All requires at least two AIV cores; "
            f"device reports {total_aiv_cores}"
        )
    producer_cores = (total_aiv_cores + 1) // 2
    consumer_cores = total_aiv_cores - producer_cores
    return producer_cores, consumer_cores


def qkv_fuse_a2a_golden(q, k, v, dsp_size, group=None):
    """Return the standard torch QKV All2All reference result."""
    q_transposed = torch.cat(torch.chunk(q, dsp_size, dim=1), dim=0)
    k_transposed = torch.cat(torch.chunk(k, dsp_size, dim=1), dim=0)
    v_transposed = torch.cat(torch.chunk(v, dsp_size, dim=1), dim=0)
    heads_per_rank = q_transposed.shape[1]
    qkv = torch.cat([q_transposed, k_transposed, v_transposed], dim=1).contiguous()
    recv = torch.empty_like(qkv)
    dist.all_to_all_single(recv, qkv, group=group)
    return recv.split([heads_per_rank] * 3, dim=1)


def _head_group_size(heads_per_rank):
    """Bound head-loop expansion while keeping enough producer tasks."""
    if heads_per_rank <= 4:
        return 0  # Block-flat path: one notify covers Q/K/V for the block.
    if heads_per_rank <= 16:
        return 2
    return 4


def _signal_groups(heads_per_rank):
    group_size = _head_group_size(heads_per_rank)
    return 1 if group_size == 0 else math.ceil(heads_per_rank / group_size)


def workspace_sizes(
    sequence_length,
    n_head,
    rank_size,
    head_dim=COMM_BLOCK_D,
    comm_block_s=None,
):
    """Return required ``(peer_elements, signal_elements, buffer_num)``."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if rank_size <= 0:
        raise ValueError("rank_size must be positive")
    if n_head <= 0 or n_head % rank_size != 0:
        raise ValueError("n_head must be positive and divisible by rank_size")
    if head_dim != COMM_BLOCK_D:
        raise ValueError(f"head_dim must be {COMM_BLOCK_D}")

    if comm_block_s is None:
        comm_block_s = COMM_BLOCK_S
    if comm_block_s != COMM_BLOCK_S:
        raise ValueError(f"comm_block_s must be {COMM_BLOCK_S}")

    buffer_num = math.ceil(sequence_length / comm_block_s)
    heads_per_rank = n_head // rank_size
    fused_heads = 3 * heads_per_rank
    peer_elements = buffer_num * comm_block_s * rank_size * fused_heads * head_dim
    # SHMEM signal addressing uses element offsets on an int32 tensor.
    signal_elements = (
        buffer_num * rank_size * _signal_groups(heads_per_rank) * SIGNAL_SLOT_STRIDE
    )
    return peer_elements, signal_elements, buffer_num


def symmetric_memory_size(rank_size, s_values, h_values) -> int:
    """Return the aligned ACLSHMEM heap capacity the sweep needs.

    The peer staging holds every rank's q/k/v payload for one case and the
    int32 signal arena one slot-stride row per (S block, rank); both are
    worst-cased over the sweep via workspace_sizes().
    """
    peer_bytes = signal_bytes = 0
    for s_value, h_value in ((s, h) for s in s_values for h in h_values):
        peer_elems, signal_elems, _ = workspace_sizes(
            s_value, h_value * rank_size, rank_size
        )
        peer_bytes = max(peer_bytes, peer_elems * 2)
        signal_bytes = max(signal_bytes, signal_elems * 4)
    required = peer_bytes + signal_bytes
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(rank_size, s_values, h_values) -> int:
    """Symmetric heap: QKV_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("QKV_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(rank_size, s_values, h_values)
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the QKV_ASH_SIZE_GB override"
        )
    return pool_bytes


@triton.jit
def kernel_qkv_fuse_a2a_block_flat(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    FUSED_HEADS: tl.constexpr,
    PRODUCER_CORES: tl.constexpr,
    CONSUMER_CORES: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Flatten every small-head ``(target rank, S block)`` QKV task."""
    pid = tl.program_id(axis=0)
    role = pid % 2
    logical_core_id = pid // 2

    num_blocks = tl.cdiv(sequence_length, BLOCK_S)
    total_tasks = rank_size * num_blocks
    peer_head_size = BLOCK_S * BLOCK_D
    block_slot_size = rank_size * FUSED_HEADS * peer_head_size
    offs_d = tl.arange(0, BLOCK_D)

    # Even AIVs produce.  Each task completes Q/K/V before its sole notify.
    if role == 0:
        for flat_idx in range(logical_core_id, total_tasks, PRODUCER_CORES):
            target_rank = flat_idx // num_blocks
            block_id = flat_idx % num_blocks
            offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
            row_mask = offs_s < sequence_length
            io_mask = row_mask[:, None]
            input_base = offs_s[:, None] * stride_a_s + offs_d[None, :] * stride_a_d
            input_head_base = target_rank * HEADS_PER_RANK

            if target_rank == rank:
                output_row = rank * sequence_length + offs_s
                output_base = (
                    output_row[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
                )
                for head in range(0, HEADS_PER_RANK):
                    input_offs = input_base + (input_head_base + head) * stride_a_h
                    output_offs = output_base + head * stride_out_h
                    q_data = tl.load(q_ptr + input_offs, mask=io_mask, other=0.0)
                    k_data = tl.load(k_ptr + input_offs, mask=io_mask, other=0.0)
                    v_data = tl.load(v_ptr + input_offs, mask=io_mask, other=0.0)
                    tl.store(q_out_ptr + output_offs, q_data, mask=io_mask)
                    tl.store(k_out_ptr + output_offs, k_data, mask=io_mask)
                    tl.store(v_out_ptr + output_offs, v_data, mask=io_mask)
            else:
                remote_peer_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                peer_block_base = block_id * block_slot_size
                peer_source_base = rank * FUSED_HEADS * peer_head_size
                peer_row_base = (
                    peer_block_base
                    + tl.arange(0, BLOCK_S)[:, None] * BLOCK_D
                    + offs_d[None, :]
                )
                for head in range(0, HEADS_PER_RANK):
                    input_offs = input_base + (input_head_base + head) * stride_a_h
                    q_data = tl.load(q_ptr + input_offs, mask=io_mask, other=0.0)
                    k_data = tl.load(k_ptr + input_offs, mask=io_mask, other=0.0)
                    v_data = tl.load(v_ptr + input_offs, mask=io_mask, other=0.0)
                    peer_head = peer_source_base + head * peer_head_size
                    tl.store(
                        remote_peer_ptr + peer_row_base + peer_head,
                        q_data,
                        mask=io_mask,
                    )
                    tl.store(
                        remote_peer_ptr
                        + peer_row_base
                        + peer_head
                        + HEADS_PER_RANK * peer_head_size,
                        k_data,
                        mask=io_mask,
                    )
                    tl.store(
                        remote_peer_ptr
                        + peer_row_base
                        + peer_head
                        + 2 * HEADS_PER_RANK * peer_head_size,
                        v_data,
                        mask=io_mask,
                    )

                libshmem_device.fence()
                signal_slot = block_id * rank_size + rank
                dl.notify(
                    signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                    target_rank,
                    1,
                )

    # Odd AIVs consume remote sources only.  Q/K/V share one acquire wait.
    if role == 1:
        for flat_idx in range(logical_core_id, total_tasks, CONSUMER_CORES):
            source_rank = flat_idx // num_blocks
            block_id = flat_idx % num_blocks
            if source_rank != rank:
                signal_slot = block_id * rank_size + source_rank
                token = dl.wait(
                    signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                    1,
                    "gpu",
                    "acquire",
                    waitValue=1,
                )
                local_peer_ptr = dl.consume_token(peer_mem_ptr, token)
                offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                row_mask = offs_s < sequence_length
                io_mask = row_mask[:, None]
                output_row = source_rank * sequence_length + offs_s
                output_base = (
                    output_row[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
                )
                peer_block_base = block_id * block_slot_size
                peer_source_base = source_rank * FUSED_HEADS * peer_head_size
                peer_row_base = (
                    peer_block_base
                    + tl.arange(0, BLOCK_S)[:, None] * BLOCK_D
                    + offs_d[None, :]
                )
                for head in range(0, HEADS_PER_RANK):
                    peer_head = peer_source_base + head * peer_head_size
                    q_data = tl.load(
                        local_peer_ptr + peer_row_base + peer_head,
                        mask=io_mask,
                        other=0.0,
                    )
                    k_data = tl.load(
                        local_peer_ptr
                        + peer_row_base
                        + peer_head
                        + HEADS_PER_RANK * peer_head_size,
                        mask=io_mask,
                        other=0.0,
                    )
                    v_data = tl.load(
                        local_peer_ptr
                        + peer_row_base
                        + peer_head
                        + 2 * HEADS_PER_RANK * peer_head_size,
                        mask=io_mask,
                        other=0.0,
                    )
                    output_offs = output_base + head * stride_out_h
                    tl.store(q_out_ptr + output_offs, q_data, mask=io_mask)
                    tl.store(k_out_ptr + output_offs, k_data, mask=io_mask)
                    tl.store(v_out_ptr + output_offs, v_data, mask=io_mask)


@triton.jit
def kernel_qkv_fuse_a2a_rank_batched(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_WAVE: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    FUSED_HEADS: tl.constexpr,
    TOTAL_CORES: tl.constexpr,
    PRODUCER_CORES: tl.constexpr,
    CONSUMER_CORES: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Low-overhead path for at most four heads per destination rank."""
    pid = tl.program_id(axis=0)
    producers_before = (pid * PRODUCER_CORES + TOTAL_CORES - 1) // TOTAL_CORES
    producers_through = ((pid + 1) * PRODUCER_CORES + TOTAL_CORES - 1) // TOTAL_CORES

    num_blocks = tl.cdiv(sequence_length, BLOCK_S)
    stride_peer_s = rank_size * FUSED_HEADS * BLOCK_D
    stride_peer_h = BLOCK_D
    block_slot_size = BLOCK_S * stride_peer_s

    for wave_start in range(0, num_blocks, S_WAVE):
        total_wave_tasks = rank_size * S_WAVE

        if producers_through > producers_before:
            logical_core_id = producers_before
            for flat_idx in range(logical_core_id, total_wave_tasks, PRODUCER_CORES):
                lane = flat_idx % S_WAVE
                target_rank = flat_idx // S_WAVE
                block_id = wave_start + lane

                if block_id < num_blocks:
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    input_base = (
                        offs_s[:, None] * stride_a_s + offs_d[None, :] * stride_a_d
                    )
                    output_row = rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    input_head_base = target_rank * HEADS_PER_RANK
                    peer_block_base = (block_id % buffer_num) * block_slot_size

                    for fused_head in range(0, FUSED_HEADS):
                        if fused_head < HEADS_PER_RANK:
                            output_head = fused_head
                            data = tl.load(
                                q_ptr
                                + input_base
                                + (input_head_base + output_head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                        elif fused_head < 2 * HEADS_PER_RANK:
                            output_head = fused_head - HEADS_PER_RANK
                            data = tl.load(
                                k_ptr
                                + input_base
                                + (input_head_base + output_head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                        else:
                            output_head = fused_head - 2 * HEADS_PER_RANK
                            data = tl.load(
                                v_ptr
                                + input_base
                                + (input_head_base + output_head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )

                        if target_rank == rank:
                            if fused_head < HEADS_PER_RANK:
                                tl.store(
                                    q_out_ptr
                                    + output_base
                                    + output_head * stride_out_h,
                                    data,
                                    mask=io_mask,
                                )
                            elif fused_head < 2 * HEADS_PER_RANK:
                                tl.store(
                                    k_out_ptr
                                    + output_base
                                    + output_head * stride_out_h,
                                    data,
                                    mask=io_mask,
                                )
                            else:
                                tl.store(
                                    v_out_ptr
                                    + output_base
                                    + output_head * stride_out_h,
                                    data,
                                    mask=io_mask,
                                )
                        else:
                            remote_peer_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                            peer_offs = (
                                peer_block_base
                                + tl.arange(0, BLOCK_S)[:, None] * stride_peer_s
                                + (rank * FUSED_HEADS + fused_head) * stride_peer_h
                                + offs_d[None, :]
                            )
                            tl.store(
                                remote_peer_ptr + peer_offs,
                                data,
                                mask=io_mask,
                            )

            libshmem_device.fence()
            for flat_idx in range(logical_core_id, total_wave_tasks, PRODUCER_CORES):
                lane = flat_idx % S_WAVE
                target_rank = flat_idx // S_WAVE
                block_id = wave_start + lane
                if block_id < num_blocks and target_rank != rank:
                    signal_slot = (block_id % buffer_num) * rank_size + rank
                    dl.notify(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        target_rank,
                        1,
                    )

        if producers_through == producers_before:
            logical_core_id = pid - producers_before
            for flat_idx in range(logical_core_id, total_wave_tasks, CONSUMER_CORES):
                lane = flat_idx % S_WAVE
                source_rank = flat_idx // S_WAVE
                block_id = wave_start + lane

                if block_id < num_blocks and source_rank != rank:
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    output_row = source_rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    peer_block_base = (block_id % buffer_num) * block_slot_size
                    signal_slot = (block_id % buffer_num) * rank_size + source_rank
                    token = dl.wait(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        1,
                        "gpu",
                        "acquire",
                        waitValue=1,
                    )
                    local_peer_ptr = dl.consume_token(peer_mem_ptr, token)

                    for fused_head in range(0, FUSED_HEADS):
                        peer_offs = (
                            peer_block_base
                            + tl.arange(0, BLOCK_S)[:, None] * stride_peer_s
                            + (source_rank * FUSED_HEADS + fused_head) * stride_peer_h
                            + offs_d[None, :]
                        )
                        data = tl.load(
                            local_peer_ptr + peer_offs,
                            mask=io_mask,
                            other=0.0,
                        )

                        if fused_head < HEADS_PER_RANK:
                            tl.store(
                                q_out_ptr + output_base + fused_head * stride_out_h,
                                data,
                                mask=io_mask,
                            )
                        elif fused_head < 2 * HEADS_PER_RANK:
                            output_head = fused_head - HEADS_PER_RANK
                            tl.store(
                                k_out_ptr + output_base + output_head * stride_out_h,
                                data,
                                mask=io_mask,
                            )
                        else:
                            output_head = fused_head - 2 * HEADS_PER_RANK
                            tl.store(
                                v_out_ptr + output_base + output_head * stride_out_h,
                                data,
                                mask=io_mask,
                            )


@triton.jit
def kernel_qkv_fuse_a2a_head_grouped(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_WAVE: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    TOTAL_CORES: tl.constexpr,
    PRODUCER_CORES: tl.constexpr,
    CONSUMER_CORES: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Parallelize bounded non-interleaved head groups across AIV cores."""
    pid = tl.program_id(axis=0)
    producers_before = (pid * PRODUCER_CORES + TOTAL_CORES - 1) // TOTAL_CORES
    producers_through = ((pid + 1) * PRODUCER_CORES + TOTAL_CORES - 1) // TOTAL_CORES

    fused_heads = 3 * HEADS_PER_RANK
    num_blocks = tl.cdiv(sequence_length, BLOCK_S)
    peer_head_size = BLOCK_S * BLOCK_D
    block_slot_size = rank_size * fused_heads * peer_head_size

    for wave_start in range(0, num_blocks, S_WAVE):
        total_wave_tasks = rank_size * HEAD_GROUPS * S_WAVE

        if producers_through > producers_before:
            logical_core_id = producers_before
            for flat_idx in range(logical_core_id, total_wave_tasks, PRODUCER_CORES):
                lane = flat_idx % S_WAVE
                task_idx = flat_idx // S_WAVE
                target_rank = task_idx % rank_size
                head_group = task_idx // rank_size
                block_id = wave_start + lane

                if block_id < num_blocks:
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    input_base = (
                        offs_s[:, None] * stride_a_s + offs_d[None, :] * stride_a_d
                    )
                    output_row = rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    input_head_base = target_rank * HEADS_PER_RANK
                    peer_block_base = (block_id % buffer_num) * block_slot_size

                    for local_head in range(0, HEADS_PER_GROUP):
                        head = head_group * HEADS_PER_GROUP + local_head
                        if head < HEADS_PER_RANK:
                            q_data = tl.load(
                                q_ptr
                                + input_base
                                + (input_head_base + head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                            k_data = tl.load(
                                k_ptr
                                + input_base
                                + (input_head_base + head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )
                            v_data = tl.load(
                                v_ptr
                                + input_base
                                + (input_head_base + head) * stride_a_h,
                                mask=io_mask,
                                other=0.0,
                            )

                            if target_rank == rank:
                                output_offs = output_base + head * stride_out_h
                                tl.store(
                                    q_out_ptr + output_offs,
                                    q_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    k_out_ptr + output_offs,
                                    k_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    v_out_ptr + output_offs,
                                    v_data,
                                    mask=io_mask,
                                )
                            else:
                                remote_peer_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                                peer_base = (
                                    peer_block_base
                                    + rank * fused_heads * peer_head_size
                                    + head * peer_head_size
                                    + tl.arange(0, BLOCK_S)[:, None] * BLOCK_D
                                    + offs_d[None, :]
                                )
                                tl.store(
                                    remote_peer_ptr + peer_base,
                                    q_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    remote_peer_ptr
                                    + peer_base
                                    + HEADS_PER_RANK * peer_head_size,
                                    k_data,
                                    mask=io_mask,
                                )
                                tl.store(
                                    remote_peer_ptr
                                    + peer_base
                                    + 2 * HEADS_PER_RANK * peer_head_size,
                                    v_data,
                                    mask=io_mask,
                                )

            libshmem_device.fence()
            for flat_idx in range(logical_core_id, total_wave_tasks, PRODUCER_CORES):
                lane = flat_idx % S_WAVE
                task_idx = flat_idx // S_WAVE
                target_rank = task_idx % rank_size
                head_group = task_idx // rank_size
                block_id = wave_start + lane
                if block_id < num_blocks and target_rank != rank:
                    signal_slot = (
                        (block_id % buffer_num) * rank_size + rank
                    ) * HEAD_GROUPS + head_group
                    dl.notify(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        target_rank,
                        1,
                    )

        if producers_through == producers_before:
            logical_core_id = pid - producers_before
            for flat_idx in range(logical_core_id, total_wave_tasks, CONSUMER_CORES):
                lane = flat_idx % S_WAVE
                task_idx = flat_idx // S_WAVE
                source_rank = task_idx % rank_size
                head_group = task_idx // rank_size
                block_id = wave_start + lane

                if block_id < num_blocks and source_rank != rank:
                    offs_s = block_id * BLOCK_S + tl.arange(0, BLOCK_S)
                    offs_d = tl.arange(0, BLOCK_D)
                    row_mask = offs_s < sequence_length
                    io_mask = row_mask[:, None]
                    output_row = source_rank * sequence_length + offs_s
                    output_base = (
                        output_row[:, None] * stride_out_s
                        + offs_d[None, :] * stride_out_d
                    )
                    peer_block_base = (block_id % buffer_num) * block_slot_size
                    signal_slot = (
                        (block_id % buffer_num) * rank_size + source_rank
                    ) * HEAD_GROUPS + head_group
                    token = dl.wait(
                        signal_mem_ptr + signal_slot * SIGNAL_STRIDE,
                        1,
                        "gpu",
                        "acquire",
                        waitValue=1,
                    )
                    local_peer_ptr = dl.consume_token(peer_mem_ptr, token)

                    for local_head in range(0, HEADS_PER_GROUP):
                        head = head_group * HEADS_PER_GROUP + local_head
                        if head < HEADS_PER_RANK:
                            peer_base = (
                                peer_block_base
                                + source_rank * fused_heads * peer_head_size
                                + head * peer_head_size
                                + tl.arange(0, BLOCK_S)[:, None] * BLOCK_D
                                + offs_d[None, :]
                            )
                            q_data = tl.load(
                                local_peer_ptr + peer_base,
                                mask=io_mask,
                                other=0.0,
                            )
                            k_data = tl.load(
                                local_peer_ptr
                                + peer_base
                                + HEADS_PER_RANK * peer_head_size,
                                mask=io_mask,
                                other=0.0,
                            )
                            v_data = tl.load(
                                local_peer_ptr
                                + peer_base
                                + 2 * HEADS_PER_RANK * peer_head_size,
                                mask=io_mask,
                                other=0.0,
                            )
                            output_offs = output_base + head * stride_out_h
                            tl.store(
                                q_out_ptr + output_offs,
                                q_data,
                                mask=io_mask,
                            )
                            tl.store(
                                k_out_ptr + output_offs,
                                k_data,
                                mask=io_mask,
                            )
                            tl.store(
                                v_out_ptr + output_offs,
                                v_data,
                                mask=io_mask,
                            )


def qkv_fuse_a2a(
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
    buffer_num=None,
    comm_block_s=None,
    comm_block_d=COMM_BLOCK_D,
):
    """Launch the shape-adaptive non-interleaved QKV All2All kernel."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.stride() != k.stride() or q.stride() != v.stride():
        raise ValueError("q, k, and v must have identical strides")
    if q.ndim != 3:
        raise ValueError("q, k, and v must have shape (S, n_head, head_dim)")

    sequence_length, n_head, head_dim = q.shape
    if n_head % rank_size != 0:
        raise ValueError("n_head must be divisible by rank_size")
    heads_per_rank = n_head // rank_size
    if comm_block_s is None:
        comm_block_s = COMM_BLOCK_S
    if comm_block_s != COMM_BLOCK_S or comm_block_d != COMM_BLOCK_D:
        raise ValueError(
            f"this implementation requires tiles ({COMM_BLOCK_S}, {COMM_BLOCK_D})"
        )

    peer_elements, signal_elements, required_buffers = workspace_sizes(
        sequence_length, n_head, rank_size, head_dim, comm_block_s
    )
    if buffer_num is not None and buffer_num < required_buffers:
        raise ValueError(
            f"buffer_num={buffer_num} is smaller than required {required_buffers}"
        )
    if peer_mem.numel() < peer_elements:
        raise ValueError(
            f"peer_mem has {peer_mem.numel()} elements; {peer_elements} required"
        )
    if signal_mem.numel() < signal_elements:
        raise ValueError(
            f"signal_mem has {signal_mem.numel()} elements; "
            f"{signal_elements} required"
        )

    expected_output_shape = (
        sequence_length * rank_size,
        heads_per_rank,
        head_dim,
    )
    for name, output in (("q_out", q_out), ("k_out", k_out), ("v_out", v_out)):
        if tuple(output.shape) != expected_output_shape:
            raise ValueError(
                f"{name} has shape {tuple(output.shape)}; "
                f"expected {expected_output_shape}"
            )
    if q_out.stride() != k_out.stride() or q_out.stride() != v_out.stride():
        raise ValueError("q_out, k_out, and v_out must have identical strides")

    vec_num = driver.active.utils.get_aivector_core_num()
    producer_cores, consumer_cores = _aiv_role_counts(vec_num)
    s_wave = 4 if rank_size <= 6 else (2 if rank_size < 24 else 1)
    group_size = _head_group_size(heads_per_rank)
    common_args = (
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
        required_buffers,
        sequence_length,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
    )

    if group_size == 0:
        kernel_qkv_fuse_a2a_block_flat[vec_num, 1, 1](
            *common_args,
            BLOCK_S=comm_block_s,
            BLOCK_D=comm_block_d,
            HEADS_PER_RANK=heads_per_rank,
            FUSED_HEADS=3 * heads_per_rank,
            PRODUCER_CORES=producer_cores,
            CONSUMER_CORES=consumer_cores,
            SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
        )
    else:
        head_groups = math.ceil(heads_per_rank / group_size)
        kernel_qkv_fuse_a2a_head_grouped[vec_num, 1, 1](
            *common_args,
            BLOCK_S=comm_block_s,
            BLOCK_D=comm_block_d,
            S_WAVE=s_wave,
            HEADS_PER_RANK=heads_per_rank,
            HEADS_PER_GROUP=group_size,
            HEAD_GROUPS=head_groups,
            TOTAL_CORES=vec_num,
            PRODUCER_CORES=producer_cores,
            CONSUMER_CORES=consumer_cores,
            SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
        )


def run_hccl_benchmark():
    """Small correctness harness; shape lists are configurable."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    if rank == 0:
        total_aiv_cores = driver.active.utils.get_aivector_core_num()
        producer_cores, consumer_cores = _aiv_role_counts(total_aiv_cores)
        print(
            "CONFIG "
            f"total_aiv_cores={total_aiv_cores} "
            f"producer_cores={producer_cores} "
            f"consumer_cores={consumer_cores} "
            f"block_s={COMM_BLOCK_S} s_wave=4"
        )
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
    pool_bytes = ash_pool_bytes(rank_size, sequence_lengths, local_heads)

    dist.barrier()
    init_aclshmem(rank, rank_size, pool_bytes, ash.OpEngineType.MTE)

    try:
        for sequence_length in sequence_lengths:
            for heads_per_rank in local_heads:
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

                q_golden, k_golden, v_golden = qkv_fuse_a2a_golden(q, k, v, rank_size)

                peer_elements, signal_elements, buffer_num = workspace_sizes(
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
                qkv_fuse_a2a(
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
                torch.npu.synchronize()

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

                if any(r.item() == 0 for r in all_passed):
                    raise AssertionError(
                        f"Parameter sweep failed at case: "
                        f"S={sequence_length}, H={n_head}"
                    )
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_hccl_benchmark()
