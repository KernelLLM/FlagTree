# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ascend UDMA MoE Dispatch implementation and correctness check.

The candidate starts from the same ``mlp_input + scatter_index`` boundary as the
Torch/HCCL golden.  A Triton pack kernel materializes the stable expert-routed
sequence without a Torch ``repeat`` or indexed assignment.  Direct exchange
batches non-blocking expert-bucket puts from one physical core per peer and
lands in final expert-major order. Small and medium payloads use one-shot pack
then UDMA; large payloads use an adaptive destination-major half-wave pipeline.
The selector depends only on input shapes and dtype, and therefore adds no
device synchronization or input-derived planning kernel. Performance timing is
provided by ``python/triton_dist/test/ascend/moe-dispatch-udma.py``.

This file is independent from the MTE baseline.  Selecting UDMA here is a source
fact; whether a particular 950 software stack carries it over URMA must be
established from target-machine build/runtime/profiler evidence.
"""

import math
import os
import random
import torch
import torch_npu
import shmem as ash
import torch.distributed as dist
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id, sub_vec_num
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Shape sweep: N (tokens) x topk x num_local_experts x hidden_size; ep_size = world_size.
_DEFAULT_N_LIST = (
    list(range(2040, 2049))
    + list(range(4088, 4097))
    + list(range(8184, 8193))
    + list(range(16376, 16385))
    + list(range(32760, 32769))
)
N_LIST = [
    int(x)
    for x in os.environ.get(
        "MOE_N_LIST", ",".join(str(x) for x in _DEFAULT_N_LIST)
    ).split(",")
    if x
]
TOPK_LIST = [int(x) for x in os.environ.get("MOE_TOPK_LIST", "8").split(",") if x]
NUM_LOCAL_EXPERTS_LIST = [
    int(x) for x in os.environ.get("MOE_NLE_LIST", "40,160").split(",") if x
]
ROUTE_MODES = [
    x.strip()
    for x in os.environ.get("MOE_ROUTE_MODES", "random").split(",")
    if x.strip()
]
HIDDEN_SIZE_LIST = [int(x) for x in os.environ.get("MOE_D", "1536").split(",") if x]
ROUTE_SEED = int(os.environ.get("MOE_ROUTE_SEED", "2026"))
PACK_BLOCK_N = int(os.environ.get("MOE_PACK_BLOCK_N", "16"))
PACK_BLOCK_D_CONFIG = os.environ.get("MOE_PACK_BLOCK_D", "auto").strip().lower()


def symmetric_memory_size(ep_size: int) -> int:
    """Return the aligned ACLSHMEM heap capacity the sweep needs.

    The receive workspace holds, per source rank, the tokens this rank
    receives from it.  Under the default uniform-random routing the
    per-source count is binomial with mean N * topk / ep_size; mean plus six
    sigma covers the maximum over all ep_size**2 (dest, source) pairs.
    Adversarial routings ("hot", "skewed") exceed that by design -- size
    those runs with the MOE_ASH_SIZE_GB override.
    """
    tokens = max(N_LIST) * max(TOPK_LIST)
    mean = tokens / ep_size
    sigma = math.sqrt(max(1.0, mean * (1.0 - 1.0 / ep_size)))
    per_source = math.ceil(mean + 6.0 * sigma) + 1
    # bf16 output_base plus one int64 recv_offsets entry per expert.
    output_bytes = ep_size * per_source * max(HIDDEN_SIZE_LIST) * 2
    recv_offsets_bytes = max(NUM_LOCAL_EXPERTS_LIST) * ep_size * 8
    required = output_bytes + recv_offsets_bytes
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(ep_size: int) -> int:
    """Symmetric heap: MOE_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("MOE_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(ep_size)
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the MOE_ASH_SIZE_GB override"
        )
    return pool_bytes
STRATEGY_OVERRIDE = os.environ.get("MOE_DISPATCH_STRATEGY", "auto").strip().lower()
WAVE_REMOTE_BYTES_THRESHOLD = 256 * 1024 * 1024
WAVE_FIRST_NUM = 1
WAVE_FIRST_DEN = 2
INVERSE_WAVE_CONTEXT = None


def _resolve_pack_block_d(config, hidden_size):
    """Resolve the fixed tile or the first data-driven hidden-size policy."""
    if hidden_size <= 0:
        raise ValueError(f"MOE_D must be > 0, got {hidden_size}")
    if config == "auto":
        # Keep one D tile for common hidden sizes without exceeding the
        # BLOCK_D=2048 candidate validated at D=1536.  The lower bound avoids
        # introducing unmeasured tiny vector tiles in the boundary sweep.
        next_power_of_two = 1 << (hidden_size - 1).bit_length()
        return min(2048, max(128, next_power_of_two))
    try:
        block_d = int(config)
    except ValueError as error:
        raise ValueError(
            "MOE_PACK_BLOCK_D must be 'auto' or a positive power of two, "
            f"got {config!r}"
        ) from error
    if block_d <= 0 or block_d & (block_d - 1):
        raise ValueError(
            "MOE_PACK_BLOCK_D must be 'auto' or a positive power of two, "
            f"got {config!r}"
        )
    return block_d


if PACK_BLOCK_N <= 0 or PACK_BLOCK_N & (PACK_BLOCK_N - 1):
    raise ValueError(
        "MOE_PACK_BLOCK_N must be a positive power of two, " f"got {PACK_BLOCK_N}"
    )
if STRATEGY_OVERRIDE not in {"auto", "serial", "wave"}:
    raise ValueError(
        "MOE_DISPATCH_STRATEGY must be auto, serial, or wave, "
        f"got {STRATEGY_OVERRIDE!r}"
    )


def select_dispatch_strategy(n, d, topk, element_size, rank_size, full_range=True):
    """Select half-wave only after its estimated remote bytes amortize overhead."""
    if STRATEGY_OVERRIDE != "auto":
        selected = STRATEGY_OVERRIDE
    else:
        remote_bytes = n * topk * d * element_size * (rank_size - 1) // rank_size
        selected = "wave" if remote_bytes >= WAVE_REMOTE_BYTES_THRESHOLD else "serial"
    if selected == "wave" and not full_range:
        return "serial"
    return selected


def prepare_inverse_wave_context(total_branches, device):
    """Allocate data-independent reusable workspace and stream events."""
    return {
        "inverse": torch.empty(total_branches, dtype=torch.int32, device=device),
        # [bucket, count], derived by the timed metadata kernel.
        "heavy_bucket": torch.empty(2, dtype=torch.int32, device=device),
        "pack_stream": torch.npu.Stream(device=device),
        "comm_stream": torch.npu.Stream(device=device),
        "start": torch.npu.Event(),
        "metadata_ready": torch.npu.Event(),
        "pack_ready": [torch.npu.Event(), torch.npu.Event()],
        "comm_done": torch.npu.Event(),
    }


@triton.jit
def kernel_dispatch_pack(
    mlp_input_ptr,
    scatter_index_ptr,
    dispatched_input_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    TOKEN_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fast no-drop pack for contiguous tensors.

    The current Dispatch contract makes ``scatter_index`` a permutation of
    ``[0, N * TOPK)``.  Making the shape constants and using direct contiguous
    offsets lets the Ascend backend lower the long row stores through the same
    efficient path measured by the standalone pack benchmark.
    """
    pid = tl.program_id(0)
    d_blocks = tl.cdiv(D, BLOCK_D)
    block_n = pid // d_blocks
    block_d = pid % d_blocks
    ns = TOKEN_OFFSET + block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ds = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (ns[:, None] < N) & (ds[None, :] < D)
    data = tl.load(
        mlp_input_ptr + ns[:, None] * D + ds[None, :],
        mask=mask,
        other=0.0,
    )
    for k in range(TOPK):
        dst_row = tl.load(
            scatter_index_ptr + ns * TOPK + k,
            mask=ns < N,
            other=0,
        )
        tl.store(
            dispatched_input_ptr + dst_row[:, None] * D + ds[None, :],
            data,
            mask=mask,
        )


@triton.jit
def kernel_dispatch_pack_dynamic(
    mlp_input_ptr,
    scatter_index_ptr,
    dispatched_input_ptr,
    N,
    D,
    topk,
    start_idx,
    end_idx,
    stride_input_n,
    stride_input_d,
    stride_si_n,
    stride_si_k,
    stride_dispatch_n,
    stride_dispatch_d,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Generic pack fallback for sliced or non-contiguous layouts.

    ``scatter_index[t, k]`` is the final row of branch ``(t, k)`` in the
    stable expert-routed sequence.  Each valid destination row is unique, so
    the pack is a disjoint gather/scatter with no atomics or host data movement.
    """
    pid = tl.program_id(0)
    d_blocks = tl.cdiv(D, BLOCK_D)
    block_n = pid // d_blocks
    block_d = pid % d_blocks
    ns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ds = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    nmask = ns < N
    dmask = ds < D
    src = ns[:, None] * stride_input_n + ds[None, :] * stride_input_d
    data = tl.load(
        mlp_input_ptr + src,
        mask=nmask[:, None] & dmask[None, :],
        other=0.0,
    )
    for k in range(topk):
        dst_row = tl.load(
            scatter_index_ptr + ns * stride_si_n + k * stride_si_k,
            mask=nmask,
            other=-1,
        )
        valid = nmask & (dst_row >= start_idx) & (dst_row < end_idx)
        local_dst_row = dst_row - start_idx
        dst = (
            local_dst_row[:, None] * stride_dispatch_n + ds[None, :] * stride_dispatch_d
        )
        tl.store(
            dispatched_input_ptr + dst,
            data,
            mask=valid[:, None] & dmask[None, :],
        )


@triton.jit
def kernel_dispatch_inverse_wave_plan(
    scatter_index_ptr,
    inverse_ptr,
    TOTAL_BRANCHES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Invert the public no-drop permutation without atomics."""
    branch = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = branch < TOTAL_BRANCHES
    row = tl.load(scatter_index_ptr + branch, mask=valid, other=0)
    tl.store(inverse_ptr + row, branch, mask=valid)


@triton.jit
def kernel_dispatch_pack_wave_inverse_adaptive(
    input_ptr,
    inverse_ptr,
    splits_ptr,
    send_offsets_ptr,
    heavy_bucket_ptr,
    packed_ptr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    FIRST_NUM: tl.constexpr,
    FIRST_DEN: tl.constexpr,
    WAVE: tl.constexpr,
    BASE_WORKERS: tl.constexpr,
    HEAVY_WORKERS: tl.constexpr,
    TIER8_MAX_ROWS: tl.constexpr,
    TIER32_MAX_ROWS: tl.constexpr,
    TIER64_MAX_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Give the largest bucket 8/32/64/128 workers based on its bytes."""
    bucket_slot = tl.program_id(0)
    worker_slot = tl.program_id(1)
    d_block = tl.program_id(2)
    helper = bucket_slot >= NUM_BUCKETS
    heavy_bucket = tl.load(heavy_bucket_ptr)
    heavy_count = tl.load(heavy_bucket_ptr + 1)
    if helper:
        helper_group = bucket_slot - NUM_BUCKETS
        worker = BASE_WORKERS + helper_group * BASE_WORKERS + worker_slot
        bucket = heavy_bucket
    else:
        worker = worker_slot
        bucket = bucket_slot

    count = tl.load(splits_ptr + bucket)
    begin = tl.load(send_offsets_ptr + bucket)
    first = count * FIRST_NUM // FIRST_DEN
    if WAVE == 0:
        wave_begin = 0
        wave_end = first
    else:
        wave_begin = first
        wave_end = count
    ds = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    if bucket != heavy_bucket:
        for row_begin in range(
            wave_begin + worker * BLOCK_N,
            wave_end,
            BASE_WORKERS * BLOCK_N,
        ):
            local_rows = row_begin + tl.arange(0, BLOCK_N)
            row_valid = local_rows < wave_end
            rows = begin + local_rows
            branch = tl.load(inverse_ptr + rows, mask=row_valid, other=0)
            token = branch // TOPK
            mask = row_valid[:, None] & (ds[None, :] < D)
            data = tl.load(
                input_ptr + token[:, None] * D + ds[None, :],
                mask=mask,
                other=0,
            )
            tl.store(packed_ptr + rows[:, None] * D + ds[None, :], data, mask=mask)
    elif heavy_count <= TIER8_MAX_ROWS:
        if worker < BASE_WORKERS:
            for row_begin in range(
                wave_begin + worker * BLOCK_N,
                wave_end,
                BASE_WORKERS * BLOCK_N,
            ):
                local_rows = row_begin + tl.arange(0, BLOCK_N)
                row_valid = local_rows < wave_end
                rows = begin + local_rows
                branch = tl.load(inverse_ptr + rows, mask=row_valid, other=0)
                token = branch // TOPK
                mask = row_valid[:, None] & (ds[None, :] < D)
                data = tl.load(
                    input_ptr + token[:, None] * D + ds[None, :],
                    mask=mask,
                    other=0,
                )
                tl.store(packed_ptr + rows[:, None] * D + ds[None, :], data, mask=mask)
    elif heavy_count <= TIER32_MAX_ROWS:
        if worker < 32:
            for row_begin in range(
                wave_begin + worker * BLOCK_N,
                wave_end,
                32 * BLOCK_N,
            ):
                local_rows = row_begin + tl.arange(0, BLOCK_N)
                row_valid = local_rows < wave_end
                rows = begin + local_rows
                branch = tl.load(inverse_ptr + rows, mask=row_valid, other=0)
                token = branch // TOPK
                mask = row_valid[:, None] & (ds[None, :] < D)
                data = tl.load(
                    input_ptr + token[:, None] * D + ds[None, :],
                    mask=mask,
                    other=0,
                )
                tl.store(packed_ptr + rows[:, None] * D + ds[None, :], data, mask=mask)
    elif heavy_count <= TIER64_MAX_ROWS:
        if worker < 64:
            for row_begin in range(
                wave_begin + worker * BLOCK_N,
                wave_end,
                64 * BLOCK_N,
            ):
                local_rows = row_begin + tl.arange(0, BLOCK_N)
                row_valid = local_rows < wave_end
                rows = begin + local_rows
                branch = tl.load(inverse_ptr + rows, mask=row_valid, other=0)
                token = branch // TOPK
                mask = row_valid[:, None] & (ds[None, :] < D)
                data = tl.load(
                    input_ptr + token[:, None] * D + ds[None, :],
                    mask=mask,
                    other=0,
                )
                tl.store(packed_ptr + rows[:, None] * D + ds[None, :], data, mask=mask)
    else:
        if worker < HEAVY_WORKERS:
            for row_begin in range(
                wave_begin + worker * BLOCK_N,
                wave_end,
                HEAVY_WORKERS * BLOCK_N,
            ):
                local_rows = row_begin + tl.arange(0, BLOCK_N)
                row_valid = local_rows < wave_end
                rows = begin + local_rows
                branch = tl.load(inverse_ptr + rows, mask=row_valid, other=0)
                token = branch // TOPK
                mask = row_valid[:, None] & (ds[None, :] < D)
                data = tl.load(
                    input_ptr + token[:, None] * D + ds[None, :],
                    mask=mask,
                    other=0,
                )
                tl.store(packed_ptr + rows[:, None] * D + ds[None, :], data, mask=mask)


@triton.jit
def kernel_prepare_local_dispatch_metadata(
    a2a_splits_ptr,
    splits_ptr,
    send_offsets_ptr,
    publish_recv_offsets_ptr,
    heavy_bucket_ptr,
    R: tl.constexpr,
    E: tl.constexpr,
    META_SIZE: tl.constexpr,
    SELECT_HEAVY_BUCKET: tl.constexpr,
):
    """Build send and receiver-owned destination prefixes from public inputs."""
    scan = tl.program_id(0)
    offsets = tl.arange(0, META_SIZE)
    valid = offsets < R * E
    if scan == 0:
        counts = tl.load(splits_ptr + offsets, mask=valid, other=0)
        inclusive = tl.cumsum(counts, axis=0)
        tl.store(send_offsets_ptr + offsets, inclusive - counts, mask=valid)
        if SELECT_HEAVY_BUCKET:
            max_count = tl.max(tl.where(valid, counts, 0), axis=0)
            first_max = tl.min(
                tl.where(valid & (counts == max_count), offsets, META_SIZE), axis=0
            )
            tl.store(heavy_bucket_ptr, first_max)
            tl.store(heavy_bucket_ptr + 1, max_count)
    else:
        source = offsets % R
        expert = offsets // R
        publish_index = source * E + expert
        counts = tl.load(a2a_splits_ptr + publish_index, mask=valid, other=0)
        inclusive = tl.cumsum(counts, axis=0)
        tl.store(
            publish_recv_offsets_ptr + publish_index,
            inclusive - counts,
            mask=valid,
        )


@triton.jit(do_not_specialize=["rank"])
def kernel_exchange_dispatch_metadata(
    publish_recv_offsets_ptr,
    recv_offsets_ptr,
    rank,
    R: tl.constexpr,
    E: tl.constexpr,
    E_BLOCK: tl.constexpr,
):
    """Publish each receiver's E destination offsets to every source rank."""
    actual_sub_vec_num = sub_vec_num()
    ncore = tl.num_programs(0) * actual_sub_vec_num // 2
    pid = tl.program_id(0) * actual_sub_vec_num // 2
    global_vec_id = pid * actual_sub_vec_num + sub_vec_id()
    experts = tl.arange(0, E_BLOCK)
    expert_mask = experts < E
    if global_vec_id % 2 == 0:
        for source_rank in range(pid, R, ncore):
            source = publish_recv_offsets_ptr + source_rank * E
            destination = recv_offsets_ptr + rank * E
            if source_rank == rank:
                values = tl.load(source + experts, mask=expert_mask, other=0)
                tl.store(destination + experts, values, mask=expert_mask)
            else:
                libshmem_device.putmem(destination, source, E * 8, source_rank)
    libshmem_device.barrier_all_vec()


@triton.jit(do_not_specialize=["rank"])
def kernel_udma_dispatch_direct(
    dispatched_input_ptr,
    output_ptr,
    splits_ptr,
    send_offsets_ptr,
    recv_offsets_ptr,
    rank,
    RANK_SIZE: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    UDMA_CHUNK_ROWS: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_BLOCK_D: tl.constexpr,
):
    """Send expert buckets directly into the final expert-major output.

    One physical core owns one remote PE.  It publishes that PE's disjoint
    expert buckets with non-blocking UDMA puts and performs one engine-specific
    completion operation after the batch.  This respects the UDMA restriction
    against concurrent operations from multiple local cores to the same PE
    while removing the source-major staging materialization and reorder kernel.
    """
    actual_sub_vec_num = sub_vec_num()
    ncore = tl.num_programs(0) * actual_sub_vec_num // 2
    pid = tl.program_id(0) * actual_sub_vec_num // 2
    global_vec_id = pid * actual_sub_vec_num + sub_vec_id()

    if global_vec_id % 2 == 0:
        for target_rank in range(pid, RANK_SIZE, ncore):
            if target_rank != rank:
                first_send_offset = tl.load(send_offsets_ptr + target_rank * E)
                first_recv_offset = tl.load(recv_offsets_ptr + target_rank * E)
                has_data = 0
                for expert in range(E):
                    metadata_index = target_rank * E + expert
                    count = tl.load(splits_ptr + metadata_index)
                    send_offset = tl.load(send_offsets_ptr + metadata_index)
                    recv_offset = tl.load(recv_offsets_ptr + metadata_index)
                    if count > 0:
                        first_send_offset = tl.where(
                            has_data == 0, send_offset, first_send_offset
                        )
                        first_recv_offset = tl.where(
                            has_data == 0, recv_offset, first_recv_offset
                        )
                        has_data = 1
                        for chunk_begin in range(0, count, UDMA_CHUNK_ROWS):
                            chunk_rows = tl.minimum(
                                count - chunk_begin, UDMA_CHUNK_ROWS
                            )
                            libshmem_device.putmem_nbi(
                                output_ptr + (recv_offset + chunk_begin) * D,
                                dispatched_input_ptr + (send_offset + chunk_begin) * D,
                                chunk_rows * D * 2,
                                target_rank,
                            )
                if has_data > 0:
                    libshmem_device.putmem(
                        output_ptr + first_recv_offset * D,
                        dispatched_input_ptr + first_send_offset * D,
                        2,
                        target_rank,
                    )
    # Keep remote-PE producer cores off the local copy critical path.
    # Sparse idle-core ownership requires all peer IDs to fit.
    # Otherwise all cores copy disjoint local stripes after sending.
    if RANK_SIZE <= ncore:
        self_worker_count = ncore - RANK_SIZE + 1
        self_worker_id = tl.where(pid == rank, 0, pid - RANK_SIZE + 1)
        is_self_worker = (pid == rank) | (pid >= RANK_SIZE)
    else:
        self_worker_count = ncore
        self_worker_id = pid
        is_self_worker = True
    d_tiles = tl.cdiv(D, COMM_BLOCK_D)
    for expert in range(E):
        metadata_index = rank * E + expert
        count = tl.load(splits_ptr + metadata_index)
        send_offset = tl.load(send_offsets_ptr + metadata_index)
        recv_offset = tl.load(recv_offsets_ptr + metadata_index)
        row_tiles = tl.cdiv(count, COMM_BLOCK_N)
        total_tasks = row_tiles * d_tiles
        if (global_vec_id % 2 == 0) & is_self_worker:
            for task in range(self_worker_id, total_tasks, self_worker_count):
                row_tile = task // d_tiles
                d_tile = task % d_tiles
                ns = row_tile * COMM_BLOCK_N + tl.arange(0, COMM_BLOCK_N)
                ds = d_tile * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (ns[:, None] < count) & (ds[None, :] < D)
                src = (send_offset + ns)[:, None] * D + ds[None, :]
                dst = (recv_offset + ns)[:, None] * D + ds[None, :]
                data = tl.load(dispatched_input_ptr + src, mask=mask, other=0.0)
                tl.store(output_ptr + dst, data, mask=mask)


@triton.jit(do_not_specialize=["rank"])
def kernel_udma_dispatch_direct_wave_inverse(
    dispatched_input_ptr,
    output_ptr,
    splits_ptr,
    send_offsets_ptr,
    recv_offsets_ptr,
    rank,
    RANK_SIZE: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    FIRST_NUM: tl.constexpr,
    FIRST_DEN: tl.constexpr,
    WAVE: tl.constexpr,
    UDMA_CHUNK_ROWS: tl.constexpr,
):
    """Send one count-derived half-wave after its pack completes."""
    actual_sub_vec_num = sub_vec_num()
    ncore = tl.num_programs(0) * actual_sub_vec_num // 2
    pid = tl.program_id(0) * actual_sub_vec_num // 2
    global_vec_id = pid * actual_sub_vec_num + sub_vec_id()
    if global_vec_id % 2 == 0:
        for target_rank in range(pid, RANK_SIZE, ncore):
            if target_rank != rank:
                first_send_offset = tl.load(send_offsets_ptr + target_rank * E)
                first_recv_offset = tl.load(recv_offsets_ptr + target_rank * E)
                has_data = 0
                for expert in range(E):
                    metadata_index = target_rank * E + expert
                    full_count = tl.load(splits_ptr + metadata_index)
                    first_count = full_count * FIRST_NUM // FIRST_DEN
                    if WAVE == 0:
                        count = first_count
                        wave_offset = 0
                    else:
                        count = full_count - first_count
                        wave_offset = first_count
                    send_offset = (
                        tl.load(send_offsets_ptr + metadata_index) + wave_offset
                    )
                    recv_offset = (
                        tl.load(recv_offsets_ptr + metadata_index) + wave_offset
                    )
                    if count > 0:
                        first_send_offset = tl.where(
                            has_data == 0, send_offset, first_send_offset
                        )
                        first_recv_offset = tl.where(
                            has_data == 0, recv_offset, first_recv_offset
                        )
                        has_data = 1
                        for chunk_begin in range(0, count, UDMA_CHUNK_ROWS):
                            chunk_rows = tl.minimum(
                                count - chunk_begin, UDMA_CHUNK_ROWS
                            )
                            libshmem_device.putmem_nbi(
                                output_ptr + (recv_offset + chunk_begin) * D,
                                dispatched_input_ptr + (send_offset + chunk_begin) * D,
                                chunk_rows * D * 2,
                                target_rank,
                            )
                if has_data > 0:
                    libshmem_device.putmem(
                        output_ptr + first_recv_offset * D,
                        dispatched_input_ptr + first_send_offset * D,
                        2,
                        target_rank,
                    )


@triton.jit(do_not_specialize=["rank"])
def kernel_dispatch_self_copy(
    dispatched_input_ptr,
    output_ptr,
    splits_ptr,
    send_offsets_ptr,
    recv_offsets_ptr,
    rank,
    RANK_SIZE: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_BLOCK_D: tl.constexpr,
):
    actual_sub_vec_num = sub_vec_num()
    ncore = tl.num_programs(0) * actual_sub_vec_num // 2
    pid = tl.program_id(0) * actual_sub_vec_num // 2
    global_vec_id = pid * actual_sub_vec_num + sub_vec_id()
    # Sparse idle-core ownership requires all peer IDs to fit.
    # Otherwise all cores copy disjoint local stripes after sending.
    if RANK_SIZE <= ncore:
        self_worker_count = ncore - RANK_SIZE + 1
        self_worker_id = tl.where(pid == rank, 0, pid - RANK_SIZE + 1)
        is_self_worker = (pid == rank) | (pid >= RANK_SIZE)
    else:
        self_worker_count = ncore
        self_worker_id = pid
        is_self_worker = True
    d_tiles = tl.cdiv(D, COMM_BLOCK_D)
    for expert in range(E):
        metadata_index = rank * E + expert
        count = tl.load(splits_ptr + metadata_index)
        send_offset = tl.load(send_offsets_ptr + metadata_index)
        recv_offset = tl.load(recv_offsets_ptr + metadata_index)
        total_tasks = tl.cdiv(count, COMM_BLOCK_N) * d_tiles
        if (global_vec_id % 2 == 0) & is_self_worker:
            for task in range(self_worker_id, total_tasks, self_worker_count):
                row_tile = task // d_tiles
                d_tile = task % d_tiles
                ns = row_tile * COMM_BLOCK_N + tl.arange(0, COMM_BLOCK_N)
                ds = d_tile * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (ns[:, None] < count) & (ds[None, :] < D)
                src = (send_offset + ns)[:, None] * D + ds[None, :]
                dst = (recv_offset + ns)[:, None] * D + ds[None, :]
                data = tl.load(dispatched_input_ptr + src, mask=mask, other=0.0)
                tl.store(output_ptr + dst, data, mask=mask)


@triton.jit
def kernel_udma_dispatch_sync():
    libshmem_device.barrier_all_vec()


def _use_no_drop_contiguous_pack(
    mlp_input, scatter_index, dispatched_input, start_idx, end_idx
):
    return (
        start_idx == 0
        and end_idx == mlp_input.shape[0] * scatter_index.shape[1]
        and mlp_input.is_contiguous()
        and scatter_index.is_contiguous()
        and dispatched_input.is_contiguous()
    )


def launch_udma_dispatch(
    mlp_input,
    scatter_index,
    dispatched_input,
    splits,
    send_offsets,
    recv_offsets,
    output_base,
    rank,
    rank_size,
    num_local_experts,
    comm_block_n,
    comm_block_d,
    start_idx,
    end_idx,
):
    """Launch device pack and direct UDMA exchange into final output order."""
    D_dim = dispatched_input.shape[1]
    R = rank_size
    E = num_local_experts
    # shmem_device_udma documents a 256 MB maximum per put request.  Use the
    # largest whole-row chunk that stays within that limit for this D.
    udma_chunk_rows = max(1, (256 * 1024 * 1024) // (D_dim * 2))

    # Reuse the driver's utility instance to avoid repeated TA 3.6 cache lookups.
    aicore_num = driver.active.utils.get_aicore_num()
    pack_block_n = PACK_BLOCK_N
    # Larger D tiles reduce repeated scatter-index/address work.  The default
    # auto policy keeps common hidden sizes in one tile where practical and
    # caps BLOCK_D at 2048; fixed power-of-two overrides remain available for
    # targeted measurements.
    pack_block_d = _resolve_pack_block_d(PACK_BLOCK_D_CONFIG, D_dim)
    token_num = mlp_input.shape[0]
    topk = scatter_index.shape[1]
    d_blocks = triton.cdiv(D_dim, pack_block_d)
    use_no_drop_contiguous = _use_no_drop_contiguous_pack(
        mlp_input,
        scatter_index,
        dispatched_input,
        start_idx,
        end_idx,
    )
    if use_no_drop_contiguous:
        # Do not launch a partially valid BLOCK_N program.  Its invalid lanes
        # load scatter_index with ``other=0`` and therefore compute row 0 as
        # their destination.  Although the store is masked in Triton, an Ascend950
        # tail case (N=4097, BLOCK_N=16) intermittently corrupted exactly the
        # complete output row 0.  Keep the aligned fast body and issue the
        # remaining tokens through BLOCK_N=1 so every scatter lane is valid.
        aligned_token_num = token_num - token_num % pack_block_n
        tail_token_num = token_num - aligned_token_num
        if aligned_token_num:
            aligned_grid = (aligned_token_num // pack_block_n * d_blocks, 1, 1)
            kernel_dispatch_pack[aligned_grid](
                mlp_input,
                scatter_index,
                dispatched_input,
                N=token_num,
                D=D_dim,
                TOPK=topk,
                TOKEN_OFFSET=0,
                BLOCK_N=pack_block_n,
                BLOCK_D=pack_block_d,
            )
        if tail_token_num:
            tail_grid = (tail_token_num * d_blocks, 1, 1)
            kernel_dispatch_pack[tail_grid](
                mlp_input,
                scatter_index,
                dispatched_input,
                N=token_num,
                D=D_dim,
                TOPK=topk,
                TOKEN_OFFSET=aligned_token_num,
                BLOCK_N=1,
                BLOCK_D=pack_block_d,
            )
    else:
        pack_grid = (triton.cdiv(token_num, pack_block_n) * d_blocks, 1, 1)
        kernel_dispatch_pack_dynamic[pack_grid](
            mlp_input,
            scatter_index,
            dispatched_input,
            token_num,
            D_dim,
            topk,
            start_idx,
            end_idx,
            mlp_input.stride(0),
            mlp_input.stride(1),
            scatter_index.stride(0),
            scatter_index.stride(1),
            dispatched_input.stride(0),
            dispatched_input.stride(1),
            BLOCK_N=pack_block_n,
            BLOCK_D=pack_block_d,
        )
    kernel_udma_dispatch_direct[aicore_num, 1, 1](
        dispatched_input,
        # Remote RMA address translation must start from the full symmetric
        # allocation on every rank. ``out`` may be a zero-length local view on
        # send-only ranks under hot routing and is not a valid remote base.
        output_base,
        splits,
        send_offsets,
        recv_offsets,
        rank,
        RANK_SIZE=R,
        E=E,
        D=D_dim,
        UDMA_CHUNK_ROWS=udma_chunk_rows,
        COMM_BLOCK_N=comm_block_n,
        COMM_BLOCK_D=comm_block_d,
    )
    kernel_udma_dispatch_sync[aicore_num, 1, 1]()


def launch_wave_dispatch(
    mlp_input,
    scatter_index,
    packed,
    splits,
    send_offsets,
    recv_offsets,
    output_base,
    rank,
    rank_size,
    experts,
    comm_block_n,
    comm_block_d,
):
    """Pipeline adaptive destination-major half packs with remote UDMA sends."""
    if INVERSE_WAVE_CONTEXT is None:
        raise RuntimeError("wave strategy needs initialized reusable workspace")
    ctx = INVERSE_WAVE_CONTEXT
    main_stream = torch.npu.current_stream()
    pack_stream = ctx["pack_stream"]
    comm_stream = ctx["comm_stream"]
    pack_block_d = _resolve_pack_block_d(PACK_BLOCK_D_CONFIG, mlp_input.shape[1])
    pack_workers = 8
    heavy_workers = 128
    helper_groups = (heavy_workers - pack_workers) // pack_workers
    pack_grid = (
        rank_size * experts + helper_groups,
        pack_workers,
        triton.cdiv(mlp_input.shape[1], pack_block_d),
    )
    row_bytes = mlp_input.shape[1] * mlp_input.element_size()
    tier8_max_rows = max(1, (16 * 1024 * 1024) // row_bytes)
    tier32_max_rows = max(tier8_max_rows + 1, (64 * 1024 * 1024) // row_bytes)
    tier64_max_rows = max(tier32_max_rows + 1, (256 * 1024 * 1024) // row_bytes)
    udma_chunk_rows = max(1, (256 * 1024 * 1024) // (mlp_input.shape[1] * 2))

    ctx["metadata_ready"].record(main_stream)
    with torch.npu.stream(pack_stream):
        pack_stream.wait_event(ctx["metadata_ready"])
        for wave in range(2):
            kernel_dispatch_pack_wave_inverse_adaptive[pack_grid](
                mlp_input,
                ctx["inverse"],
                splits,
                send_offsets,
                ctx["heavy_bucket"],
                packed,
                D=mlp_input.shape[1],
                TOPK=scatter_index.shape[1],
                NUM_BUCKETS=rank_size * experts,
                FIRST_NUM=WAVE_FIRST_NUM,
                FIRST_DEN=WAVE_FIRST_DEN,
                WAVE=wave,
                BASE_WORKERS=pack_workers,
                HEAVY_WORKERS=heavy_workers,
                TIER8_MAX_ROWS=tier8_max_rows,
                TIER32_MAX_ROWS=tier32_max_rows,
                TIER64_MAX_ROWS=tier64_max_rows,
                BLOCK_N=PACK_BLOCK_N,
                BLOCK_D=pack_block_d,
            )
            ctx["pack_ready"][wave].record(pack_stream)

    with torch.npu.stream(comm_stream):
        comm_stream.wait_event(ctx["metadata_ready"])
        for wave in range(2):
            comm_stream.wait_event(ctx["pack_ready"][wave])
            kernel_udma_dispatch_direct_wave_inverse[(rank_size,)](
                packed,
                output_base,
                splits,
                send_offsets,
                recv_offsets,
                rank,
                RANK_SIZE=rank_size,
                E=experts,
                D=mlp_input.shape[1],
                FIRST_NUM=WAVE_FIRST_NUM,
                FIRST_DEN=WAVE_FIRST_DEN,
                WAVE=wave,
                UDMA_CHUNK_ROWS=udma_chunk_rows,
            )
        ctx["comm_done"].record(comm_stream)

    main_stream.wait_event(ctx["pack_ready"][1])
    kernel_dispatch_self_copy[(driver.active.utils.get_aicore_num(),)](
        packed,
        output_base,
        splits,
        send_offsets,
        recv_offsets,
        rank,
        RANK_SIZE=rank_size,
        E=experts,
        D=mlp_input.shape[1],
        COMM_BLOCK_N=comm_block_n,
        COMM_BLOCK_D=comm_block_d,
    )
    main_stream.wait_event(ctx["comm_done"])
    kernel_udma_dispatch_sync[(driver.active.utils.get_aicore_num(),)]()


def generate_random_exp_indices(token_num, total_num_experts, topk, drop_ratio=0.0):
    """Generate random expert routing indices, mirroring NVIDIA test_ep_ll_a2a.py.

    Returns (token_num, topk) int64 tensor on CPU.
    """
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        for i, _ in enumerate(top_selected):
            if random.uniform(0, 1) < drop_ratio:
                top_selected[i] = total_num_experts  # dropped token sentinel
        exp_indices.append(top_selected)
    return torch.tensor(exp_indices, dtype=torch.int64)


def generate_hot_exp_indices(token_num, total_num_experts, topk, target_expert=0):
    """Route every choice to one global expert to stress a single hot bucket."""
    target_expert = int(target_expert) % total_num_experts
    return torch.full((token_num, topk), target_expert, dtype=torch.int64)


def generate_skewed_exp_indices(token_num, total_num_experts, topk, target_expert=0):
    """Route one choice per token to a hotspot and keep top-k experts unique."""
    if topk > total_num_experts:
        raise ValueError(
            f"topk={topk} cannot be unique with total_num_experts={total_num_experts}"
        )
    target_expert = int(target_expert) % total_num_experts
    other_experts = [
        expert for expert in range(total_num_experts) if expert != target_expert
    ]
    routing = []
    for _ in range(token_num):
        selected = [target_expert]
        selected.extend(random.sample(other_experts, topk - 1))
        random.shuffle(selected)
        routing.append(selected)
    return torch.tensor(routing, dtype=torch.int64)


def torch_golden_dispatch(
    start_idx,
    end_idx,
    splits_cpu,
    a2a_splits_cpu,
    mlp_input,
    scatter_index,
    ep_size,
    ep_group,
    num_local_experts,
):
    """Torch/HCCL golden synchronized with ``tutorial/torch_golden.py``."""
    repeated_input = (
        mlp_input.unsqueeze(1)
        .repeat(1, scatter_index.shape[1], 1)
        .view(-1, mlp_input.shape[-1])
    )
    dispatched_input = torch.empty_like(repeated_input)
    dispatched_input[scatter_index.flatten()] = repeated_input
    dispatched_input = dispatched_input[start_idx:end_idx]
    a2a_expert_input = torch.empty(
        (int(a2a_splits_cpu.sum()), dispatched_input.shape[-1]),
        device=dispatched_input.device,
        dtype=dispatched_input.dtype,
    )
    handle = dist.all_to_all_single(
        a2a_expert_input,
        dispatched_input,
        output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
        input_split_sizes=splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
        group=ep_group,
        async_op=True,
    )
    handle.wait()
    inputs_list = torch.split(a2a_expert_input, a2a_splits_cpu.tolist())
    outputs_list = []
    for local_expert in range(num_local_experts):
        for src_rank in range(ep_size):
            outputs_list.append(
                inputs_list[src_rank * num_local_experts + local_expert]
            )
    return torch.cat(outputs_list, dim=0)


def udma_dispatch_recipe(
    start_idx,
    end_idx,
    mlp_input,
    scatter_index,
    dispatched_input,
    splits,
    a2a_splits,
    send_offsets,
    recv_offsets,
    publish_recv_offsets,
    output_base,
    rank,
    rank_size,
    num_local_experts,
    comm_block_n,
    comm_block_d,
    strategy,
):
    """Full candidate contract matching ``torch_golden_dispatch`` inputs."""
    if strategy == "wave":
        if INVERSE_WAVE_CONTEXT is None:
            raise RuntimeError("wave strategy needs initialized reusable workspace")
        main_stream = torch.npu.current_stream()
        pack_stream = INVERSE_WAVE_CONTEXT["pack_stream"]
        INVERSE_WAVE_CONTEXT["start"].record(main_stream)
        pack_stream.wait_event(INVERSE_WAVE_CONTEXT["start"])
        with torch.npu.stream(pack_stream):
            block = 256
            kernel_dispatch_inverse_wave_plan[
                (triton.cdiv(scatter_index.numel(), block),)
            ](
                scatter_index,
                INVERSE_WAVE_CONTEXT["inverse"],
                TOTAL_BRANCHES=scatter_index.numel(),
                BLOCK_SIZE=block,
            )

    num_buckets = rank_size * num_local_experts
    heavy_bucket = (
        INVERSE_WAVE_CONTEXT["heavy_bucket"] if strategy == "wave" else send_offsets
    )
    kernel_prepare_local_dispatch_metadata[(2,)](
        a2a_splits,
        splits,
        send_offsets,
        publish_recv_offsets,
        heavy_bucket,
        R=rank_size,
        E=num_local_experts,
        META_SIZE=triton.next_power_of_2(num_buckets),
        SELECT_HEAVY_BUCKET=strategy == "wave",
    )
    kernel_exchange_dispatch_metadata[driver.active.utils.get_aicore_num(), 1, 1](
        publish_recv_offsets,
        recv_offsets,
        rank,
        R=rank_size,
        E=num_local_experts,
        E_BLOCK=triton.next_power_of_2(num_local_experts),
    )
    if strategy == "wave":
        launch_wave_dispatch(
            mlp_input,
            scatter_index,
            dispatched_input,
            splits,
            send_offsets,
            recv_offsets,
            output_base,
            rank,
            rank_size,
            num_local_experts,
            comm_block_n,
            comm_block_d,
        )
        return
    launch_udma_dispatch(
        mlp_input,
        scatter_index,
        dispatched_input,
        splits,
        send_offsets,
        recv_offsets,
        output_base,
        rank,
        rank_size,
        num_local_experts,
        comm_block_n,
        comm_block_d,
        start_idx,
        end_idx,
    )


def _init_aclshmem(rank, world_size):
    """Initialize ACL symmetric shared memory environment."""
    init_aclshmem(rank, world_size, ash_pool_bytes(world_size), ash.OpEngineType.UDMA)


def _make_data(
    token_num,
    hidden,
    num_experts,
    topk,
    rank,
    route_mode="random",
    dtype=torch.bfloat16,
):
    """Generate a single test case.

    Mirrors NVIDIA test_ep_ll_a2a._make_data.

    Args:
        token_num: number of tokens on this rank (N).
        hidden: hidden dimension (D).
        num_experts: total global experts (= ep_size * num_local_experts).
        topk: number of experts each token is routed to.
        rank: current rank (for seed-offset reproducible data).
        route_mode: "random" for normal random routing, "hot" for one hot bucket.
        dtype: tensor data type.

    Returns:
        mlp_input: (N, D) on NPU.
        routing: (N, topk) int64 on CPU.
        topk_weights: (N, topk) float32 on CPU (softmax-normalized).
    """
    # Deterministic per-rank seed offset
    gen_in = torch.Generator(device="cpu").manual_seed(1000 + rank)
    mlp_input = torch.randn(
        token_num, hidden, dtype=torch.float32, generator=gen_in
    ).to(dtype)

    if route_mode == "random":
        routing = generate_random_exp_indices(token_num, num_experts, topk)
    elif route_mode == "hot":
        routing = generate_hot_exp_indices(
            token_num, num_experts, topk, target_expert=0
        )
    elif route_mode == "skewed":
        routing = generate_skewed_exp_indices(
            token_num, num_experts, topk, target_expert=0
        )
    else:
        raise ValueError(f"Unsupported route_mode: {route_mode}")

    weight = torch.randn(token_num, topk, dtype=torch.float32)
    weight = torch.nn.functional.softmax(weight, dim=1)

    return mlp_input, routing, weight


def _verify_across_ranks(golden, candidate, pe, world_size, rtol=1e-3, atol=1e-3):
    """Check numerical accuracy and all-gather pass/fail across ranks.

    Returns True if all ranks passed.
    """
    passed = torch.tensor([1], dtype=torch.int32).npu()
    error_msg = ""
    try:
        torch.testing.assert_close(golden, candidate, rtol=rtol, atol=atol)
    except AssertionError as e:
        passed[0] = 0
        error_msg = str(e)

    all_passed = [torch.zeros(1, dtype=torch.int32).npu() for _ in range(world_size)]
    dist.all_gather(all_passed, passed)

    # Rank-sequential print to avoid interleaved output
    for rank_id in range(world_size):
        if pe == rank_id:
            if all_passed[rank_id].item() == 1:
                print(f"  {GREEN}[PASS]{RESET} Rank {pe}: matches golden.", flush=True)
            else:
                print(
                    f"  {RED}[FAIL]{RESET} Rank {pe}: mismatch.\n"
                    f"    Details: {error_msg[:500]}",
                    flush=True,
                )
        dist.barrier()

    return all(r.item() == 1 for r in all_passed)


def run_test_distributed(case_executor=None, *, case_selector=None, verify=True):
    """Run the correctness sweep, optionally delegating A/B execution.

    ``case_executor`` is the narrow interface used by ``python/triton_dist/test/ascend/``.  It receives
    the golden and candidate recipes plus case metadata, and returns the final
    golden output.  Benchmark policy and timing therefore stay out of this
    formal operator file.

    ``case_selector`` optionally filters the complete case list; ``verify``
    controls output comparison. Standalone calls retain full verification.

    Architecture mirrors NVIDIA test_ep_ll_a2a:
      _make_data -> torch_moe_a2a_dispatch (golden) -> moe_a2a_dispatch_inline_offsets_consumer_local_bypass (candidate) -> verify.
    """
    global INVERSE_WAVE_CONTEXT

    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    ep_size = world_size
    ep_group = None  # default group == world

    _init_aclshmem(pe, world_size)

    # Build the Cartesian sweep of (route_mode, N, topk, E, hidden_size)
    test_cases = []
    for route_mode in ROUTE_MODES:
        for n_val in N_LIST:
            for topk in TOPK_LIST:
                for e_val in NUM_LOCAL_EXPERTS_LIST:
                    for hidden_size in HIDDEN_SIZE_LIST:
                        test_cases.append((route_mode, n_val, topk, e_val, hidden_size))
    if case_selector is not None:
        test_cases = case_selector(test_cases)
    total_cases = len(test_cases)
    if pe == 0:
        print(
            f"{BOLD}[START]{RESET} MoE A2A dispatch sweep ({total_cases} cases) on "
            f"world_size={world_size}, hidden_sizes={HIDDEN_SIZE_LIST}, dtype=bfloat16, route_modes={ROUTE_MODES}, "
            f"mode={'perf' if case_executor else 'correctness'}, "
            "transport=UDMA, candidate=production_auto_direct",
            flush=True,
        )

    COMM_BLOCK_N = int(os.environ.get("MOE_COMM_BLOCK_N", "64"))
    COMM_BLOCK_D = int(os.environ.get("MOE_COMM_BLOCK_D", "256"))
    for case_idx, (route_mode, N, topk, E, hidden_size) in enumerate(test_cases):
        dist.barrier()
        num_experts = ep_size * E
        pack_block_d = _resolve_pack_block_d(PACK_BLOCK_D_CONFIG, hidden_size)

        if pe == 0:
            print(
                f"\n{BOLD}Case {case_idx + 1}/{total_cases}:{RESET} N={N}, topk={topk}, "
                f"E={E}, D={hidden_size}, ep={ep_size}, route={route_mode}, "
                f"transport=UDMA",
                flush=True,
            )

        dtype = torch.bfloat16

        # ---- 1. Generate deterministic data (identical across A/B files) ----
        # Keep the routing stream fixed across adjacent N values so an N-boundary
        # sweep changes shape rather than changing shape and routing together.
        # Each rank still has a distinct, deterministic route stream.
        case_seed = ROUTE_SEED + pe
        random.seed(case_seed)
        torch.manual_seed(case_seed)
        mlp_input, routing, _ = _make_data(
            N, hidden_size, num_experts, topk, pe, route_mode, dtype
        )
        mlp_input = mlp_input.npu()

        # ---- 2. Pre-compute host-side splits and scatter_index ----
        splits_cpu = torch.bincount(routing.flatten(), minlength=num_experts).to(
            torch.int32
        )
        splits = splits_cpu.npu()

        # scatter_index is the inverse stable-sort permutation required by the
        # benchmark: original flattened branch -> rank-major permuted position.
        _, index_choosed = routing.flatten().sort(stable=True)
        scatter_index_cpu = torch.empty(N * topk, dtype=torch.int64)
        scatter_index_cpu[index_choosed] = torch.arange(N * topk, dtype=torch.int64)
        scatter_index = scatter_index_cpu.reshape(N, topk).npu()
        start_idx = 0
        end_idx = int(splits_cpu.sum().item())

        # ---- 3. Gather all send counts and derive receive-count metadata ----
        # splits_all[src, dest, expert] contains the number of tokens that src
        # sends to one local expert on dest. Transpose it to the layout consumed
        # by the golden and Triton launcher: [dest, src, expert].
        rows = [torch.empty_like(splits) for _ in range(ep_size)]
        dist.all_gather(rows, splits, group=ep_group)
        splits_all = torch.stack(rows).reshape(ep_size, ep_size, E)
        a2a_splits_all = (
            splits_all.permute(1, 0, 2).contiguous().reshape(ep_size, num_experts)
        )
        a2a_splits = a2a_splits_all[pe].contiguous()

        # Only capacity workspaces are prepared outside timing.  Every offset
        # is rebuilt from the public local splits inside the candidate recipe.
        send_offsets = torch.empty(num_experts, dtype=torch.int64, device=splits.device)
        dist.barrier()
        recv_offsets = ash.aclshmem_create_tensor(
            [num_experts], dtype=torch.int64, device_id=local_pe
        )
        publish_recv_offsets = torch.empty_like(send_offsets)
        strategy = select_dispatch_strategy(
            N,
            hidden_size,
            topk,
            mlp_input.element_size(),
            ep_size,
            full_range=(
                start_idx == 0
                and end_idx == N * topk
                and mlp_input.is_contiguous()
                and scatter_index.is_contiguous()
            ),
        )

        peer_capacity = max(
            1,
            int(a2a_splits_all.reshape(ep_size, ep_size, E).sum(-1).max().item()),
        )
        a2a_splits_cpu = a2a_splits.cpu()
        if pe == 0:
            timing_boundary = (
                "timing_boundary=common_inputs_full_recipe, " if case_executor else ""
            )
            print(
                f"  symmetric_output_rows={ep_size * peer_capacity}, "
                f"pack_block_n={PACK_BLOCK_N}, "
                f"pack_block_d_config={PACK_BLOCK_D_CONFIG}, "
                f"pack_block_d_effective={pack_block_d}, "
                "pack_mode=no_drop_auto, exchange_mode=direct, "
                f"strategy={strategy}, "
                f"{timing_boundary}"
                f"selector_remote_mib="
                f"{N * topk * hidden_size * 2 * (ep_size - 1) / ep_size / 2**20:.1f}, "
                f"selector_threshold_mib={WAVE_REMOTE_BYTES_THRESHOLD / 2**20:.0f}, "
                f"comm_block_n={COMM_BLOCK_N}, comm_block_d={COMM_BLOCK_D}",
                flush=True,
            )

        # Sanity: send total must equal N*topk (no token dropping)
        assert (
            int(splits_cpu.sum().item()) == N * topk
        ), f"Rank {pe}: splits sum {splits_cpu.sum()} != N*topk {N*topk}"

        def run_golden():
            return torch_golden_dispatch(
                start_idx,
                end_idx,
                splits_cpu,
                a2a_splits_cpu,
                mlp_input,
                scatter_index,
                ep_size,
                ep_group,
                E,
            )

        # ---- 4. Candidate storage and recipe ----
        total_recv = int(a2a_splits.sum().item())
        dist.barrier()
        output_base = ash.aclshmem_create_tensor(
            [ep_size * peer_capacity * hidden_size],
            dtype=dtype,
            device_id=local_pe,
        )
        out = output_base[: total_recv * hidden_size].view(total_recv, hidden_size)
        dispatched_input = torch.empty(
            (end_idx - start_idx, hidden_size),
            dtype=dtype,
            device=mlp_input.device,
        )
        if pe == 0:
            if _use_no_drop_contiguous_pack(
                mlp_input,
                scatter_index,
                dispatched_input,
                start_idx,
                end_idx,
            ):
                if strategy == "wave":
                    actual_pack_variant = "inverse_destination_adaptive_half_wave"
                else:
                    actual_pack_variant = (
                        "no_drop_contiguous_split_tail"
                        if mlp_input.shape[0] % PACK_BLOCK_N
                        else "no_drop_contiguous"
                    )
            else:
                actual_pack_variant = "dynamic"
            print(f"  pack_variant={actual_pack_variant}", flush=True)
        output_base.fill_(float("nan"))
        dispatched_input.fill_(float("nan"))
        torch.npu.synchronize()
        if strategy == "wave":
            INVERSE_WAVE_CONTEXT = prepare_inverse_wave_context(
                scatter_index.numel(), mlp_input.device
            )

        def run_candidate():
            udma_dispatch_recipe(
                start_idx,
                end_idx,
                mlp_input,
                scatter_index,
                dispatched_input,
                splits,
                a2a_splits,
                send_offsets,
                recv_offsets,
                publish_recv_offsets,
                output_base,
                pe,
                ep_size,
                E,
                COMM_BLOCK_N,
                COMM_BLOCK_D,
                strategy,
            )

        # ---- 5. Execute once for correctness, or delegate timing to python/triton_dist/test/ascend/ ----
        case_info = {
            "route": route_mode,
            "n": N,
            "d": hidden_size,
            "topk": topk,
            "local_experts": E,
            "ranks": world_size,
            "strategy": strategy,
        }
        if case_executor is None:
            dist.barrier()
            torch.npu.synchronize()
            golden = run_golden()
            torch.npu.synchronize()
            dist.barrier()
            torch.npu.synchronize()
            run_candidate()
            torch.npu.synchronize()
        else:
            golden = case_executor(run_golden, run_candidate, case_info)
        assert golden is not None

        dist.barrier()

        # ---- 6. Verify ----
        all_passed_flag = (
            _verify_across_ranks(golden, out, pe, world_size) if verify else True
        )

        del out
        ash.aclshmem_free_tensor(output_base)
        ash.aclshmem_free_tensor(recv_offsets)

        if not all_passed_flag:
            _ = ash.aclshmem_finialize()
            raise AssertionError(
                f"Sweep failed at case: route={route_mode}, N={N}, D={hidden_size}, topk={topk}, E={E}"
            )

    _ = ash.aclshmem_finialize()
    if pe == 0:
        print(
            f"\n{GREEN}[SUCCESS]{RESET} All {total_cases} MoE A2A dispatch cases "
            f"{'verified' if verify else 'completed'}."
        )


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)
    world_size = dist.get_world_size()

    print(f"[INFO] Rank {local_pe} of {world_size} initialized")
    dist.barrier()
    run_test_distributed()
