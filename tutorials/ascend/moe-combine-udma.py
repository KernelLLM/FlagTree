# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ascend UDMA MoE Combine implementation and correctness check.

Expert-major buckets are sent directly to their final rank-major offsets in a
symmetric returned buffer.  This removes the intermediate peer-major pack and
retains the existing scatter/sum kernel, so exchange-layout optimization stays
separate from the mathematical Combine contract. Performance timing is
provided by ``python/triton_dist/test/ascend/moe-combine-udma.py``.
"""

import os
import random

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.runtime import driver
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id, sub_vec_num
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


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
E_LIST = [int(x) for x in os.environ.get("MOE_NLE_LIST", "40,160").split(",") if x]
ROUTE_MODES = [x for x in os.environ.get("MOE_ROUTE_MODES", "random").split(",") if x]
HIDDEN_SIZE_LIST = [int(x) for x in os.environ.get("MOE_D", "1536").split(",") if x]
SEED = int(os.environ.get("MOE_ROUTE_SEED", "2026"))


def symmetric_memory_size() -> int:
    """Return the aligned ACLSHMEM heap capacity the sweep needs.

    The only symmetric allocation is the returned buffer: up to N * topk
    rows (every routed copy of this rank's tokens) of `hidden` bf16
    elements each.
    """
    rows = max(1, max(N_LIST) * max(TOPK_LIST))
    required = rows * max(HIDDEN_SIZE_LIST) * 2
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def _ash_pool_size() -> int:
    """Symmetric heap: MOE_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("MOE_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size()
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the MOE_ASH_SIZE_GB override"
        )
    return pool_bytes


g_ash_size = _ash_pool_size()

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


@triton.jit(do_not_specialize=["rank"])
def kernel_prepare_combine_metadata(
    all_splits_ptr,
    recv_splits_ptr,
    bucket_counts_ptr,
    send_offsets_ptr,
    recv_offsets_ptr,
    rank,
    R: tl.constexpr,
    E: tl.constexpr,
    META_SIZE: tl.constexpr,
):
    """Derive direct-UDMA bucket metadata from the public split inputs."""
    bucket = tl.program_id(0)
    target_source = bucket // E
    expert = bucket % E
    meta = tl.arange(0, META_SIZE)
    valid = meta < R * E

    # ``send`` is expert-major [expert][target source], while recv_splits is
    # rank-major [target source][expert].
    source_index = meta % R
    expert_index = meta // R
    send_values = tl.load(
        recv_splits_ptr + source_index * E + expert_index,
        mask=valid,
        other=0,
    )
    before_send = (expert_index < expert) | (
        (expert_index == expert) & (source_index < target_source)
    )
    send_offset = tl.sum(tl.where(valid & before_send, send_values, 0), axis=0)

    # The destination is rank-major [expert rank][expert].  all_splits is the
    # timed all-gather of every source rank's local split tensor.
    recv_values = tl.load(
        all_splits_ptr + target_source * R * E + meta,
        mask=valid,
        other=0,
    )
    recv_offset = tl.sum(
        tl.where(valid & (meta < rank * E + expert), recv_values, 0), axis=0
    )
    count = tl.load(recv_splits_ptr + target_source * E + expert)
    tl.store(bucket_counts_ptr + bucket, count)
    tl.store(send_offsets_ptr + bucket, send_offset)
    tl.store(recv_offsets_ptr + bucket, recv_offset)


@triton.jit(do_not_specialize=["rank"])
def kernel_udma_combine_direct(
    send_ptr,
    recv_ptr,
    bucket_counts_ptr,
    send_offsets_ptr,
    recv_offsets_ptr,
    rank,
    R: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    UDMA_CHUNK_ROWS: tl.constexpr,
    COMM_BLOCK_N: tl.constexpr,
    COMM_BLOCK_D: tl.constexpr,
):
    """Send expert-major buckets directly to final returned offsets."""
    # Widen row offsets before multiplying by D: large BF16 buffers can
    # exceed the signed int32 element-offset range.
    actual_sub_vec_num = sub_vec_num()
    ncore = tl.num_programs(0) * actual_sub_vec_num // 2
    pid = tl.program_id(0) * actual_sub_vec_num // 2
    global_vec_id = pid * actual_sub_vec_num + sub_vec_id()

    # One physical core owns each remote PE.  The source remains in
    # expert-major [expert][target source] order, while explicit offsets place
    # each bucket in the target's final [dispatch rank][expert] order.
    if global_vec_id % 2 == 0:
        for target_source in range(pid, R, ncore):
            if target_source != rank:
                first_index = target_source * E
                first_send_offset = tl.load(send_offsets_ptr + first_index).to(tl.int64)
                first_recv_offset = tl.load(recv_offsets_ptr + first_index).to(tl.int64)
                has_data = 0
                for expert in range(E):
                    metadata_index = target_source * E + expert
                    count = tl.load(bucket_counts_ptr + metadata_index)
                    send_offset = tl.load(send_offsets_ptr + metadata_index).to(
                        tl.int64
                    )
                    recv_offset = tl.load(recv_offsets_ptr + metadata_index).to(
                        tl.int64
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
                                recv_ptr + (recv_offset + chunk_begin) * D,
                                send_ptr + (send_offset + chunk_begin) * D,
                                chunk_rows * D * 2,
                                target_source,
                            )
                # A synchronous duplicate of the first BF16 element performs
                # the engine-specific completion for all preceding NBI puts to
                # this PE, matching the validated Dispatch direct protocol.
                if has_data > 0:
                    libshmem_device.putmem(
                        recv_ptr + first_recv_offset * D,
                        send_ptr + first_send_offset * D,
                        2,
                        target_source,
                    )

    # UDMA cannot self-send.  Copy this rank's expert buckets from their
    # expert-major source offsets into final returned offsets on active even
    # Vector subcores.
    d_tiles = tl.cdiv(D, COMM_BLOCK_D)
    for expert in range(E):
        metadata_index = rank * E + expert
        count = tl.load(bucket_counts_ptr + metadata_index)
        send_offset = tl.load(send_offsets_ptr + metadata_index).to(tl.int64)
        recv_offset = tl.load(recv_offsets_ptr + metadata_index).to(tl.int64)
        row_tiles = tl.cdiv(count, COMM_BLOCK_N)
        total_tasks = row_tiles * d_tiles
        if global_vec_id % 2 == 0:
            for task in range(pid, total_tasks, ncore):
                row_tile = task // d_tiles
                d_tile = task % d_tiles
                ns = row_tile * COMM_BLOCK_N + tl.arange(0, COMM_BLOCK_N)
                ds = d_tile * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (ns[:, None] < count) & (ds[None, :] < D)
                src = (send_offset + ns)[:, None] * D + ds[None, :]
                dst = (recv_offset + ns)[:, None] * D + ds[None, :]
                data = tl.load(send_ptr + src, mask=mask, other=0.0)
                tl.store(recv_ptr + dst, data, mask=mask)

    # Every remote batch has an explicit completion put.  The vector-only
    # collective establishes cross-rank visibility before scatter consumes the
    # returned allocation.
    libshmem_device.barrier_all_vec()


@triton.jit
def kernel_scatter_sum(
    returned_ptr,
    scatter_index_ptr,
    output_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    TILED_D: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fast path for complete 16-row by 512-column regions."""
    pid = tl.program_id(0)
    blocks_d = TILED_D // BLOCK_D
    D_TILES_PER_PROGRAM: tl.constexpr = 2
    block_groups_d = blocks_d // D_TILES_PER_PROGRAM
    block_n = pid // block_groups_d
    block_group_d = pid % block_groups_d
    ns = (block_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    ds0 = (block_group_d * D_TILES_PER_PROGRAM) * BLOCK_D + tl.arange(0, BLOCK_D)
    ds1 = (block_group_d * D_TILES_PER_PROGRAM + 1) * BLOCK_D + tl.arange(0, BLOCK_D)
    nmask = ns < N
    dmask0 = ds0 < TILED_D
    dmask1 = ds1 < TILED_D
    acc0 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc1 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for k in range(TOPK):
        idx = tl.load(
            scatter_index_ptr + ns * TOPK + k,
            mask=nmask,
            other=0,
        ).to(tl.int64)
        offs0 = idx[:, None] * D + ds0[None, :]
        offs1 = idx[:, None] * D + ds1[None, :]
        acc0 += tl.load(
            returned_ptr + offs0,
            mask=nmask[:, None] & dmask0[None, :],
            other=0.0,
        )
        acc1 += tl.load(
            returned_ptr + offs1,
            mask=nmask[:, None] & dmask1[None, :],
            other=0.0,
        )

    out_offs0 = ns[:, None] * D + ds0[None, :]
    out_offs1 = ns[:, None] * D + ds1[None, :]
    tl.store(
        output_ptr + out_offs0,
        acc0.to(tl.bfloat16),
        mask=nmask[:, None] & dmask0[None, :],
    )
    tl.store(
        output_ptr + out_offs1,
        acc1.to(tl.bfloat16),
        mask=nmask[:, None] & dmask1[None, :],
    )


@triton.jit(do_not_specialize=["N", "row_end", "end_idx"])
def kernel_scatter_sum_tail(
    returned_ptr,
    scatter_index_ptr,
    output_ptr,
    N,
    row_end,
    end_idx,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    ROW_START: tl.constexpr,
    COL_START: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Masked fallback for the N or D region outside the fast rectangle.

    ``scatter_index`` is a permutation of ``[0, N * TOPK)`` under the current
    public Combine contract.  Only tail launches pay for the validity checks.
    """
    pid = tl.program_id(0)
    blocks_d = tl.cdiv(D - COL_START, BLOCK_D)
    block_n = pid // blocks_d
    block_d = pid % blocks_d
    ns = (ROW_START + block_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    ds = COL_START + block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    nmask = (ns < N) & (ns < row_end)
    dmask = ds < D
    acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for k in range(TOPK):
        idx = tl.load(
            scatter_index_ptr + ns * TOPK + k,
            mask=nmask,
            other=-1,
        ).to(tl.int64)
        valid = nmask & (idx >= 0) & (idx < end_idx)
        safe_idx = tl.where(valid, idx, 0)
        offs = safe_idx[:, None] * D + ds[None, :]
        data = tl.load(
            returned_ptr + offs,
            mask=valid[:, None] & dmask[None, :],
            other=0.0,
        )
        acc += data

    out_offs = ns[:, None] * D + ds[None, :]
    tl.store(
        output_ptr + out_offs,
        acc.to(tl.bfloat16),
        mask=nmask[:, None] & dmask[None, :],
    )


def torch_golden_combine(
    start_idx,
    end_idx,
    splits_cpu,
    a2a_splits_cpu,
    permute_a2a_expert_output,
    scatter_index,
    ep_size,
    ep_group,
    num_local_experts,
):
    """Exact Combine contract from dispatch_and_combine_docs/torch_golden.py."""
    expert_major_splits = (
        a2a_splits_cpu.reshape(-1, num_local_experts).permute(-1, -2).flatten()
    )
    chunks = torch.split(permute_a2a_expert_output, expert_major_splits.tolist())
    rank_major = []
    for source_rank in range(ep_size):
        for local_expert in range(num_local_experts):
            rank_major.append(chunks[local_expert * ep_size + source_rank])
    a2a_expert_output = torch.cat(rank_major, dim=0)
    expert_output = torch.empty(
        (int(splits_cpu.sum()), a2a_expert_output.shape[-1]),
        device=a2a_expert_output.device,
        dtype=a2a_expert_output.dtype,
    )
    dist.all_to_all_single(
        expert_output,
        a2a_expert_output,
        output_split_sizes=splits_cpu.reshape(ep_size, -1).sum(-1).tolist(),
        input_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(-1).tolist(),
        group=ep_group,
    )
    if start_idx == end_idx:
        return torch.zeros(
            (scatter_index.shape[0], expert_output.shape[-1]),
            device=expert_output.device,
            dtype=expert_output.dtype,
        )
    valid = (scatter_index >= start_idx) & (scatter_index < end_idx)
    safe = torch.where(valid, scatter_index, torch.zeros_like(scatter_index))
    values = expert_output[safe.flatten()].reshape(
        scatter_index.shape[0], scatter_index.shape[1], -1
    )
    return (values * valid.unsqueeze(-1).to(values.dtype)).sum(1)


def launch_udma_combine(
    send,
    returned_base,
    returned,
    output,
    scatter_index,
    splits,
    recv_splits,
    all_splits,
    all_splits_views,
    bucket_counts,
    send_offsets,
    recv_offsets,
    rank,
    rank_size,
    experts,
    comm_block_n,
    comm_block_d,
):
    """Build direct metadata, exchange buckets, then scatter/sum."""
    # Reuse the driver's utility instance instead of constructing NPUUtils.
    aicore_num = driver.active.utils.get_aicore_num()
    # The public Combine inputs contain this rank's local send/receive splits,
    # not every remote rank's destination prefixes.  Recover the global split
    # matrix and derive all direct-address metadata inside the timed recipe.
    dist.all_gather(all_splits_views, splits)
    num_buckets = rank_size * experts
    kernel_prepare_combine_metadata[(num_buckets,)](
        all_splits,
        recv_splits,
        bucket_counts,
        send_offsets,
        recv_offsets,
        rank,
        R=rank_size,
        E=experts,
        META_SIZE=triton.next_power_of_2(num_buckets),
    )
    # shmem_device_udma documents a 256 MB maximum per put request.  Use the
    # largest whole-row chunk that stays within that limit for this D.
    udma_chunk_rows = max(1, (256 * 1024 * 1024) // (send.shape[1] * 2))
    kernel_udma_combine_direct[aicore_num, 1, 1](
        send,
        # Address translation for remote RMA must start from the complete
        # symmetric allocation on every rank.  ``returned`` is a rank-local
        # length-dependent view and is not a valid cross-rank base.
        returned_base,
        bucket_counts,
        send_offsets,
        recv_offsets,
        rank,
        R=rank_size,
        E=experts,
        D=send.shape[1],
        UDMA_CHUNK_ROWS=udma_chunk_rows,
        COMM_BLOCK_N=comm_block_n,
        COMM_BLOCK_D=comm_block_d,
    )
    block_n = 16
    block_d = 256
    tile_span_d = 2 * block_d
    n = scatter_index.shape[0]
    d = send.shape[1]
    n_full = (n // block_n) * block_n
    d_full = (d // tile_span_d) * tile_span_d

    # The customer case N=32768,D=1536 is entirely covered here and therefore
    # retains the single-launch constexpr/direct-address fast path.
    if n_full > 0 and d_full > 0:
        fast_grid = ((n_full // block_n) * (d_full // tile_span_d), 1, 1)
        kernel_scatter_sum[fast_grid](
            returned,
            scatter_index,
            output,
            N=n_full,
            D=d,
            TILED_D=d_full,
            TOPK=scatter_index.shape[1],
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )

    # At most 15 token rows take this path.  BLOCK_N=1 avoids the failing
    # partially valid 16-row vector observed at N=2050.
    if n_full < n:
        n_tail_grid = ((n - n_full) * triton.cdiv(d, block_d), 1, 1)
        kernel_scatter_sum_tail[n_tail_grid](
            returned,
            scatter_index,
            output,
            N=n,
            row_end=n,
            end_idx=scatter_index.numel(),
            D=d,
            TOPK=scatter_index.shape[1],
            ROW_START=n_full,
            COL_START=0,
            BLOCK_N=1,
            BLOCK_D=block_d,
        )

    # Full token blocks but fewer than 512 remaining columns use one masked
    # tile per program.  This covers D=3073 without an all-masked second tile.
    if d_full < d and n_full > 0:
        d_tail_grid = ((n_full // block_n) * triton.cdiv(d - d_full, block_d), 1, 1)
        kernel_scatter_sum_tail[d_tail_grid](
            returned,
            scatter_index,
            output,
            N=n,
            row_end=n_full,
            end_idx=scatter_index.numel(),
            D=d,
            TOPK=scatter_index.shape[1],
            ROW_START=0,
            COL_START=d_full,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )


def make_routing(n, total_experts, topk, mode, rank, seed):
    rng = random.Random(seed + rank)
    rows = []
    for token in range(n):
        if mode == "hot":
            choices = [0]
            remaining = [x for x in range(1, total_experts)]
            rng.shuffle(remaining)
            choices.extend(remaining[: topk - 1])
        else:
            choices = rng.sample(range(total_experts), topk)
        rows.append(choices)
    return torch.tensor(rows, dtype=torch.int64)


def make_metadata(n, topk, experts, rank, world_size, mode, seed):
    routing = make_routing(n, world_size * experts, topk, mode, rank, seed)
    splits_cpu = torch.bincount(routing.flatten(), minlength=world_size * experts).to(
        torch.int32
    )
    _, sorted_original = routing.flatten().sort(stable=True)
    inverse = torch.empty(n * topk, dtype=torch.int64)
    inverse[sorted_original] = torch.arange(n * topk, dtype=torch.int64)
    scatter_index = inverse.reshape(n, topk).npu()

    splits = splits_cpu.npu()
    gathered = [torch.empty_like(splits) for _ in range(world_size)]
    dist.all_gather(gathered, splits)
    counts = torch.stack(gathered).reshape(world_size, world_size, experts)
    recv_splits = counts[:, rank, :].contiguous().reshape(-1)
    return splits_cpu, splits, recv_splits, scatter_index


def verify_all_ranks(expected, actual, rank, world_size):
    passed = torch.ones(1, dtype=torch.int32).npu()
    detail = ""
    try:
        torch.testing.assert_close(expected, actual, rtol=1e-3, atol=1e-3)
    except AssertionError as exc:
        passed.zero_()
        detail = str(exc)
    all_passed = [torch.empty_like(passed) for _ in range(world_size)]
    dist.all_gather(all_passed, passed)
    for current in range(world_size):
        if rank == current:
            label = f"{GREEN}[PASS]{RESET}" if passed.item() else f"{RED}[FAIL]{RESET}"
            print(
                (
                    f"  {label} rank {rank} matches golden."
                    if passed.item()
                    else f"  {label} rank {rank}: {detail[:600]}"
                ),
                flush=True,
            )
        dist.barrier()
    return all(x.item() == 1 for x in all_passed)


def init_shmem(rank, world_size):
    init_aclshmem(rank, world_size, g_ash_size, ash.OpEngineType.UDMA)


def run(case_executor=None, *, case_selector=None, verify=True):
    """Run cases, optionally delegating timing and selection to the perf script.

    Standalone calls run all cases with verification enabled.
    """
    rank = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    init_shmem(rank, world_size)
    comm_block_n = int(os.environ.get("MOE_COMM_BLOCK_N", "64"))
    comm_block_d = int(os.environ.get("MOE_COMM_BLOCK_D", "128"))
    cases = [
        (mode, n, topk, experts, hidden_size)
        for mode in ROUTE_MODES
        for n in N_LIST
        for topk in TOPK_LIST
        for experts in E_LIST
        for hidden_size in HIDDEN_SIZE_LIST
        if topk <= world_size * experts
    ]
    if case_selector is not None:
        cases = case_selector(cases)
    if rank == 0:
        print(
            f"{BOLD}[START]{RESET} common-input UDMA Combine: "
            f"{len(cases)} cases, R={world_size}, hidden_sizes={HIDDEN_SIZE_LIST}, "
            f"mode={'perf' if case_executor else 'correctness'}, "
            f"candidate=direct_bucket_exchange_with_timed_metadata",
            flush=True,
        )

    try:
        for case_id, (mode, n, topk, experts, hidden_size) in enumerate(cases):
            dist.barrier()
            splits_cpu, splits, recv_splits, scatter_index = make_metadata(
                n, topk, experts, rank, world_size, mode, SEED
            )
            recv_splits_cpu = recv_splits.cpu()
            start_idx = 0
            end_idx = int(splits_cpu.sum())
            max_bucket = int(recv_splits.max().item())
            send_rows = int(recv_splits.sum().item())
            max_returned = max(1, n * topk)

            # Reusable storage may be allocated outside timing, but every value
            # derived from the common split inputs is regenerated per recipe.
            all_splits = torch.empty(
                (world_size, world_size * experts),
                dtype=torch.int32,
                device=splits.device,
            )
            all_splits_views = list(all_splits.unbind(0))
            bucket_counts = torch.empty_like(recv_splits)
            send_offsets = torch.empty_like(recv_splits)
            recv_offsets = torch.empty_like(recv_splits)

            # Use one deterministic stream across adjacent N values so the
            # boundary sweep changes shape, not shape and data together.
            torch.manual_seed(SEED + rank)
            send = torch.randn((send_rows, hidden_size), dtype=torch.bfloat16).npu()
            output = torch.empty((n, hidden_size), dtype=torch.bfloat16).npu()
            dist.barrier()
            peer_mem = ash.aclshmem_create_tensor(
                [max_returned * hidden_size],
                dtype=torch.bfloat16,
                device_id=local_pe,
            )
            # UDMA writes land directly in the final rank-major returned buffer.
            returned = peer_mem[: end_idx * hidden_size].view(end_idx, hidden_size)
            peer_mem.fill_(float("nan"))
            returned.fill_(float("nan"))
            output.fill_(float("nan"))
            torch.npu.synchronize()
            dist.barrier()

            if rank == 0:
                timing_boundary = (
                    "timing_boundary=common_inputs_full_recipe"
                    if case_executor
                    else "correctness_only"
                )
                print(
                    f"\nCase {case_id + 1}/{len(cases)}: mode={mode}, "
                    f"N={n}, K={topk}, E={experts}, D={hidden_size}, "
                    f"max_bucket={max_bucket}, "
                    f"{timing_boundary}",
                    flush=True,
                )

            def run_golden():
                return torch_golden_combine(
                    start_idx,
                    end_idx,
                    splits_cpu,
                    recv_splits_cpu,
                    send,
                    scatter_index,
                    world_size,
                    None,
                    experts,
                )

            def run_candidate():
                launch_udma_combine(
                    send,
                    peer_mem,
                    returned,
                    output,
                    scatter_index,
                    splits,
                    recv_splits,
                    all_splits,
                    all_splits_views,
                    bucket_counts,
                    send_offsets,
                    recv_offsets,
                    rank,
                    world_size,
                    experts,
                    comm_block_n,
                    comm_block_d,
                )

            case_info = {
                "route": mode,
                "n": n,
                "d": hidden_size,
                "topk": topk,
                "local_experts": experts,
                "ranks": world_size,
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
            if verify and not verify_all_ranks(golden, output, rank, world_size):
                raise AssertionError(
                    f"Combine failed: mode={mode}, N={n}, D={hidden_size}, K={topk}, E={experts}"
                )
            torch.npu.synchronize()
            dist.barrier()
            ash.aclshmem_free_tensor(peer_mem)
            dist.barrier()
    finally:
        ash.aclshmem_finialize()

    if rank == 0:
        print(
            f"{GREEN}[SUCCESS]{RESET} all {len(cases)} Combine cases "
            f"{'verified' if verify else 'completed'}.",
            flush=True,
        )


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=pe)
    print(f"[INFO] rank {local_rank}/{dist.get_world_size()} initialized", flush=True)
    dist.barrier()
    run()
