# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AllToAllV-GMM tutorial using MTE remote stores and device-side completion.

Producers derive their source and destination offsets from device-resident
``token_counts`` and dispatch one ``BLOCK_M`` tile into symmetric memory.
A separate pure Vector collective establishes a cross-rank completion frontier
before grouped GEMM consumes the receive layout.

Run on a two-NPU Ascend host:

    torchrun --standalone --nproc_per_node=2 \
      tutorials/ascend/alltoallv-gmm.py

The tutorial uses a fixed-capacity output ``[world_size * M, N]``.  Its valid
prefix is grouped by local expert and source rank; its unused tail is zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.runtime import driver
from triton_dist.language.extra import libshmem_device
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem

import shmem as ash

# Core counts are constant for the lifetime of a process; cache them so the
# per-round launch path never queries the driver utilities again.
_CORE_COUNTS: dict[str, int] = {}


def _aicore_num() -> int:
    """Return the Cube core count for this process."""
    if "aicore" not in _CORE_COUNTS:
        _CORE_COUNTS["aicore"] = driver.active.utils.get_aicore_num()
    return _CORE_COUNTS["aicore"]


def _aivector_core_num() -> int:
    """Return the Vector core count for this process."""
    if "aivector" not in _CORE_COUNTS:
        _CORE_COUNTS["aivector"] = driver.active.utils.get_aivector_core_num()
    return _CORE_COUNTS["aivector"]


@triton.jit
def _alltoallv_dispatch_kernel(
    activations,
    receive_buffer,
    send_group_offsets,
    send_destination_offsets,
    send_group_tile_offsets,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Dispatch source expert tiles to the destination rank's symmetric buffer."""
    program = tl.program_id(axis=0)
    total_experts: tl.constexpr = world_size * experts_per_rank
    k_block = program % tl.cdiv(k, block_k)
    tile_id = program // tl.cdiv(k, block_k)
    global_expert = 0
    for index in tl.static_range(1, total_experts + 1):
        global_expert += tl.where(
            tl.load(send_group_tile_offsets + index) <= tile_id, 1, 0
        )
    global_expert = tl.minimum(global_expert, total_experts - 1)
    peer = global_expert // experts_per_rank
    block = tile_id - tl.load(send_group_tile_offsets + global_expert)
    source_offset = tl.load(send_group_offsets + global_expert)
    token_count = tl.load(send_group_offsets + global_expert + 1) - source_offset
    destination_offset = tl.load(send_destination_offsets + global_expert)

    rows = block * block_m + tl.arange(0, block_m)
    columns = k_block * block_k + tl.arange(0, block_k)
    offsets = rows[:, None] * k + columns[None, :]
    mask = (rows[:, None] < token_count) & (columns[None, :] < k)
    values = tl.load(activations + source_offset * k + offsets, mask=mask, other=0.0)
    if peer == rank:
        tl.store(
            receive_buffer
            + (destination_offset + rows[:, None]) * k
            + columns[None, :],
            values,
            mask=mask,
        )
    else:
        remote_buffer = dl.symm_at(receive_buffer, peer)
        tl.store(
            remote_buffer + (destination_offset + rows[:, None]) * k + columns[None, :],
            values,
            mask=mask,
        )
        libshmem_device.quiet()


@triton.jit
def _grouped_matmul_kernel(
    receive_buffer,
    weights,
    output,
    group_offsets,
    group_tile_offsets,
    experts_per_rank: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_n_tiles: tl.constexpr,
):
    """Compute one grouped-GEMM output tile for a local expert."""
    tile_idx = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    total_tiles = tl.load(group_tile_offsets + experts_per_rank) * num_n_tiles
    while tile_idx < total_tiles:
        tile_m = tile_idx // num_n_tiles
        tile_n = tile_idx % num_n_tiles
        expert = 0
        for index in tl.static_range(1, experts_per_rank + 1):
            expert += tl.where(tl.load(group_tile_offsets + index) <= tile_m, 1, 0)
        expert = tl.minimum(expert, experts_per_rank - 1)
        local_tile_m = tile_m - tl.load(group_tile_offsets + expert)
        group_start = tl.load(group_offsets + expert)
        group_end = tl.load(group_offsets + expert + 1)
        rows = local_tile_m * block_m + tl.arange(0, block_m)
        columns = tile_n * block_n + tl.arange(0, block_n)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_block in range(tl.cdiv(k, block_k)):
            k_offsets = k_block * block_k + tl.arange(0, block_k)
            lhs_ptrs = (
                receive_buffer + (group_start + rows[:, None]) * k + k_offsets[None, :]
            )
            rhs_ptrs = (
                weights + expert * k * n + k_offsets[:, None] * n + columns[None, :]
            )
            tl.multiple_of(lhs_ptrs, [16, 16])
            tl.multiple_of(rhs_ptrs, [16, 16])
            lhs = tl.load(
                lhs_ptrs,
                mask=(rows[:, None] < group_end - group_start)
                & (k_offsets[None, :] < k),
                other=0.0,
            )
            rhs = tl.load(
                rhs_ptrs,
                mask=(k_offsets[:, None] < k) & (columns[None, :] < n),
                other=0.0,
            )
            accumulator += tl.dot(lhs, rhs)
        tl.store(
            output + (group_start + rows[:, None]) * n + columns[None, :],
            accumulator.to(output.dtype.element_ty),
            mask=(rows[:, None] < group_end - group_start) & (columns[None, :] < n),
        )
        tile_idx += num_programs


@triton.jit
def _mte_completion_barrier_kernel():
    """Make all remote MTE stores visible before the following grouped GEMM."""
    if tl.program_id(axis=0) == 0:
        libshmem_device.barrier_all_vec()


@triton.jit
def _zero_output_rows_kernel(
    output,
    row_start,
    row_end,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    n_tiles: tl.constexpr,
):
    """Zero the output row band [row_start, row_end) with a persistent Vector grid."""
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    band_tiles = tl.cdiv(row_end - row_start, block_m)
    for task in range(pid, band_tiles * n_tiles, ncore):
        tile_m = task // n_tiles
        tile_n = task % n_tiles
        rows = row_start + tile_m * block_m + tl.arange(0, block_m)
        columns = tile_n * block_n + tl.arange(0, block_n)
        tl.store(
            output + rows[:, None] * n + columns[None, :],
            tl.zeros((block_m, block_n), dtype=output.dtype.element_ty),
            mask=(rows[:, None] < row_end) & (columns[None, :] < n),
        )


def alltoallv_dispatch(
    activations,
    receive_buffer,
    send_group_offsets,
    send_destination_offsets,
    send_group_tile_offsets,
    total_send_tiles,
    rank,
    world_size,
    experts_per_rank,
    block_m,
    block_k,
):
    """Launch the self-contained AllToAllV dispatch kernel."""
    if total_send_tiles == 0:
        return
    _, k = activations.shape
    _alltoallv_dispatch_kernel[(total_send_tiles * triton.cdiv(k, block_k),)](
        activations,
        receive_buffer,
        send_group_offsets,
        send_destination_offsets,
        send_group_tile_offsets,
        rank=rank,
        world_size=world_size,
        experts_per_rank=experts_per_rank,
        k=k,
        block_m=block_m,
        block_k=block_k,
    )


def grouped_matmul(
    receive_buffer,
    weights,
    output,
    group_offsets,
    group_tile_offsets,
    total_m_tiles,
    experts_per_rank,
    k,
    n,
    block_m,
    block_n,
    block_k,
):
    """Launch the self-contained grouped-GEMM kernel."""
    num_n_tiles = triton.cdiv(n, block_n)
    total_tiles = total_m_tiles * num_n_tiles
    num_programs = min(_aicore_num(), total_tiles)
    _grouped_matmul_kernel[(max(1, num_programs),)](
        receive_buffer,
        weights,
        output,
        group_offsets,
        group_tile_offsets,
        experts_per_rank=experts_per_rank,
        k=k,
        n=n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_n_tiles=num_n_tiles,
    )


def _clear_output_rows(
    output: torch.Tensor,
    row_start: int,
    row_end: int,
    block_m: int,
    block_n: int,
) -> None:
    """Zero output rows [row_start, row_end); skip the launch for an empty band."""
    if row_end <= row_start:
        return
    _, n = output.shape
    n_tiles = triton.cdiv(n, block_n)
    band_tiles = triton.cdiv(row_end - row_start, block_m)
    _zero_output_rows_kernel[(min(_aivector_core_num(), band_tiles * n_tiles),)](
        output,
        row_start,
        row_end,
        n=n,
        block_m=block_m,
        block_n=block_n,
        n_tiles=n_tiles,
    )


@dataclass
class OutputTailGuard:
    """Keep the unused output tail zero while avoiding per-round clear launches.

    ``valid_rows`` is the row count written by the previous invocation and every
    row at or beyond it is guaranteed to be zero.  Only a shrinking route makes
    rows dirty, so a fixed-route benchmark issues no clear launch at all.
    """

    capacity_rows: int
    valid_rows: int

    @classmethod
    def create(cls, capacity_rows: int) -> "OutputTailGuard":
        """Start with the whole buffer treated as tail so the first round clears it."""
        return cls(capacity_rows=capacity_rows, valid_rows=capacity_rows)


def _maintain_output_tail(
    output: torch.Tensor,
    valid_rows: int,
    tail_guard: OutputTailGuard,
    block_m: int,
    block_n: int,
) -> None:
    """Zero the dirty tail band after a shrinking route, then advance the watermark."""
    if valid_rows > tail_guard.capacity_rows:
        raise ValueError(
            f"valid rows {valid_rows} exceed output capacity {tail_guard.capacity_rows}"
        )
    if valid_rows < tail_guard.valid_rows:
        _clear_output_rows(output, valid_rows, tail_guard.valid_rows, block_m, block_n)
    tail_guard.valid_rows = valid_rows


def gather_timings(timings: list[float], world_size: int) -> list[list[float]]:
    """Gather fixed-size timing vectors through HCCL tensors."""
    local = torch.tensor(timings, dtype=torch.float32).npu()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [
        [float(gathered[rank][index].item()) for rank in range(world_size)]
        for index in range(len(timings))
    ]


def build_routing_metadata(
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    dispatch_block_m: int,
    gmm_block_m: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, int, int
]:
    """Build dispatch and grouped-GEMM offsets for the local rank."""
    total_experts = world_size * experts_per_rank
    send_counts = token_counts[rank]
    send_group_offsets = torch.empty(total_experts + 1, dtype=torch.int32)
    send_group_offsets[0] = 0
    send_group_offsets[1:] = torch.cumsum(send_counts, dim=0)
    send_group_tile_offsets = torch.empty(total_experts + 1, dtype=torch.int32)
    send_group_tile_offsets[0] = 0
    send_group_tile_offsets[1:] = torch.cumsum(
        (send_counts + dispatch_block_m - 1) // dispatch_block_m, dim=0
    )

    send_destination_offsets = torch.empty(total_experts, dtype=torch.int32)
    for destination_rank in range(world_size):
        expert_start = destination_rank * experts_per_rank
        destination_counts = token_counts[
            :, expert_start : expert_start + experts_per_rank
        ]
        expert_offsets = torch.empty(experts_per_rank + 1, dtype=torch.int32)
        expert_offsets[0] = 0
        expert_offsets[1:] = torch.cumsum(destination_counts.sum(dim=0), dim=0)
        for local_expert in range(experts_per_rank):
            send_destination_offsets[expert_start + local_expert] = (
                expert_offsets[local_expert]
                + destination_counts[:rank, local_expert].sum()
            )

    local_expert_start = rank * experts_per_rank
    group_sizes = token_counts[
        :, local_expert_start : local_expert_start + experts_per_rank
    ].sum(dim=0)
    group_offsets = torch.empty(experts_per_rank + 1, dtype=torch.int32)
    group_offsets[0] = 0
    group_offsets[1:] = torch.cumsum(group_sizes, dim=0)
    group_tile_offsets = torch.empty(experts_per_rank + 1, dtype=torch.int32)
    group_tile_offsets[0] = 0
    group_tile_offsets[1:] = torch.cumsum(
        (group_sizes + gmm_block_m - 1) // gmm_block_m, dim=0
    )

    return (
        send_group_offsets.npu(),
        send_destination_offsets.npu(),
        send_group_tile_offsets.npu(),
        int(send_group_tile_offsets[-1].item()),
        group_offsets.npu(),
        group_tile_offsets.npu(),
        int(group_tile_offsets[-1].item()),
        int(group_sizes.sum().item()),
    )


def launch_alltoallv_gmm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    receive_buffer: torch.Tensor,
    output: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    dispatch_block_m: int,
    dispatch_block_k: int,
    gmm_block_m: int,
    gmm_block_n: int,
    gmm_block_k: int,
    tail_guard: OutputTailGuard,
    routing_metadata: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ],
) -> None:
    """Maintain the output tail, then dispatch, synchronize, and compute."""
    (
        send_group_offsets,
        send_destination_offsets,
        send_group_tile_offsets,
        total_send_tiles,
        group_offsets,
        group_tile_offsets,
        total_m_tiles,
        valid_rows,
    ) = routing_metadata
    # Grouped GEMM only rewrites the valid prefix, so rows left behind by a
    # longer previous route must be zeroed before this round's result is read.
    _maintain_output_tail(output, valid_rows, tail_guard, gmm_block_m, gmm_block_n)
    alltoallv_dispatch(
        activations,
        receive_buffer,
        send_group_offsets,
        send_destination_offsets,
        send_group_tile_offsets,
        total_send_tiles,
        rank,
        world_size,
        experts_per_rank,
        dispatch_block_m,
        dispatch_block_k,
    )
    _mte_completion_barrier_kernel[(1,)]()
    grouped_matmul(
        receive_buffer,
        weights,
        output,
        group_offsets,
        group_tile_offsets,
        total_m_tiles,
        experts_per_rank,
        activations.shape[1],
        weights.shape[-1],
        gmm_block_m,
        gmm_block_n,
        gmm_block_k,
    )


def symmetric_memory_size(world_size: int, m: int, k: int, buffer_count: int) -> int:
    """Return aligned ACLSHMEM capacity for the receive buffers in this tutorial."""
    dtype_bytes = torch.tensor([], dtype=torch.float16).element_size()
    required_bytes = buffer_count * world_size * m * k * dtype_bytes
    pool_bytes = (
        (required_bytes + SHMEM_PAGE_SIZE - 1) // SHMEM_PAGE_SIZE
    ) * SHMEM_PAGE_SIZE
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the swept cases"
        )
    return pool_bytes


ROUTE_SEED = 20260828
ROUTE_MODES = ("random", "hot", "skewed", "static")


@dataclass(frozen=True)
class RouteSample:
    """One expert-major activation tensor and its global count matrix."""

    activations: torch.Tensor
    token_counts: torch.Tensor
    mode: str
    epoch: int


def _static_token_counts(
    m: int, world_size: int, experts_per_rank: int, sparse: bool
) -> torch.Tensor:
    """Build the fixed static route for one snapshot."""
    total_experts = world_size * experts_per_rank
    if not sparse:
        counts = torch.full(
            (world_size, total_experts), m // total_experts, dtype=torch.int32
        )
        for source_rank in range(world_size):
            for index in range(m % total_experts):
                counts[source_rank, (source_rank + index) % total_experts] += 1
        return counts
    counts = torch.zeros((world_size, total_experts), dtype=torch.int32)
    for source_rank in range(world_size):
        first = (source_rank * experts_per_rank + 1) % total_experts
        second = (first + 1) % total_experts
        counts[source_rank, first] = m // 2
        counts[source_rank, second] = m - m // 2
    return counts


def _expert_ids(
    m: int, total_experts: int, rank: int, epoch: int, mode: str
) -> torch.Tensor:
    """Generate a fixed-seed top-1 route with controllable skew."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ROUTE_SEED + rank * 1000003 + epoch * 10007)
    if mode == "random":
        return torch.randint(
            total_experts, (m,), generator=generator, dtype=torch.int64
        )
    if mode == "hot":
        return torch.zeros((m,), dtype=torch.int64)
    if mode == "skewed":
        ids = torch.randint(total_experts, (m,), generator=generator, dtype=torch.int64)
        ids[: (m * 3) // 4] = 0
        return ids
    raise ValueError(f"unsupported dynamic route mode: {mode}")


def _exchange_counts(
    local_counts: torch.Tensor, world_size: int
) -> torch.Tensor:
    """Gather token counts through HCCL and return the CPU matrix in rank order."""
    local = local_counts.npu()
    rows = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(rows, local)
    return torch.stack(rows).cpu()


def prepare_route(
    m: int,
    k: int,
    world_size: int,
    experts_per_rank: int,
    rank: int,
    epoch: int,
    mode: str,
    sparse: bool,
) -> RouteSample:
    """Create one route epoch and return expert-major activations plus global counts."""
    if mode not in ROUTE_MODES:
        raise ValueError(
            f"TRITON_ROUTE_MODES must contain only: {', '.join(ROUTE_MODES)}"
        )
    total_experts = world_size * experts_per_rank
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ROUTE_SEED + rank * 1000003 + epoch * 10007 + 1)
    activations = torch.randn((m, k), generator=generator, dtype=torch.float16)
    if mode == "static":
        token_counts = _static_token_counts(m, world_size, experts_per_rank, sparse)
    else:
        expert_ids = _expert_ids(m, total_experts, rank, epoch, mode)
        order = torch.argsort(expert_ids, stable=True)
        activations = activations[order]
        local_counts = torch.bincount(expert_ids, minlength=total_experts).to(
            torch.int32
        )
        token_counts = _exchange_counts(local_counts, world_size)
    if token_counts.shape != (world_size, total_experts) or torch.any(token_counts < 0):
        raise ValueError("invalid global token-count matrix")
    if not torch.equal(
        token_counts.sum(dim=1), torch.full((world_size,), m, dtype=torch.int32)
    ):
        raise ValueError("every source rank must route exactly M tokens")
    return RouteSample(
        activations=activations.npu(), token_counts=token_counts, mode=mode, epoch=epoch
    )


def _reference_output(
    activations: torch.Tensor,
    weights: torch.Tensor,
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
) -> torch.Tensor:
    """Build the torch golden output with all_to_all_single and a fused grouped GEMM."""
    counts = token_counts.to(torch.int64)
    local_expert_start = rank * experts_per_rank
    send_splits = counts[rank].reshape(world_size, experts_per_rank).sum(dim=1)
    receive_splits = counts[
        :, local_expert_start : local_expert_start + experts_per_rank
    ].sum(dim=1)
    received = torch.empty(
        (int(receive_splits.sum().item()), activations.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    dist.all_to_all_single(
        received,
        activations,
        output_split_sizes=receive_splits.tolist(),
        input_split_sizes=send_splits.tolist(),
    )
    source_chunks = []
    source_offset = 0
    for source_rank in range(world_size):
        expert_offset = source_offset
        expert_chunks = []
        for local_expert in range(experts_per_rank):
            count = int(counts[source_rank, local_expert_start + local_expert].item())
            expert_chunks.append(received[expert_offset : expert_offset + count])
            expert_offset += count
        source_chunks.append(expert_chunks)
        source_offset += int(receive_splits[source_rank].item())
    x_list = []
    for local_expert in range(experts_per_rank):
        expert_input = torch.cat(
            [
                source_chunks[source_rank][local_expert]
                for source_rank in range(world_size)
            ],
            dim=0,
        )
        x_list.append(expert_input)
    out_list = torch_npu.npu_grouped_matmul(
        x=x_list,
        weight=[weights[local_expert] for local_expert in range(experts_per_rank)],
        group_type=0,
        split_item=2,
    )
    output = out_list[0]
    padded = torch.zeros(
        (world_size * activations.shape[0], weights.shape[-1]),
        dtype=output.dtype,
        device=output.device,
    )
    padded[: output.shape[0]] = output
    return padded


def run() -> None:
    """Initialize distributed state and run the fixed-seed self-test."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    m = int(os.environ.get("TRITON_M", 64))
    k = int(os.environ.get("TRITON_K", 128))
    n = int(os.environ.get("TRITON_N", 128))
    experts_per_rank = int(os.environ.get("TRITON_EXPERTS_PER_RANK", 2))
    dispatch_block_m = min(int(os.environ.get("TRITON_DISPATCH_BLOCK_M", 128)), m)
    dispatch_block_k = min(int(os.environ.get("TRITON_DISPATCH_BLOCK_K", 256)), k)
    gmm_block_m = min(int(os.environ.get("TRITON_GMM_BLOCK_M", 128)), m)
    gmm_block_n = min(int(os.environ.get("TRITON_GMM_BLOCK_N", 256)), n)
    gmm_block_k = min(int(os.environ.get("TRITON_GMM_BLOCK_K", 256)), k)
    validation_iterations = int(os.environ.get("TRITON_VALIDATION_ITERS", 1))
    sparse = os.environ.get("TRITON_SPARSE", "1") == "1"
    route_modes = tuple(
        value.strip()
        for value in os.environ.get("TRITON_ROUTE_MODES", "random").split(",")
        if value.strip()
    )
    if len(route_modes) != 1:
        raise ValueError(
            "TRITON_ROUTE_MODES must contain exactly one mode when running the tutorial script"
        )
    route_mode = route_modes[0]
    if (
        min(
            m,
            k,
            n,
            experts_per_rank,
            dispatch_block_m,
            dispatch_block_k,
            gmm_block_m,
            gmm_block_n,
            gmm_block_k,
        )
        <= 0
    ):
        raise ValueError("shape and block-size variables must be positive")
    if validation_iterations < 0:
        raise ValueError("TRITON_VALIDATION_ITERS must be non-negative")
    buffer_count = 2
    local_mem_size = symmetric_memory_size(world_size, m, k, buffer_count)
    if rank == 0:
        print(
            f"ALLTOALLV_GMM_LOCAL_MEM_SIZE_BYTES={local_mem_size}",
            flush=True,
        )
    init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.MTE)

    receive_buffers = []
    try:
        seed = 20260811
        torch.manual_seed(seed + rank)
        weights = torch.randn((experts_per_rank, k, n), dtype=torch.float16).npu()
        for _ in range(buffer_count):
            receive_buffers.append(
                ash.aclshmem_create_tensor(
                    [world_size * m, k], dtype=torch.float16, device_id=local_rank
                )
            )
        output = torch.empty((world_size * m, n), dtype=torch.float16).npu()
        tail_guard = OutputTailGuard.create(world_size * m)
        invocation_count = 0

        def prepare_invocation(
            epoch: int,
        ) -> tuple[
            RouteSample,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
            ],
        ]:
            sample = prepare_route(
                m,
                k,
                world_size,
                experts_per_rank,
                rank,
                epoch,
                route_mode,
                sparse,
            )
            metadata = build_routing_metadata(
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
                dispatch_block_m,
                gmm_block_m,
            )
            return sample, metadata

        def invoke(
            activations: torch.Tensor,
            metadata: tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
            ],
        ) -> torch.Tensor:
            nonlocal invocation_count
            buffer_index = invocation_count % buffer_count
            invocation_count += 1
            launch_alltoallv_gmm(
                activations,
                weights,
                receive_buffers[buffer_index],
                output,
                rank,
                world_size,
                experts_per_rank,
                dispatch_block_m,
                dispatch_block_k,
                gmm_block_m,
                gmm_block_n,
                gmm_block_k,
                tail_guard,
                metadata,
            )
            return output

        if validation_iterations:
            for iteration in range(validation_iterations):
                sample, metadata = prepare_invocation(iteration)
                actual = invoke(sample.activations, metadata)
                expected = _reference_output(
                    sample.activations,
                    weights,
                    sample.token_counts,
                    rank,
                    world_size,
                    experts_per_rank,
                )
                torch.testing.assert_close(actual, expected, rtol=2**-8, atol=2**-8)
                dist.barrier()
                if rank == 0:
                    print(
                        f"AllToAllV-GMM MTE tutorial iteration {iteration + 1}/{validation_iterations} passed",
                        flush=True,
                    )
    finally:
        for receive_buffer in receive_buffers:
            ash.aclshmem_free_tensor(receive_buffer)
        ash.aclshmem_finalize()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
