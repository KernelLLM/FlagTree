# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

# Run from the repository root (two Ascend950 devices):
# OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 tutorials/ascend/alltoallv-gmm-udma.py

"""Fused AllToAllV-GMM using Ascend950 UDMA communication.

TRITON_M/K/N select one example (defaults: 4096/4096/4096). One mixed kernel handles
remote dispatch, local copy, ready waits and grouped GEMM. The caller initializes
the unused output tail once for each fixed route.
The caller resets signals between globally completed invocations. GEMM uses
expert-local gemm_swizzle2d scheduling with a default offset of 7.
Batch verification and event timing live in test/ascend/alltoallv-gmm-udma.py.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.ascend import libaclshmem_device
from triton_dist.language.extra.ascend.algorithm import gemm_swizzle2d

import shmem as ash


# Ascend 950 UDMA uses uint64 notify/wait slots. Eight elements preserve one 64-byte cache line per source rank and avoid signal false sharing.
SIGNAL_DTYPE = torch.uint64
SIGNAL_STRIDE = 8
SIGNAL_ZERO_BLOCK = 128


def direct_grouped_input_offsets(
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
) -> tuple[list[int], list[int]]:
    """Build source and target offsets for direct expert-major UDMA writes."""
    total_experts = world_size * experts_per_rank
    local_counts = token_counts[rank]
    source_offsets = []
    running = 0
    for count in local_counts.tolist():
        source_offsets.append(running)
        running += int(count)
    direct_offsets = []
    for global_expert in range(total_experts):
        destination_rank = global_expert // experts_per_rank
        local_expert = global_expert % experts_per_rank
        expert_start = destination_rank * experts_per_rank
        expert_prefix = sum(
            int(token_counts[source_rank, expert_start + previous].item())
            for previous in range(local_expert)
            for source_rank in range(world_size)
        )
        source_prefix = sum(
            int(token_counts[source_rank, global_expert].item())
            for source_rank in range(rank)
        )
        direct_offsets.append(expert_prefix + source_prefix)
    return source_offsets, direct_offsets


def symmetric_memory_size(
    world_size: int,
    m: int,
    k: int,
    buffer_count: int = 1,
    extra_bytes: int = 0,
) -> int:
    """Return aligned ACLSHMEM capacity for all symmetric allocations."""
    dtype_bytes = torch.tensor([], dtype=torch.float16).element_size()
    required_bytes = buffer_count * world_size * m * k * dtype_bytes + extra_bytes
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


@triton.jit
def _zero_ready_signals_kernel(
    ready_signals,
    signal_elements: tl.constexpr,
    block: tl.constexpr,
):
    """Clear ACLSHMEM signal slots without invoking an unsupported ACLNN op."""
    offsets = tl.program_id(axis=0) * block + tl.arange(0, block)
    tl.store(ready_signals + offsets, 0, mask=offsets < signal_elements)


def _clear_ready_signals(ready_signals: torch.Tensor) -> None:
    """Clear the symmetric ready slots through a pure Vector Triton kernel."""
    signal_elements = ready_signals.numel()
    _zero_ready_signals_kernel[(triton.cdiv(signal_elements, SIGNAL_ZERO_BLOCK),)](
        ready_signals,
        signal_elements=signal_elements,
        block=SIGNAL_ZERO_BLOCK,
    )


UDMA_MAX_BYTES = 256 * 1024 * 1024


@triton_dist.jit()
def _alltoallv_gmm_single_kernel(
    activations,
    grouped_input,
    weights,
    output,
    expert_source_offsets,
    expert_direct_offsets,
    expert_token_counts,
    group_offsets,
    group_tile_offsets,
    ready_signals,
    rank_id: tl.constexpr,
    world_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    signal_stride: tl.constexpr,
    element_bytes: tl.constexpr,
    dispatch_block_m: tl.constexpr,
    dispatch_block_k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_n_tiles: tl.constexpr,
    max_put_elements: tl.constexpr,
    group_m: tl.constexpr,
):
    """Dispatch/copy -> acquire every source -> persistent grouped GEMM.

    One AIV1 program owns each target, including the local target. A source
    slot is set once, even for an empty payload. Waiting does not clear the
    slots, so multiple AICs can wait on the same slot. The caller must finish
    the previous round on all ranks before resetting or reusing
    any signal/input storage.
    """
    program = tl.program_id(0)
    num_programs = tl.num_programs(0)
    subblock = sub_vec_id()

    if subblock == 1:
        # Striding also covers world_size > grid_size without shared issuers. All sends precede all waits, so an empty-output rank still produces.
        for destination_rank in range(program, world_size, num_programs):
            for local_expert in tl.static_range(experts_per_rank):
                global_expert = destination_rank * experts_per_rank + local_expert
                rows = tl.load(expert_token_counts + global_expert)
                source_offset = tl.load(expert_source_offsets + global_expert)
                direct_offset = tl.load(expert_direct_offsets + global_expert)
                if rows > 0:
                    if destination_rank != rank_id:
                        # Bound individual UDMA requests, retaining one issuer per peer even when a large expert needs many requests.
                        for begin in range(0, rows * k, max_put_elements):
                            count = tl.minimum(rows * k - begin, max_put_elements)
                            libaclshmem_device.putmem(
                                grouped_input + direct_offset * k + begin,
                                activations + source_offset * k + begin,
                                count * element_bytes,
                                destination_rank,
                            )
                    else:
                        # Local memory copies must use MTE, not UDMA to self.
                        for m_begin in range(0, rows, dispatch_block_m):
                            copy_rows = m_begin + tl.arange(0, dispatch_block_m)
                            for k_begin in range(0, k, dispatch_block_k):
                                copy_columns = k_begin + tl.arange(0, dispatch_block_k)
                                copy_offsets = copy_rows[:, None] * k + copy_columns[None, :]
                                copy_mask = (copy_rows[:, None] < rows) & (copy_columns[None, :] < k)
                                values = tl.load(
                                    activations + source_offset * k + copy_offsets,
                                    mask=copy_mask, other=0.0,
                                )
                                tl.store(
                                    grouped_input + direct_offset * k + copy_offsets,
                                    values, mask=copy_mask,
                                )
            # Order data transfers before publishing readiness for this peer.
            libaclshmem_device.quiet_lowlevel(
                transport=libaclshmem_device.ACLSHMEMTransport.MTE
            )
            dl.notify(
                ready_signals + rank_id * signal_stride,
                destination_rank.to(tl.int32),
                1, sig_op="set", comm_scope="intra_node",
            )

    # Ascend multi-barrier wait visits slots at 64-byte intervals. Every source publishes a slot, including self and empty sends, so a single wait covers all sources. Bind the acquire token to the GEMM input.
    tl.static_assert(signal_stride == 8, "uint64 ready slots must be 64 bytes apart")
    token = dl.wait(
        ready_signals, world_size, "gpu", "acquire", waitValue=1,
    )
    ready_grouped_input = dl.consume_token(grouped_input, token)

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
        expert_tile_start = tl.load(group_tile_offsets + expert)
        local_task = tile_idx - expert_tile_start * num_n_tiles
        group_start = tl.load(group_offsets + expert)
        group_end = tl.load(group_offsets + expert + 1)
        if group_m == 0:
            local_tile_m = tile_m - expert_tile_start
        else:
            local_tile_m, tile_n = gemm_swizzle2d(
                local_task,
                group_end - group_start,
                n,
                block_m,
                block_n,
                swizzle_offset=group_m,
                swizzle_direction=0,
            )
        rows = local_tile_m * block_m + tl.arange(0, block_m)
        columns = tile_n * block_n + tl.arange(0, block_n)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_tile in range(tl.cdiv(k, block_k)):
            k_offsets = k_tile * block_k + tl.arange(0, block_k)
            lhs_ptrs = (
                ready_grouped_input
                + (group_start + rows[:, None]).to(tl.int64) * k
                + k_offsets[None, :]
            )
            rhs_ptrs = (
                weights + expert * k * n + k_offsets[:, None] * n + columns[None, :]
            )
            lhs = tl.load(
                lhs_ptrs,
                mask=(rows[:, None] < group_end - group_start) & (k_offsets[None, :] < k),
                other=0.0,
            )
            rhs = tl.load(
                rhs_ptrs,
                mask=(k_offsets[:, None] < k) & (columns[None, :] < n),
                other=0.0,
            )
            accumulator += tl.dot(lhs, rhs)
        tl.store(
            output + (group_start + rows[:, None]).to(tl.int64) * n + columns[None, :],
            accumulator.to(output.dtype.element_ty),
            mask=(rows[:, None] < group_end - group_start) & (columns[None, :] < n),
        )
        tile_idx += num_programs


def _build_gmm_metadata(token_counts, rank, experts_per_rank, block_m):
    """Build only the row/tile boundaries consumed by the fused kernel."""
    start = rank * experts_per_rank
    sizes = token_counts[:, start : start + experts_per_rank].sum(dim=0)
    offsets = torch.zeros(experts_per_rank + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(sizes, dim=0)
    tile_offsets = torch.zeros_like(offsets)
    tile_offsets[1:] = torch.cumsum((sizes + block_m - 1) // block_m, dim=0)
    return GmmMetadata(offsets.npu(), tile_offsets.npu(), int(offsets[-1].item()))


def validate_against_torch_npu(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    """Compare outputs and gather failure flags through HCCL before raising on all ranks."""
    error = ""
    try:
        torch.testing.assert_close(actual, expected, rtol=2**-8, atol=2**-8)
    except AssertionError as exception:
        error = str(exception)
    failed = torch.tensor([int(bool(error))], dtype=torch.int32, device=actual.device)
    flags = [torch.empty_like(failed) for _ in range(world_size)]
    dist.all_gather(flags, failed)
    failed_ranks = [peer for peer, flag in enumerate(torch.cat(flags).cpu().tolist()) if flag]
    if failed_ranks:
        for peer in failed_ranks:
            if rank == peer:
                print(f"UDMA_VALIDATION_MISMATCH rank{rank}: {error}", flush=True)
            dist.barrier()
        raise AssertionError(
            f"Triton output does not match torch_npu on rank(s): {failed_ranks}"
        )


@dataclass(frozen=True)
class DispatchRoutes:
    """Device metadata for direct expert-major UDMA dispatch."""

    expert_source_offsets: torch.Tensor
    expert_direct_offsets: torch.Tensor
    expert_token_counts: torch.Tensor


@dataclass(frozen=True)
class GmmMetadata:
    """Expert row boundaries and the persistent GEMM task boundaries."""

    group_offsets: torch.Tensor
    group_tile_offsets: torch.Tensor
    valid_rows: int


def _build_dispatch_routes(host_token_counts, rank, world_size, experts_per_rank):
    """Build the local sender's route once and reuse it for every invocation."""
    source, direct = direct_grouped_input_offsets(
        host_token_counts, rank, world_size, experts_per_rank
    )
    return DispatchRoutes(
        torch.tensor(source, dtype=torch.int64).npu(),
        torch.tensor(direct, dtype=torch.int64).npu(),
        host_token_counts[rank].to(dtype=torch.int32).npu(),
    )


def launch_alltoallv_gmm_udma(
    activations: torch.Tensor,
    weights: torch.Tensor,
    ready_signals: torch.Tensor,
    grouped_input: torch.Tensor,
    output: torch.Tensor,
    rank: int,
    world_size: int,
    dispatch_block_m: int,
    dispatch_block_k: int,
    gmm_block_m: int,
    gmm_block_n: int,
    gmm_block_k: int,
    routes: DispatchRoutes,
    metadata: GmmMetadata,
    ncore: int,
    swizzle_offset: int = 7,
) -> None:
    """Launch fused dispatch, acquire and grouped GEMM.

    The caller must finish work on every rank before resetting ready signals.
    Route and GEMM metadata are prepared once, outside the timed invocation.
    The caller must zero output[metadata.valid_rows:] once when preparing each route.
    """
    if activations.dtype != torch.float16 or weights.dtype != torch.float16:
        raise TypeError(
            "the UDMA tutorial currently supports float16 activations and weights only"
        )
    valid_rows = metadata.valid_rows
    if not 0 <= valid_rows <= output.shape[0]:
        raise ValueError(
            f"invalid valid_rows={valid_rows}, capacity={output.shape[0]}"
        )
    _, k = activations.shape
    n = output.shape[-1]
    _alltoallv_gmm_single_kernel[(ncore,)](
        activations,
        grouped_input,
        weights,
        output,
        routes.expert_source_offsets,
        routes.expert_direct_offsets,
        routes.expert_token_counts,
        metadata.group_offsets,
        metadata.group_tile_offsets,
        ready_signals,
        rank_id=rank,
        world_size=world_size,
        experts_per_rank=routes.expert_token_counts.numel() // world_size,
        k=k,
        n=n,
        signal_stride=SIGNAL_STRIDE,
        element_bytes=activations.element_size(),
        dispatch_block_m=dispatch_block_m,
        dispatch_block_k=dispatch_block_k,
        block_m=gmm_block_m,
        block_n=gmm_block_n,
        block_k=gmm_block_k,
        num_n_tiles=triton.cdiv(n, gmm_block_n),
        max_put_elements=UDMA_MAX_BYTES // activations.element_size(),
        group_m=swizzle_offset,
        enable_dynamic_cv_pipeline=False,
    )


def require_udma_device(local_rank: int) -> None:
    """Reject non-950 systems instead of silently running an unsupported path."""
    name = os.environ.get("ASCEND_SOC_VERSION", "")
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is not None:
        name = str(get_device_name(local_rank)) or name
    if "950" not in name:
        raise RuntimeError(
            "AllToAllV-GMM UDMA requires Ascend 950; detected "
            f"{name or 'an unknown SoC'}. Use alltoallv-gmm.py for the MTE path."
        )


def _prepare_inputs(args, case, route, rank, world_size):
    m, k, n = map(int, case)
    torch.manual_seed(20260811 + rank)
    weights = torch.randn((args.experts_per_rank, k, n), dtype=torch.float16).npu()
    sample = prepare_route(
        m,
        k,
        world_size,
        args.experts_per_rank,
        rank,
        args.epoch,
        route,
        bool(args.sparse),
    )
    return sample, weights


def _prepare_dispatch(args, sample, rank, world_size, tiles):
    dispatch = _build_dispatch_routes(
        sample.token_counts, rank, world_size, args.experts_per_rank
    )
    metadata = _build_gmm_metadata(
        sample.token_counts, rank, args.experts_per_rank, tiles[2]
    )
    return dispatch, metadata


def shmem_init(local_rank, rank, world_size, max_elements):
    """Initialize SHMEM and allocate the reusable input/signal buffers once."""
    signal_elements = world_size * SIGNAL_STRIDE
    local_mem_size = symmetric_memory_size(
        1, max_elements, 1, extra_bytes=signal_elements * 8
    )
    if rank == 0:
        print(
            f"SHMEM_INIT_ONCE bytes={local_mem_size} input_buffer_elements={max_elements}",
            flush=True,
        )
    init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.UDMA)
    input_buffer = None
    try:
        input_buffer = ash.aclshmem_create_tensor(
            [max_elements], dtype=torch.float16, device_id=local_rank
        )
        ready_signals = ash.aclshmem_create_tensor(
            [signal_elements], dtype=SIGNAL_DTYPE, device_id=local_rank
        )
    except Exception:
        if input_buffer is not None:
            ash.aclshmem_free_tensor(input_buffer)
        ash.aclshmem_finalize()
        raise
    return input_buffer, ready_signals


def case_options():
    """Read the route settings shared by the tutorial and its test script."""
    options = SimpleNamespace(
        experts_per_rank=int(os.environ.get("TRITON_EXPERTS_PER_RANK", "2")),
        epoch=int(os.environ.get("TRITON_BENCHMARK_EPOCH", "1")),
        sparse=int(os.environ.get("TRITON_SPARSE", "1")),
        routes=tuple(
            dict.fromkeys(
                mode.strip()
                for mode in os.environ.get("TRITON_ROUTE_MODES", "random").split(",")
                if mode.strip()
            )
        ),
    )
    if options.experts_per_rank <= 0 or options.epoch < 0:
        raise ValueError("expert count must be positive and epoch must be non-negative")
    if options.sparse not in (0, 1):
        raise ValueError("TRITON_SPARSE must be 0 or 1")
    if not options.routes or any(mode not in ROUTE_MODES for mode in options.routes):
        raise ValueError(f"TRITON_ROUTE_MODES must contain modes from {ROUTE_MODES}")
    return options


def run():
    """Run one shape, with HCCL reference work completed before UDMA starts."""
    options = case_options()
    if len(options.routes) != 1:
        raise ValueError("the tutorial requires exactly one TRITON_ROUTE_MODES value")
    case = tuple(
        int(os.environ.get(name, default))
        for name, default in (
            ("TRITON_M", "4096"),
            ("TRITON_K", "4096"),
            ("TRITON_N", "4096"),
        )
    )
    if min(case) <= 0:
        raise ValueError("M/K/N must be positive")
    # Dispatch M/K and GEMM M/N/K tiles for the 4096/4096/4096 example.
    tiles = (128, 256, 128, 256, 128)
    m, k, n = case
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    ncore = driver.active.utils.get_aicore_num()
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    require_udma_device(local_rank)
    dist.init_process_group("hccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        dist.barrier()
        sample, weights = _prepare_inputs(
            options, case, options.routes[0], rank, world_size
        )
        dispatch, metadata = _prepare_dispatch(options, sample, rank, world_size, tiles)
        expected = _reference_output(
            sample.activations,
            weights,
            sample.token_counts,
            rank,
            world_size,
            options.experts_per_rank,
        )
        torch.npu.synchronize()
        dist.barrier()
        input_buffer, ready_signals = shmem_init(
            local_rank, rank, world_size, world_size * m * k
        )
        try:
            grouped_input = input_buffer.view(world_size * m, k)
            output = torch.empty((world_size * m, n), dtype=torch.float16).npu()
            output[metadata.valid_rows:].zero_()
            _clear_ready_signals(ready_signals)
            torch.npu.synchronize()
            dist.barrier()
            launch_alltoallv_gmm_udma(
                sample.activations,
                weights,
                ready_signals,
                grouped_input,
                output,
                rank,
                world_size,
                *tiles,
                routes=dispatch,
                metadata=metadata,
                ncore=ncore,
                swizzle_offset=7,
            )
            torch.npu.synchronize()
            validate_against_torch_npu(output, expected, rank, world_size)
            dist.barrier()
            if rank == 0:
                print(
                    f"AllToAllV-GMM single-kernel verification passed: M={m} K={k} N={n}",
                    flush=True,
                )
        finally:
            torch.npu.synchronize()
            ash.aclshmem_free_tensor(ready_signals)
            ash.aclshmem_free_tensor(input_buffer)
            ash.aclshmem_finalize()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
