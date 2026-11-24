# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Grouped-MatMul fused with AllToAllV on the MTE engine (Ascend 910B3).

This file is the correctness path: kernels, host-side metadata, the fp32 golden
and the same-semantics torch_npu reference, plus a fixed-shape self-test under
``__main__``.  For event timing and speedup numbers run the matching script in
python/triton_dist/test/ascend/gmm-alltoallv.py.
"""

import json
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


def _env_flag(name: str, default: str) -> bool:
    """Read a documented 0/1 environment flag and reject any other value."""
    raw = os.environ.get(name, default)
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


class KernelLauncher:
    """Launch one compiled Triton kernel without re-binding arguments per call.

    ``kernel[grid](...)`` re-runs the argument binder, rebuilds the
    specialization key and looks the kernel up in the JIT cache on every launch,
    which costs about 0.13 ms of host time for the kernels used here.  Compiling
    the kernel once and calling the compiled runner skips that work (about
    0.06 ms), and reusing a stream captured once per round also skips the two
    driver queries per launch (about 0.04 ms).

    Only kernels whose runtime arguments are tensors may use this path: a
    scalar argument that happens to equal ``1`` would be baked into the compiled
    code, and the runner cannot re-specialize it.
    """

    def __init__(self, kernel) -> None:
        self._kernel = kernel
        self._compiled = None
        self._signature = None
        self._runners: dict[tuple, object] = {}
        self._stream = None

    def set_stream(self, stream) -> None:
        """Pin the stream captured for the current round."""
        self._stream = stream

    def launch(self, grid: tuple[int, int, int], constexprs: tuple, runtime_args: tuple) -> None:
        """Launch on ``grid`` with the specialization compiled on first use."""
        # Route changes move ``max_row_tiles`` and the grid, so the cached runner
        # has to be keyed by the constexprs as well: they are baked into the
        # compiled code, while the grid is not.
        key = (grid, constexprs)
        runner = self._runners.get(key)
        if runner is None:
            if self._stream is None:
                device = driver.active.get_current_device()
                self._stream = driver.active.get_current_stream(device)
            if self._compiled is None or self._signature != constexprs:
                self._compiled = self._kernel.warmup(*runtime_args, *constexprs, grid=grid)
                self._signature = constexprs
            runner = self._compiled[grid]
            self._runners[key] = runner
        # The Ascend C launcher's arity check counts constexprs too, so forward
        # them along with the runtime args; they are parsed and then ignored.
        runner(*runtime_args, *constexprs, stream=self._stream)


@dataclass
class RoundLaunchers:
    """Compiled runner for the single kernel a round launches.

    The compute/scatter kernel publishes the cross-rank frontier itself, so a
    round costs one launch: the kernel holds the stream it was captured with.
    """

    compute: KernelLauncher

    def start_round(self) -> None:
        """Refresh the cached stream once for the per-round kernel."""
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        self.compute.set_stream(stream)


def effective_tiles(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Return grouped-GEMM tiles, widening parallelism for small shapes."""
    block_m = min(int(os.environ.get("TRITON_GMM_BLOCK_M", "128")), m)
    block_n = min(int(os.environ.get("TRITON_GMM_BLOCK_N", "256")), n)
    block_k = min(int(os.environ.get("TRITON_GMM_BLOCK_K", "256")), k)
    aicore = _aicore_num()
    tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    if aicore > 0 and tiles < 2 * aicore:
        for candidate_m, candidate_n in ((block_m // 2, block_n), (block_m, block_n // 2)):
            if candidate_m < 32 or candidate_n < 64 or candidate_m * candidate_n > 128 * 256:
                continue
            if triton.cdiv(m, candidate_m) * triton.cdiv(n, candidate_n) > tiles:
                block_m, block_n = candidate_m, candidate_n
                break
    return block_m, block_n, block_k


def _resolve_schedule(
    num_m_tiles: int,
    num_n_tiles: int,
    block_m: int,
    block_n: int,
    k: int,
) -> tuple[int, int]:
    """Return the super-tile shape chosen for one shape.

    Both operands cost the same per tile, so whichever one is left hot decides
    the HBM traffic: M-major keeps one activation tile hot and streams the
    weights ``num_m_tiles`` times, N-major keeps one weight tile hot and streams
    the activation ``num_n_tiles`` times.
    """
    activation_bytes = num_m_tiles * block_m * k * 2
    weight_bytes = num_n_tiles * block_n * k * 2
    m_major_traffic = activation_bytes + num_m_tiles * weight_bytes
    n_major_traffic = num_n_tiles * activation_bytes + weight_bytes
    if n_major_traffic < m_major_traffic:
        return max(1, num_m_tiles), 1
    return 1, 1


@triton.jit
def _grouped_matmul_scatter_kernel(
    receive_buffer,
    weights,
    output,
    group_offsets,
    group_tile_offsets,
    token_counts,
    dst_offsets,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_n_tiles: tl.constexpr,
    group_m: tl.constexpr,
    group_n: tl.constexpr,
):
    """Compute one grouped-GEMM tile and scatter it straight to its owners.

    The tile is written directly into the owning source ranks' symmetric output
    buffers, so no intermediate workspace is needed: an expert's rows are the
    source segments in order, and the rows of this tile that fall into a segment
    are stored locally for this rank and remotely (``symm_at`` + ``quiet``) for
    the other source ranks.
    """
    tile_idx = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_m_tiles = tl.load(group_tile_offsets + experts_per_rank)
    total_tiles = num_m_tiles * num_n_tiles
    if group_m > 1 or group_n > 1:
        tiles_per_super: tl.constexpr = group_m * group_n
        super_n_tiles: tl.constexpr = (num_n_tiles + group_n - 1) // group_n
        loop_end = tl.cdiv(num_m_tiles, group_m) * super_n_tiles * tiles_per_super
    else:
        loop_end = total_tiles
    total_experts = world_size * experts_per_rank
    while tile_idx < loop_end:
        if group_m > 1 or group_n > 1:
            super_index = tile_idx // tiles_per_super
            super_local = tile_idx % tiles_per_super
            tile_m = (super_index // super_n_tiles) * group_m + super_local % group_m
            tile_n = (super_index % super_n_tiles) * group_n + super_local // group_m
            tile_active = (tile_m < num_m_tiles) & (tile_n < num_n_tiles)
        else:
            tile_m = tile_idx // num_n_tiles
            tile_n = tile_idx % num_n_tiles
            tile_active = True
        if tile_active:
            expert = 0
            for index in tl.static_range(1, experts_per_rank + 1):
                expert += tl.where(tl.load(group_tile_offsets + index) <= tile_m, 1, 0)
            expert = tl.minimum(expert, experts_per_rank - 1)
            local_tile_m = tile_m - tl.load(group_tile_offsets + expert)
            group_start = tl.load(group_offsets + expert)
            group_end = tl.load(group_offsets + expert + 1)
            global_expert = rank * experts_per_rank + expert
            rows = local_tile_m * block_m + tl.arange(0, block_m)
            columns = tile_n * block_n + tl.arange(0, block_n)
            row_valid = rows[:, None] < group_end - group_start
            column_valid = columns[None, :] < n
            accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k_block in range(tl.cdiv(k, block_k)):
                k_offsets = k_block * block_k + tl.arange(0, block_k)
                lhs_ptrs = receive_buffer + (group_start + rows[:, None]) * k + k_offsets[None, :]
                rhs_ptrs = weights + expert * k * n + k_offsets[:, None] * n + columns[None, :]
                tl.multiple_of(lhs_ptrs, [16, 16])
                tl.multiple_of(rhs_ptrs, [16, 16])
                lhs = tl.load(
                    lhs_ptrs,
                    mask=row_valid & (k_offsets[None, :] < k),
                    other=0.0,
                )
                rhs = tl.load(
                    rhs_ptrs,
                    mask=(k_offsets[:, None] < k) & column_valid,
                    other=0.0,
                )
                accumulator += tl.dot(lhs, rhs)
            values = accumulator.to(output.dtype.element_ty)
            # Store segment by segment: the tile only overlaps the one or two
            # segments that its row range crosses.
            segment_start = 0
            for source_rank in tl.static_range(world_size):
                count = tl.load(token_counts + source_rank * total_experts + global_expert)
                if count > 0:
                    tile_start = local_tile_m * block_m
                    if (segment_start < tile_start + block_m) and (tile_start < segment_start + count):
                        mask = (
                            row_valid
                            & column_valid
                            & (rows[:, None] >= segment_start)
                            & (rows[:, None] < segment_start + count)
                        )
                        destination = tl.load(dst_offsets + source_rank * total_experts + global_expert)
                        destination_ptrs = (
                            destination + rows[:, None] - segment_start
                        ) * n + columns[None, :]
                        if source_rank == rank:
                            tl.store(output + destination_ptrs, values, mask=mask)
                        else:
                            remote_buffer = dl.symm_at(output, source_rank)
                            tl.store(remote_buffer + destination_ptrs, values, mask=mask)
                segment_start += count
        tile_idx += num_programs
    if world_size > 1:
        # Drain this program's remote stores once instead of once per tile, then
        # join the cross-rank frontier.  Every program calls the barrier exactly
        # once; that is the form the project matrix measures as usable inside a
        # MIX kernel that contains tl.dot, and it keeps the round at one launch.
        libshmem_device.quiet()
        libshmem_device.barrier_all()


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
    # This kernel carries scalar runtime arguments, so it keeps the regular
    # launch path: the fast launcher cannot re-specialize a scalar that was
    # baked into the compiled code.
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

    ``valid_rows`` is the row count written by the previous invocation, and
    every row at or beyond it is guaranteed to be zero.  Only a shrinking route
    makes rows dirty, so a fixed route issues no clear launch after the first.
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
        raise ValueError(f"valid rows {valid_rows} exceed output capacity {tail_guard.capacity_rows}")
    if valid_rows < tail_guard.valid_rows:
        _clear_output_rows(output, valid_rows, tail_guard.valid_rows, block_m, block_n)
    tail_guard.valid_rows = valid_rows


def grouped_matmul_scatter(
    receive_buffer,
    weights,
    output,
    group_offsets,
    group_tile_offsets,
    token_counts,
    dst_offsets,
    total_m_tiles,
    rank,
    world_size,
    experts_per_rank,
    k,
    n,
    block_m,
    block_n,
    block_k,
    group_m,
    group_n,
    launcher,
):
    """Launch the fused grouped-GEMM plus result-scatter kernel."""
    num_n_tiles = triton.cdiv(n, block_n)
    total_tiles = total_m_tiles * num_n_tiles
    if group_m > 1 or group_n > 1:
        super_n_tiles = triton.cdiv(num_n_tiles, group_n)
        total_tiles = triton.cdiv(total_m_tiles, group_m) * super_n_tiles * group_m * group_n
    num_programs = max(1, min(_aicore_num(), total_tiles))
    launcher.launch(
        (num_programs, 1, 1),
        (rank, world_size, experts_per_rank, k, n, block_m, block_n, block_k, num_n_tiles,
         group_m, group_n),
        (receive_buffer, weights, output, group_offsets, group_tile_offsets, token_counts, dst_offsets),
    )


def host_store():
    """Return the torchrun TCPStore without submitting an HCCL operation."""
    from torch.distributed.distributed_c10d import _get_default_store

    return _get_default_store()


def build_gmm_metadata(
    token_counts: torch.Tensor,
    rank: int,
    experts_per_rank: int,
    gmm_block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the grouped-GEMM offsets and the number of tiles to run."""
    local_expert_start = rank * experts_per_rank
    group_sizes = token_counts[:, local_expert_start:local_expert_start + experts_per_rank].sum(dim=0)
    group_offsets = torch.empty(experts_per_rank + 1, dtype=torch.int32)
    group_offsets[0] = 0
    group_offsets[1:] = torch.cumsum(group_sizes, dim=0)
    group_tile_offsets = torch.empty(experts_per_rank + 1, dtype=torch.int32)
    group_tile_offsets[0] = 0
    group_tile_offsets[1:] = torch.cumsum((group_sizes + gmm_block_m - 1) // gmm_block_m, dim=0)
    return (
        group_offsets.npu(),
        group_tile_offsets.npu(),
        int(group_tile_offsets[-1].item()),
    )


def build_result_scatter_metadata(
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
    """Build the destination offsets plus the source-side output valid row count.

    Only the destination table stays on the device: the fused kernel derives each
    segment's start inside its expert by accumulating the per-source counts, so
    it needs no workspace-offset table.
    """
    total_experts = world_size * experts_per_rank
    local_expert_start = rank * experts_per_rank
    group_sizes = token_counts[:, local_expert_start:local_expert_start + experts_per_rank].sum(dim=0).tolist()
    ws_offsets = []
    running_le = 0
    for local_expert in range(experts_per_rank):
        running_source = 0
        for source_rank in range(world_size):
            ws_offsets.append(running_le + running_source)
            running_source += int(token_counts[source_rank, local_expert_start + local_expert].item())
        running_le += group_sizes[local_expert]
    dst_offsets = []
    for source_rank in range(world_size):
        running_expert = 0
        for global_expert in range(total_experts):
            dst_offsets.append(running_expert)
            running_expert += int(token_counts[source_rank, global_expert].item())
    return (
        torch.tensor(dst_offsets, dtype=torch.int32).npu(),
        tuple(ws_offsets),
        tuple(dst_offsets),
        int(token_counts[rank].sum().item()),
    )


def launch_gmm_alltoallv(
    receive_buffer: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    gmm_block_m: int,
    gmm_block_n: int,
    gmm_block_k: int,
    gmm_metadata: tuple[torch.Tensor, torch.Tensor, int],
    scatter_metadata: tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int],
    tail_guard: OutputTailGuard,
    launchers: RoundLaunchers,
) -> None:
    """Run one round: a single launch that computes, scatters and joins the frontier."""
    group_offsets, group_tile_offsets, total_m_tiles = gmm_metadata
    dst_offsets, _, _, valid_rows = scatter_metadata
    launchers.start_round()
    # The fused kernel only rewrites the valid prefix, so rows left behind by a
    # longer previous route must be zeroed before this round's result is read.
    _maintain_output_tail(output, valid_rows, tail_guard, gmm_block_m, gmm_block_n)
    local_expert_start = rank * experts_per_rank
    expert_m_tiles = max(
        triton.cdiv(
            int(token_counts[:, local_expert_start + expert].sum().item()), gmm_block_m
        )
        for expert in range(experts_per_rank)
    )
    num_n_tiles = triton.cdiv(output.shape[-1], gmm_block_n)
    group_m, group_n = _resolve_schedule(
        max(1, expert_m_tiles), num_n_tiles, gmm_block_m, gmm_block_n, receive_buffer.shape[1]
    )
    grouped_matmul_scatter(
        receive_buffer,
        weights,
        output,
        group_offsets,
        group_tile_offsets,
        token_counts,
        dst_offsets,
        total_m_tiles,
        rank,
        world_size,
        experts_per_rank,
        receive_buffer.shape[1],
        output.shape[-1],
        gmm_block_m,
        gmm_block_n,
        gmm_block_k,
        group_m,
        group_n,
        launchers.compute,
    )


def symmetric_memory_size(world_size: int, m: int, n: int) -> int:
    """Return aligned ACLSHMEM capacity for the symmetric output buffer."""
    dtype_bytes = torch.tensor([], dtype=torch.float16).element_size()
    required_bytes = world_size * m * n * dtype_bytes
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


def _static_token_counts(m: int, world_size: int, experts_per_rank: int, sparse: bool) -> torch.Tensor:
    """Build the fixed static route for one snapshot."""
    total_experts = world_size * experts_per_rank
    if not sparse:
        counts = torch.full((world_size, total_experts), m // total_experts, dtype=torch.int32)
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


def _expert_ids(m: int, total_experts: int, rank: int, epoch: int, mode: str) -> torch.Tensor:
    """Generate a fixed-seed top-1 route with controllable skew."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ROUTE_SEED + rank * 1000003 + epoch * 10007)
    if mode == "random":
        return torch.randint(total_experts, (m,), generator=generator, dtype=torch.int64)
    if mode == "hot":
        return torch.zeros((m,), dtype=torch.int64)
    if mode == "skewed":
        ids = torch.randint(total_experts, (m,), generator=generator, dtype=torch.int64)
        ids[: (m * 3) // 4] = 0
        return ids
    raise ValueError(f"unsupported dynamic route mode: {mode}")


def _exchange_counts(store, local_counts: torch.Tensor, epoch: int, rank: int, world_size: int) -> torch.Tensor:
    """Exchange CPU count rows through the torchrun TCPStore."""
    key_prefix = f"gmm_alltoallv/route/{epoch}"
    store.set(f"{key_prefix}/{rank}", json.dumps(local_counts.tolist()))
    rows = [json.loads(store.get(f"{key_prefix}/{peer}").decode("utf-8")) for peer in range(world_size)]
    return torch.tensor(rows, dtype=torch.int32)


def prepare_route(
    m: int,
    k: int,
    world_size: int,
    experts_per_rank: int,
    rank: int,
    epoch: int,
    mode: str,
    store,
    sparse: bool,
) -> RouteSample:
    """Create one route epoch and return expert-major activations plus global counts."""
    if mode not in ROUTE_MODES:
        raise ValueError(f"TRITON_ROUTE_MODES must contain only: {', '.join(ROUTE_MODES)}")
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
        local_counts = torch.bincount(expert_ids, minlength=total_experts).to(torch.int32)
        token_counts = _exchange_counts(store, local_counts, epoch, rank, world_size)
    if token_counts.shape != (world_size, total_experts) or torch.any(token_counts < 0):
        raise ValueError("invalid global token-count matrix")
    if not torch.equal(token_counts.sum(dim=1), torch.full((world_size,), m, dtype=torch.int32)):
        raise ValueError("every source rank must route exactly M tokens")
    return RouteSample(activations=activations.npu(), token_counts=token_counts, mode=mode, epoch=epoch)


def build_receive_buffer(
    own_activations: torch.Tensor,
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
) -> torch.Tensor:
    """Assemble this rank's receive buffer (local experts from every source)."""
    gathered = [torch.empty_like(own_activations) for _ in range(world_size)]
    dist.all_gather(gathered, own_activations)
    local_expert_start = rank * experts_per_rank
    chunks = []
    for local_expert in range(experts_per_rank):
        global_expert = local_expert_start + local_expert
        for source_rank in range(world_size):
            start = int(token_counts[source_rank, :global_expert].sum().item())
            count = int(token_counts[source_rank, global_expert].item())
            chunks.append(gathered[source_rank][start:start + count])
    if chunks:
        return torch.cat(chunks, dim=0)
    return torch.empty(
        (0, own_activations.shape[1]), dtype=own_activations.dtype, device=own_activations.device
    )


def golden_fp32(
    own_activations: torch.Tensor,
    full_weights: torch.Tensor,
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
) -> torch.Tensor:
    """Build the AscendC-style fp32 golden for the source-side output layout."""
    total_experts = world_size * experts_per_rank
    counts = token_counts.to(torch.int64)
    padded = torch.zeros(
        (world_size * own_activations.shape[0], full_weights.shape[-1]),
        dtype=torch.float32,
        device=own_activations.device,
    )
    start = 0
    for global_expert in range(total_experts):
        count = int(counts[rank, global_expert].item())
        if count:
            segment = own_activations[start:start + count].float()
            padded[start:start + count] = torch.matmul(segment, full_weights[global_expert].float())
        start += count
    return padded


def precision_rtol(k: int) -> float:
    """Return the CatCCOS reference tolerance for a float16 reduction depth."""
    return 2 ** (-8) if k < 2048 else 2 ** (-7)


@dataclass(frozen=True)
class BaselinePlan:
    """Host tables for the same-semantics torch_npu performance baseline."""

    gemm_sizes: tuple[int, ...]
    payload_slices: tuple[tuple[tuple[int, int], ...], ...]
    input_split_sizes: tuple[int, ...]
    receive_split_sizes: tuple[int, ...]
    placement: tuple[tuple[int, int, int], ...]


def build_baseline_plan(
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    ws_offsets_host: tuple[int, ...],
    dst_offsets_host: tuple[int, ...],
) -> BaselinePlan:
    """Build the GEMM, exchange and placement tables of the torch baseline."""
    total_experts = world_size * experts_per_rank
    local_expert_start = rank * experts_per_rank
    gemm_sizes = tuple(
        int(token_counts[:, local_expert_start + local_expert].sum().item())
        for local_expert in range(experts_per_rank)
    )
    payload_slices = []
    input_split_sizes = []
    for source_rank in range(world_size):
        slices = []
        total = 0
        for local_expert in range(experts_per_rank):
            count = int(token_counts[source_rank, local_expert_start + local_expert].item())
            start = ws_offsets_host[local_expert * world_size + source_rank]
            slices.append((start, count))
            total += count
        payload_slices.append(tuple(slices))
        input_split_sizes.append(total)
    receive_split_sizes = [
        sum(
            int(token_counts[rank, writer * experts_per_rank + local_expert].item())
            for local_expert in range(experts_per_rank)
        )
        for writer in range(world_size)
    ]
    placement = []
    cursor = 0
    for writer in range(world_size):
        for local_expert in range(experts_per_rank):
            global_expert = writer * experts_per_rank + local_expert
            count = int(token_counts[rank, global_expert].item())
            placement.append((dst_offsets_host[rank * total_experts + global_expert], cursor, count))
            cursor += count
    return BaselinePlan(
        gemm_sizes=gemm_sizes,
        payload_slices=tuple(payload_slices),
        input_split_sizes=tuple(input_split_sizes),
        receive_split_sizes=tuple(receive_split_sizes),
        placement=tuple(placement),
    )


def baseline_output(
    receive_buffer: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    plan: BaselinePlan,
) -> torch.Tensor:
    """Run the same-semantics torch_npu baseline: GEMM, result exchange, placement."""
    columns = output.shape[-1]
    x_list = []
    running = 0
    for size in plan.gemm_sizes:
        x_list.append(receive_buffer[running:running + size])
        running += size
    local_result = torch_npu.npu_grouped_matmul(
        x=x_list,
        weight=[weights[local_expert] for local_expert in range(len(plan.gemm_sizes))],
        group_type=0,
        split_item=2,
    )[0]
    payloads = []
    for source_slices in plan.payload_slices:
        parts = [local_result[start:start + count] for start, count in source_slices if count]
        if parts:
            payloads.append(torch.cat(parts, dim=0))
        else:
            payloads.append(torch.empty((0, columns), dtype=local_result.dtype, device=local_result.device))
    packed = torch.cat(payloads, dim=0)
    incoming = torch.empty(
        (sum(plan.receive_split_sizes), columns), dtype=local_result.dtype, device=local_result.device
    )
    if packed.shape[0] or incoming.shape[0]:
        dist.all_to_all_single(
            incoming,
            packed,
            output_split_sizes=list(plan.receive_split_sizes),
            input_split_sizes=list(plan.input_split_sizes),
        )
    for dst_offset, incoming_offset, count in plan.placement:
        if count:
            output[dst_offset:dst_offset + count].copy_(incoming[incoming_offset:incoming_offset + count])
    return output


def assert_precision(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float, eb_threshold: float = 2 ** -10
) -> None:
    """Apply the CatCCOS elementwise tolerance plus error-bias acceptance rule."""
    actual_f32 = actual.float().cpu()
    expected_f32 = expected.float().cpu()
    difference = actual_f32 - expected_f32
    scale = torch.maximum(torch.ones_like(expected_f32), expected_f32.abs())
    tolerance = difference.abs() - rtol * scale
    exceeded = int((tolerance > 0).sum().item())
    if exceeded:
        index = int(tolerance.argmax().item())
        row, column = divmod(index, expected_f32.shape[-1])
        raise AssertionError(
            f"precision failed: {exceeded}/{tolerance.numel()} elements exceed rtol={rtol:g}; "
            f"worst at ({row}, {column}) actual={actual_f32[row, column]:.6g} "
            f"expected={expected_f32[row, column]:.6g}"
        )
    error_bias = float((difference / scale).mean().abs().item())
    if error_bias > eb_threshold:
        raise AssertionError(f"precision failed: error bias {error_bias:.6g} exceeds {eb_threshold:g}")


def run() -> None:
    """Initialize distributed state and run the fixed-seed self-test."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    store = host_store()
    m = int(os.environ.get("TRITON_M", "4096"))
    k = int(os.environ.get("TRITON_K", "4096"))
    n = int(os.environ.get("TRITON_N", "4096"))
    experts_per_rank = int(os.environ.get("TRITON_EXPERTS_PER_RANK", "2"))
    gmm_block_m, gmm_block_n, gmm_block_k = effective_tiles(m, k, n)
    sparse = _env_flag("TRITON_SPARSE", "1")
    route_modes = tuple(
        value.strip() for value in os.environ.get("TRITON_ROUTE_MODES", "random").split(",") if value.strip()
    )
    if len(route_modes) != 1:
        raise ValueError("TRITON_ROUTE_MODES must contain exactly one mode when running the tutorial script")
    route_mode = route_modes[0]
    if min(m, k, n, experts_per_rank, gmm_block_m, gmm_block_n, gmm_block_k) <= 0:
        raise ValueError("shape and block-size variables must be positive")
    local_mem_size = symmetric_memory_size(world_size, m, n)
    if rank == 0:
        print(f"GMM_ALLTOALLV_LOCAL_MEM_SIZE_BYTES={local_mem_size}", flush=True)
        print(
            "CONFIG "
            f"engine=mte ranks={world_size} shape=m{m}_k{k}_n{n} "
            f"blocks={gmm_block_m}x{gmm_block_n}x{gmm_block_k} "
            f"route={route_mode} sparse={int(sparse)} experts_per_rank={experts_per_rank} "
            f"aicore_num={_aicore_num()} vector_cores={_aivector_core_num()}",
            flush=True,
        )
    init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.MTE)

    seed = 20260811
    try:
        torch.manual_seed(seed + rank)
        weights = torch.randn((experts_per_rank, k, n), dtype=torch.float16).npu()
        # Gather every rank's local weights for the source-side golden reference.
        full_weights_list = [torch.empty_like(weights) for _ in range(world_size)]
        dist.all_gather(full_weights_list, weights)
        full_weights = torch.cat(full_weights_list, dim=0)
        output = ash.aclshmem_create_tensor([world_size * m, n], dtype=torch.float16, device_id=local_rank)
        tail_guard = OutputTailGuard.create(world_size * m)
        launchers = RoundLaunchers(compute=KernelLauncher(_grouped_matmul_scatter_kernel))

        def prepare_invocation(
            epoch: int,
        ) -> tuple[
            RouteSample,
            torch.Tensor,
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor, int],
            tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int],
            BaselinePlan,
        ]:
            sample = prepare_route(
                m, k, world_size, experts_per_rank, rank, epoch, route_mode, store, sparse
            )
            device_token_counts = sample.token_counts.npu()
            recv = build_receive_buffer(
                sample.activations, sample.token_counts, rank, world_size, experts_per_rank
            )
            gmm_metadata = build_gmm_metadata(
                sample.token_counts, rank, experts_per_rank, gmm_block_m
            )
            scatter_metadata = build_result_scatter_metadata(
                sample.token_counts, rank, world_size, experts_per_rank
            )
            baseline_plan = build_baseline_plan(
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
                scatter_metadata[1],
                scatter_metadata[2],
            )
            return sample, recv, device_token_counts, gmm_metadata, scatter_metadata, baseline_plan

        def invoke(
            recv: torch.Tensor,
            gmm_metadata: tuple[torch.Tensor, torch.Tensor, int],
            scatter_metadata: tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int],
            token_counts: torch.Tensor,
        ) -> torch.Tensor:
            launch_gmm_alltoallv(
                recv,
                weights,
                output,
                token_counts,
                rank,
                world_size,
                experts_per_rank,
                gmm_block_m,
                gmm_block_n,
                gmm_block_k,
                gmm_metadata,
                scatter_metadata,
                tail_guard,
                launchers,
            )
            return output

        (
            sample,
            recv,
            device_token_counts,
            gmm_metadata,
            scatter_metadata,
            baseline_plan,
        ) = prepare_invocation(0)
        actual = invoke(recv, gmm_metadata, scatter_metadata, device_token_counts)
        expected = golden_fp32(
            sample.activations, full_weights, sample.token_counts, rank, world_size, experts_per_rank
        )
        assert_precision(actual, expected, precision_rtol(k))
        # Guard: the same-semantics baseline must reproduce the fp32 golden.
        baseline_actual = baseline_output(recv, weights, output, baseline_plan)
        torch.npu.synchronize()
        assert_precision(baseline_actual, expected, precision_rtol(k))
        dist.barrier()
        if rank == 0:
            print("GMM-AllToAllV MTE tutorial passed", flush=True)

    finally:
        ash.aclshmem_free_tensor(output)
        ash.aclshmem_finalize()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
