# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Grouped-MatMul fused with AllToAllV on the UDMA engine (Ascend 950).

This file is the correctness path: kernels, host-side metadata, the fp32 golden
and the same-semantics torch_npu reference, plus a fixed-shape self-test under
``__main__``.  For event timing and speedup numbers run the matching script in
python/triton_dist/test/ascend/gmm-alltoallv-udma.py.
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend import libaclshmem_device
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem

import shmem as ash


# Ascend 950 UDMA uses uint64 notify/wait slots. Eight elements preserve one
# 64-byte cache line per source rank and avoid signal false sharing.
def _env_bytes(name: str, default: int) -> int:
    """Read a byte limit from the environment and keep it int32-safe."""
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return min(value, 1 << 30)


# Low-level putmem is only safe up to this many bytes per call: 02 chunks above
# 256 MB, and 06 records that a single put around 512 MB corrupts data and drops
# into a ~20x slow path (an int32 offset/length limit in the UDMA engine).  One
# put therefore ships at most `min(PUT_CHUNK_BYTES, MAX_PUT_BYTES)` bytes, and a
# segment larger than that is split into row-aligned pieces; lower PUT_CHUNK_BYTES
# (for example to 2 MB, 06's proven choice for multi-hundred-MB slots) if a
# slotted shape ever shows the slow path.
MAX_PUT_BYTES = _env_bytes("TRITON_UDMA_PUT_MAX_BYTES", 256 * 1024 * 1024)
PUT_CHUNK_BYTES = _env_bytes("TRITON_UDMA_PUT_CHUNK_BYTES", MAX_PUT_BYTES)


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

    def launch(
        self,
        grid: tuple[int, int, int],
        constexprs: tuple,
        runtime_args: tuple,
        options: dict | None = None,
    ) -> None:
        """Launch on ``grid`` with the specialization compiled on first use."""
        # Route changes move the tile constants and the grid, so the cached
        # runner has to be keyed by them as well: constexprs and options are
        # baked into the compiled code, while the grid is not.
        signature = (constexprs, tuple(sorted(dict(options or {}).items())))
        key = (grid, signature)
        runner = self._runners.get(key)
        if runner is None:
            if self._stream is None:
                device = driver.active.get_current_device()
                self._stream = driver.active.get_current_stream(device)
            if self._compiled is None or self._signature != signature:
                self._compiled = self._kernel.warmup(
                    *runtime_args, *constexprs, grid=grid, **dict(options or {})
                )
                self._signature = signature
            runner = self._compiled[grid]
            self._runners[key] = runner
        # The Ascend C launcher's arity check counts constexprs too, so forward
        # them along with the runtime args; they are parsed and then ignored.
        runner(*runtime_args, *constexprs, stream=self._stream)


@dataclass
class RoundLaunchers:
    """Compiled runner of the one-shot UDMA kernel launched per round."""

    oneshot: KernelLauncher

    def start_round(self) -> None:
        """Refresh the cached stream once for every per-round kernel."""
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        self.oneshot.set_stream(stream)


def _effective_tiles(
    m: int, k: int, n: int,
) -> tuple[int, int, int]:
    """Return grouped-GEMM tiles, widening parallelism for small shapes."""
    gmm_block_m = min(int(os.environ.get("TRITON_GMM_BLOCK_M", "128")), m)
    gmm_block_n = min(int(os.environ.get("TRITON_GMM_BLOCK_N", "256")), n)
    gmm_block_k = min(int(os.environ.get("TRITON_GMM_BLOCK_K", "256")), k)
    aicore = _aicore_num()
    tiles = triton.cdiv(m, gmm_block_m) * triton.cdiv(n, gmm_block_n)
    if aicore > 0 and tiles < 2 * aicore:
        for candidate_m, candidate_n in (
            (gmm_block_m // 2, gmm_block_n),
            (gmm_block_m, gmm_block_n // 2),
        ):
            if candidate_m < 32 or candidate_n < 64 or candidate_m * candidate_n > 128 * 256:
                continue
            if triton.cdiv(m, candidate_m) * triton.cdiv(n, candidate_n) > tiles:
                gmm_block_m, gmm_block_n = candidate_m, candidate_n
                break
    return gmm_block_m, gmm_block_n, gmm_block_k


def symmetric_memory_size(
    world_size: int,
    m: int,
    n: int,
) -> int:
    """Return aligned ACLSHMEM capacity for the symmetric output and signals."""
    dtype_bytes = torch.tensor([], dtype=torch.float16).element_size()
    output_bytes = world_size * m * n * dtype_bytes
    pool_bytes = (
        (output_bytes + SHMEM_PAGE_SIZE - 1) // SHMEM_PAGE_SIZE
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


def precision_error(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float, eb_threshold: float = 2 ** -10
) -> str:
    """Return an empty string on success, or the CatCCOS-style failure detail."""
    actual_f32 = actual.float().cpu()
    expected_f32 = expected.float().cpu()
    difference = actual_f32 - expected_f32
    scale = torch.maximum(torch.ones_like(expected_f32), expected_f32.abs())
    tolerance = difference.abs() - rtol * scale
    exceeded = int((tolerance > 0).sum().item())
    if exceeded:
        index = int(tolerance.argmax().item())
        row, column = divmod(index, expected_f32.shape[-1])
        return (
            f"{exceeded}/{tolerance.numel()} elements exceed rtol={rtol:g}; "
            f"worst at ({row}, {column}) actual={actual_f32[row, column]:.6g} "
            f"expected={expected_f32[row, column]:.6g}"
        )
    error_bias = float((difference / scale).mean().abs().item())
    if error_bias > eb_threshold:
        return f"error bias {error_bias:.6g} exceeds {eb_threshold:g}"
    return ""


@dataclass(frozen=True)
class BaselinePlan:
    """Host tables for the same-semantics torch_npu performance baseline."""

    gemm_sizes: tuple[int, ...]
    payload_slices: tuple[tuple[tuple[int, int], ...], ...]
    input_split_sizes: tuple[int, ...]
    receive_split_sizes: tuple[int, ...]
    placement: tuple[tuple[int, int, int], ...]


def _baseline_offsets(token_counts, rank: int, world_size: int, experts_per_rank: int):
    """Build host offsets needed only by the torch_npu reference."""
    total_experts = world_size * experts_per_rank
    local_expert_start = rank * experts_per_rank
    ws_offsets = []
    running_local = 0
    for local_expert in range(experts_per_rank):
        running_source = 0
        for source_rank in range(world_size):
            ws_offsets.append(running_local + running_source)
            running_source += int(token_counts[source_rank, local_expert_start + local_expert])
        running_local += running_source
    dst_offsets = []
    for source_rank in range(world_size):
        running_expert = 0
        for global_expert in range(total_experts):
            dst_offsets.append(running_expert)
            running_expert += int(token_counts[source_rank, global_expert])
    return tuple(ws_offsets), tuple(dst_offsets)


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


def host_store():
    """Return the torchrun TCPStore without submitting an HCCL operation."""
    from torch.distributed.distributed_c10d import _get_default_store

    return _get_default_store()


@dataclass
class OutputTailGuard:
    """Keep the unused output tail zero while avoiding per-round clear launches.

    ``valid_rows`` is the row count written by the previous invocation, and
    every row at or beyond it is guaranteed to be zero.  Only a shrinking route
    makes rows dirty, so a fixed route needs clearing only during output preparation.
    """

    capacity_rows: int
    valid_rows: int

    @classmethod
    def create(cls, capacity_rows: int) -> "OutputTailGuard":
        """Treat the initial buffer as dirty until prepare() clears its unused rows."""
        return cls(capacity_rows=capacity_rows, valid_rows=capacity_rows)

    def prepare(self, output: torch.Tensor, valid_rows: int) -> None:
        """Clear a new or shrinking output tail on the current stream.

        Benchmarks call this before warmup/events. Calls with an unchanged
        route issue no device work. Previous uses must have completed before
        reusing the output for a different route.
        """
        if valid_rows < self.valid_rows:
            output[valid_rows:].zero_()
        self.valid_rows = valid_rows


@triton.jit
def _gmm_alltoallv_oneshot_kernel(
    receive_buffer,
    weights,
    output,
    workspace,
    group_offsets,
    group_tile_offsets,
    ws_offsets,
    dst_offsets,
    token_counts,
    local_tile_offsets,
    local_ws_offsets,
    local_dst_offsets,
    local_counts,
    rank_id: tl.constexpr,
    world_size: tl.constexpr,
    experts_per_rank: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    element_bytes: tl.constexpr,
    max_put_bytes: tl.constexpr,
    put_chunk_bytes: tl.constexpr,
):
    """Compute all experts, then scatter their complete source segments."""
    program = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_n_tiles = tl.cdiv(n, block_n)
    total_tiles = tl.load(group_tile_offsets + experts_per_rank) * num_n_tiles
    tile_idx = program
    while tile_idx < total_tiles:
        tile_m = tile_idx // num_n_tiles
        tile_n = tile_idx % num_n_tiles
        expert = 0
        for index in range(1, experts_per_rank + 1):
            expert += tl.where(tl.load(group_tile_offsets + index) <= tile_m, 1, 0)
        expert = tl.minimum(expert, experts_per_rank - 1)
        local_tile_m = tile_m - tl.load(group_tile_offsets + expert)
        group_start = tl.load(group_offsets + expert)
        group_end = tl.load(group_offsets + expert + 1)
        rows = local_tile_m * block_m + tl.arange(0, block_m)
        columns = tile_n * block_n + tl.arange(0, block_n)
        row_mask = rows[:, None] < group_end - group_start
        column_mask = columns[None, :] < n
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_tile in range(tl.cdiv(k, block_k)):
            k_offsets = k_tile * block_k + tl.arange(0, block_k)
            lhs_ptrs = (
                receive_buffer
                + (group_start + rows[:, None]).to(tl.int64) * k
                + k_offsets[None, :]
            )
            rhs_ptrs = (
                weights + expert * k * n + k_offsets[:, None] * n + columns[None, :]
            )
            lhs = tl.load(lhs_ptrs, mask=row_mask & (k_offsets[None, :] < k), other=0.0)
            rhs = tl.load(rhs_ptrs, mask=(k_offsets[:, None] < k) & column_mask, other=0.0)
            accumulator += tl.dot(lhs, rhs)
        tl.extra.cann.extension.compile_hint(accumulator, "matmul_at_least_once")
        tl.store(
            workspace
            + (group_start + rows[:, None]).to(tl.int64) * n
            + columns[None, :],
            accumulator.to(workspace.dtype.element_ty),
            mask=row_mask & column_mask,
        )
        tile_idx += num_programs

    libshmem_device.barrier_all()

    peer_programs: tl.constexpr = world_size - 1
    if program < peer_programs:
        destination_rank = program + (program >= rank_id)
        expert_start: tl.constexpr = rank_id * experts_per_rank
        total_experts: tl.constexpr = world_size * experts_per_rank
        peer_rows = 0
        for expert in range(experts_per_rank):
            global_expert = expert_start + expert
            rows = tl.load(token_counts + destination_rank * total_experts + global_expert)
            peer_rows += rows
            if rows > 0:
                source_offset = tl.load(ws_offsets + expert * world_size + destination_rank)
                destination_offset = tl.load(dst_offsets + destination_rank * total_experts + global_expert)
                # A single put carries at most one chunk, so the segment takes the
                # one-call path only when it fits in a chunk; otherwise it is split
                # into row-aligned pieces.  The chunk honours both limits, which is
                # what makes lowering either knob take effect.
                rows_per_chunk = max(1, min(put_chunk_bytes, max_put_bytes) // (n * element_bytes))
                if rows <= rows_per_chunk:
                    libaclshmem_device.putmem(
                        output + destination_offset * n,
                        workspace + source_offset * n,
                        rows * n * element_bytes,
                        destination_rank.to(tl.int32),
                    )
                else:
                    for chunk in range(tl.cdiv(rows, rows_per_chunk)):
                        row_start = chunk * rows_per_chunk
                        chunk_rows = tl.minimum(rows_per_chunk, rows - row_start)
                        libaclshmem_device.putmem(
                            output + (destination_offset + row_start) * n,
                            workspace + (source_offset + row_start) * n,
                            chunk_rows * n * element_bytes,
                            destination_rank.to(tl.int32),
                        )
        if peer_rows > 0:
            libaclshmem_device.quiet_lowlevel(
                transport=libaclshmem_device.ACLSHMEMTransport.MTE
            )
    else:
        copy_workers = num_programs - peer_programs
        copy_task = program - peer_programs
        total_copy_tasks = tl.load(local_tile_offsets + experts_per_rank) * num_n_tiles
        while copy_task < total_copy_tasks:
            tile_m = copy_task // num_n_tiles
            tile_n = copy_task % num_n_tiles
            expert = 0
            for index in range(1, experts_per_rank + 1):
                expert += tl.where(tl.load(local_tile_offsets + index) <= tile_m, 1, 0)
            expert = tl.minimum(expert, experts_per_rank - 1)
            local_tile_m = tile_m - tl.load(local_tile_offsets + expert)
            source_offset = tl.load(local_ws_offsets + expert)
            destination_offset = tl.load(local_dst_offsets + expert)
            count = tl.load(local_counts + expert)
            rows = local_tile_m * block_m + tl.arange(0, block_m)
            columns = tile_n * block_n + tl.arange(0, block_n)
            mask = (rows[:, None] < count) & (columns[None, :] < n)
            values = tl.load(
                workspace + (source_offset + rows[:, None]) * n + columns[None, :],
                mask=mask,
                other=0.0,
            )
            tl.store(
                output + (destination_offset + rows[:, None]) * n + columns[None, :],
                values,
                mask=mask,
            )
            copy_task += copy_workers

    libshmem_device.barrier_all()


def _build_oneshot_metadata(
    token_counts: torch.Tensor,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    block_m: int,
) -> "OneshotMetadata":
    """Move the expert and source segment tables to device tensors."""
    plan = _oneshot_plan_lists(
        token_counts.tolist(), rank, world_size, experts_per_rank, block_m
    )
    return OneshotMetadata(
        group_offsets=torch.tensor(plan["group_offsets"], dtype=torch.int32).npu(),
        group_tile_offsets=torch.tensor(plan["group_tile_offsets"], dtype=torch.int32).npu(),
        ws_offsets=torch.tensor(plan["ws_offsets"], dtype=torch.int64).npu(),
        dst_offsets=torch.tensor(plan["dst_offsets"], dtype=torch.int64).npu(),
        token_counts=token_counts.npu(),
        local_tile_offsets=torch.tensor(
            plan["local_tile_offsets"], dtype=torch.int32
        ).npu(),
        local_ws_offsets=torch.tensor(
            plan["local_ws_offsets"], dtype=torch.int64
        ).npu(),
        local_dst_offsets=torch.tensor(
            plan["local_dst_offsets"], dtype=torch.int64
        ).npu(),
        local_counts=torch.tensor(plan["local_counts"], dtype=torch.int32).npu(),
        experts_per_rank=experts_per_rank,
        total_m_tiles=plan["group_tile_offsets"][-1],
        valid_rows=plan["valid_rows"],
    )


def host_barrier(store, tag: str, rank: int, world_size: int) -> None:
    """Synchronize ranks through TCPStore inside the timed ACLSHMEM UDMA rounds.

    A round boundary is a pure host rendezvous: it must not enqueue any device
    work, because anything it enqueues lands between the timed launch and the
    next one.  A collective does enqueue work of its own (``HcclLaunchAicpuKernel``),
    so every boundary of the timed rounds stays on TCPStore.

    That is a positioning choice, not a statement that the two runtimes cannot
    coexist: the case script
    ``python/triton_dist/test/ascend/gmm-alltoallv-udma.py`` keeps its *kernel*
    rounds on this barrier but times the torch_npu reference with the default
    collective barrier while ACLSHMEM UDMA is live, and that mixing was
    re-measured on the current stack as passing.
    """
    key_prefix = f"gmm_alltoallv_udma/{tag}"
    store.set(f"{key_prefix}/{rank}", "1")
    for peer in range(world_size):
        store.get(f"{key_prefix}/{peer}")
def validate_against_torch_npu(
    store,
    tag: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rank: int,
    world_size: int,
    rtol: float,
) -> None:
    """Validate all ranks against the fp32 golden without an HCCL collective."""
    error = precision_error(actual, expected, rtol)
    key_prefix = f"gmm_alltoallv_udma/{tag}"
    store.set(f"{key_prefix}/{rank}", error)
    errors = [store.get(f"{key_prefix}/{peer}").decode("utf-8") for peer in range(world_size)]
    if any(errors):
        details = "; ".join(
            f"rank{peer}: {message}" for peer, message in enumerate(errors) if message
        )
        # Every rank holds the same gathered errors, so only rank 0 reports them.
        if rank == 0:
            print(f"UDMA_VALIDATION_MISMATCH {details}", flush=True)
        raise AssertionError(
            f"Triton output does not match torch_npu on rank(s): {details}"
        )


def _oneshot_plan_lists(
    token_counts,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    block_m: int,
):
    """Build complete expert and source segments without splitting GEMM rows."""
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    total_experts = world_size * experts_per_rank
    if len(token_counts) != world_size or any(len(row) != total_experts for row in token_counts):
        raise ValueError("token_counts has an invalid shape")

    group_offsets = [0]
    group_tile_offsets = [0]
    ws_offsets = []
    local_ws_offsets = []
    local_dst_offsets = []
    local_counts = []
    local_tile_offsets = [0]
    for expert in range(experts_per_rank):
        global_expert = rank * experts_per_rank + expert
        group_rows = 0
        for source in range(world_size):
            ws_offsets.append(group_offsets[-1] + group_rows)
            group_rows += int(token_counts[source][global_expert])
        group_offsets.append(group_offsets[-1] + group_rows)
        group_tile_offsets.append(group_tile_offsets[-1] + (group_rows + block_m - 1) // block_m)
        local_ws_offsets.append(ws_offsets[expert * world_size + rank])
        local_dst_offsets.append(sum(int(value) for value in token_counts[rank][:global_expert]))
        count = int(token_counts[rank][global_expert])
        local_counts.append(count)
        local_tile_offsets.append(local_tile_offsets[-1] + (count + block_m - 1) // block_m)

    dst_offsets = []
    for source in range(world_size):
        source_prefix = 0
        for expert in range(total_experts):
            dst_offsets.append(source_prefix)
            source_prefix += int(token_counts[source][expert])
    return {
        "group_offsets": group_offsets,
        "group_tile_offsets": group_tile_offsets,
        "ws_offsets": ws_offsets,
        "dst_offsets": dst_offsets,
        "local_tile_offsets": local_tile_offsets,
        "local_ws_offsets": local_ws_offsets,
        "local_dst_offsets": local_dst_offsets,
        "local_counts": local_counts,
        "valid_rows": sum(int(value) for value in token_counts[rank]),
    }


@dataclass(frozen=True)
class OneshotMetadata:
    """Device grouped-GEMM and scatter tables for a complete round."""

    group_offsets: torch.Tensor
    group_tile_offsets: torch.Tensor
    ws_offsets: torch.Tensor
    dst_offsets: torch.Tensor
    token_counts: torch.Tensor
    local_tile_offsets: torch.Tensor
    local_ws_offsets: torch.Tensor
    local_dst_offsets: torch.Tensor
    local_counts: torch.Tensor
    experts_per_rank: int
    total_m_tiles: int
    valid_rows: int


def launch_gmm_alltoallv_udma(
    receive_buffer: torch.Tensor,
    weights: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    rank: int,
    world_size: int,
    gmm_block_m: int,
    gmm_block_n: int,
    gmm_block_k: int,
    metadata: OneshotMetadata,
    tail_guard: OutputTailGuard,
    launchers: RoundLaunchers,
) -> None:
    """Compute every local expert and scatter its results in one launch."""
    if receive_buffer.dtype != torch.float16 or weights.dtype != torch.float16:
        raise TypeError("the UDMA tutorial currently supports float16 activations and weights only")
    launchers.start_round()
    if world_size <= 1 or world_size > _aicore_num():
        raise ValueError("UDMA one-shot requires 2..aicore_num ranks")
    if workspace.shape[0] < receive_buffer.shape[0]:
        raise ValueError("one-shot workspace must hold the complete grouped result")
    tail_guard.prepare(output, metadata.valid_rows)
    launchers.oneshot.launch(
        (_aicore_num(), 1, 1),
        (
            rank, world_size, metadata.experts_per_rank, receive_buffer.shape[1], output.shape[1],
            gmm_block_m, gmm_block_n, gmm_block_k, output.element_size(), MAX_PUT_BYTES, PUT_CHUNK_BYTES,
        ),
        (
            receive_buffer, weights, output, workspace, metadata.group_offsets,
            metadata.group_tile_offsets, metadata.ws_offsets, metadata.dst_offsets,
            metadata.token_counts, metadata.local_tile_offsets, metadata.local_ws_offsets,
            metadata.local_dst_offsets, metadata.local_counts,
        ),
    )


def require_udma_device(local_rank: int) -> None:
    """Reject non-950 systems instead of silently running an unsupported path."""
    name = os.environ.get("ASCEND_SOC_VERSION", "")
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is not None:
        name = str(get_device_name(local_rank)) or name
    if "950" not in name:
        raise RuntimeError(
            "GMM-AllToAllV UDMA requires Ascend 950; detected "
            f"{name or 'an unknown SoC'}. Use gmm-alltoallv.py for the MTE path."
        )


def run() -> None:
    """Initialize distributed state and run the fixed-seed self-test."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    require_udma_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Create the HCCL communicator before ACLSHMEM.  The HCCL work this self-test
    # needs (the weights all-gather and the reference pass) is submitted before
    # ``aclshmem_init`` for the same reason.
    dist.barrier(device_ids=[local_rank])
    store = host_store()
    m = int(os.environ.get("TRITON_M", "4096"))
    k = int(os.environ.get("TRITON_K", "4096"))
    n = int(os.environ.get("TRITON_N", "4096"))
    experts_per_rank = int(os.environ.get("TRITON_EXPERTS_PER_RANK", "2"))
    gmm_block_m, gmm_block_n, gmm_block_k = _effective_tiles(m, k, n)
    sparse = _env_flag("TRITON_SPARSE", "1")
    route_modes = tuple(
        value.strip() for value in os.environ.get("TRITON_ROUTE_MODES", "random").split(",") if value.strip()
    )
    if len(route_modes) != 1:
        raise ValueError("TRITON_ROUTE_MODES must contain exactly one mode when running the tutorial script")
    route_mode = route_modes[0]
    if min(m, k, n, experts_per_rank, gmm_block_m, gmm_block_n, gmm_block_k) <= 0:
        raise ValueError("shape and block-size variables must be positive")

    output: torch.Tensor | None = None
    seed = 20260811
    try:
        torch.manual_seed(seed + rank)
        weights = torch.randn((experts_per_rank, k, n), dtype=torch.float16).npu()
        # The same-semantics baseline runs before ACLSHMEM starts, so it needs
        # its own plain output buffer instead of the symmetric one.
        baseline_output_buffer = torch.zeros((world_size * m, n), dtype=torch.float16).npu()

        def prepare_invocation(
            epoch: int,
        ) -> tuple[
            RouteSample,
            torch.Tensor,
            OneshotMetadata,
            BaselinePlan,
        ]:
            sample = prepare_route(
                m, k, world_size, experts_per_rank, rank, epoch, route_mode, store, sparse
            )
            recv = build_receive_buffer(
                sample.activations, sample.token_counts, rank, world_size, experts_per_rank
            )
            metadata = _build_oneshot_metadata(
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
                gmm_block_m,
            )
            ws_offsets, dst_offsets = _baseline_offsets(
                sample.token_counts, rank, world_size, experts_per_rank
            )
            baseline_plan = build_baseline_plan(
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
                ws_offsets,
                dst_offsets,
            )
            return sample, recv, metadata, baseline_plan

        weight_list = [torch.empty_like(weights) for _ in range(world_size)]
        dist.all_gather(weight_list, weights)
        full_weights = torch.cat(weight_list, dim=0)
        sample, recv, metadata, baseline_plan = prepare_invocation(0)
        expected = golden_fp32(
            sample.activations, full_weights, sample.token_counts, rank, world_size, experts_per_rank
        )
        # Guard: the same-semantics baseline must reproduce the fp32 golden.
        baseline_result = baseline_output(recv, weights, baseline_output_buffer, baseline_plan)
        torch.npu.synchronize()
        baseline_error = precision_error(baseline_result, expected, precision_rtol(k))
        if baseline_error:
            raise AssertionError(f"torch_npu baseline does not match the fp32 golden: {baseline_error}")
        local_mem_size = symmetric_memory_size(world_size, m, n)
        if rank == 0:
            print(f"GMM_ALLTOALLV_LOCAL_MEM_SIZE_BYTES={local_mem_size}", flush=True)
            print(
                "CONFIG "
                f"engine=udma ranks={world_size} shape=m{m}_k{k}_n{n} "
                f"blocks={gmm_block_m}x{gmm_block_n}x{gmm_block_k} "
                f"route={route_mode} sparse={int(sparse)} experts_per_rank={experts_per_rank} "
                f"aicore_num={_aicore_num()} vector_cores={_aivector_core_num()}",
                flush=True,
            )
        init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.UDMA)
        output = ash.aclshmem_create_tensor(
            [world_size * m, n], dtype=torch.float16, device_id=local_rank
        )
        tail_guard = OutputTailGuard.create(world_size * m)
        tail_guard.prepare(output, metadata.valid_rows)
        workspace = torch.empty((world_size * m, n), dtype=torch.float16).npu()
        launchers = RoundLaunchers(
            oneshot=KernelLauncher(_gmm_alltoallv_oneshot_kernel),
        )
        torch.npu.synchronize()
        host_barrier(store, "oneshot_init", rank, world_size)

        def invoke(recv: torch.Tensor, metadata: OneshotMetadata) -> torch.Tensor:
            launch_gmm_alltoallv_udma(
                recv,
                weights,
                workspace,
                output,
                rank,
                world_size,
                gmm_block_m,
                gmm_block_n,
                gmm_block_k,
                metadata=metadata,
                tail_guard=tail_guard,
                launchers=launchers,
            )
            return output

        actual = invoke(recv, metadata)
        torch.npu.synchronize()
        validate_against_torch_npu(store, "validation", actual, expected, rank, world_size, precision_rtol(k))
        host_barrier(store, "validation_done", rank, world_size)
        if rank == 0:
            print("GMM-AllToAllV UDMA tutorial passed", flush=True)
    finally:
        if output is not None:
            ash.aclshmem_free_tensor(output)
        ash.aclshmem_finalize()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
