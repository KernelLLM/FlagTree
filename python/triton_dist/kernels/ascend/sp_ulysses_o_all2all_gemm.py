# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fused Ulysses output All2All and GEMM for Ascend950.

The kernel overlaps UDMA communication on AIV1 with source-major GEMM on
AIC.  The output-M dimension is divided into fixed-size pipeline stages.  Two
symmetric staging buffers are alternated across stages, and one ``barrier_all``
per stage provides both remote-data readiness and safe buffer reuse.

All ranks accumulate source/K partitions in the canonical global order
``0 .. RANK_SIZE-1``, matching the source order in Golden's logical K axis.

Input and output layouts for ``P`` ranks are:

* ``attn_out``: ``[S_total, H_local, D]``;
* ``weight``: ``[N, P * H_local * D]``;
* ``output``: ``[S_total / P, N]``.

The local All2All source is read directly from ``attn_out``.  Only remote
sources are copied into symmetric staging memory, avoiding a local self-copy.

The caller initializes aclshmem with the UDMA engine and owns the symmetric
workspace and output tensors. Reuse the operator and its workspace across
calls; synchronize all ranks before releasing the workspace.
"""

import math

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id

from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend.algorithm import gemm_swizzle2d

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128
STAGE_M_TILES_PER_RANK = 4

BUFFER_COUNT = 2

ELEMENT_BYTES = tl.constexpr(2)


@triton.jit
def kernel_sp_ulysses_o_all2all_gemm(
    attn_ptr,
    staging_ptr,
    weight_ptr,
    output_ptr,
    rank,
    M,
    N,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    RANK_SIZE: tl.constexpr,
    LOCAL_CHUNK: tl.constexpr,
    PEER_ELEMS: tl.constexpr,
    PIPE_M: tl.constexpr,
    PIPE_ELEMS: tl.constexpr,
    BUFFER_ELEMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Execute a putmem All2All and source-major GEMM pipeline.

    AIV1 program ``pid == target_rank`` is the only UDMA issuer for that
    target.  For stage ``s``, source rank ``r`` writes its rows into
    ``staging[s % 2, r]`` on every remote target rank.

    The stage barrier has two roles:

    * all remote writes for the current stage are visible before AIC reads;
    * before a buffer is reused two stages later, every AIC has completed the
      previous stage that used the same buffer.
    """
    subblock_idx = sub_vec_id()
    pid = tl.program_id(0)
    core_count = tl.num_programs(0)

    n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    stage_m_tiles = tl.cdiv(PIPE_M, BLOCK_SIZE_M)
    stage_tiles = stage_m_tiles * n_tiles
    local_k_tiles = tl.cdiv(LOCAL_CHUNK, BLOCK_SIZE_K)

    for stage_id in range(0, NUM_STAGES):
        buffer_id = stage_id % NUM_BUFFERS
        stage_m_begin = stage_id * PIPE_M
        stage_rows = tl.minimum(M - stage_m_begin, PIPE_M)
        buffer_base = buffer_id * BUFFER_ELEMS

        # The verified Ascend950 mapping uses AIV1 for UDMA and assigns exactly one
        # program to each target rank.  The local target is not copied.
        if subblock_idx == 1:
            if pid < RANK_SIZE:
                target_rank = pid
                if target_rank != rank:
                    libshmem_device.putmem(
                        staging_ptr
                        + buffer_base
                        + rank * PIPE_ELEMS,
                        attn_ptr
                        + target_rank * PEER_ELEMS
                        + stage_m_begin * LOCAL_CHUNK,
                        stage_rows * LOCAL_CHUNK * ELEMENT_BYTES,
                        target_rank,
                    )

        libshmem_device.barrier_all()

        for stage_tile_id in range(pid, stage_tiles, core_count):
            tile_m, tile_n = gemm_swizzle2d(
                stage_tile_id,
                PIPE_M,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                swizzle_direction=1,
            )
            stage_m_offsets = (
                tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            )
            m_offsets = stage_m_begin + stage_m_offsets
            n_offsets = (
                tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            )
            m_mask = m_offsets[:, None] < M
            n_mask = n_offsets[None, :] < N
            accumulator = tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_N),
                dtype=tl.float32,
            )

            # Match Golden's logical K layout on every rank by accumulating
            # source 0, 1, ..., RANK_SIZE - 1.  The local source still avoids
            # a staging self-copy; only its input pointer is different.
            for source_rank in range(0, RANK_SIZE):
                for local_k_tile in range(0, local_k_tiles):
                    local_k_offsets = (
                        local_k_tile * BLOCK_SIZE_K
                        + tl.arange(0, BLOCK_SIZE_K)
                    )
                    global_k_offsets = (
                        source_rank * LOCAL_CHUNK + local_k_offsets
                    )
                    k_mask = local_k_offsets < LOCAL_CHUNK
                    local_a_offsets = (
                        source_rank * PEER_ELEMS
                        + m_offsets[:, None] * LOCAL_CHUNK
                        + local_k_offsets[None, :]
                    )
                    remote_a_offsets = (
                        buffer_base
                        + source_rank * PIPE_ELEMS
                        + stage_m_offsets[:, None] * LOCAL_CHUNK
                        + local_k_offsets[None, :]
                    )
                    b_offsets = (
                        n_offsets[None, :] * stride_wn
                        + global_k_offsets[:, None] * stride_wk
                    )
                    if source_rank == rank:
                        a = tl.load(
                            attn_ptr + local_a_offsets,
                            mask=m_mask & k_mask[None, :],
                            other=0.0,
                        )
                    else:
                        a = tl.load(
                            staging_ptr + remote_a_offsets,
                            mask=m_mask & k_mask[None, :],
                            other=0.0,
                        )
                    b = tl.load(
                        weight_ptr + b_offsets,
                        mask=k_mask[:, None] & n_mask,
                        other=0.0,
                    )
                    accumulator += tl.dot(a, b)

            tl.extra.cann.extension.compile_hint(
                accumulator,
                "matmul_at_least_once",
            )
            output_offsets = (
                m_offsets[:, None] * stride_om
                + n_offsets[None, :] * stride_on
            )
            tl.store(
                output_ptr + output_offsets,
                accumulator.to(tl.bfloat16),
                mask=m_mask & n_mask,
            )


def _pipeline_layout(local_sequence_length, local_chunk, rank_size):
    """Return element counts for the fixed two-buffer M pipeline."""
    pipe_m = BLOCK_M * STAGE_M_TILES_PER_RANK * rank_size
    pipe_elems = pipe_m * local_chunk
    buffer_elems = rank_size * pipe_elems
    staging_elems = BUFFER_COUNT * buffer_elems
    stage_count = (local_sequence_length + pipe_m - 1) // pipe_m
    return pipe_m, pipe_elems, buffer_elems, staging_elems, stage_count


class SpUlyssesOAll2AllGemmKernel:
    """Launch the fused operator using a caller-owned symmetric workspace.

    All ranks must use matching shapes and enter calls in the same order.
    Inputs, weights, output, and workspace use BF16. The workspace must not
    be shared by concurrent launches.
    """

    def __init__(self, rank, world_size, staging):
        self.core_count = self._validate_world_size(world_size)
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if staging.dtype != torch.bfloat16 or not staging.is_contiguous():
            raise ValueError("staging must be a contiguous BF16 symmetric tensor")
        self.rank = rank
        self.world_size = world_size
        self.staging = staging
        self._launcher = kernel_sp_ulysses_o_all2all_gemm[self.core_count, 1, 1]

    @staticmethod
    def _validate_world_size(world_size):
        core_count = driver.active.utils.get_aicore_num()
        if core_count <= 0:
            raise RuntimeError(f"Invalid AI core count: {core_count}")
        # Each target rank needs its own program in the launch grid.
        if not 1 <= world_size <= core_count:
            raise ValueError(
                f"world_size must be between 1 and {core_count}: "
                "the launch grid needs one program per target rank"
            )
        return core_count

    @staticmethod
    def workspace_size(world_size, local_heads, head_dim):
        """Return the required symmetric workspace size in BF16 elements.

        The fixed pipeline workspace does not depend on sequence length.
        """
        SpUlyssesOAll2AllGemmKernel._validate_world_size(world_size)
        if local_heads <= 0 or head_dim <= 0:
            raise ValueError("local_heads and head_dim must be positive")
        return _pipeline_layout(1, local_heads * head_dim, world_size)[3]

    def forward(self, attn_out, weight, output):
        """Write and return output with shape [S_total / world_size, N]."""
        if attn_out.ndim != 3 or weight.ndim != 2 or output.ndim != 2:
            raise ValueError("attn_out, weight, and output must be 3D, 2D, and 2D")
        total_sequence_length, local_heads, head_dim = attn_out.shape
        if min(total_sequence_length, local_heads, head_dim, weight.shape[0]) <= 0:
            raise ValueError("input dimensions must be positive")
        if total_sequence_length % self.world_size:
            raise ValueError("S_total must be divisible by world_size")
        local_sequence_length = total_sequence_length // self.world_size
        local_chunk = local_heads * head_dim
        out_features = weight.shape[0]
        if weight.shape[1] != self.world_size * local_chunk:
            raise ValueError("weight must have shape [N, world_size * H_local * D]")
        if tuple(output.shape) != (local_sequence_length, out_features):
            raise ValueError("output must have shape [S_total / world_size, N]")
        for name, tensor in (
            ("attn_out", attn_out),
            ("weight", weight),
            ("output", output),
        ):
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"{name} must use BF16")
            if tensor.device != self.staging.device:
                raise ValueError(f"{name} and staging must be on the same device")
        if not attn_out.is_contiguous():
            raise ValueError("attn_out must be contiguous")
        (
            pipe_m,
            pipe_elems,
            buffer_elems,
            staging_elems,
            stage_count,
        ) = _pipeline_layout(local_sequence_length, local_chunk, self.world_size)
        if self.staging.numel() < staging_elems:
            raise ValueError(
                f"staging requires {staging_elems} elements, "
                f"got {self.staging.numel()}"
            )

        self._launcher(
            attn_out,
            self.staging,
            weight,
            output,
            self.rank,
            local_sequence_length,
            out_features,
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            RANK_SIZE=self.world_size,
            LOCAL_CHUNK=local_chunk,
            PEER_ELEMS=local_sequence_length * local_chunk,
            PIPE_M=pipe_m,
            PIPE_ELEMS=pipe_elems,
            BUFFER_ELEMS=buffer_elems,
            NUM_STAGES=stage_count,
            NUM_BUFFERS=BUFFER_COUNT,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
        )
        return output


def reference_gemm_input(attn_out, weight):
    """Reference HCCL All2All and layout conversion."""
    world_size = dist.get_world_size()
    total_sequence_length, local_heads, head_dim = attn_out.shape
    if total_sequence_length % world_size:
        raise ValueError("S_total must be divisible by world_size")
    local_sequence_length = total_sequence_length // world_size
    hidden_size = world_size * local_heads * head_dim
    if weight.ndim != 2 or weight.shape[1] != hidden_size:
        raise ValueError("weight must have shape [N, world_size * H_local * D]")
    recv = torch.empty_like(attn_out)
    dist.all_to_all_single(recv, attn_out.contiguous())
    gemm_input = (
        recv.reshape(world_size, local_sequence_length, local_heads, head_dim)
        .permute(1, 0, 2, 3)
        .reshape(local_sequence_length, hidden_size)
        .contiguous()
    )
    return gemm_input



def _cpu_reference(attn_out, weight):
    """Complete the collective on every rank before returning lazy CPU chunks."""
    gemm_input = reference_gemm_input(attn_out, weight)

    def chunks():
        weight_fp64 = weight.detach().cpu().double().T
        for start in range(0, gemm_input.shape[0], 64):
            rows = gemm_input[start:start + 64].detach().cpu().double()
            yield start, torch.matmul(rows, weight_fp64)

    return chunks()



def _check_cpu_reference(actual, expected, chunks):
    """
    Chunks must cover every output row in order. Aggregate metrics over the
    complete output, rather than checking only direct-comparison mismatches.
    """
    statistics = [
        {"max_relative": 0.0, "sum_relative": 0.0, "sum_squared": 0.0, "small_errors": 0}
        for _ in range(2)
    ]
    normal_count = small_count = next_row = 0
    for start, truth in chunks:
        truth = truth.detach().cpu().double()
        if (truth.ndim != 2 or start != next_row or truth.shape[0] == 0
                or truth.shape[1] != actual.shape[1]
                or start + truth.shape[0] > actual.shape[0]):
            raise ValueError("CPU reference chunks must cover output rows in order")
        next_row = start + truth.shape[0]
        if not torch.isfinite(truth).all():
            raise ValueError("CPU FP64 reference contains NaN or Inf")
        small = truth.abs() < 2 ** (-8)
        normal = ~small
        normal_count += int(normal.sum())
        small_count += int(small.sum())
        for result, stats in zip((actual, expected), statistics):
            values = result[start:next_row].detach().cpu().double()
            if not torch.isfinite(values).all():
                raise ValueError("result contains NaN or Inf")
            difference = values - truth
            relative = difference[normal].abs() / (truth[normal].abs() + 1e-7)
            if relative.numel():
                stats["max_relative"] = max(stats["max_relative"], float(relative.max()))
                stats["sum_relative"] += float(relative.sum())
                stats["sum_squared"] += float(difference[normal].square().sum())
            stats["small_errors"] += int((difference[small].abs() > 2 ** (-16)).sum())
    if next_row != actual.shape[0] or actual.shape != expected.shape:
        raise ValueError("CPU reference must cover the complete output")

    def ratio(custom, reference):
        return custom / reference if reference else (1.0 if custom == 0 else float("inf"))

    def within(value, limit):
        return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)

    metrics = []
    for stats in statistics:
        metrics.append({
            "mare": stats["max_relative"],
            "mere": stats["sum_relative"] / max(normal_count, 1),
            "rmse": math.sqrt(stats["sum_squared"] / max(normal_count, 1)),
        })
    ratios = {name: ratio(metrics[0][name], metrics[1][name]) for name in metrics[0]}
    normal_pass = all(within(ratios[name], limit)
                      for name, limit in (("mare", 2.0), ("mere", 1.2), ("rmse", 1.2)))
    small_ratio = statistics[0]["small_errors"] / max(statistics[1]["small_errors"], 1)
    passed = normal_pass and within(small_ratio, 2.0)
    return passed, (
        f"CPU FP64: normal_count={normal_count}, small_count={small_count}, "
        f"kernel_metrics={metrics[0]}, torch_metrics={metrics[1]}, ratios={ratios} "
        f"(limits: mare=2.0, mere=1.2, rmse=1.2), "
        f"small_errors={statistics[0]['small_errors']}/{statistics[1]['small_errors']}, "
        f"small_count_ratio={small_ratio} (limit=2.0)"
    )


def check_accuracy(actual, expected, attn_out, weight, rtol=1e-3, atol=1e-3, context="result"):
    """Collectively check Ulysses output, falling back to full CPU FP64 metrics.

    All ranks must call this function. Direct comparison is tried first; only
    failing ranks compute CPU GEMMs, after all ranks exchange reference inputs.
    This diagnostic is separate from forward() and is never run implicitly.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    passed = torch.tensor([1], dtype=torch.int32, device=actual.device)
    error_msg = ""
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as err:
        passed[0] = 0
        error_msg = str(err)
    if world_size == 1:
        all_passed = [passed]
    else:
        all_passed = [
            torch.zeros(1, dtype=torch.int32, device=actual.device) for _ in range(world_size)
        ]
        dist.all_gather(all_passed, passed)
    cpu_details = ""
    if any(r.item() == 0 for r in all_passed):
        try:
            chunks = _cpu_reference(attn_out, weight)
            if passed.item() == 0:
                cpu_passed, cpu_details = _check_cpu_reference(actual, expected, chunks)
                passed[0] = int(cpu_passed)
                if not cpu_passed:
                    error_msg += "\n" + cpu_details
        except Exception as err:
            passed[0] = 0
            error_msg += f"\nCPU reference failed: {type(err).__name__}: {err}"
        if world_size > 1:
            dist.all_gather(all_passed, passed)
    for rank_id in range(world_size):
        if rank == rank_id:
            if all_passed[rank_id].item() == 1:
                print(
                    f"[PASS] Rank {rank}: {context} "
                    + (f"passed CPU fallback after direct comparison failed. {cpu_details}"
                       if cpu_details else
                       f"matches golden within tolerances (rtol={rtol}, atol={atol})."),
                    flush=True,
                )
            else:
                print(
                    f"[FAIL] Rank {rank}: {context} does NOT match "
                    f"golden. Details:\n{error_msg}",
                    flush=True,
                )
        if distributed:
            dist.barrier()
    if any(r.item() == 0 for r in all_passed):
        raise AssertionError(error_msg or f"{context} correctness check failed")
