# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""UDMA reverse all-to-all + two RMSNorms + MatMul tutorial.

Run with, for example:

    torchrun --nproc-per-node=2 \
      tutorials/ascend/reverse-a2a-norm-matmul-2.py

Set FUSED_RUN_BENCH=1 to run the built-in boundary-shape batch cases.
Event timing for this example lives in python/triton_dist/test/ascend/reverse-a2a-norm-matmul-2.py
(warmup/iterations controlled there by ASCEND_PROFILE_WARMUP/ITERS).

The operation implemented by this file is:

    recv_1 = reverse_all_to_all(tensor_1)
    recv_2 = reverse_all_to_all(tensor_2)
    norm_1 = rms_norm(recv_1, norm_weight, eps)
    norm_2 = rms_norm(recv_2, norm_weight, eps)
    output = cat([norm_1, norm_2], dim=-1) @ linear_weight.T

RMSNorm norm_weight.shape == [D]
linear_weight.shape == [N, 2 * D]
tensor_1 and tensor_2 are different inputs.
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
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device
from utils import init_aclshmem

DEFAULT_ASH_SIZE_MB = 1024

SIG_STRIDE = 16  # int32 elements per 64-byte signal slot.

# Accuracy verification is divided into two levels: First,
# compare the Triton output and Torch Golden with rtol=atol=5e-3;
# If the direct comparison fails, then use the CPU FP64 result as the
# true value to perform MARE, MERE, and RMSE error ratio verification.
ACCURACY_RTOL = 5e-3
ACCURACY_ATOL = 5e-3
MARE_RATIO_LIMIT = 2.0
MERE_RATIO_LIMIT = 1.2
RMSE_RATIO_LIMIT = 1.2
RELATIVE_ERROR_EPSILON = 1e-7

SEQUENCE_LENGTHS = tuple(
    list(range(2040, 2049)) + list(range(4088, 4097)) + list(range(8184, 8196))
)
HIDDEN_GROUPS = 128
FEATURE_DIMS = (12, 24, 28, 32, 40, 48, 56)


@triton.jit(do_not_specialize=["S", "H", "D"])
def kernel_reverse_a2a_2_udma(
    a1_ptr,
    a2_ptr,
    recv1_ptr,
    recv2_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    S,
    H,
    D,
    stride_as,
    stride_ah,
    stride_ad,
    stride_rs,
    stride_rh,
    stride_rd,
    COMM_BLOCK_S: tl.constexpr,
    TILE_HD: tl.constexpr,
    SUB_ROWS: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
    PEER2_BASE: tl.constexpr,
):
    """Reverse all-to-all two inputs with one ready signal per data pair."""
    n_vec = 1
    total_vec = tl.num_programs(axis=0) * n_vec
    global_vec_id = tl.program_id(axis=0) * n_vec + sub_vec_id()
    logical_core_id = global_vec_id // 2
    n_role_cores = total_vec // 2
    elem_bytes = 2  # This tutorial intentionally supports bfloat16 only.

    S = S.to(tl.int64)
    H = H.to(tl.int64)
    D = D.to(tl.int64)
    HD = H * D
    num_blocks_s = tl.cdiv(S, COMM_BLOCK_S)
    row_tiles_per_block = COMM_BLOCK_S // SUB_ROWS

    # Each input's peer_mem region carries rank_size - 1 contiguous
    # [S, H, D] tensors, one per remote sender. The sender-to-region mapping
    # must stay identical on the producer and consumer sides.

    for global_id_s in range(0, num_blocks_s):
        cur_block_s = tl.minimum(COMM_BLOCK_S, S - global_id_s * COMM_BLOCK_S)

        # Even Vector Cores assigned to peers issue both UDMA puts, then one
        # notification.  The consumer therefore cannot observe a half-ready
        # pair of inputs.
        if global_vec_id % 2 == 0 and logical_core_id < rank_size:
            for target_rank in range(logical_core_id, rank_size, n_role_cores):
                if target_rank != rank:
                    src1_offset = (
                        target_rank * S * HD + global_id_s * COMM_BLOCK_S * stride_as
                    )
                    dst_region = rank if rank < target_rank else rank - 1
                    dst1_offset = dst_region * S * HD + global_id_s * COMM_BLOCK_S * HD
                    nbytes = cur_block_s * HD * elem_bytes
                    libshmem_device.putmem(
                        peer_mem_ptr + dst1_offset,
                        a1_ptr + src1_offset,
                        nbytes,
                        target_rank,
                    )
                    # Keep the offset SSA values for the two source pointers
                    # independent.  Ascend's pointer-source analysis does not
                    # support one offset value shared by unrelated base ptrs.
                    src2_offset = (
                        target_rank * S * HD + global_id_s * COMM_BLOCK_S * stride_as
                    )
                    dst2_offset = (
                        PEER2_BASE
                        + dst_region * S * HD
                        + global_id_s * COMM_BLOCK_S * HD
                    )
                    libshmem_device.putmem(
                        peer_mem_ptr + dst2_offset,
                        a2_ptr + src2_offset,
                        nbytes,
                        target_rank,
                    )
            for target_rank in range(logical_core_id, rank_size, n_role_cores):
                if target_rank != rank:
                    dl.notify(
                        signal_mem_ptr
                        + (global_id_s * rank_size + rank) * SIGNAL_STRIDE,
                        target_rank.to(tl.int32),
                    )

        # Odd Vector Cores consume the pair and materialize the two reverse
        # all-to-all outputs. Flattening H and D makes each transfer a
        # contiguous burst instead of a per-D-column strided access.
        if global_vec_id % 2 == 1:
            offs_row = tl.arange(0, SUB_ROWS)
            offs_hd = tl.arange(0, TILE_HD)

            for sender_rank in range(0, rank_size):
                if sender_rank != rank:
                    ready_signal = (
                        signal_mem_ptr
                        + (global_id_s * rank_size + sender_rank) * SIGNAL_STRIDE
                    )
                    token = dl.wait(ready_signal, 1, "sys", "acquire")
                    load_ptr = dl.consume_token(peer_mem_ptr, token)
                    sender_region = (
                        sender_rank if sender_rank < rank else sender_rank - 1
                    )
                    remote_src_base = (
                        sender_region * S * HD + global_id_s * COMM_BLOCK_S * HD
                    )
                    for row_tile in range(
                        logical_core_id, row_tiles_per_block, n_role_cores
                    ):
                        rows = (
                            global_id_s * COMM_BLOCK_S + row_tile * SUB_ROWS + offs_row
                        )
                        mask = (rows < S)[:, None] & (offs_hd < HD)[None, :]
                        src_rows = row_tile * SUB_ROWS + offs_row
                        src_offsets = (
                            remote_src_base + src_rows[:, None] * HD + offs_hd[None, :]
                        )
                        data1 = tl.load(load_ptr + src_offsets, mask=mask, other=0.0)
                        dst_offsets = (
                            rows[:, None] * stride_rs
                            + sender_rank * HD
                            + offs_hd[None, :]
                        )
                        tl.store(recv1_ptr + dst_offsets, data1, mask=mask)
                        data2 = tl.load(
                            load_ptr + PEER2_BASE + src_offsets, mask=mask, other=0.0
                        )
                        tl.store(recv2_ptr + dst_offsets, data2, mask=mask)

            # Self data never crosses ranks, so bypass peer_mem and signaling.
            self_src_base = rank * S * HD + global_id_s * COMM_BLOCK_S * HD
            for row_tile in range(logical_core_id, row_tiles_per_block, n_role_cores):
                rows = global_id_s * COMM_BLOCK_S + row_tile * SUB_ROWS + offs_row
                mask = (rows < S)[:, None] & (offs_hd < HD)[None, :]
                src_rows = row_tile * SUB_ROWS + offs_row
                src_offsets = self_src_base + src_rows[:, None] * HD + offs_hd[None, :]
                data1 = tl.load(a1_ptr + src_offsets, mask=mask, other=0.0)
                dst_offsets = rows[:, None] * stride_rs + rank * HD + offs_hd[None, :]
                tl.store(recv1_ptr + dst_offsets, data1, mask=mask)
                data2 = tl.load(a2_ptr + src_offsets, mask=mask, other=0.0)
                tl.store(recv2_ptr + dst_offsets, data2, mask=mask)


@triton.jit(do_not_specialize=["M", "N", "D"])
def kernel_rmsnorm_matmul_2(
    recv1_ptr,
    recv2_ptr,
    norm_weight_ptr,
    linear_weight_ptr,
    output_ptr,
    M,
    N,
    D,
    stride_rm,
    stride_rd,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """Normalize two inputs and multiply them without materializing either."""
    pid = tl.program_id(axis=0)
    nprograms = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)

    for tile_id in range(pid, num_tiles_m * num_tiles_n, nprograms):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        input_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        weight_mask = (offs_d[:, None] < D) & (offs_n[None, :] < N)

        input1_offsets = offs_m[:, None] * stride_rm + offs_d[None, :] * stride_rd
        weight1_offsets = offs_n[None, :] * stride_wn + offs_d[:, None] * stride_wk
        input1 = tl.load(recv1_ptr + input1_offsets, mask=input_mask, other=0.0).to(
            tl.float32
        )
        variance1 = tl.sum(input1 * input1, axis=1, keep_dims=True) / D
        inv_rms1 = 1.0 / tl.sqrt(variance1 + EPS)
        gamma = tl.load(norm_weight_ptr + offs_d, mask=offs_d < D, other=0.0).to(
            tl.float32
        )
        input1 = (input1 * inv_rms1 * gamma[None, :]).to(tl.bfloat16)
        weight1 = tl.load(
            linear_weight_ptr + weight1_offsets, mask=weight_mask, other=0.0
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        accumulator = tl.dot(input1, weight1, accumulator)

        input2_offsets = offs_m[:, None] * stride_rm + offs_d[None, :] * stride_rd
        weight2_offsets = (
            offs_n[None, :] * stride_wn + (D + offs_d[:, None]) * stride_wk
        )
        input2 = tl.load(recv2_ptr + input2_offsets, mask=input_mask, other=0.0).to(
            tl.float32
        )
        variance2 = tl.sum(input2 * input2, axis=1, keep_dims=True) / D
        inv_rms2 = 1.0 / tl.sqrt(variance2 + EPS)
        input2 = (input2 * inv_rms2 * gamma[None, :]).to(tl.bfloat16)
        weight2 = tl.load(
            linear_weight_ptr + weight2_offsets, mask=weight_mask, other=0.0
        )
        accumulator = tl.dot(input2, weight2, accumulator)

        output_offsets = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            output_ptr + output_offsets, accumulator.to(tl.bfloat16), mask=output_mask
        )


_TUNED_MATMUL_BLOCKS = {
    2048: {
        3: (512, 16, 16),
        6: (1024, 32, 16),
        8: (1024, 16, 16),
        12: (1024, 16, 16),
        16: (1024, 16, 16),
        24: (512, 32, 32),
        28: (512, 32, 32),
        32: (256, 32, 32),
        40: (256, 64, 64),
        48: (256, 64, 64),
        56: (256, 64, 64),
    },
    4096: {
        3: (512, 16, 16),
        6: (1024, 32, 16),
        8: (1024, 16, 16),
        12: (1024, 16, 16),
        16: (512, 32, 16),
        24: (512, 32, 32),
        28: (512, 32, 32),
        32: (256, 32, 32),
        40: (256, 64, 64),
        48: (256, 64, 64),
        56: (256, 64, 64),
    },
    8192: {
        3: (1024, 16, 16),
        6: (1024, 32, 16),
        8: (1024, 16, 16),
        12: (1024, 16, 16),
        16: (1024, 16, 16),
        24: (512, 32, 32),
        28: (512, 32, 32),
        32: (256, 32, 32),
        40: (256, 64, 64),
        48: (256, 64, 64),
        56: (256, 64, 64),
    },
}


def _select_matmul_blocks(S, D, N):
    sequence_bucket = 2048 if S <= 2048 else 4096 if S <= 4096 else 8192
    if D == N and D in _TUNED_MATMUL_BLOCKS[sequence_bucket]:
        return _TUNED_MATMUL_BLOCKS[sequence_bucket][D]

    tile_width = max(
        max(16, triton.next_power_of_2(D)),
        min(128, max(16, triton.next_power_of_2(N))),
    )
    if tile_width <= 32:
        block_m = 512
    elif tile_width <= 64:
        block_m = 256
    else:
        block_m = 128
    block_n = min(128, max(16, triton.next_power_of_2(N)))
    block_d = max(16, triton.next_power_of_2(D))
    return block_m, block_n, block_d


def _validate_inputs(
    tensor_1,
    tensor_2,
    norm_weight,
    linear_weight,
    output,
    recv_1,
    recv_2,
    peer_mem,
    signal_mem,
    rank_size,
    buffer_num,
    comm_block_s,
    comm_block_d,
    block_m,
    block_n,
):
    if tensor_1.shape != tensor_2.shape or tensor_1.ndim != 3:
        raise ValueError(
            "tensor_1 and tensor_2 must have the same [S_total, H, D] shape"
        )
    if not tensor_1.is_contiguous() or not tensor_2.is_contiguous():
        raise ValueError("UDMA source tensors must be contiguous")
    if tensor_1.dtype != torch.bfloat16 or tensor_2.dtype != torch.bfloat16:
        raise TypeError("this tutorial currently supports bfloat16 inputs only")
    S_total, H, D = tensor_1.shape
    if S_total % rank_size != 0:
        raise ValueError("S_total must be divisible by rank_size")
    S = S_total // rank_size
    if norm_weight.shape != (D,):
        raise ValueError(
            f"norm_weight must have shape [{D}], got {tuple(norm_weight.shape)}"
        )
    if linear_weight.ndim != 2 or linear_weight.shape[1] != 2 * D:
        raise ValueError(
            f"linear_weight must have shape [N, {2 * D}], got {tuple(linear_weight.shape)}"
        )
    N = linear_weight.shape[0]
    expected_output = (S, H * rank_size, N)
    if output.shape != expected_output:
        raise ValueError(
            f"output must have shape {expected_output}, got {tuple(output.shape)}"
        )
    expected_recv_shape = (S, H * rank_size, D)
    if recv_1.shape != expected_recv_shape or recv_2.shape != expected_recv_shape:
        raise ValueError(f"recv workspaces must both have shape {expected_recv_shape}")

    tensors = (norm_weight, linear_weight, output)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("weights and output must use torch.bfloat16")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("norm_weight, linear_weight, and output must be contiguous")
    if (
        recv_1.dtype != torch.bfloat16
        or recv_2.dtype != torch.bfloat16
        or not recv_1.is_contiguous()
        or not recv_2.is_contiguous()
    ):
        raise ValueError("recv workspaces must be contiguous bfloat16 tensors")
    if peer_mem.dtype != torch.bfloat16:
        raise TypeError("peer_mem must use torch.bfloat16")
    if signal_mem.dtype != torch.int32:
        raise TypeError("signal_mem must use torch.int32 for Ascend dl.notify")

    devices = {
        tensor.device
        for tensor in (
            tensor_1,
            tensor_2,
            norm_weight,
            linear_weight,
            output,
            recv_1,
            recv_2,
            peer_mem,
            signal_mem,
        )
    }
    if len(devices) != 1:
        raise ValueError("inputs, outputs, and workspaces must be on the same NPU")

    selected_block_m, selected_block_n, selected_block_d = _select_matmul_blocks(
        S, D, N
    )
    if comm_block_d is None:
        comm_block_d = selected_block_d
    if comm_block_d < D:
        raise ValueError("comm_block_d must cover the complete D dimension")
    if block_m is None:
        block_m = selected_block_m
    if block_n is None:
        block_n = selected_block_n
    if min(buffer_num, comm_block_s, block_m, block_n) <= 0:
        raise ValueError("buffer and block sizes must be positive")
    block_sizes = (comm_block_s, comm_block_d, block_m, block_n)
    if any(triton.next_power_of_2(value) != value for value in block_sizes):
        raise ValueError("all Triton block sizes must be powers of two")

    num_blocks_s = triton.cdiv(S, comm_block_s)
    required_peer_elements = 2 * max(rank_size - 1, 1) * S * H * D
    required_signal_elements = num_blocks_s * rank_size * SIG_STRIDE
    if peer_mem.numel() < required_peer_elements:
        raise ValueError(
            f"peer_mem needs at least {required_peer_elements} bfloat16 elements"
        )
    if signal_mem.numel() < required_signal_elements:
        raise ValueError(
            f"signal_mem needs at least {required_signal_elements} int32 elements"
        )

    return S, H, D, N, comm_block_d, block_m, block_n


def hccl_reverse_a2a_norm_matmul_2(
    tensor_1,
    tensor_2,
    norm_weight,
    linear_weight,
    output,
    recv_1,
    recv_2,
    peer_mem,
    signal_mem,
    rank,
    rank_size,
    eps=1e-6,
    buffer_num=2,
    comm_block_s=1024,
    comm_block_d=None,
    block_m=None,
    block_n=None,
):
    """Launch UDMA reverse A2A, then fused RMSNorm and MatMul."""
    S, H, D, N, comm_block_d, block_m, block_n = _validate_inputs(
        tensor_1,
        tensor_2,
        norm_weight,
        linear_weight,
        output,
        recv_1,
        recv_2,
        peer_mem,
        signal_mem,
        rank_size,
        buffer_num,
        comm_block_s,
        comm_block_d,
        block_m,
        block_n,
    )

    # Flatten each row's [H, D] span so MTE2/MTE3 accesses are contiguous.
    # Keep each row tile within 64 KiB and expose enough row tiles to occupy
    # the consumer Vector Cores.
    tile_hd = triton.next_power_of_2(H * D)
    max_rows = max(1, 32768 // tile_hd)
    parallel_cap = triton.next_power_of_2(max(1, comm_block_s // (2 * rank_size)))
    sub_rows = max(1, min(comm_block_s, max_rows, parallel_cap))

    aivectorcore_num = driver.active.utils.get_aivector_core_num()
    if aivectorcore_num // 2 <= rank_size:
        raise RuntimeError("not enough Vector Cores for the peer senders")

    kernel_reverse_a2a_2_udma[aivectorcore_num, 1, 1](
        tensor_1,
        tensor_2,
        recv_1,
        recv_2,
        peer_mem,
        signal_mem,
        rank,
        rank_size,
        S,
        H,
        D,
        tensor_1.stride(0),
        tensor_1.stride(1),
        tensor_1.stride(2),
        recv_1.stride(0),
        recv_1.stride(1),
        recv_1.stride(2),
        COMM_BLOCK_S=comm_block_s,
        TILE_HD=tile_hd,
        SUB_ROWS=sub_rows,
        SIGNAL_STRIDE=SIG_STRIDE,
        PEER2_BASE=max(rank_size - 1, 1) * S * H * D,
    )

    M = S * H * rank_size
    block_d = max(16, triton.next_power_of_2(D))
    aicore_num = driver.active.utils.get_aicore_num()
    kernel_rmsnorm_matmul_2[aicore_num, 1, 1](
        recv_1,
        recv_2,
        norm_weight,
        linear_weight,
        output,
        M,
        N,
        D,
        D,
        1,
        linear_weight.stride(0),
        linear_weight.stride(1),
        N,
        1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        EPS=eps,
    )


def _high_precision_golden(recv_1, recv_2, norm_weight, linear_weight, eps):
    # CPU FP64 recalculates RMSNorm and MatMul as high-precision truth values
    # for secondary verification. This path is only executed if the direct comparison
    # in 5e-3 fails and is not included in performance test time.
    recv_1 = recv_1.cpu().to(torch.float64)
    recv_2 = recv_2.cpu().to(torch.float64)
    norm_weight = norm_weight.cpu().to(torch.float64)
    linear_weight = linear_weight.cpu().to(torch.float64)

    norm_1 = recv_1 * torch.rsqrt(recv_1.square().mean(dim=-1, keepdim=True) + eps)
    norm_2 = recv_2 * torch.rsqrt(recv_2.square().mean(dim=-1, keepdim=True) + eps)
    norm_1 *= norm_weight
    norm_2 *= norm_weight
    return torch.cat([norm_1, norm_2], dim=-1) @ linear_weight.T


def _error_metrics(result, truth):
    # MARE: Maximum relative error;
    # MERE: Mean relative error;
    # RMSE: Root mean square error.
    result = result.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(result).all().item()):
        raise AssertionError("result contains NaN or Inf")
    if not bool(torch.isfinite(truth).all().item()):
        raise AssertionError("high-precision truth contains NaN or Inf")
    difference = result - truth
    relative_error = difference.abs() / (truth.abs() + RELATIVE_ERROR_EPSILON)
    return {
        "mare": relative_error.max().item(),
        "mere": relative_error.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def _metric_ratio(custom_value, reference_value):
    # Returns to the Triton error metric / Torch Golden error metric.
    if reference_value == 0.0:
        return 1.0 if custom_value == 0.0 else float("inf")
    return custom_value / reference_value


def _assert_accuracy(
    golden,
    output,
    recv_1,
    recv_2,
    norm_weight,
    linear_weight,
    eps,
):
    # Level 1: Triton output is directly compared with Torch Golden with a tolerance of 5e-3.
    try:
        torch.testing.assert_close(
            golden.float().cpu(),
            output.float().cpu(),
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )
        return "assert_close", {}
    except AssertionError as direct_error:
        # Level 2: Calculate the errors of Triton and Torch Golden relative to the CPU FP64 true value.
        # A pass requires the error ratios of MARE, MERE, and RMSE to
        # all be no more than 2.0, 1.2, and 1.2, respectively.
        truth = _high_precision_golden(recv_1, recv_2, norm_weight, linear_weight, eps)
        custom_metrics = _error_metrics(output, truth)
        reference_metrics = _error_metrics(golden, truth)
        ratios = {
            name: _metric_ratio(custom_metrics[name], reference_metrics[name])
            for name in ("mare", "mere", "rmse")
        }
        limits = {
            "mare": MARE_RATIO_LIMIT,
            "mere": MERE_RATIO_LIMIT,
            "rmse": RMSE_RATIO_LIMIT,
        }
        if all(
            ratios[name] <= limits[name]
            or math.isclose(ratios[name], limits[name], rel_tol=1e-12, abs_tol=1e-12)
            for name in ratios
        ):
            return "error_ratio_fallback", ratios
        ratio_summary = ", ".join(
            f"{name}={ratios[name]:.6g} (limit={limits[name]:.6g})" for name in ratios
        )
        raise AssertionError(
            f"{direct_error}\nError-ratio fallback failed: {ratio_summary}"
        ) from direct_error


def _run_single_case(
    rank,
    rank_size,
    device,
    S,
    H,
    D,
    eps,
    ash_size_mb,
    comm_block_s,
    buffer_num,
):
    """Run one S/H/D case with N equal to D. Returns True on accuracy pass."""
    local_rank = int(os.environ["LOCAL_RANK"])
    N = D
    if min(S, H, D, N) <= 0:
        raise ValueError("S/H/D/N must be positive")
    if ash_size_mb <= 0 or ash_size_mb % 2 != 0:
        raise ValueError("FUSED_ASH_SIZE_MB must be a positive multiple of 2 MB")

    comm_block_d = max(D, triton.next_power_of_2(D))
    peer_mem = None
    signal_mem = None
    try:
        S_total = S * rank_size
        peer_mem_elems = 2 * max(rank_size - 1, 1) * S * H * D
        peer_mem = ash.aclshmem_create_tensor(
            [peer_mem_elems], dtype=torch.bfloat16, device_id=local_rank
        )
        num_blocks_s = triton.cdiv(S, comm_block_s)
        signal_elements = num_blocks_s * rank_size * SIG_STRIDE
        signal_mem = ash.aclshmem_create_tensor(
            [signal_elements], dtype=torch.int32, device_id=local_rank
        )

        torch.manual_seed(2026)
        norm_weight = torch.randn([D], dtype=torch.bfloat16, device=device)
        linear_weight = torch.randn([N, 2 * D], dtype=torch.bfloat16, device=device)
        torch.manual_seed(42 + rank)
        tensor_1 = torch.randn([S_total, H, D], dtype=torch.bfloat16, device=device)
        tensor_2 = torch.randn([S_total, H, D], dtype=torch.bfloat16, device=device)
        recv_1 = torch.empty([S, H * rank_size, D], dtype=torch.bfloat16, device=device)
        recv_2 = torch.empty_like(recv_1)
        output = torch.empty([S, H * rank_size, N], dtype=torch.bfloat16, device=device)

        def golden_launch(return_received=False):
            send_1 = torch.cat(
                torch.chunk(tensor_1, rank_size, dim=0), dim=1
            ).contiguous()
            send_1 = send_1.transpose(1, 0).contiguous()
            send_2 = torch.cat(
                torch.chunk(tensor_2, rank_size, dim=0), dim=1
            ).contiguous()
            send_2 = send_2.transpose(1, 0).contiguous()

            golden_recv_1 = torch.empty_like(send_1)
            golden_recv_2 = torch.empty_like(send_2)
            dist.all_to_all_single(golden_recv_1, send_1)
            dist.all_to_all_single(golden_recv_2, send_2)

            golden_recv_1 = golden_recv_1.transpose(0, 1).contiguous()
            golden_recv_2 = golden_recv_2.transpose(0, 1).contiguous()
            norm_1 = torch_npu.npu_rms_norm(golden_recv_1, norm_weight, eps)[0]
            norm_2 = torch_npu.npu_rms_norm(golden_recv_2, norm_weight, eps)[0]
            result = torch.cat([norm_1, norm_2], dim=-1) @ linear_weight.T
            if return_received:
                return result, golden_recv_1, golden_recv_2
            return result

        def kernel_launch():
            hccl_reverse_a2a_norm_matmul_2(
                tensor_1,
                tensor_2,
                norm_weight,
                linear_weight,
                output,
                recv_1,
                recv_2,
                peer_mem,
                signal_mem,
                rank,
                rank_size,
                eps=eps,
                buffer_num=buffer_num,
                comm_block_s=comm_block_s,
                comm_block_d=comm_block_d,
            )

        golden, golden_recv_1, golden_recv_2 = golden_launch(return_received=True)
        torch.npu.synchronize()
        signal_mem.fill_(0)
        torch.npu.synchronize()
        dist.barrier()
        kernel_launch()
        torch.npu.synchronize()

        passed = torch.tensor([1], dtype=torch.int32, device=device)
        error_message = ""
        accuracy_method = ""
        accuracy_details = {}
        try:
            accuracy_method, accuracy_details = _assert_accuracy(
                golden,
                output,
                golden_recv_1,
                golden_recv_2,
                norm_weight,
                linear_weight,
                eps,
            )
        except AssertionError as error:
            passed[0] = 0
            error_message = str(error)
        all_passed = [torch.zeros_like(passed) for _ in range(rank_size)]
        dist.all_gather(all_passed, passed)
        if rank == 0:
            print(
                "CONFIG "
                f"engine=UDMA ranks={rank_size} S={S} H={H} D={D} N={N} "
                f"ash_heap={ash_size_mb}MB "
                f"norm_weight=[{D}] linear_weight=[{N}, {2 * D}]",
                flush=True,
            )
        for rank_id in range(rank_size):
            if rank == rank_id:
                if passed.item():
                    details = " ".join(
                        f"{name}_ratio={value:.6g}"
                        for name, value in accuracy_details.items()
                    )
                    suffix = f" {details}" if details else ""
                    print(
                        f"[PASS] Rank {rank}: method={accuracy_method}{suffix}",
                        flush=True,
                    )
                else:
                    print(f"[FAIL] Rank {rank}:\n{error_message}", flush=True)
            dist.barrier()
        if any(value.item() == 0 for value in all_passed):
            return False
        return True
    finally:
        if peer_mem is not None:
            ash.aclshmem_free_tensor(peer_mem)
        if signal_mem is not None:
            ash.aclshmem_free_tensor(signal_mem)


def run_tutorial():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")

    # FUSED_RUN_BENCH enables one case per (S, D) pair built from
    # SEQUENCE_LENGTHS / HIDDEN_GROUPS / FEATURE_DIMS. Without it, a single
    # case runs with the default or env-overridden S/H/D values. Every case
    # uses N = D.
    bench_mode = "FUSED_RUN_BENCH" in os.environ
    eps = float(os.environ.get("FUSED_EPS", "1e-6"))
    ash_size_mb = int(os.environ.get("FUSED_ASH_SIZE_MB", str(DEFAULT_ASH_SIZE_MB)))
    comm_block_s = int(os.environ.get("FUSED_COMM_BLOCK_S", "1024"))
    buffer_num = 2
    if bench_mode:
        cases = [(S, HIDDEN_GROUPS, D) for S in SEQUENCE_LENGTHS for D in FEATURE_DIMS]
    else:
        cases = [
            (
                int(os.environ.get("FUSED_S", "128")),
                int(os.environ.get("FUSED_H", "4")),
                int(os.environ.get("FUSED_D", "128")),
            )
        ]

    dist.barrier()
    init_aclshmem(rank, rank_size, ash_size_mb * 1024 * 1024, ash.OpEngineType.UDMA)

    try:
        failed_cases = []
        for case_id, (S, H, D) in enumerate(cases):
            N = D
            if rank == 0 and len(cases) > 1:
                print(
                    f"CASE {case_id + 1}/{len(cases)}: S={S} H={H} D={D} N={N}",
                    flush=True,
                )
            case_passed = _run_single_case(
                rank,
                rank_size,
                device,
                S,
                H,
                D,
                eps,
                ash_size_mb,
                comm_block_s,
                buffer_num,
            )
            if not case_passed:
                failed_cases.append((S, H, D, N))
        if failed_cases:
            summary = ", ".join(
                f"S={S} H={H} D={D} N={N}" for S, H, D, N in failed_cases
            )
            raise AssertionError(
                "hccl_reverse_a2a_norm_matmul_2 accuracy check failed for cases: "
                f"{summary}"
            )
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tutorial()
