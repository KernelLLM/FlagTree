"""MACA FlashAttention forward, D64/D128: kernel6 with register-only P fragments.

Each CTA keeps Q in registers and visits KV tiles in reverse order. A tile
computes QK, updates FP32 online softmax state, then accumulates PV through
four K16 fragments. QK also reads K in K16 fragments, prefetching one ahead.
QK's accumulator layout and PV's operand layout jointly
keep P in its producing lanes; assert_trivial guards this relationship.

Shared memory totals 16/32 KiB for D64/D128: Q/O reuse K's storage and, for
M128, V's storage. V has one/two D64 pages, each with four K16 fragments.
Noncausal Q/K use asynchronous copies; causal Q/K use register staging.
Row sums stay lane-local until the final normalization.
Both causal branches use the gluon-flashattn-fwd compiler scenario.

launch_flash_fwd accepts BHSD FP16/BF16 tensors. Running this file checks and
benchmarks the kernel against flash_attn with the same command-line interface
as the earlier kernels.

For the measured BSM-offset configuration, set TRITON_DISABLE_BSM_OFFSET_0=1
in this process's launch environment. Use a separate TRITON_CACHE_DIR for
BSM-enabled runs: this environment switch is not part of the compiler cache
key. No environment variables are changed by importing or calling this module.

Example from this directory (append --causal for causal attention):
    env TRITON_DISABLE_BSM_OFFSET_0=1 TRITON_CACHE_DIR=/tmp/fa-kernel6-bsm0 python 10-maca-flash-fwd-fused-single-kernel_6.py --dtype fp16 --head-dim 64 128 --seq-len 2048 8192
"""

import argparse
import math

import torch
import triton
from flash_attn.flash_attn_interface import _flash_attn_forward

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


BM64 = 64
BM128 = 128
BN = 64
D = 128
NUM_WARPS = 4
HEADS = 32
LOG2E = 1.4426950408889634
TEST_SEQ_LENS = (1024, 4096, 10240)
# Fall back to automatic backend scheduling for dtypes without a measured preset.
IGLP_DEFAULT = (0, 0, 0, 0, 0, 0, 0, 0)
# Measured with FP16, B1/H32/D128, 2K/8K on MetaX C550.
IGLP_FP16_NONCAUSAL = (2, 16, 1, -1, -1, 1, 1, 8)
IGLP_FP16_CAUSAL = (2, 16, 2, -1, -1, 1, 1, 8)


@gluon.constexpr_function
def _qk_mma_layout():
    # A lane owns four adjacent score columns in each K16 group. Keep this
    # coupled to _pv_mma_layout and _lane_row_values.
    return gl.MACAMmaLayout(
        version_major=2, version_minor=0, warps_per_cta=[4, 1],
        elements_mnk=[1, 1, 4], col_major=1,
    )


@gluon.constexpr_function
def _pv_mma_layout():
    # Dot-A accepts QK's four-column groups without changing lane ownership.
    return gl.MACAMmaLayout(
        version_major=2, version_minor=0, warps_per_cta=[4, 1],
        elements_mnk=[1, 2, 4], col_major=1,
    )


@gluon.constexpr_function
def _k_shared_layout(BLOCK_D):
    # C++ D64-paged K storage, viewed as [D, N] by the QK reader.
    bases = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [8, 1], [16, 2], [32, 4], [0, 8], [0, 16], [0, 32],
    ]
    if BLOCK_D == 128:
        bases += [[64, 0]]
    return gl.SharedLinearLayout(
        offset_bases=bases,
        alignment=16,
    )


@gluon.constexpr_function
def _v_shared_layout(BLOCK_D):
    # C++ V packing in D64 pages. Each basis maps a shared offset bit to
    # [row, column]; the first eight half elements are V[0:4, 0], V[0:4, 4].
    bases = [
        [1, 0], [2, 0], [0, 4], [0, 8], [0, 16], [0, 32],
        [0, 1], [0, 2], [4, 0], [8, 0], [16, 0], [32, 0],
    ]
    if BLOCK_D == 128:
        bases += [[0, 64]]
    return gl.SharedLinearLayout(
        offset_bases=bases,
        alignment=16,
    )


@gluon.constexpr_function
def _v_copy_layout(BLOCK_D):
    # Four consecutive rows per lane form V's packed stores. D64 uses four
    # columns per lane; D128 uses eight, retaining its original ownership.
    if BLOCK_D == 64:
        return gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [1, 0], [2, 0]],
            lane_bases=[[0, 4], [0, 8], [0, 16], [0, 32], [4, 0], [32, 0]],
            warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 64],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [32, 0]],
        warp_bases=[[8, 0], [16, 0]], block_bases=[], shape=[64, 128],
    )


@gluon.constexpr_function
def _v_fragment_shared_layout():
    # The first ten offset bits of _v_shared_layout describe one [K16, D64]
    # fragment. The remaining bits select K16 within D64, then the D64 page.
    return gl.SharedLinearLayout(
        offset_bases=[
            [1, 0], [2, 0], [0, 4], [0, 8], [0, 16], [0, 32],
            [0, 1], [0, 2], [4, 0], [8, 0],
        ],
        alignment=16,
    )


@gluon.constexpr_function
def _row_sum_layout(BLOCK_M):
    # The singleton middle axis keeps the partial-state layout independent
    # of the rank-2 MMA carrier in RemoveLayoutConversions. It has no extra
    # elements: each lane still carries one sum per row that it owns.
    if BLOCK_M == 128:
        reg_bases = [[64, 0, 0]]
    else:
        assert BLOCK_M == 64
        reg_bases = []
    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1], [0, 0, 2]],
        warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[BLOCK_M, 1, 4],
    )


@gluon.constexpr_function
def _paired_sum_layout(BLOCK_M):
    reg_bases = [[0, 0, 0, 1]]
    if BLOCK_M == 128:
        reg_bases += [[64, 0, 0, 0]]
    else:
        assert BLOCK_M == 64
    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=[[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0],
                    [0, 0, 1, 0], [0, 0, 2, 0]],
        warp_bases=[[16, 0, 0, 0], [32, 0, 0, 0]], block_bases=[],
        shape=[BLOCK_M, 1, 4, 2],
    )


@gluon.jit
def _lane_row_values(p):
    # Four lanes share each row. The last axis selects one lane's 16 values:
    # four adjacent columns from each of the four K16 groups.
    gl.static_assert(p.shape[1] == 64)
    grouped = gl.reshape(p, (p.shape[0], 4, 4, 4))
    return gl.reshape(grouped.permute(0, 2, 1, 3), (p.shape[0], 1, 4, 16))


@gluon.jit
def _split_register_halves(values):
    # Split the register-only final axis into its low and high halves.
    grouped = gl.reshape(values, (values.shape[0], 1, 4, 2, values.shape[3] // 2))
    return gl.split(grouped.permute(0, 1, 2, 4, 3))


@gluon.jit
def _accumulate_row_sum(p, previous_sum, IS_FIRST: gl.constexpr):
    # Sum sixteen lane-local probabilities using two independent chains.
    # Seed one chain with the rescaled previous sum, then combine the chains.
    values = _lane_row_values(p)
    low8, high8 = _split_register_halves(values)
    low4, midlow4 = _split_register_halves(low8)
    midhigh4, high4 = _split_register_halves(high8)
    pair0, pair1 = _split_register_halves(low4)
    pair2, pair3 = _split_register_halves(midlow4)
    pair4, pair5 = _split_register_halves(midhigh4)
    pair6, pair7 = _split_register_halves(high4)
    pairs = (pair0, pair1, pair2, pair3, pair4, pair5, pair6, pair7)
    pair_layout: gl.constexpr = _paired_sum_layout(p.shape[0])
    if IS_FIRST:
        pair_sum = gl.zeros((p.shape[0], 1, 4, 2), gl.float32, pair_layout)
    else:
        # previous_sum has already been rescaled by the new row maximum.
        pair_sum = gl.convert_layout(
            gl.join(previous_sum, gl.zeros_like(previous_sum)), pair_layout,
            assert_trivial=True,
        )
    for pair in gl.static_range(8):
        pair_values = gl.convert_layout(pairs[pair], pair_layout, assert_trivial=True)
        pair_sum = pair_sum + pair_values
    first, second = gl.split(pair_sum)
    partial_layout: gl.constexpr = _row_sum_layout(p.shape[0])
    return gl.convert_layout(first + second, partial_layout, assert_trivial=True)


@gluon.jit
def _lane_row_max(score, row_max, IS_FIRST: gl.constexpr):
    """Combine this lane's score maximum with the previous row maximum."""
    BLOCK_M: gl.constexpr = score.shape[0]
    partial_layout: gl.constexpr = _row_sum_layout(BLOCK_M)
    partial_row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, partial_layout))
    local_max = gl.convert_layout(
        gl.max(_lane_row_values(score), axis=3), partial_layout, assert_trivial=True,
    )
    if not IS_FIRST:
        previous_max = gl.convert_layout(row_max, partial_row_layout, assert_trivial=True)
        local_max = gl.maximum(local_max, previous_max[:, None, None])
    return local_max


@gluon.jit
def _update_softmax(
    score, local_max, output_acc, row_max, row_sum, q_rows, SQ, scale_log2, point_mask,
    MASKED: gl.constexpr, IS_FIRST: gl.constexpr,
):
    """Update the FP32 online state and return unnormalized probabilities."""
    BLOCK_M: gl.constexpr = score.shape[0]
    pv_mma_layout: gl.constexpr = output_acc.type.layout
    partial_layout: gl.constexpr = _row_sum_layout(BLOCK_M)
    partial_row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, partial_layout))
    new_max = gl.convert_layout(
        gl.max(gl.reshape(local_max, (BLOCK_M, 4)), axis=1),
        row_max.type.layout, assert_trivial=True,
    )
    row_has_key = (q_rows < SQ) & (new_max != -float("inf"))
    safe_max = gl.where(row_has_key, new_max, 0.0)

    if not IS_FIRST:
        alpha = gl.where(row_has_key, gl.exp2((row_max - safe_max) * scale_log2), 0.0)
        sum_rescale = gl.convert_layout(alpha, partial_row_layout, assert_trivial=True)
        row_sum = row_sum * sum_rescale[:, None, None]
        output_rescale = gl.convert_layout(alpha[:, None], pv_mma_layout)
        output_acc = output_acc * output_rescale

    # Keep raw QK maxima. Scale only at exp2, using the C++ FMA expression.
    max_scaled = safe_max * scale_log2
    probabilities = gl.exp2(gl.fma(score, scale_log2, -max_scaled[:, None]))
    if MASKED:
        probabilities = gl.where(point_mask, probabilities, 0.0)
    row_sum = _accumulate_row_sum(probabilities, row_sum, IS_FIRST)
    row_max = gl.where(row_has_key, new_max, row_max)

    return probabilities, output_acc, row_max, row_sum


@gluon.jit
def _accumulate_pv(probabilities, v_smem, output_acc):
    """Read four K16 V fragments and consume P entirely from registers."""
    BLOCK_M: gl.constexpr = probabilities.shape[0]
    BLOCK_D: gl.constexpr = output_acc.shape[1]
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(0, output_acc.type.layout, 0)
    pv_b_layout: gl.constexpr = gl.DotOperandLayout(1, output_acc.type.layout, 0)
    # Matrix instructions consume P in V's dtype. QK owns four contiguous
    # columns per lane, matching K16 PV dot-A; the layout retag and all four
    # fragment selections must stay register-only.
    p_half = gl.convert_layout(probabilities.to(v_smem.dtype), pv_a_layout, assert_trivial=True)
    p_low32, p_high32 = gl.split(gl.reshape(p_half, (BLOCK_M, 2, 32)).permute(0, 2, 1))
    p0, p1 = gl.split(gl.reshape(p_low32, (BLOCK_M, 2, 16)).permute(0, 2, 1))
    p2, p3 = gl.split(gl.reshape(p_high32, (BLOCK_M, 2, 16)).permute(0, 2, 1))
    # Reinterpret the existing packed bytes; no shared-memory copy is needed.
    # Indexing the small fragments exposes constant offsets from a shared base.
    v_fragment_shared: gl.constexpr = _v_fragment_shared_layout()
    v_fragments = v_smem._reinterpret(v_smem.dtype, [4 * (BLOCK_D // 64), 16, 64], v_fragment_shared)
    fragments = (p0, p1, p2, p3)
    for fragment in gl.static_range(4):
        p_fragment = gl.convert_layout(fragments[fragment], pv_a_layout, assert_trivial=True)
        v_low = v_fragments.index(fragment).load(
            layout=pv_b_layout, intrinsic=True, is_constant_offs=True,
        )
        if BLOCK_D == 128:
            v_high = v_fragments.index(fragment + 4).load(
                layout=pv_b_layout, intrinsic=True, is_constant_offs=True,
            )
            v_fragment = gl.join(v_low, v_high).permute(0, 2, 1).reshape([16, 128])
        else:
            v_fragment = v_low
        v_fragment = gl.convert_layout(v_fragment, pv_b_layout, assert_trivial=True)
        output_acc = gl.dot(p_fragment, v_fragment, output_acc, input_precision="tf32")
    return output_acc


@gluon.jit
def _load_v_tile(v_head, n_block, v_stride_n, v_stride_d, SK,
                 BLOCK_N: gl.constexpr, BLOCK_D: gl.constexpr,
                 v_copy_layout: gl.constexpr, MASKED: gl.constexpr):
    rows = n_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_copy_layout))
    cols = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, v_copy_layout))
    ptrs = v_head + rows[:, None] * v_stride_n + cols[None, :] * v_stride_d
    if MASKED:
        return gl.load(ptrs, mask=(n_block >= 0) & (rows[:, None] < SK), other=0.0)
    else:
        return gl.load(ptrs)


@gluon.jit
def _copy_k_async(k_head, n_block, k_stride_n, k_stride_d, SK, k_d64_pages,
                  BLOCK_N: gl.constexpr, copy_layout: gl.constexpr,
                  MASKED: gl.constexpr):
    # Copy each D64 page into the K reader's shared storage.
    NUM_PAGES: gl.constexpr = k_d64_pages.shape[0]
    rows = n_block * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, copy_layout))
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, copy_layout))
    for page in gl.static_range(NUM_PAGES):
        ptrs = k_head + rows[:, None] * k_stride_n + (page * 64 + cols[None, :]) * k_stride_d
        if MASKED:
            mask = (n_block >= 0) & (rows[:, None] < SK)
            gl.metax.async_copy_global_to_shared(
                k_d64_pages.index(page), ptrs, mask=mask, other=0.0, intrinsic=True)
        else:
            gl.metax.async_copy_global_to_shared(k_d64_pages.index(page), ptrs, intrinsic=True)


@gluon.jit
def _qk_k16(q_tile, k_smem):
    """Keep Q in registers and prefetch one K16 shared fragment ahead."""
    BLOCK_M: gl.constexpr = q_tile.shape[0]
    BLOCK_N: gl.constexpr = k_smem.shape[1]
    BLOCK_D: gl.constexpr = q_tile.shape[1]
    qk_mma_layout: gl.constexpr = q_tile.type.layout.parent
    qk_a_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma_layout, 0)
    qk_b_layout: gl.constexpr = gl.DotOperandLayout(1, qk_mma_layout, 0)
    # Split/reshape selects each lane's Q registers without moving values.
    if BLOCK_D == 128:
        q_low64, q_high64 = gl.split(q_tile.reshape((BLOCK_M, 2, 64)).permute(0, 2, 1))
    else:
        q_low64 = q_tile
    q0_32, q1_32 = gl.split(q_low64.reshape((BLOCK_M, 2, 32)).permute(0, 2, 1))
    q0, q1 = gl.split(q0_32.reshape((BLOCK_M, 2, 16)).permute(0, 2, 1))
    q2, q3 = gl.split(q1_32.reshape((BLOCK_M, 2, 16)).permute(0, 2, 1))
    if BLOCK_D == 128:
        q2_32, q3_32 = gl.split(q_high64.reshape((BLOCK_M, 2, 32)).permute(0, 2, 1))
        q4, q5 = gl.split(q2_32.reshape((BLOCK_M, 2, 16)).permute(0, 2, 1))
        q6, q7 = gl.split(q3_32.reshape((BLOCK_M, 2, 16)).permute(0, 2, 1))
        q_fragments = (q0, q1, q2, q3, q4, q5, q6, q7)
    else:
        q_fragments = (q0, q1, q2, q3)
    score = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, qk_mma_layout)
    current_k = k_smem.slice(0, 16, dim=0).load(
        layout=qk_b_layout, intrinsic=True, is_constant_offs=True,
    )
    for fragment in gl.static_range(BLOCK_D // 16):
        if fragment + 1 < BLOCK_D // 16:
            next_k = k_smem.slice((fragment + 1) * 16, 16, dim=0).load(
                layout=qk_b_layout, intrinsic=True, is_constant_offs=True,
            )
        q_fragment = gl.convert_layout(q_fragments[fragment], qk_a_layout, assert_trivial=True)
        score = gl.dot(q_fragment, current_k, score, input_precision="tf32")
        if fragment + 1 < BLOCK_D // 16:
            current_k = next_k
    return score


@gluon.jit
def _attention_tile(
    n_block, state, query, kv_heads, kv_strides, sequence_lengths, scale_log2,
    kv_shared, first_v_register,
    copy_layout: gl.constexpr, v_copy_layout: gl.constexpr,
    MASKED: gl.constexpr, CAUSAL: gl.constexpr,
    IS_FIRST: gl.constexpr, PREFETCH_NEXT: gl.constexpr,
):
    """Run one KV tile, overlapping next-K prefetch with softmax and PV."""
    k_register, output_acc, row_max, row_sum = state
    q_tile, q_rows = query
    k_head, v_head = kv_heads
    k_stride_n, k_stride_d, v_stride_n, v_stride_d = kv_strides
    SQ, SK = sequence_lengths
    k_smem, k_d64_pages, v_smem = kv_shared
    BLOCK_N: gl.constexpr = v_smem.shape[0]
    BLOCK_D: gl.constexpr = v_smem.shape[1]
    head_cols = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, copy_layout))

    if CAUSAL:
        k_smem.store(k_register.permute(1, 0))
    else:
        # Complete and publish K before reading shared or issuing V loads.
        gl.metax.gvm_arrive(0)
        gl.metax.barrier()
    if IS_FIRST:
        v_register = first_v_register
    else:
        v_register = _load_v_tile(
            v_head, n_block, v_stride_n, v_stride_d, SK,
            BLOCK_N, BLOCK_D, v_copy_layout, MASKED=MASKED,
        )

    if CAUSAL:
        gl.metax.barrier_shared()
    score = _qk_k16(q_tile, k_smem)

    point_mask = True
    if MASKED:
        n_full = n_block * BLOCK_N + gl.arange(0, BLOCK_N)
        point_mask = (q_rows[:, None] < SQ) & (n_full[None, :] < SK)
        if CAUSAL:
            causal_limit = q_rows[:, None] + SK - SQ
            point_mask = point_mask & (n_full[None, :] <= causal_limit)
        score = gl.where(point_mask, score, -float("inf"))

    v_smem.store(v_register)

    # Causal K prefetch lives in registers until the next tile's shared store.
    next_k_register = k_register
    if PREFETCH_NEXT and CAUSAL:
        next_n = (n_block - 1) * BLOCK_N + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(1, copy_layout))
        next_k_ptrs = k_head + next_n[:, None] * k_stride_n + head_cols[None, :] * k_stride_d
        if MASKED:
            if n_block > 0:
                next_k_register = gl.load(next_k_ptrs)
        else:
            next_k_register = gl.load(next_k_ptrs)

    local_max = _lane_row_max(score, row_max, IS_FIRST)
    # Publish V and finish all reads of current K before reusing K's buffer.
    gl.metax.barrier_shared()
    if PREFETCH_NEXT and not CAUSAL:
        if MASKED:
            if n_block > 0:
                _copy_k_async(
                    k_head, n_block - 1, k_stride_n, k_stride_d, SK,
                    k_d64_pages, BLOCK_N, copy_layout, MASKED=False,
                )
        else:
            _copy_k_async(
                k_head, n_block - 1, k_stride_n, k_stride_d, SK,
                k_d64_pages, BLOCK_N, copy_layout, MASKED=False,
            )
    probabilities, output_acc, row_max, row_sum = _update_softmax(
        score, local_max, output_acc, row_max, row_sum, q_rows, SQ, scale_log2, point_mask,
        MASKED=MASKED, IS_FIRST=IS_FIRST,
    )
    output_acc = _accumulate_pv(probabilities, v_smem, output_acc)
    return next_k_register, output_acc, row_max, row_sum


@gluon.jit
def _normalize_output(output_acc, row_max, row_sum, q_rows, SQ, scale_softmax):
    """Finish the cross-lane sum and normalize output; empty rows return 0/+inf."""
    BLOCK_M: gl.constexpr = output_acc.shape[0]
    qk_row_layout: gl.constexpr = row_max.type.layout
    pv_mma_layout: gl.constexpr = output_acc.type.layout
    # The only cross-lane sum is after all KV tiles have completed.
    row_sum = gl.reshape(row_sum, (BLOCK_M, 4))
    row_sum = gl.convert_layout(gl.sum(row_sum, axis=1), qk_row_layout, assert_trivial=True)
    row_has_key = (q_rows < SQ) & (row_sum > 0.0)
    safe_sum = gl.where(row_has_key, row_sum, 1.0)
    row_has_key_for_o = gl.convert_layout(
        row_has_key[:, None],
        pv_mma_layout,
    )
    inv_sum = 1.0 / safe_sum
    output_inv_sum = gl.convert_layout(inv_sum[:, None], pv_mma_layout)
    out = gl.where(
        row_has_key_for_o,
        output_acc * output_inv_sum,
        0.0,
    )
    lse = gl.where(
        row_has_key,
        row_max * scale_softmax + gl.log(safe_sum),
        float("inf"),
    )

    return out, lse


@gluon.jit
def _store_output(
    out, lse, o_smem, o_head, lse_head, q_rows, pid_m, SQ,
    o_stride_m, o_stride_d, lse_stride_m, copy_layout: gl.constexpr,
):
    """Redistribute the accumulator through shared for coalesced global stores."""
    BLOCK_M: gl.constexpr = out.shape[0]
    BLOCK_D: gl.constexpr = out.shape[1]
    # Finish K/V reads before the arena becomes O storage.
    gl.metax.barrier_shared()
    o_smem.store(out.to(o_head.dtype.element_ty))
    gl.metax.barrier_shared()
    out_coalesced = o_smem.load(
        layout=copy_layout,
        intrinsic=True,
        is_constant_offs=True,
    )

    # Build output offsets in the coalesced writer's layout.
    o_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=gl.SliceLayout(1, copy_layout),
    )
    o_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, copy_layout),
    )
    o_ptrs = (
        o_head
        + o_m[:, None] * o_stride_m
        + o_d[None, :] * o_stride_d
    )
    gl.store(o_ptrs, out_coalesced, mask=o_m[:, None] < SQ)
    gl.store(
        lse_head + q_rows * lse_stride_m,
        lse,
        mask=q_rows < SQ,
    )


@gluon.jit
def flash_fwd_hdim128_generic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    SQ,
    SK,
    H: gl.constexpr,
    q_stride_b,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_n,
    v_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_m,
    o_stride_d,
    lse_stride_b,
    lse_stride_h,
    lse_stride_m,
    scale_log2,
    scale_softmax,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    CAUSAL: gl.constexpr,
    EVEN_MN: gl.constexpr,
    IGLP_C0: gl.constexpr = 0,
    IGLP_C1: gl.constexpr = 0,
    IGLP_C2: gl.constexpr = 0,
    IGLP_C3: gl.constexpr = 0,
    IGLP_C4: gl.constexpr = 0,
    IGLP_C5: gl.constexpr = 0,
    IGLP_C6: gl.constexpr = 0,
    IGLP_C7: gl.constexpr = 0,
):
    # Match C++'s (head, batch, Q-block) grid and reverse Q-block mapping.
    pid_h = gl.program_id(0)
    pid_b = gl.program_id(1)
    pid_m = gl.cdiv(SQ, BLOCK_M) - 1 - gl.program_id(2)

    q_head = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    k_head = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h
    v_head = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h
    o_head = o_ptr + pid_b * o_stride_b + pid_h * o_stride_h
    lse_head = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h

    gl.static_assert(BLOCK_N == 64 and (BLOCK_D == 64 or BLOCK_D == 128))
    qk_mma_layout: gl.constexpr = _qk_mma_layout()
    pv_mma_layout: gl.constexpr = _pv_mma_layout()
    qk_a_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma_layout, 0)
    qk_row_layout: gl.constexpr = gl.SliceLayout(1, qk_mma_layout)
    # Q/K/O copies: eight contiguous half elements per lane.
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    v_copy_layout: gl.constexpr = _v_copy_layout(BLOCK_D)
    # Swizzle<3,3,3> independently in each Q/O D64 page.
    qo_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [2, 0, 1])
    k_shared_layout: gl.constexpr = _k_shared_layout(BLOCK_D)
    v_shared_layout: gl.constexpr = _v_shared_layout(BLOCK_D)

    # M64 Q/K/O fit in K's allocation. Give V its own allocation so its full-tile
    # store also proves initialization of every indexed fragment view.
    # M128 Q/O span both K and V, so retain a single arena there.
    if BLOCK_M == 64:
        shared_arena = gl.local_alloc(q_ptr.dtype.element_ty, [BLOCK_N, BLOCK_D])
        k_storage = shared_arena
        v_smem = gl.local_alloc(
            v_ptr.dtype.element_ty, [BLOCK_N, BLOCK_D], layout=v_shared_layout,
        )
    else:
        shared_arena = gl.local_alloc(
            q_ptr.dtype.element_ty, [BLOCK_N, BLOCK_D], num_buffers=2,
        )
        k_storage = shared_arena.index(0)
        v_smem = shared_arena.index(1)._reinterpret(
            v_ptr.dtype.element_ty, [BLOCK_N, BLOCK_D], v_shared_layout,
        )
    qo_d64_pages = shared_arena._reinterpret(
        q_ptr.dtype.element_ty, [BLOCK_M, BLOCK_D // 64, 64], qo_shared_layout,
    )
    q_smem = qo_d64_pages.reshape([BLOCK_M, BLOCK_D])
    o_smem = q_smem
    k_smem = k_storage._reinterpret(
        k_ptr.dtype.element_ty,
        [BLOCK_D, BLOCK_N],
        k_shared_layout,
    )
    if not CAUSAL:
        k_page_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
        k_d64_pages = k_storage._reinterpret(
            k_ptr.dtype.element_ty, [BLOCK_D // 64, BLOCK_N, 64], k_page_layout,
        )
    else:
        # Unused on the compile-time causal branch.
        k_d64_pages = k_smem
    # Prologue: stage Q, start the first K/V loads, then keep Q in registers.
    # Q/O need explicit copy layouts because they alias the same shared view.
    q_load_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=gl.SliceLayout(1, copy_layout),
    )
    q_load_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, copy_layout),
    )
    q_ptrs = (
        q_head
        + q_load_m[:, None] * q_stride_m
        + q_load_d[None, :] * q_stride_d
    )
    if CAUSAL:
        if EVEN_MN:
            q_register = gl.load(q_ptrs)
        else:
            q_register = gl.load(q_ptrs, mask=q_load_m[:, None] < SQ, other=0.0)
    else:
        # The async writer views Q storage as contiguous D64 pages.
        q_copy_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
        q_copy_pages = shared_arena._reinterpret(
            q_ptr.dtype.element_ty, [BLOCK_D // 64, BLOCK_M, 64], q_copy_layout,
        )
        q_copy_d = gl.arange(0, 64, layout=gl.SliceLayout(0, copy_layout))
        for page in gl.static_range(BLOCK_D // 64):
            q_copy_ptrs = (
                q_head + q_load_m[:, None] * q_stride_m
                + (page * 64 + q_copy_d[None, :]) * q_stride_d
            )
            if EVEN_MN:
                gl.metax.async_copy_global_to_shared(
                    q_copy_pages.index(page), q_copy_ptrs, intrinsic=True)
            else:
                gl.metax.async_copy_global_to_shared(
                    q_copy_pages.index(page), q_copy_ptrs,
                    mask=q_load_m[:, None] < SQ, other=0.0, intrinsic=True)

    q_rows = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=qk_row_layout,
    )

    num_n_blocks = gl.cdiv(SK, BLOCK_N)
    if CAUSAL:
        causal_hi = (pid_m + 1) * BLOCK_M + SK - SQ
        active_n_blocks = min(num_n_blocks, max(0, gl.cdiv(causal_hi, BLOCK_N)))
    else:
        active_n_blocks = num_n_blocks
    first_n_block = active_n_blocks - 1

    first_n = first_n_block * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=gl.SliceLayout(1, copy_layout),
    )
    head_cols = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, copy_layout),
    )
    if CAUSAL:
        q_smem.store(q_register)
    first_k_ptrs = (
        k_head
        + first_n[:, None] * k_stride_n
        + head_cols[None, :] * k_stride_d
    )
    if CAUSAL:
        first_valid = (first_n_block >= 0) & (first_n[:, None] < SK)
        k_register = gl.load(first_k_ptrs, mask=first_valid, other=0.0)
    else:
        # No K data tensor is carried across loop iterations.
        k_register = 0

    if CAUSAL:
        gl.metax.barrier_shared()
    else:
        # Complete Q's global-to-shared transfers before any wave reads Q.
        gl.metax.gvm_arrive(0)
        gl.metax.barrier()
    q_tile = q_smem.load(
        layout=qk_a_layout,
        intrinsic=True,
        is_constant_offs=True,
    )

    first_v_register = _load_v_tile(
        v_head, first_n_block, v_stride_n, v_stride_d, SK,
        BLOCK_N, BLOCK_D, v_copy_layout, MASKED=not EVEN_MN or CAUSAL,
    )

    # Finish Q reads before K/V reuse its shared storage.
    gl.metax.barrier_shared()
    if not CAUSAL:
        _copy_k_async(
            k_head, first_n_block, k_stride_n, k_stride_d, SK,
            k_d64_pages, BLOCK_N, copy_layout, MASKED=not EVEN_MN,
        )

    output_acc = gl.zeros(
        (BLOCK_M, BLOCK_D),
        gl.float32,
        layout=pv_mma_layout,
    )
    row_max = gl.full(
        (BLOCK_M,),
        -float("inf"),
        gl.float32,
        layout=qk_row_layout,
    )
    partial_layout: gl.constexpr = _row_sum_layout(BLOCK_M)
    row_sum = gl.zeros((BLOCK_M, 1, 4), gl.float32, layout=partial_layout)

    if CAUSAL:
        if EVEN_MN:
            masking_steps: gl.constexpr = gl.cdiv(BLOCK_M, BLOCK_N)
        else:
            masking_steps: gl.constexpr = gl.cdiv(BLOCK_M, BLOCK_N) + 1
    else:
        masking_steps: gl.constexpr = 1
    masked_blocks = min(active_n_blocks, masking_steps)

    if IGLP_C0 != 0:
        gl.metax.iglp(IGLP_C0, IGLP_C1, IGLP_C2, IGLP_C3,
                      IGLP_C4, IGLP_C5, IGLP_C6, IGLP_C7)

    # State: prefetched K, unnormalized output, raw-score maximum, lane-local sum.
    state = (k_register, output_acc, row_max, row_sum)
    query = (q_tile, q_rows)
    kv_heads = (k_head, v_head)
    kv_strides = (k_stride_n, k_stride_d, v_stride_n, v_stride_d)
    sequence_lengths = (SQ, SK)
    kv_shared = (k_smem, k_d64_pages, v_smem)

    # Reverse traversal starts at the sequence/causal boundary. Peel the first
    # tile to initialize softmax, and the final dense tile to stop K prefetch.
    if active_n_blocks > 0:
        n_block = first_n_block
        state = _attention_tile(
            n_block, state, query, kv_heads, kv_strides, sequence_lengths, scale_log2,
            kv_shared, first_v_register, copy_layout, v_copy_layout,
            MASKED=True, CAUSAL=CAUSAL, IS_FIRST=True, PREFETCH_NEXT=True,
        )

    for n_iteration in range(1, masked_blocks):
        n_block = first_n_block - n_iteration
        state = _attention_tile(
            n_block, state, query, kv_heads, kv_strides, sequence_lengths, scale_log2,
            kv_shared, first_v_register, copy_layout, v_copy_layout,
            MASKED=True, CAUSAL=CAUSAL, IS_FIRST=False, PREFETCH_NEXT=True,
        )

    for n_iteration in range(masked_blocks, active_n_blocks - 1):
        n_block = first_n_block - n_iteration
        state = _attention_tile(
            n_block, state, query, kv_heads, kv_strides, sequence_lengths, scale_log2,
            kv_shared, first_v_register, copy_layout, v_copy_layout,
            MASKED=False, CAUSAL=CAUSAL, IS_FIRST=False, PREFETCH_NEXT=True,
        )

    if active_n_blocks > masked_blocks:
        n_block = 0
        state = _attention_tile(
            n_block, state, query, kv_heads, kv_strides, sequence_lengths, scale_log2,
            kv_shared, first_v_register, copy_layout, v_copy_layout,
            MASKED=False, CAUSAL=CAUSAL, IS_FIRST=False, PREFETCH_NEXT=False,
        )

    if IGLP_C0 != 0:
        gl.metax.iglp(0, -1, -1, -1, -1, -1, -1, -1)

    _, output_acc, row_max, row_sum = state
    out, lse = _normalize_output(output_acc, row_max, row_sum, q_rows, SQ, scale_softmax)
    _store_output(
        out, lse, o_smem, o_head, lse_head, q_rows, pid_m, SQ,
        o_stride_m, o_stride_d, lse_stride_m, copy_layout,
    )


def _validate_inputs(q, k, v):
    if q.ndim != 4:
        raise ValueError("Q must be a BHSD tensor")
    batch, heads, seqlen_q, head_dim = q.shape
    if head_dim not in (64, 128):
        raise ValueError(f"head dimension must be 64 or 128, got {head_dim}")
    if k.ndim != 4 or k.shape[:2] != (batch, heads) or k.shape[3] != head_dim:
        raise ValueError(f"invalid K shape {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError("V must have the same shape as K")
    if seqlen_q <= 0 or k.shape[2] <= 0:
        raise ValueError("query and key sequence lengths must be positive")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported dtype {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K, and V must have the same dtype")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous in the head dimension")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("Q, K, and V must be on the same device")
    return batch, heads, seqlen_q, k.shape[2]


def launch_flash_fwd(q, k, v, out=None, lse=None, scale=None, causal=False,
                     iglp=None):
    batch, heads, seqlen_q, seqlen_k = _validate_inputs(q, k, v)
    head_dim = q.shape[3]
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(
            (batch, heads, seqlen_q),
            device=q.device,
            dtype=torch.float32,
        )
    if out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("output must match Q and be contiguous in head dimension")
    if lse.shape != (batch, heads, seqlen_q) or lse.dtype != torch.float32:
        raise ValueError("LSE must be fp32 [B,H,SQ]")
    if not lse.is_contiguous():
        raise ValueError("LSE must be contiguous")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if iglp is None:
        if q.dtype == torch.float16:
            iglp = IGLP_FP16_CAUSAL if causal else IGLP_FP16_NONCAUSAL
        else:
            iglp = IGLP_DEFAULT

    # Match the xcore1000 reference kernel's M128 causal tile.
    block_m = BM128
    grid = (heads, batch, triton.cdiv(seqlen_q, block_m))
    flash_fwd_hdim128_generic_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        seqlen_q,
        seqlen_k,
        heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        float(scale * LOG2E),
        float(scale),
        BLOCK_M=block_m,
        BLOCK_N=BN,
        BLOCK_D=head_dim,
        CAUSAL=causal,
        # Full sequence tiles allow unmasked prologue copies; causal boundary
        # tiles and the first softmax tile still use the masked specialization.
        EVEN_MN=(seqlen_q % block_m == 0 and seqlen_k % BN == 0),
        IGLP_C0=iglp[0], IGLP_C1=iglp[1], IGLP_C2=iglp[2], IGLP_C3=iglp[3],
        IGLP_C4=iglp[4], IGLP_C5=iglp[5], IGLP_C6=iglp[6], IGLP_C7=iglp[7],
        num_warps=NUM_WARPS,
        num_stages=2,
        num_ctas=1,
        pipeline="basic",
        scenario="gluon-flashattn-fwd",
    )
    return out, lse


def _prepare_flash_attn_forward(q, k, v, scale, causal):
    # flash_attn consumes BSHD tensors. Keep layout conversion outside both
    # measured regions so the table compares device attention calls directly.
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    def launch():
        return _flash_attn_forward(
            q_fa,
            k_fa,
            v_fa,
            0.0,
            scale,
            causal,
            (-1, -1),
        )

    (
        _,
        _,
        _,
        _,
        out_fa,
        lse_fa,
        _,
        _,
        _,
    ) = launch()
    return out_fa.transpose(1, 2), lse_fa, launch


def _run_case(seq_len, dtype, causal, head_dim=D):
    torch.manual_seed(0)
    shape = (1, HEADS, seq_len, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    out, lse = launch_flash_fwd(q, k, v, scale=scale, causal=causal)
    flash_out, flash_lse, flash_launch = _prepare_flash_attn_forward(
        q, k, v, scale, causal
    )
    result = {
        "allclose": torch.allclose(
            out, flash_out, atol=3e-2, rtol=3e-2
        )
        and torch.allclose(lse, flash_lse, atol=3e-2, rtol=3e-2),
    }
    result["gluon_ms"] = triton.testing.do_bench(
        lambda: launch_flash_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            scale=scale,
            causal=causal,
        ),
        warmup=10,
        rep=30,
    )
    result["flash_ms"] = triton.testing.do_bench(
        flash_launch,
        warmup=10,
        rep=30,
    )
    result["speedup_vs_flash"] = result["flash_ms"] / result["gluon_ms"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", nargs="+", type=int, default=TEST_SEQ_LENS)
    parser.add_argument("--head-dim", nargs="+", type=int, choices=(64, 128), default=(D,))
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    rows = []
    results = []
    for head_dim in tuple(dict.fromkeys(args.head_dim)):
        for seq_len in tuple(dict.fromkeys(args.seq_len)):
            result = _run_case(seq_len, dtype, args.causal, head_dim)
            results.append(result)
            row = [
                f"1x{HEADS}x{seq_len}x{head_dim}",
                dtype,
                args.causal,
                f"{result['gluon_ms']:.6f}",
                f"{result['flash_ms']:.6f}",
                f"{result['speedup_vs_flash']:.3f}",
                result["allclose"],
            ]
            rows.append(row)

    headers = [
        "BxHxSxD",
        "dtype",
        "causal",
        "gluon_ms",
        "_flash_attn_forward_ms",
        "gluon_speedup_vs_flash",
        "accuracy_ok",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")
    if not all(result["allclose"] for result in results):
        raise AssertionError(f"FlashAttention forward accuracy failure: {rows}")


if __name__ == "__main__":
    main()
