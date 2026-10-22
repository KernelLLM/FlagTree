"""C500 FlashAttention forward - kernel5: MACA C non-MMA alignment experiment.

Based on kernel4, preserving QK/PV MMA layouts and their register ownership.
Q/K/O use C++ D64-paged shared storage; V retains kernel4's C++ packing.
Softmax keeps raw-score maxima, applies scale with FMA before exp2, and
combines per-lane row sums after the KV loop. First and final tiles follow
separate paths; the final tile does not prefetch K. Q/K/V prologue order and
local-max / barrier / cross-lane-max ordering follow the C++ source.

FP16 launches select separate measured IGLP presets for causal and noncausal
execution. Other dtypes retain automatic backend scheduling. An explicit iglp
argument overrides either default. MMA fragments remain compiler-generated.
"""

import argparse
import math

import torch
import triton
from flash_attn.flash_attn_interface import _flash_attn_forward

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from maca_compose_shared_layout import (
    compose_maca_shared_layout_for_operand,
)


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
IGLP_FP16_NONCAUSAL = (2, 16, 4, -1, -1, 1, 1, 4)
IGLP_FP16_CAUSAL = (2, 16, 2, -1, -1, 1, 1, 8)


@gluon.constexpr_function
def _partial_sum_layout(BLOCK_M):
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


@gluon.jit
def _thread_row_values(p):
    # Four lane groups per row; the last sixteen values are lane-local.
    gl.static_assert(p.shape[1] == 64)
    grouped = gl.reshape(p, (p.shape[0], 2, 4, 8))
    return gl.reshape(grouped.permute(0, 2, 1, 3), (p.shape[0], 1, 4, 16))


@gluon.jit
def _thread_row_sum(p):
    partial_layout: gl.constexpr = _partial_sum_layout(p.shape[0])
    return gl.convert_layout(gl.sum(_thread_row_values(p), axis=3),
                             partial_layout, assert_trivial=True)


@gluon.jit
def _load_v_tile(v_head, n_block, v_stride_n, v_stride_d, SK,
                 BLOCK_N: gl.constexpr, BLOCK_D: gl.constexpr,
                 v_global_layout: gl.constexpr, MASKED: gl.constexpr):
    rows = n_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_global_layout))
    cols = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, v_global_layout))
    ptrs = v_head + rows[:, None] * v_stride_n + cols[None, :] * v_stride_d
    if MASKED:
        return gl.load(ptrs, mask=(n_block >= 0) & (rows[:, None] < SK), other=0.0)
    else:
        return gl.load(ptrs)


@gluon.jit
def _flash_fwd_tile(
    n_block,
    k_register,
    acc_o,
    m_i,
    l_i,
    q_for_score,
    q_m,
    k_head,
    v_head,
    k_stride_n,
    k_stride_d,
    v_stride_n,
    v_stride_d,
    SQ,
    SK,
    scale_log2,
    first_d,
    k_tn_smem,
    v_whole,
    tn_layout: gl.constexpr,
    global_layout: gl.constexpr,
    v_global_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    MASKED: gl.constexpr,
    CAUSAL: gl.constexpr,
    qk_mma_layout: gl.constexpr,
    qk_b_layout: gl.constexpr,
    pv_mma_layout: gl.constexpr,
    pv_a_layout: gl.constexpr,
    pv_b_layout: gl.constexpr,
    first_v_register,
    IS_FIRST: gl.constexpr,
    PREFETCH_NEXT: gl.constexpr,
):
    # Whole-block K: one store, one load, one QK MMA (no N32 fragment split).
    current_n = n_block * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=gl.SliceLayout(1, global_layout),
    )

    # Whole-block K store (same view as load below).
    k_tn_smem.store(k_register.permute(1, 0))
    if IS_FIRST:
        v_register = first_v_register
    else:
        v_register = _load_v_tile(v_head, n_block, v_stride_n, v_stride_d, SK,
                                  BLOCK_N, BLOCK_D, v_global_layout, MASKED)

    gl.metax.barrier_shared()
    # Whole-block K load → one QK MMA.
    kt_for_score = k_tn_smem.load(
        layout=qk_b_layout,
        intrinsic=True,
        is_constant_offs=True,
    )
    score = gl.dot(
        q_for_score,
        kt_for_score,
        gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout=qk_mma_layout),
        input_precision="tf32",
    )

    if MASKED:
        n_full = n_block * BLOCK_N + gl.arange(0, BLOCK_N)
        point_mask = (q_m[:, None] < SQ) & (n_full[None, :] < SK)
        if CAUSAL:
            causal_limit = q_m[:, None] + SK - SQ
            point_mask = point_mask & (n_full[None, :] <= causal_limit)
        score = gl.where(point_mask, score, -float("inf"))

    # Whole-block V store (same view as load below).
    v_whole.store(v_register)

    # C peels the final unmasked tile and does not prefetch past it.
    next_k_register = k_register
    if PREFETCH_NEXT:
        next_n = (n_block - 1) * BLOCK_N + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(1, global_layout))
        next_k_ptrs = k_head + next_n[:, None] * k_stride_n + first_d[None, :] * k_stride_d
        if MASKED:
            if n_block > 0:
                next_k_register = gl.load(next_k_ptrs)
        else:
            next_k_register = gl.load(next_k_ptrs)

    partial_layout: gl.constexpr = _partial_sum_layout(BLOCK_M)
    partial_row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, partial_layout))
    # C's local maximum includes the old maximum before the cross-lane step.
    local_max = gl.convert_layout(gl.max(_thread_row_values(score), axis=3),
                                  partial_layout, assert_trivial=True)
    if not IS_FIRST:
        previous_max = gl.convert_layout(m_i, partial_row_layout, assert_trivial=True)
        local_max = gl.maximum(local_max, previous_max[:, None, None])
    gl.metax.barrier_shared()
    m_ij = gl.convert_layout(gl.max(gl.reshape(local_max, (BLOCK_M, 4)), axis=1),
                             m_i.type.layout, assert_trivial=True)
    row_has_key = (q_m < SQ) & (m_ij != -float("inf"))
    stable_m = gl.where(row_has_key, m_ij, 0.0)

    if not IS_FIRST:
        alpha = gl.where(row_has_key, gl.exp2((m_i - stable_m) * scale_log2), 0.0)
        alpha_for_l = gl.convert_layout(alpha, partial_row_layout, assert_trivial=True)
        l_i = l_i * alpha_for_l[:, None, None]
        alpha_for_o = gl.convert_layout(alpha[:, None], pv_mma_layout)
        acc_o = acc_o * alpha_for_o

    # Keep raw QK maxima. Scale only at exp2, using the C++ FMA expression.
    max_scaled = stable_m * scale_log2
    p_qk = gl.exp2(gl.fma(score, scale_log2, -max_scaled[:, None]))
    if MASKED:
        p_qk = gl.where(point_mask, p_qk, 0.0)
    partial = _thread_row_sum(p_qk)
    if IS_FIRST:
        l_i = partial
    else:
        l_i = l_i + partial
    m_i = gl.where(row_has_key, m_ij, m_i)

    # Whole-block V load → one PV MMA.
    p = gl.convert_layout(
        p_qk.to(v_head.dtype.element_ty),
        pv_a_layout,
        assert_trivial=True,
    )
    v_full = v_whole.load(
        layout=pv_b_layout,
        intrinsic=True,
        is_constant_offs=True,
    )
    acc_o = gl.dot(p, v_full, acc_o, input_precision="tf32")
    return next_k_register, acc_o, m_i, l_i


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
    pid_m = gl.program_id(0)
    pid_bh = gl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_head = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    k_head = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h
    v_head = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h
    o_head = o_ptr + pid_b * o_stride_b + pid_h * o_stride_h
    lse_head = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h

    qk_elements_mnk: gl.constexpr = [1, 2, 8]
    pv_elements_mnk: gl.constexpr = [1, 2, 8]

    # C Q/O: Swizzle<3,3,3> independently in each D64 page.
    qo_d64_page_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[2, 0, 1]
    )
    # N64/D128 V packing, paired with v_global_layout below. The first eight
    # half elements are V[0:4, 0], then V[0:4, 4]. Each basis maps a bit of
    # the shared element offset to [row, column].
    v_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[
            [1, 0], [2, 0], [0, 4], [0, 8], [0, 16], [0, 32],
            [0, 1], [0, 2], [4, 0], [8, 0], [16, 0], [32, 0], [0, 64],
        ],
        alignment=16,
    )
    # C K shared: D64 pages; descriptor axes are [D, N].
    tn_layout: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [8, 1], [16, 2], [32, 4], [0, 8], [0, 16], [0, 32], [64, 0]], alignment=16,
    )
    qk_mma_layout: gl.constexpr = gl.MACAMmaLayout(
        version_major=2,
        version_minor=0,
        # Four warps partition the selected M tile without duplicating the
        # N64 score tile. PV retains the same ownership.
        warps_per_cta=[4, 1],
        elements_mnk=qk_elements_mnk,
        # C's atom owns M by lane%16 and N by 4*(lane//16)+value, and lowers
        # the intrinsic as (B, A, C).  That is MACA col_major=1, not the c0
        # layout used by kernels that stage QK-C through shared before PV-A.
        col_major=1,
    )
    qk_a_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma_layout, 0)
    qk_b_layout: gl.constexpr = gl.DotOperandLayout(1, qk_mma_layout, 0)
    qk_row_layout: gl.constexpr = gl.SliceLayout(1, qk_mma_layout)
    pv_mma_layout: gl.constexpr = gl.MACAMmaLayout(
        version_major=2,
        version_minor=0,
        # Match QK-C ownership so each score fragment becomes PV-A by a
        # trivial register retag.  V uses the ordinary Dot-B load contract.
        warps_per_cta=[4, 1],
        elements_mnk=pv_elements_mnk,
        # Keep PV-C in the same C-aligned accumulator orientation as QK-C.
        col_major=1,
    )
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma_layout, 0)
    pv_b_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma_layout, 0)
    # C's B128 tiled copy uses eight lanes per row and eight contiguous half
    # values per lane; four warps then cover the 32-row copy atom.
    global_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [8, 8],
        [4, 1],
        [1, 0],
    )
    # C's V producer owns four consecutive rows x eight columns per lane.
    # For wave 0, lane 0 reads V[0:4, 0:8]; lane 1 reads V[0:4, 8:16].
    # This ownership lets each lane form the cross-row packed shared stores.
    v_global_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[
            [0, 1], [0, 2], [0, 4], [1, 0], [2, 0],
        ],
        lane_bases=[
            [0, 8], [0, 16], [0, 32], [0, 64], [4, 0], [32, 0],
        ],
        warp_bases=[[8, 0], [16, 0]],
        block_bases=[],
        shape=[64, 128],
    )

    # Exactly 32 KiB for fp16/bf16. Q/O use buffer 0 in their epochs, while
    # K and V use disjoint buffers 0 and 1 in the main-loop epoch.
    shared_arena = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_D],
        num_buffers=2,
    )
    qo_d64_pages = shared_arena._reinterpret(
        q_ptr.dtype.element_ty, [BLOCK_M, BLOCK_D // 64, 64], qo_d64_page_layout,
    )
    q_smem = qo_d64_pages.reshape([BLOCK_M, BLOCK_D])
    o_smem = q_smem
    k_tn_smem = shared_arena.index(0)._reinterpret(
        k_ptr.dtype.element_ty,
        [BLOCK_D, BLOCK_N],
        tn_layout,
    )
    v_whole = shared_arena.index(1)._reinterpret(
        v_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_D],
        v_layout,
    )
    # Q and O are both D-contiguous in global memory.  Keep this producer /
    # sink layout explicit because layout propagation currently merges the Q
    # and O register constraints across their lifetime-disjoint shared alias.
    q_load_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=gl.SliceLayout(1, global_layout),
    )
    q_load_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, global_layout),
    )
    q_ptrs = (
        q_head
        + q_load_m[:, None] * q_stride_m
        + q_load_d[None, :] * q_stride_d
    )
    if EVEN_MN:
        q_register = gl.load(q_ptrs)
    else:
        q_register = gl.load(q_ptrs, mask=q_load_m[:, None] < SQ, other=0.0)

    q_m = pid_m * BLOCK_M + gl.arange(
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
        layout=gl.SliceLayout(1, global_layout),
    )
    first_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, global_layout),
    )
    q_smem.store(q_register)
    first_k_ptrs = (
        k_head
        + first_n[:, None] * k_stride_n
        + first_d[None, :] * k_stride_d
    )
    if EVEN_MN and not CAUSAL:
        k_register = gl.load(first_k_ptrs)
    else:
        first_valid = (first_n_block >= 0) & (first_n[:, None] < SK)
        k_register = gl.load(first_k_ptrs, mask=first_valid, other=0.0)

    gl.metax.barrier_shared()
    q_for_score = q_smem.load(
        layout=qk_a_layout,
        intrinsic=True,
        is_constant_offs=True,
    )

    first_v_register = _load_v_tile(
        v_head, first_n_block, v_stride_n, v_stride_d, SK,
        BLOCK_N, BLOCK_D, v_global_layout, not EVEN_MN or CAUSAL,
    )

    # Every thread must finish reading sQ before the first K tile aliases it.
    gl.metax.barrier_shared()

    acc_o = gl.zeros(
        (BLOCK_M, BLOCK_D),
        gl.float32,
        layout=pv_mma_layout,
    )
    m_i = gl.full(
        (BLOCK_M,),
        -float("inf"),
        gl.float32,
        layout=qk_row_layout,
    )
    partial_layout: gl.constexpr = _partial_sum_layout(BLOCK_M)
    l_i = gl.zeros((BLOCK_M, 1, 4), gl.float32, layout=partial_layout)

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

    if active_n_blocks > 0:
        n_block = first_n_block
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block, k_register, acc_o, m_i, l_i, q_for_score, q_m,
            k_head, v_head, k_stride_n, k_stride_d, v_stride_n, v_stride_d,
            SQ, SK, scale_log2, first_d, k_tn_smem, v_whole, tn_layout,
            global_layout, v_global_layout, BLOCK_M, BLOCK_N, BLOCK_D,
            True, CAUSAL, qk_mma_layout, qk_b_layout, pv_mma_layout,
            pv_a_layout, pv_b_layout, first_v_register, True, True,
        )

    for n_iteration in range(1, masked_blocks):
        n_block = first_n_block - n_iteration
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block, k_register, acc_o, m_i, l_i, q_for_score, q_m,
            k_head, v_head, k_stride_n, k_stride_d, v_stride_n, v_stride_d,
            SQ, SK, scale_log2, first_d, k_tn_smem, v_whole, tn_layout,
            global_layout, v_global_layout, BLOCK_M, BLOCK_N, BLOCK_D,
            True, CAUSAL, qk_mma_layout, qk_b_layout, pv_mma_layout,
            pv_a_layout, pv_b_layout, first_v_register, False, True,
        )

    for n_iteration in range(masked_blocks, active_n_blocks - 1):
        n_block = first_n_block - n_iteration
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block, k_register, acc_o, m_i, l_i, q_for_score, q_m,
            k_head, v_head, k_stride_n, k_stride_d, v_stride_n, v_stride_d,
            SQ, SK, scale_log2, first_d, k_tn_smem, v_whole, tn_layout,
            global_layout, v_global_layout, BLOCK_M, BLOCK_N, BLOCK_D,
            False, CAUSAL, qk_mma_layout, qk_b_layout, pv_mma_layout,
            pv_a_layout, pv_b_layout, first_v_register, False, True,
        )

    if active_n_blocks > masked_blocks:
        n_block = 0
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block, k_register, acc_o, m_i, l_i, q_for_score, q_m,
            k_head, v_head, k_stride_n, k_stride_d, v_stride_n, v_stride_d,
            SQ, SK, scale_log2, first_d, k_tn_smem, v_whole, tn_layout,
            global_layout, v_global_layout, BLOCK_M, BLOCK_N, BLOCK_D,
            False, CAUSAL, qk_mma_layout, qk_b_layout, pv_mma_layout,
            pv_a_layout, pv_b_layout, first_v_register, False, False,
        )

    if IGLP_C0 != 0:
        gl.metax.iglp(0, -1, -1, -1, -1, -1, -1, -1)

    # The only cross-lane sum is after all KV tiles have completed.
    l_i = gl.reshape(l_i, (BLOCK_M, 4))
    l_i = gl.convert_layout(gl.sum(l_i, axis=1), qk_row_layout, assert_trivial=True)
    row_has_key = (q_m < SQ) & (l_i > 0.0)
    safe_l = gl.where(row_has_key, l_i, 1.0)
    row_has_key_for_o = gl.convert_layout(
        row_has_key[:, None],
        pv_mma_layout,
    )
    inv_l = 1.0 / safe_l
    inv_l_for_o = gl.convert_layout(inv_l[:, None], pv_mma_layout)
    out = gl.where(
        row_has_key_for_o,
        acc_o * inv_l_for_o,
        0.0,
    )
    lse = gl.where(
        row_has_key,
        m_i * scale_softmax + gl.log(safe_l),
        float("inf"),
    )

    # C epilogue: the complete arena changes ownership from K|V to O.  Stage O
    # through shared so the final global writer has the coalesced Q/O layout.
    gl.metax.barrier_shared()
    o_smem.store(out.to(o_ptr.dtype.element_ty))
    gl.metax.barrier_shared()
    out_coalesced = o_smem.load(
        layout=global_layout,
        intrinsic=True,
        is_constant_offs=True,
    )

    # The shared round-trip changes ownership from the PV accumulator to a
    # D-contiguous global writer.  Rebuild offsets and mask in that ownership
    # domain instead of propagating the QK row layout into the final store.
    o_m = pid_m * BLOCK_M + gl.arange(
        0,
        BLOCK_M,
        layout=gl.SliceLayout(1, global_layout),
    )
    o_d = gl.arange(
        0,
        BLOCK_D,
        layout=gl.SliceLayout(0, global_layout),
    )
    o_ptrs = (
        o_head
        + o_m[:, None] * o_stride_m
        + o_d[None, :] * o_stride_d
    )
    gl.store(o_ptrs, out_coalesced, mask=o_m[:, None] < SQ)
    gl.store(
        lse_head + q_m * lse_stride_m,
        lse,
        mask=q_m < SQ,
    )


def _validate_inputs(q, k, v):
    if q.ndim != 4:
        raise ValueError("Q must be a BHSD tensor")
    batch, heads, seqlen_q, head_dim = q.shape
    if head_dim != D:
        raise ValueError(f"head dimension must be {D}, got {head_dim}")
    if k.ndim != 4 or k.shape[:2] != (batch, heads) or k.shape[3] != D:
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
        scale = 1.0 / math.sqrt(D)
    if iglp is None:
        if q.dtype == torch.float16:
            iglp = IGLP_FP16_CAUSAL if causal else IGLP_FP16_NONCAUSAL
        else:
            iglp = IGLP_DEFAULT

    block_m = BM64 if causal and q.dtype == torch.float16 else BM128
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
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
        BLOCK_D=D,
        CAUSAL=causal,
        # Mirrors C's Is_even_MN specialization.  Only proven full tiles can
        # select the dense noncausal load/score path without point masks.
        EVEN_MN=(seqlen_q % block_m == 0 and seqlen_k % BN == 0),
        IGLP_C0=iglp[0], IGLP_C1=iglp[1], IGLP_C2=iglp[2], IGLP_C3=iglp[3],
        IGLP_C4=iglp[4], IGLP_C5=iglp[5], IGLP_C6=iglp[6], IGLP_C7=iglp[7],
        num_warps=NUM_WARPS,
        num_stages=2,
        num_ctas=1,
        pipeline="cpasync",
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


def _run_case(seq_len, dtype, causal):
    torch.manual_seed(0)
    shape = (1, HEADS, seq_len, D)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(D)
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
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    rows = []
    results = []
    for seq_len in tuple(dict.fromkeys(args.seq_len)):
        result = _run_case(seq_len, dtype, args.causal)
        results.append(result)
        row = [
            f"1x{HEADS}x{seq_len}x{D}",
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
