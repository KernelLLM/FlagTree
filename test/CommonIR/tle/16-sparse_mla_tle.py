"""Self-contained DSA sparse-MLA forward kernel for MetaX C550 (Triton + TLE).

Kernel + input builders + FP32 reference + TLE/FlagGems A/B harness in one file.

API
    out, lse = sparse_mla_fwd(q, kv, indices, sm_scale=576 ** -0.5, d_v=512)

    q        [B, SQ, H, DQK]      bf16, contiguous
    kv       [B, SKV, VG, DQK]    bf16, contiguous
    indices  [B, SQ, VG, topk]    int32; padding must be SKV (or any value
             >= SKV or < 0). Slot j of query t is valid iff indices <= t.
    out      [B, SQ, H, d_v]      bf16
    lse      [B, SQ, H]           bf16, **log2 domain** (the vendor C kernel
             returns natural-log lse -- multiply by log2(e) to compare)

DISPATCH (plan(), a pure function of shape)
    topk <  256                                 -> _fwd_bk32  (BK=32, D split 4x)
    topk >= 256, G < 32 or topk < 2048          -> _fwd_bk64  (BK=64, 4 warps)
    topk >= 2048 and G >= 32 (GLM)              -> _fwd_bk64  (BH=32, 8 warps,
                                                   mma_warps (2,4), v_from_trans)
    The BK crossover at topk 256 is measured: a causal BK=64 tile wastes up to
    63 keys, so the narrow tile wins below it and the wide tile wins above.
    The GLM branch's 8 warps halve the per-thread live set, which is what takes
    n_spills from 26 to 0; mma_warps (2,4) is the only 8-warp MMA grid that
    also bottomed out the allocator ((1,8)/(4,2) stay at 256 regs, spill 26/78).

TOPK SPLIT (GLM branch only)
    Axis 0 of the launch becomes N_GROUPS * VG * NH; each CTA owns topk tiles
    [g*NK/NG, (g+1)*NK/NG) and writes partial_o/partial_lse, and one extra
    launch merges them in the log2-LSE domain.  This is a *grid-occupancy* fix,
    not a work reduction, so it only triggers when the unsplit grid cannot fill
    the machine:
        unsplit_CTA = VG * cdiv(G,BH) * SQ * B
        split iff unsplit_CTA < GLM_DS_WAVE_GATE(=4) * AP(=104)  ->  < 416
        NG = 4 (halved until it divides NK = cdiv(topk, BK))
    The 13 official GLM shapes all have SQ=4096, so they stay NG=1 and the
    partial stores are compile-time dead code.  The rule is kept because it is
    the right fix if a short-SQ GLM shape ever lands (it was worth 1.35x on the
    now-retired glm_sq64).

ACCURACY
    s1-s4 and all 13 GLM shapes pass against an FP32 reference (maxerr <= 0.0005
    for GLM, <= 0.0156 bf16-level for DSA).  TD == 0 (DQK == d_v, e.g. DQK=512)
    is handled: the header QK dot is skipped and its contribution is exactly
    zero.  Note _next_pow2(0) == 1 here, so TDP would be 1 and the dot's K
    extent would fall below Triton's minimum -- that was a real compile failure
    on 8 of the 13 official shapes.

RESOURCES (C550, from _return_meta)
    DSA (bk32, D split 4x): smem 26624 B -> 2 CTA/AP, n_regs 194-256, spills 0
    GLM (bk64, D split 8x): smem 53248 B -> 1 CTA/AP, n_regs 202-204, spills 0
    53248 B > the 32768 B half-SMEM mark, so the GLM body is ONE CTA per AP.

WHY IT WINS
    A softmax with no running max and no in-loop reduction: logits stay within
    +-6.7 in log2 units, so a fixed zero offset is safe and the row sum rides
    along as a PV dot against a ones column.  BK=64 above topk 256, MMA element
    slices of 8 columns.  Known limitation: the Python wrapper costs ~58 us of
    host time per call at s1 (Triton JIT launch), so small-shape latency is
    partly launch-bound.

STRUCTURAL CEILING on the SQ=4096 GLM shapes (~0.53x of the vendor C kernel)
    The gap is KV READ TRAFFIC, not latency hiding (a pure-read floor reached
    1677 GB/s, sequential addresses saved only 9%, and scaling BH 16->32 scaled
    time 1.878/1.889/1.893 -- strictly linear in NH, the number of times a CTA
    re-reads the same KV).  With BH=32 and G=64, NH=2; the vendor packs query
    tokens and heads into one M dimension so it reads KV once (NH=1).  Reaching
    NH=1 needs BH=64, whose Q [64,576] bf16 is 73728 B > the 64 KB smem limit;
    its only compiling form (a dynamic inner loop) regressed 1.9x for a 0.5x
    traffic win.  Exhausted and falsified -- do not retry: BH=64 static/alloc,
    BH=48/24, D-chunk static unroll (90112 B), BK 64->32/16, iglp, loop_stages,
    unroll=2, mma_warps (1,8)/(4,2), 2-CTA/AP Vgather form (26624 B, 15 spills,
    29-34% slower), 16 warps (over the 512-thread cap), async K double-buffer
    (needs 16384 B vs 12288 B slack), and topk splitting on large SQ (5-30%).

BASELINE
    ``flag_gems`` is the reference baseline (resolve_impl); ``torch`` (the FP32
    reference) is the accuracy oracle.  The former ``maca`` column was removed
    on 2026-11-23: it was never mcoplib (mcoplib ships no sparse-MLA
    implementation), it was the separate ``flash_mla`` wheel, it ran only 5 of
    the 13 GLM shapes, and it is not the project's baseline.  FlagGems-vllm's
    own sparse-MLA kernels need 4 source patches plus small-tile overrides to
    run on C550 and land 22-29x slower -- correctness cross-check only.

RUN (from anywhere)
    python 16-sparse_mla_tle.py --list-cases
    python 16-sparse_mla_tle.py --suite dsa                      # s1-s4
    python 16-sparse_mla_tle.py --impls sparse_mla_tle,torch     # accuracy
    python 16-sparse_mla_tle.py --impls sparse_mla_tle,flag_gems --csv
    python 16-sparse_mla_tle.py --case s1 --no-check             # latency only

Each case runs in its own worker process (FlagGems autotuning stays isolated);
timings are do_bench medians, warmup=50 ms / rep=100 ms.  TLE correctness is
always verified against the FP32 reference, never against the baseline.
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.experimental.tle.language.gpu.metax import DotOperandEncoding, MmaEncoding

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton-cache"))


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


# --- topk-split auto dispatch -------------------------------------------------
# Splitting topk across CTAs is a *grid-occupancy* fix, not a work reduction: it
# writes N_GROUPS copies of (partial_o, partial_lse) and adds one merge launch.
# It only pays while the unsplit launch cannot fill the machine.  Measured on the
# now-retired short-SQ GLM shapes: glm_sq64 (128 CTA, far under one wave) went
# 0.8182 -> 0.6039 ms (1.35x); every shape that already filled the machine
# regressed 5-30%.  So: only split below GLM_DS_WAVE_GATE waves.
# The 13 official GLM shapes all have SQ=4096, so none of them split.
GLM_DS_NG = 4          # splits per query when a split is taken (NK must divide)
GLM_DS_WAVE_GATE = 4   # unsplit CTA count < this many waves -> split
_AP_COUNT = None


def _ap_count():
    """AP count of the current device (cached)."""
    global _AP_COUNT
    if _AP_COUNT is None:
        _AP_COUNT = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _AP_COUNT


def _pick_n_groups(cta_unsplit, NK):
    """Topk-split count for an unsplit grid of `cta_unsplit` CTAs."""
    if cta_unsplit >= GLM_DS_WAVE_GATE * _ap_count():
        return 1
    ng = GLM_DS_NG
    while ng > 1 and NK % ng != 0:
        ng //= 2
    return ng


# ===========================================================================
# Body 1: BK=32, D split four ways (topk < 256)
# ===========================================================================
@triton.jit
def _sparse_mla_bk32_kernel(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvg: tl.constexpr,
    stride_kvn: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_tb: tl.constexpr,
    stride_tg: tl.constexpr,
    stride_tm: tl.constexpr,
    stride_tt: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lm: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    SKV: tl.constexpr,
    is_causal: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    V_FROM_TRANS: tl.constexpr,
    SUM_MODE: tl.constexpr,
    UNROLL: tl.constexpr,
    IGLP: tl.constexpr,
    A_LAYOUT: tl.constexpr,
    B_LAYOUT: tl.constexpr,
    MMA: tl.constexpr,
):
    DC: tl.constexpr = DP // 4
    i_gbh, i_sq_raw, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Heaviest query first: the causal tile count grows with i_sq.
    i_sq = SQ - 1 - i_sq_raw
    NH = tl.cdiv(G, BH)
    i_g, i_bh = i_gbh // NH, i_gbh % NH
    q_base = q + i_b * stride_qb + i_sq * stride_qm + i_gbh * (BH * stride_qh)
    tq_base = q_base + D * stride_qd
    kv_base = kv + i_b * stride_kvb + i_g * stride_kvg
    tkv_base = kv_base + D * stride_kvd
    t_base = indices + i_b * stride_tb + i_sq * stride_tm + i_g * stride_tg
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_gbh * (BH * stride_oh)
    l_base = lse + i_b * stride_lb + i_sq * stride_lm + i_gbh * (BH * stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d1 = tl.arange(0, DC)
    offs_td = tl.arange(0, TDP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_td = offs_td < TD
    md0 = offs_d1 < D
    md1 = (DC + offs_d1) < D
    md2 = (2 * DC + offs_d1) < D
    md3 = (3 * DC + offs_d1) < D

    # Q is loaded one D-chunk at a time (see the module docstring): each chunk is
    # an independent global->dot-operand conversion whose shared staging buffer
    # is dead as soon as the chunk is in registers.
    q0_ptr = q_base + offs_h[:, None] * stride_qh + offs_d1[None, :] * stride_qd
    q0 = tle.gpu.set_layout(tl.load(q0_ptr, mask_h[:, None] & md0[None, :], other=0.0),
                            A_LAYOUT)
    q1 = tle.gpu.set_layout(tl.load(q0_ptr + DC * stride_qd,
                                    mask_h[:, None] & md1[None, :], other=0.0), A_LAYOUT)
    q2 = tle.gpu.set_layout(tl.load(q0_ptr + 2 * DC * stride_qd,
                                    mask_h[:, None] & md2[None, :], other=0.0), A_LAYOUT)
    q3 = tle.gpu.set_layout(tl.load(q0_ptr + 3 * DC * stride_qd,
                                    mask_h[:, None] & md3[None, :], other=0.0), A_LAYOUT)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tle.gpu.set_layout(tl.load(tq_ptr, tq_msk, other=0.0), A_LAYOUT)

    # Row statistics stay rank-2 ([BH, 1]): a rank-1 value that picks up
    # SliceEncoding(1, mma) and is then expand_dims'ed back builds
    # SliceEncoding(0, SliceEncoding(...)), whose parent rank is 1 and which
    # trips SliceEncodingAttr::verify ("parent layout must have at least
    # rank >= 2") during ConvertTritonToTritonGPU.
    sum_exp = tl.full([BH, 1], 0.0, dtype=tl.float32)
    acc0 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc1 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc2 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc3 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    qk0 = tle.gpu.set_layout(tl.zeros([BH, BK], dtype=tl.float32), MMA)
    if SUM_MODE == 1:
        # A ones column, so one extra PV dot accumulates the row sums and the
        # loop needs no reduction at all.  All 16 columns get the same value; the
        # epilogue reads column 0.
        ones_b = tle.gpu.set_layout(
            tl.full([BK, 16], 1.0, dtype=tl.bfloat16), B_LAYOUT)
        acc_s = tle.gpu.set_layout(tl.zeros([BH, 16], dtype=tl.float32), MMA)

    log_scale: tl.constexpr = sm_scale * 1.44269504

    max_col = i_sq if is_causal else SKV - 1

    NK = tl.cdiv(K, BK)
    nk_eff = tl.minimum(NK, max_col // BK + 1)
    for ck in tl.range(nk_eff, num_stages=LOOP_STAGES, loop_unroll_factor=UNROLL):
        if True:
            t_ptr = (BK * ck + offs_t) * stride_tt
            t_msk = t_ptr < K
            t_ptr += t_base
            kv_ids = tl.load(t_ptr, t_msk, other=-1)
            mask_ids = (kv_ids <= max_col) & (kv_ids >= 0) & (kv_ids < SKV)

            # TD == 0 (DQK == d_v, e.g. DQK=512) means there is no rope tail:
            # TDP = _next_pow2(0) = 1, so the header dot's K extent is 1, below
            # Triton's minimum dot size.  The contribution is exactly zero then,
            # so the accumulation starts from qk0 unchanged.
            if TD > 0:
                tkv_blk = tle.gpu.set_layout(
                    tl.load(
                        tkv_base + offs_td[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn,
                        mask_td[:, None] & mask_ids[None, :],
                        other=0.0,
                    ),
                    B_LAYOUT,
                )
                qk = tl.dot(tq_blk, tkv_blk, qk0, out_dtype=tl.float32)
            else:
                qk = qk0

            k0 = tl.load(
                kv_base + offs_d1[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn,
                md0[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q0, tle.gpu.set_layout(k0, B_LAYOUT), qk, out_dtype=tl.float32)
            k1 = tl.load(
                kv_base + (DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md1[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q1, tle.gpu.set_layout(k1, B_LAYOUT), qk, out_dtype=tl.float32)
            k2 = tl.load(
                kv_base + (2 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md2[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q2, tle.gpu.set_layout(k2, B_LAYOUT), qk, out_dtype=tl.float32)
            k3 = tl.load(
                kv_base + (3 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md3[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q3, tle.gpu.set_layout(k3, B_LAYOUT), qk, out_dtype=tl.float32)

            if V_FROM_TRANS:
                # V is the same tile as K, read transposed.  If the compiler
                # lowers ``trans`` of the smem staging to ``memdesc_trans``
                # this costs no second buffer and no second global gather.
                vt0 = tle.gpu.set_layout(tl.trans(k0), B_LAYOUT)
                vt1 = tle.gpu.set_layout(tl.trans(k1), B_LAYOUT)
                vt2 = tle.gpu.set_layout(tl.trans(k2), B_LAYOUT)
                vt3 = tle.gpu.set_layout(tl.trans(k3), B_LAYOUT)
            else:
                # V is gathered in [key, dim] order: the PV dot's B operand.
                vt0 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + offs_d1[None, :] * stride_kvd,
                            mask_ids[:, None] & md0[None, :], other=0.0), B_LAYOUT)
                vt1 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md1[None, :], other=0.0), B_LAYOUT)
                vt2 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (2 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md2[None, :], other=0.0), B_LAYOUT)
                vt3 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (3 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md3[None, :], other=0.0), B_LAYOUT)

            # Invalid keys get -inf, and exp2(-inf) is exactly 0 -- the same
            # value the reference produces for a masked key.
            qk = tl.where(mask_ids[None, :], qk, float("-inf"))
            exp_qk = tl.math.exp2(qk * log_scale)
            if SUM_MODE == 0:
                # Reduce while the values are still fp32 in the MMA layout the
                # QK dot produced.  Converting to the A operand layout first would
                # make the reduction cross layouts as well as waves.
                sum_exp += tl.sum(exp_qk, axis=1, keep_dims=True)
            exp_qk = tle.gpu.set_layout(exp_qk.to(tl.bfloat16), A_LAYOUT)
            if SUM_MODE == 1:
                # No reduction: the same exp_qk that feeds PV also feeds a dot
                # against a ones column, so the row sum arrives as an
                # accumulator.  bf16 rounding of exp_qk is the same rounding the
                # PV dots already apply to it.
                acc_s = tl.dot(exp_qk, ones_b, acc_s, out_dtype=tl.float32)
            acc0 = tl.dot(exp_qk, vt0, acc0, out_dtype=tl.float32)
            acc1 = tl.dot(exp_qk, vt1, acc1, out_dtype=tl.float32)
            acc2 = tl.dot(exp_qk, vt2, acc2, out_dtype=tl.float32)
            acc3 = tl.dot(exp_qk, vt3, acc3, out_dtype=tl.float32)
            if IGLP == 1:
                # fused-flash config: 16 prefetched shared loads, 4 MMA between
                # shared-load groups, 1 shared load per group, 1 MMA between
                # global loads, 8 MMA between the last shared load and ARRIVE.
                tle.gpu.metax.iglp(config_0=2, config_1=16, config_2=4,
                                   config_5=1, config_6=1, config_7=8)
            elif IGLP == 2:
                tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=5)

    if SUM_MODE == 1:
        # All 16 columns of acc_s hold the row sum; read column 0.
        sum_exp = tl.sum(acc_s, axis=1, keep_dims=True) * 0.0625
    # A row whose every key was invalid sums to exactly 0.  Dividing would give
    # 0/0, so the select supplies the reference's out=0 / lse=-inf instead.
    ok = sum_exp > 0.0
    inv = 1.0 / tl.where(ok, sum_exp, 1.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d1[None, :] * stride_od
    keep = mask_h[:, None] & ok
    tl.store(o_ptr, tl.where(keep, acc0 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md0[None, :])
    tl.store(o_ptr + DC * stride_od, tl.where(keep, acc1 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md1[None, :])
    tl.store(o_ptr + 2 * DC * stride_od, tl.where(keep, acc2 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md2[None, :])
    tl.store(o_ptr + 3 * DC * stride_od, tl.where(keep, acc3 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md3[None, :])

    fin_log = tl.where(ok, tl.math.log2(tl.where(ok, sum_exp, 1.0)), float("-inf"))
    l_ptr = l_base + offs_h[:, None] * stride_lh
    tl.store(l_ptr, fin_log.to(tl.bfloat16), mask_h[:, None])


def _fwd_bk32(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum=False,
    d_v=512,
    BH=16,
    BK=32,
    loop_stages=0,
    v_from_trans=False,
    sum_mode=0,
    unroll=1,
    num_warps=4,
    num_stages=1,
    mma_warps=(1, 4),
    mma_elements=(1, 1, 4),
    col_major=False,
    iglp=2,
    compile_opts=None,
    _return_meta=False,
    is_causal=True,
):
    """_fwd_bk32: BK/dsplit body, see the module docstring."""
    assert not return_p_sum, "forward only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, SKV, VG, _ = kv.shape

    D = d_v
    assert kv.shape[-1] == DT
    TD = DT - D
    DP = _next_pow2(D)
    TDP = _next_pow2(TD)
    assert kv.shape[0] == B
    _, _, _, K = indices.shape
    assert indices.shape == (B, SQ, VG, K)
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5
    assert DP % 4 == 0, "the BK=32 body splits D four ways"

    NH = triton.cdiv(G, BH)

    mma = MmaEncoding((2, 0), mma_warps, mma_elements, col_major)
    a_layout = DotOperandEncoding(0, mma, 0)
    b_layout = DotOperandEncoding(1, mma, 0)

    # The epilogue stores every element of every row it owns (masked rows get an
    # explicit 0 / -inf), so a pinned default is only needed for the head rows
    # that fall outside G -- i.e. when G does not tile BH.  Zeroing the output
    # costs 10.5 / 15.4 / 26.1 / 47.6 us on s1-s4, which is the same order as
    # the whole remaining gap to the vendor, and the vendor allocates with
    # torch::empty.
    if G % BH == 0 and D == DP:
        output = torch.empty((B, SQ, H, D), device=q.device, dtype=q.dtype)
        lse = torch.empty((B, SQ, H), device=q.device, dtype=q.dtype)
    else:
        output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
        lse = torch.full((B, SQ, H), float("-inf"), device=q.device, dtype=q.dtype)
    grid = (VG * NH, SQ, B)

    compiled = _sparse_mla_bk32_kernel[grid](
        q,
        kv,
        indices,
        sm_scale,
        output,
        lse,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        q.stride(3),
        kv.stride(0),
        kv.stride(2),
        kv.stride(1),
        kv.stride(3),
        indices.stride(0),
        indices.stride(2),
        indices.stride(1),
        indices.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        output.stride(3),
        lse.stride(0),
        lse.stride(2),
        lse.stride(1),
        SQ,
        K,
        D,
        TD,
        DP,
        TDP,
        G,
        BK,
        BH,
        SKV,
        is_causal,
        loop_stages,
        v_from_trans,
        sum_mode,
        unroll,
        iglp,
        A_LAYOUT=a_layout,
        B_LAYOUT=b_layout,
        MMA=mma,
        num_warps=num_warps,
        num_stages=num_stages,
        **(compile_opts or {}),
    )
    if _return_meta:
        return output, lse, compiled
    return output, lse


# ===========================================================================
# Body 2: BK=64, D split eight ways (topk >= 256)
# ===========================================================================
@triton.jit
def _sparse_mla_bk64_kernel(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    partial_o,
    partial_lse,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvg: tl.constexpr,
    stride_kvn: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_tb: tl.constexpr,
    stride_tg: tl.constexpr,
    stride_tm: tl.constexpr,
    stride_tt: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_pob: tl.constexpr,
    stride_pom: tl.constexpr,
    stride_pog: tl.constexpr,
    stride_poh: tl.constexpr,
    stride_pod: tl.constexpr,
    stride_plb: tl.constexpr,
    stride_plm: tl.constexpr,
    stride_plg: tl.constexpr,
    stride_plh: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    SKV: tl.constexpr,
    is_causal: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    V_FROM_TRANS: tl.constexpr,
    SUM_MODE: tl.constexpr,
    UNROLL: tl.constexpr,
    IGLP: tl.constexpr,
    N_GROUPS: tl.constexpr,
    A_LAYOUT: tl.constexpr,
    B_LAYOUT: tl.constexpr,
    MMA: tl.constexpr,
):
    DC: tl.constexpr = DP // 8
    i_gbh_all, i_sq_raw, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NH = tl.cdiv(G, BH)
    # Topk split: when N_GROUPS > 1, axis 0 is (group, kv-group, head-block).
    # With N_GROUPS == 1, i_grp is 0 and folds away at compile time, so the
    # unsplit launch is unchanged.
    if N_GROUPS > 1:
        i_grp = i_gbh_all // (tl.num_programs(0) // N_GROUPS)
        i_gbh = i_gbh_all % (tl.num_programs(0) // N_GROUPS)
    else:
        i_grp = 0
        i_gbh = i_gbh_all
    # Heaviest query first: the causal tile count grows with i_sq.
    i_sq = SQ - 1 - i_sq_raw
    i_g, i_bh = i_gbh // NH, i_gbh % NH
    q_base = q + i_b * stride_qb + i_sq * stride_qm + i_gbh * (BH * stride_qh)
    tq_base = q_base + D * stride_qd
    kv_base = kv + i_b * stride_kvb + i_g * stride_kvg
    tkv_base = kv_base + D * stride_kvd
    t_base = indices + i_b * stride_tb + i_sq * stride_tm + i_g * stride_tg
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_gbh * (BH * stride_oh)
    l_base = lse + i_b * stride_lb + i_sq * stride_lm + i_gbh * (BH * stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d1 = tl.arange(0, DC)
    offs_td = tl.arange(0, TDP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_td = offs_td < TD
    md0 = (offs_d1) < D
    md1 = (DC + offs_d1) < D
    md2 = (2 * DC + offs_d1) < D
    md3 = (3 * DC + offs_d1) < D
    md4 = (4 * DC + offs_d1) < D
    md5 = (5 * DC + offs_d1) < D
    md6 = (6 * DC + offs_d1) < D
    md7 = (7 * DC + offs_d1) < D

    # Q is loaded one D-chunk at a time: each chunk is an independent
    # global->dot-operand conversion whose shared staging buffer is dead as soon
    # as the chunk is in registers.
    q0_ptr = q_base + offs_h[:, None] * stride_qh + offs_d1[None, :] * stride_qd
    q0 = tle.gpu.set_layout(
        tl.load(q0_ptr, mask_h[:, None] & md0[None, :], other=0.0), A_LAYOUT)
    q1 = tle.gpu.set_layout(
        tl.load(q0_ptr + DC * stride_qd, mask_h[:, None] & md1[None, :], other=0.0), A_LAYOUT)
    q2 = tle.gpu.set_layout(
        tl.load(q0_ptr + 2 * DC * stride_qd, mask_h[:, None] & md2[None, :], other=0.0), A_LAYOUT)
    q3 = tle.gpu.set_layout(
        tl.load(q0_ptr + 3 * DC * stride_qd, mask_h[:, None] & md3[None, :], other=0.0), A_LAYOUT)
    q4 = tle.gpu.set_layout(
        tl.load(q0_ptr + 4 * DC * stride_qd, mask_h[:, None] & md4[None, :], other=0.0), A_LAYOUT)
    q5 = tle.gpu.set_layout(
        tl.load(q0_ptr + 5 * DC * stride_qd, mask_h[:, None] & md5[None, :], other=0.0), A_LAYOUT)
    q6 = tle.gpu.set_layout(
        tl.load(q0_ptr + 6 * DC * stride_qd, mask_h[:, None] & md6[None, :], other=0.0), A_LAYOUT)
    q7 = tle.gpu.set_layout(
        tl.load(q0_ptr + 7 * DC * stride_qd, mask_h[:, None] & md7[None, :], other=0.0), A_LAYOUT)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tle.gpu.set_layout(tl.load(tq_ptr, tq_msk, other=0.0), A_LAYOUT)

    # Row statistics stay rank-2 ([BH, 1]): a rank-1 value that picks up
    # SliceEncoding(1, mma) and is then expand_dims'ed back builds
    # SliceEncoding(0, SliceEncoding(...)), whose parent rank is 1 and which
    # trips SliceEncodingAttr::verify during ConvertTritonToTritonGPU.
    sum_exp = tl.full([BH, 1], 0.0, dtype=tl.float32)
    acc0 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc1 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc2 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc3 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc4 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc5 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc6 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    acc7 = tle.gpu.set_layout(tl.zeros([BH, DC], dtype=tl.float32), MMA)
    qk0 = tle.gpu.set_layout(tl.zeros([BH, BK], dtype=tl.float32), MMA)
    if SUM_MODE == 1:
        # One extra PV dot accumulates the row sums, so the tile loop needs no
        # reduction at all.  All 16 columns hold the same value.
        ones_b = tle.gpu.set_layout(tl.full([BK, 16], 1.0, dtype=tl.bfloat16), B_LAYOUT)
        acc_s = tle.gpu.set_layout(tl.zeros([BH, 16], dtype=tl.float32), MMA)

    log_scale: tl.constexpr = sm_scale * 1.44269504

    max_col = i_sq if is_causal else SKV - 1

    NK = tl.cdiv(K, BK)
    nk_eff = tl.minimum(NK, max_col // BK + 1)
    # Topk split: this CTA owns tiles [ck_lo, ck_hi) of the topk range.  The
    # LSE-domain merge in _merge_partial_kernel makes the split exact, so any
    # divisor of NK is legal; N_GROUPS == 1 gives ck_lo=0, ck_hi=nk_eff, i.e.
    # the original loop bound.
    if N_GROUPS > 1:
        ck_lo = i_grp * (NK // N_GROUPS)
        ck_hi = tl.minimum(ck_lo + (NK // N_GROUPS), nk_eff)
    else:
        ck_lo = 0
        ck_hi = nk_eff
    for ck in tl.range(ck_lo, ck_hi, num_stages=LOOP_STAGES, loop_unroll_factor=UNROLL):
        if True:
            t_ptr = (BK * ck + offs_t) * stride_tt
            t_msk = t_ptr < K
            t_ptr += t_base
            kv_ids = tl.load(t_ptr, t_msk, other=-1)
            mask_ids = (kv_ids <= max_col) & (kv_ids >= 0) & (kv_ids < SKV)

            # TD == 0 (DQK == d_v, e.g. DQK=512) means there is no rope tail:
            # TDP = _next_pow2(0) = 1, so the header dot's K extent is 1, below
            # Triton's minimum dot size.  The contribution is exactly zero then,
            # so the accumulation starts from qk0 unchanged.
            if TD > 0:
                tkv_blk = tle.gpu.set_layout(
                    tl.load(
                        tkv_base + offs_td[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn,
                        mask_td[:, None] & mask_ids[None, :],
                        other=0.0,
                    ),
                    B_LAYOUT,
                )
                qk = tl.dot(tq_blk, tkv_blk, qk0, out_dtype=tl.float32)
            else:
                qk = qk0

            k0 = tl.load(
                kv_base + offs_d1[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn,
                md0[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q0, tle.gpu.set_layout(k0, B_LAYOUT), qk, out_dtype=tl.float32)
            k1 = tl.load(
                kv_base + (DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md1[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q1, tle.gpu.set_layout(k1, B_LAYOUT), qk, out_dtype=tl.float32)
            k2 = tl.load(
                kv_base + (2 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md2[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q2, tle.gpu.set_layout(k2, B_LAYOUT), qk, out_dtype=tl.float32)
            k3 = tl.load(
                kv_base + (3 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md3[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q3, tle.gpu.set_layout(k3, B_LAYOUT), qk, out_dtype=tl.float32)
            k4 = tl.load(
                kv_base + (4 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md4[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q4, tle.gpu.set_layout(k4, B_LAYOUT), qk, out_dtype=tl.float32)
            k5 = tl.load(
                kv_base + (5 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md5[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q5, tle.gpu.set_layout(k5, B_LAYOUT), qk, out_dtype=tl.float32)
            k6 = tl.load(
                kv_base + (6 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md6[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q6, tle.gpu.set_layout(k6, B_LAYOUT), qk, out_dtype=tl.float32)
            k7 = tl.load(
                kv_base + (7 * DC + offs_d1)[:, None] * stride_kvd
                + kv_ids[None, :] * stride_kvn,
                md7[:, None] & mask_ids[None, :], other=0.0)
            qk = tl.dot(q7, tle.gpu.set_layout(k7, B_LAYOUT), qk, out_dtype=tl.float32)

            if V_FROM_TRANS:
                # V is the same tile as K, read transposed.  If the compiler
                # lowers ``trans`` of the smem staging to ``memdesc_trans``
                # this costs no second buffer and no second global gather.
                vt0 = tle.gpu.set_layout(tl.trans(k0), B_LAYOUT)
                vt1 = tle.gpu.set_layout(tl.trans(k1), B_LAYOUT)
                vt2 = tle.gpu.set_layout(tl.trans(k2), B_LAYOUT)
                vt3 = tle.gpu.set_layout(tl.trans(k3), B_LAYOUT)
                vt4 = tle.gpu.set_layout(tl.trans(k4), B_LAYOUT)
                vt5 = tle.gpu.set_layout(tl.trans(k5), B_LAYOUT)
                vt6 = tle.gpu.set_layout(tl.trans(k6), B_LAYOUT)
                vt7 = tle.gpu.set_layout(tl.trans(k7), B_LAYOUT)
            else:
                # V is gathered in [key, dim] order: the PV dot's B operand.
                vt0 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + offs_d1[None, :] * stride_kvd,
                            mask_ids[:, None] & md0[None, :], other=0.0), B_LAYOUT)
                vt1 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md1[None, :], other=0.0), B_LAYOUT)
                vt2 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (2 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md2[None, :], other=0.0), B_LAYOUT)
                vt3 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (3 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md3[None, :], other=0.0), B_LAYOUT)
                vt4 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (4 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md4[None, :], other=0.0), B_LAYOUT)
                vt5 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (5 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md5[None, :], other=0.0), B_LAYOUT)
                vt6 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (6 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md6[None, :], other=0.0), B_LAYOUT)
                vt7 = tle.gpu.set_layout(
                    tl.load(kv_base + kv_ids[:, None] * stride_kvn
                            + (7 * DC + offs_d1)[None, :] * stride_kvd,
                            mask_ids[:, None] & md7[None, :], other=0.0), B_LAYOUT)

            # Invalid keys get -inf, and exp2(-inf) is exactly 0 -- the same
            # value the reference produces for a masked key.
            qk = tl.where(mask_ids[None, :], qk, float("-inf"))
            exp_qk = tl.math.exp2(qk * log_scale)
            if SUM_MODE == 0:
                # Reduce while the values are still fp32 in the MMA layout the
                # QK dot produced.  Converting to the A operand layout first would
                # make the reduction cross layouts as well as waves.
                sum_exp += tl.sum(exp_qk, axis=1, keep_dims=True)
            exp_qk = tle.gpu.set_layout(exp_qk.to(tl.bfloat16), A_LAYOUT)
            if SUM_MODE == 1:
                # No reduction: the same exp_qk that feeds PV also feeds a dot
                # against a ones column, so the row sum arrives as an
                # accumulator.  bf16 rounding of exp_qk is the same rounding the
                # PV dots already apply to it.
                acc_s = tl.dot(exp_qk, ones_b, acc_s, out_dtype=tl.float32)
            acc0 = tl.dot(exp_qk, vt0, acc0, out_dtype=tl.float32)
            acc1 = tl.dot(exp_qk, vt1, acc1, out_dtype=tl.float32)
            acc2 = tl.dot(exp_qk, vt2, acc2, out_dtype=tl.float32)
            acc3 = tl.dot(exp_qk, vt3, acc3, out_dtype=tl.float32)
            acc4 = tl.dot(exp_qk, vt4, acc4, out_dtype=tl.float32)
            acc5 = tl.dot(exp_qk, vt5, acc5, out_dtype=tl.float32)
            acc6 = tl.dot(exp_qk, vt6, acc6, out_dtype=tl.float32)
            acc7 = tl.dot(exp_qk, vt7, acc7, out_dtype=tl.float32)
            if IGLP == 1:
                # fused-flash config: 16 prefetched shared loads, 4 MMA between
                # shared-load groups, 1 shared load per group, 1 MMA between
                # global loads, 8 MMA between the last shared load and ARRIVE.
                tle.gpu.metax.iglp(config_0=2, config_1=16, config_2=4,
                                   config_5=1, config_6=1, config_7=8)
            elif IGLP == 2:
                tle.gpu.metax.iglp(config_2=2, config_5=1, config_6=5)

    if SUM_MODE == 1:
        # All 16 columns of acc_s hold the row sum; read column 0.
        sum_exp = tl.sum(acc_s, axis=1, keep_dims=True) * 0.0625
    # A row whose every key was invalid sums to exactly 0.  Dividing would give
    # 0/0, so the select supplies the reference's out=0 / lse=-inf instead.
    ok = sum_exp > 0.0
    if N_GROUPS > 1:
        # Topk-split partial epilogue: normalise this split, then publish it in
        # the LSE domain (log2(sum)+max is already our log2 lse, so the merge is
        # a plain exp2-weighted sum -- no scaling conversion needed).  An
        # all-masked split writes -inf lse, which the merge weights to exactly
        # zero; partial_o is written as 0 so a 0-weighted product is never NaN.
        inv_p = 1.0 / tl.where(ok, sum_exp, 1.0)
        p_ptr = (partial_o
                 + i_b * stride_pob + i_sq * stride_pom
                 + i_grp * stride_pog + i_gbh * (BH * stride_poh)
                 + offs_h[:, None] * stride_poh + offs_d1[None, :] * stride_pod)
        keep_p = mask_h[:, None] & ok
        tl.store(p_ptr, tl.where(keep_p, acc0 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md0[None, :])
        tl.store(p_ptr + DC * stride_pod, tl.where(keep_p, acc1 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md1[None, :])
        tl.store(p_ptr + 2 * DC * stride_pod, tl.where(keep_p, acc2 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md2[None, :])
        tl.store(p_ptr + 3 * DC * stride_pod, tl.where(keep_p, acc3 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md3[None, :])
        tl.store(p_ptr + 4 * DC * stride_pod, tl.where(keep_p, acc4 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md4[None, :])
        tl.store(p_ptr + 5 * DC * stride_pod, tl.where(keep_p, acc5 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md5[None, :])
        tl.store(p_ptr + 6 * DC * stride_pod, tl.where(keep_p, acc6 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md6[None, :])
        tl.store(p_ptr + 7 * DC * stride_pod, tl.where(keep_p, acc7 * inv_p, 0.0).to(tl.bfloat16),
                 mask_h[:, None] & md7[None, :])
        pl_ptr = (partial_lse
                  + i_b * stride_plb + i_sq * stride_plm
                  + i_grp * stride_plg + i_gbh * (BH * stride_plh)
                  + offs_h[:, None] * stride_plh)
        pl_val = tl.where(ok, tl.math.log2(tl.where(ok, sum_exp, 1.0)), float("-inf"))
        tl.store(pl_ptr, pl_val, mask_h[:, None])
        return  # the merge kernel owns output/lse for a split launch
    inv = 1.0 / tl.where(ok, sum_exp, 1.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d1[None, :] * stride_od
    keep = mask_h[:, None] & ok
    tl.store(o_ptr, tl.where(keep, acc0 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md0[None, :])
    tl.store(o_ptr + DC * stride_od, tl.where(keep, acc1 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md1[None, :])
    tl.store(o_ptr + 2 * DC * stride_od, tl.where(keep, acc2 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md2[None, :])
    tl.store(o_ptr + 3 * DC * stride_od, tl.where(keep, acc3 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md3[None, :])
    tl.store(o_ptr + 4 * DC * stride_od, tl.where(keep, acc4 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md4[None, :])
    tl.store(o_ptr + 5 * DC * stride_od, tl.where(keep, acc5 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md5[None, :])
    tl.store(o_ptr + 6 * DC * stride_od, tl.where(keep, acc6 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md6[None, :])
    tl.store(o_ptr + 7 * DC * stride_od, tl.where(keep, acc7 * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md7[None, :])

    fin_log = tl.where(ok, tl.math.log2(tl.where(ok, sum_exp, 1.0)), float("-inf"))
    l_ptr = l_base + offs_h[:, None] * stride_lh
    tl.store(l_ptr, fin_log.to(tl.bfloat16), mask_h[:, None])


@triton.jit
def _merge_partial_kernel(
    partial_o,
    partial_lse,
    output,
    lse,
    stride_pob: tl.constexpr,
    stride_pom: tl.constexpr,
    stride_pog: tl.constexpr,
    stride_poh: tl.constexpr,
    stride_pod: tl.constexpr,
    stride_plb: tl.constexpr,
    stride_plm: tl.constexpr,
    stride_plg: tl.constexpr,
    stride_plh: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lm: tl.constexpr,
    SQ: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    G: tl.constexpr,
    BH: tl.constexpr,
    N_GROUPS: tl.constexpr,
):
    """Topk-split merge: out = sum_g o_g*exp2(l_g-m) / sum_g exp2(l_g-m).

    partial_lse is already in our log2 lse domain (log2(sum_exp)+0 offset, since
    this kernel family uses no running max), so the merge is a plain exp2-weighted
    sum.  A split that saw only invalid keys carries lse = -inf; exp2(-inf-m) = 0
    so it contributes exactly nothing.  Ties to the reference's out=0 / lse=-inf
    for an all-invalid row fall out for free: m = -inf, both sums are 0, and the
    ok-mask supplies 0 / -inf.
    """
    i_gbh, i_sq_raw, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_sq = SQ - 1 - i_sq_raw
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    mask_h = i_gbh * BH + offs_h < G
    md = offs_d < D

    l_base = (partial_lse + i_b * stride_plb + i_sq * stride_plm
              + i_gbh * (BH * stride_plh))
    l_ptr = l_base + offs_h[:, None] * stride_plh
    # Pass 1: running max over splits (all -inf for an all-invalid row).
    m = tl.full([BH, 1], float("-inf"), dtype=tl.float32)
    for g in tl.static_range(N_GROUPS):
        lg = tl.load(l_ptr + g * stride_plg, mask_h[:, None], other=float("-inf"))
        m = tl.maximum(m, lg)
    m = tl.where(m == float("-inf"), 0.0, m)   # avoid -inf - -inf = NaN

    o_base = (partial_o + i_b * stride_pob + i_sq * stride_pom
              + i_gbh * (BH * stride_poh))
    o_ptr = o_base + offs_h[:, None] * stride_poh + offs_d[None, :] * stride_pod
    acc = tl.zeros([BH, DP], dtype=tl.float32)
    den = tl.zeros([BH, 1], dtype=tl.float32)
    for g in tl.static_range(N_GROUPS):
        lg = tl.load(l_ptr + g * stride_plg, mask_h[:, None], other=float("-inf"))
        w = tl.math.exp2(lg - m)                       # 0 for an empty split
        den += w
        og = tl.load(o_ptr + g * stride_pog, mask_h[:, None] & md[None, :], other=0.0)
        acc += og.to(tl.float32) * w

    ok = den > 0.0
    inv = 1.0 / tl.where(ok, den, 1.0)
    keep = mask_h[:, None] & ok
    out_ptr = (output + i_b * stride_ob + i_sq * stride_om
               + i_gbh * (BH * stride_oh) + offs_h[:, None] * stride_oh
               + offs_d[None, :] * stride_od)
    tl.store(out_ptr, tl.where(keep, acc * inv, 0.0).to(tl.bfloat16),
             mask_h[:, None] & md[None, :])

    # lse = log2(sum_g exp2(l_g)) = m + log2(den), already log2 domain.
    fin = tl.where(ok, m + tl.math.log2(tl.where(ok, den, 1.0)), float("-inf"))
    l_out = (lse + i_b * stride_lb + i_sq * stride_lm
             + i_gbh * (BH * stride_lh) + offs_h[:, None] * stride_lh)
    tl.store(l_out, fin.to(tl.bfloat16), mask_h[:, None])


def _fwd_bk64(
    q, kv, indices, sm_scale=None, return_p_sum=False, d_v=512,
    BH=16, BK=64, loop_stages=0, v_from_trans=False, sum_mode=1, unroll=1,
    num_warps=4, num_stages=1, mma_warps=(1, 4), mma_elements=(1, 1, 4),
    col_major=False, iglp=2, n_groups=0, compile_opts=None, _return_meta=False,
    is_causal=True,
):
    """_fwd_bk64: BK/dsplit body, see the module docstring."""
    assert not return_p_sum, "forward only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, SKV, VG, _ = kv.shape
    D = d_v
    assert kv.shape[-1] == DT
    TD = DT - D
    DP = _next_pow2(D)
    TDP = _next_pow2(TD)
    assert kv.shape[0] == B
    _, _, _, K = indices.shape
    assert indices.shape == (B, SQ, VG, K)
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5
    assert DP % 8 == 0, "the BK=64 body splits D eight ways"

    NH = triton.cdiv(G, BH)
    # n_groups == 0 means "auto" (see _pick_n_groups).  The gate is a pure
    # function of the launch shape and topk, so identical cases always take the
    # same branch.
    if n_groups == 0:
        n_groups = _pick_n_groups(VG * NH * SQ, triton.cdiv(K, BK))

    mma = MmaEncoding((2, 0), mma_warps, mma_elements, col_major)
    a_layout = DotOperandEncoding(0, mma, 0)
    b_layout = DotOperandEncoding(1, mma, 0)

    # The epilogue stores every element of every row it owns (masked rows get an
    # explicit 0 / -inf), so a pinned default is only needed for the head rows
    # that fall outside G -- i.e. when G does not tile BH.  Zeroing the output
    # costs 10.5 / 15.4 / 26.1 / 47.6 us on s1-s4, which is the same order as
    # the whole remaining gap to the vendor, and the vendor allocates with
    # torch::empty.
    if G % BH == 0 and D == DP:
        output = torch.empty((B, SQ, H, D), device=q.device, dtype=q.dtype)
        lse = torch.empty((B, SQ, H), device=q.device, dtype=q.dtype)
    else:
        output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
        lse = torch.full((B, SQ, H), float("-inf"), device=q.device, dtype=q.dtype)

    # Topk split: n_groups > 1 splits each query's topk range across CTAs and
    # merges the per-split (partial_o, partial_lse) in the LSE domain.  With
    # n_groups == 1 the buffers are dummies (never written, never read) and the
    # launch is unchanged.
    if n_groups > 1:
        NK = triton.cdiv(K, BK)
        assert NK % n_groups == 0, "N_GROUPS must divide cdiv(topk, BK)"
        partial_o = torch.empty((B, SQ, n_groups, H, D), device=q.device, dtype=q.dtype)
        partial_lse = torch.empty((B, SQ, n_groups, H), device=q.device, dtype=torch.float32)
        grid = (n_groups * VG * NH, SQ, B)
    else:
        partial_o = output          # unused
        partial_lse = lse           # unused
        grid = (VG * NH, SQ, B)
    # The partial strides are compile-time constants; give the unused NG==1
    # launch something arithmetically harmless (the branch is dead code).
    if n_groups > 1:
        pod_s = partial_o.stride()
        pls_s = partial_lse.stride()
    else:
        pod_s = (0, 0, 0, 0, 0)
        pls_s = (0, 0, 0, 0)

    compiled = _sparse_mla_bk64_kernel[grid](
        q, kv, indices, sm_scale, output, lse, partial_o, partial_lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        kv.stride(0), kv.stride(2), kv.stride(1), kv.stride(3),
        indices.stride(0), indices.stride(2), indices.stride(1), indices.stride(3),
        output.stride(0), output.stride(2), output.stride(1), output.stride(3),
        lse.stride(0), lse.stride(2), lse.stride(1),
        pod_s[0], pod_s[1], pod_s[2], pod_s[3], pod_s[4],
        pls_s[0], pls_s[1], pls_s[2], pls_s[3],
        SQ, K, D, TD, DP, TDP, G, BK, BH, SKV, is_causal,
        loop_stages, v_from_trans, sum_mode, unroll, iglp, n_groups,
        A_LAYOUT=a_layout, B_LAYOUT=b_layout, MMA=mma,
        num_warps=num_warps, num_stages=num_stages, **(compile_opts or {}),
    )
    if n_groups > 1:
        # One CTA per (query, head block): merge n_groups splits.
        _merge_partial_kernel[(VG * NH, SQ, B)](
            partial_o, partial_lse, output, lse,
            partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
            partial_o.stride(3), partial_o.stride(4),
            partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
            partial_lse.stride(3),
            output.stride(0), output.stride(2), output.stride(1), output.stride(3),
            lse.stride(0), lse.stride(2), lse.stride(1),
            SQ, D, DP, G, BH, N_GROUPS=n_groups,
            num_warps=num_warps,
        )
    if _return_meta:
        return output, lse, compiled
    return output, lse


# ---------------------------------------------------------------------------
# Shape dispatch: measured crossover at topk 256.  Below it the narrower tile
# wins because a causal BK=64 tile computes up to 63 keys the mask discards;
# above it the wider tile halves the per-key operand conversions.
# ---------------------------------------------------------------------------
WIDE_TOPK = 256


def plan(topk, BH=16, G=0):
    """Return (host wrapper, kwargs) for a topk -- the whole dispatch decision.

    Three configs, each measured rather than guessed:

    DSA (topk < 256): BK=32, 4 warps, BH=16.  s1-s4 must not inherit the GLM
    branch's BH=32 -- smem doubles and occupancy drops.

    GLM wide (topk >= 2048 and G >= 32, i.e. head duplication across CTAs):
    BH=32 + v_from_trans=True with 8 warps and mma_warps (2,4).  The per-thread
    live set halves (acc 64->32, V 64->32, Q 32->16 regs), which is what takes
    n_spills from 26 to 0.  (2,4) is the only 8-warp MMA grid that also
    bottomed out the allocator; (1,8)/(4,2) stay at 256 regs and spill 26/78.
    Measured 1.56x/1.70x over the old BH=16 config.

    GLM narrow / everything else in the wide range: BK=64, 4 warps, BH=16.
    """
    if topk >= WIDE_TOPK:
        if topk >= 2048 and G >= 32:
            return _fwd_bk64, dict(BH=32, BK=64, sum_mode=1,
                                   v_from_trans=True,
                                   num_warps=8, mma_warps=(2, 4),
                                   mma_elements=(1, 1, 4), col_major=False,
                                   iglp=0)
        return _fwd_bk64, dict(BH=BH, BK=64, sum_mode=1, v_from_trans=False,
                               num_warps=4, mma_warps=(1, 4),
                               mma_elements=(1, 1, 8), col_major=False, iglp=0)
    return _fwd_bk32, dict(BH=BH, BK=32, sum_mode=1, v_from_trans=False,
                           num_warps=4, mma_warps=(1, 4),
                           mma_elements=(1, 1, 8), col_major=True, iglp=0)


def sparse_mla_fwd(q, kv, indices, sm_scale=None, return_p_sum=False, d_v=512,
                   BH=16, compile_opts=None, _return_meta=False, **overrides):
    """Fused DSA sparse-MLA forward. Returns (output, lse), lse in log2/bf16."""
    topk = indices.shape[-1]
    G = q.shape[2] // kv.shape[2]
    fwd, kw = plan(topk, BH=BH, G=G)
    kw.update(overrides)
    return fwd(q, kv, indices, sm_scale=sm_scale, return_p_sum=return_p_sum,
               d_v=d_v, compile_opts=compile_opts, _return_meta=_return_meta, **kw)


# Input generators adapted from FlagGems/FlagGems-vLLM (Apache-2.0).
# All 13 shape records, generators and FP32 references are embedded below.
CASE_SPEC = {'description': 'Upstream sparse-MLA forward shapes: 4 DSA (FlagGems) + 13 GLM (FlagGems-vllm). B=1, HKV=1, '
                'DQK=576 (DSA; GLM up00-04) or 512 (GLM up05-12), DV=512.',
 'dtype': 'bfloat16',
 'seed': 42,
 'reference': '/workspace/FlagGems/benchmark/test_sparse_mla_fwd_interface.py',
 'cases': [{'name': 's1',
            'B': 1,
            'SQ': 64,
            'SKV': 1024,
            'H': 128,
            'HKV': 1,
            'DQK': 576,
            'DV': 512,
            'topk': 64},
           {'name': 's2',
            'B': 1,
            'SQ': 128,
            'SKV': 2048,
            'H': 128,
            'HKV': 1,
            'DQK': 576,
            'DV': 512,
            'topk': 128},
           {'name': 's3',
            'B': 1,
            'SQ': 256,
            'SKV': 4096,
            'H': 128,
            'HKV': 1,
            'DQK': 576,
            'DV': 512,
            'topk': 256},
           {'name': 's4',
            'B': 1,
            'SQ': 512,
            'SKV': 8192,
            'H': 128,
            'HKV': 1,
            'DQK': 576,
            'DV': 512,
            'topk': 512}],
 'flashmla_cases': [
                    {'name': 'glm_up00_skv8192_h128_d576',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 8192,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 576,
                     'DV': 512,
                     'topk': 2048,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 0},
                    {'name': 'glm_up01_skv32768_h128_d576',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 32768,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 576,
                     'DV': 512,
                     'topk': 2048,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 1},
                    {'name': 'glm_up02_skv65536_h128_d576',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 65536,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 576,
                     'DV': 512,
                     'topk': 2048,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 2},
                    {'name': 'glm_up03_skv98304_h128_d576',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 98304,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 576,
                     'DV': 512,
                     'topk': 2048,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 3},
                    {'name': 'glm_up04_skv131072_h128_d576',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 131072,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 576,
                     'DV': 512,
                     'topk': 2048,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 4},
                    {'name': 'glm_up05_skv8192_h64_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 8192,
                     'H': 64,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 512,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 5},
                    {'name': 'glm_up06_skv32768_h64_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 32768,
                     'H': 64,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 512,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 6},
                    {'name': 'glm_up07_skv49152_h64_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 49152,
                     'H': 64,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 512,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 7},
                    {'name': 'glm_up08_skv65536_h64_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 65536,
                     'H': 64,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 512,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 8},
                    {'name': 'glm_up09_skv8192_h128_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 8192,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 1024,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 9},
                    {'name': 'glm_up10_skv32768_h128_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 32768,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 1024,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 10},
                    {'name': 'glm_up11_skv49152_h128_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 49152,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 1024,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 11},
                    {'name': 'glm_up12_skv65536_h128_d512',
                     'B': 1,
                     'SQ': 4096,
                     'SKV': 65536,
                     'H': 128,
                     'HKV': 1,
                     'DQK': 512,
                     'DV': 512,
                     'topk': 1024,
                     'sm_scale': 0.5,
                     'causal': False,
                     'input_kind': 'flashmla_glm',
                     'reference': '/workspace/FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py',
                     'upstream_index': 12},
                    ]}


def load_cases():
    return {c['name']: dict(c) for c in CASE_SPEC['cases'] + CASE_SPEC['flashmla_cases']}, CASE_SPEC


def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sparse_mla_input(batch_size, seq_len_q, seq_len_kv, num_heads, num_kv_heads, qk_dim, topk, dtype, device, seed=42):
    """Return (q, kv, indices) matching the FlagGems generator."""
    init_seed(seed)
    B, S, H = (batch_size, seq_len_q, num_heads)
    SKV, HKV, DQK = (seq_len_kv, num_kv_heads, qk_dim)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device)
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i
    return (q, kv, indices)


def _randperm_batch(batch_size, perm_range, perm_size, paddings):
    """Row-wise unique random permutations, padded where range < perm_size.

    Adapted from FlagGems-vllm's generator; kept verbatim because the GLM
    shapes' recorded numbers depend on its exact RNG consumption.
    """
    if perm_range.numel() != batch_size:
        raise ValueError('perm_range must contain one value per batch row')
    ranges = perm_range.to(device='cpu', dtype=torch.int64).reshape(-1)
    positions = torch.arange(perm_size, dtype=torch.int64)
    padding_values = torch.tensor(paddings, dtype=torch.int32)
    res = torch.empty((batch_size, perm_size), dtype=torch.int32)
    for row, range_value in enumerate(ranges.tolist()):
        valid_size = min(range_value, perm_size)
        if valid_size:
            offset = random.randrange(range_value)
            step = 1
            if range_value > 1:
                step = random.randrange(1, range_value)
                while math.gcd(step, range_value) != 1:
                    step += 1
                    if step == range_value:
                        step = 1
            values = (offset + step * positions[:valid_size]) % range_value
            res[row, :valid_size] = values.to(torch.int32)
        if valid_size < perm_size:
            if len(paddings) == 1:
                res[row, valid_size:].fill_(paddings[0])
            else:
                filler = torch.randint(0, len(paddings), (perm_size - valid_size,))
                res[row, valid_size:] = padding_values[filler]
    return res


# Padding candidates, in the upstream generator's exact ORDER (s_kv sits at
# index 3): _randperm_batch draws fillers with torch.randint over this list,
# so reordering it changes the generated indices.


def _invalid_indices(s_kv):
    return [-2147483648, -123456, -1, s_kv, 114514, 1919810, 2147480000, 2147483647]


def _make_flashmla_glm_input(case, device, dtype):
    """Reproduce FlagGems-vllm's flashmla generator for one GLM shape.

    The RNG stream (seed -> counter, one random.random() for q, one for kv,
    then _randperm_batch) matches the upstream generator exactly, so shapes
    generated here are bit-identical to theirs.
    """
    seed = case['upstream_index']   # upstream seeds each shape by its index
    random.seed(seed)
    torch.manual_seed(seed)
    s_q, s_kv, h_q, h_kv, d_qk, topk = (case['SQ'], case['SKV'], case['H'],
                                        case['HKV'], case['DQK'], case['topk'])
    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10 + (random.random() - 0.5) / 10
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10 + (random.random() - 0.5) / 10
    q = q.clamp_(-10, 10)
    kv = kv.clamp_(-10, 10)
    paddings = _invalid_indices(s_kv)
    idx = _randperm_batch(s_q, torch.full((s_q,), s_kv, dtype=torch.int32), topk, paddings)
    # Order matters: move to the device BEFORE unsqueeze.  On this backend
    # .to(device) on an unsqueezed view returns a non-contiguous tensor with
    # scrambled strides, and the kernel asserts contiguity.
    idx = idx.view(s_q, h_kv, topk).to(device).unsqueeze(0)
    return q.unsqueeze(0), kv.unsqueeze(0), idx


def build_case(case, device, dtype=torch.bfloat16, seed=42):
    if case.get('input_kind') == 'flashmla_glm':
        return _make_flashmla_glm_input(case, device, dtype)
    q, kv, indices = make_sparse_mla_input(
        batch_size=case['B'], seq_len_q=case['SQ'], seq_len_kv=case['SKV'],
        num_heads=case['H'], num_kv_heads=case['HKV'], qk_dim=case['DQK'],
        topk=case['topk'], dtype=dtype, device=device, seed=seed)
    return (q, kv, indices)


def ref_sparse_mla_fwd(q, kv, indices, sm_scale=None, d_v=512, query_chunk=64):
    """Reference (out, lse) in FlagGems convention.

    out: [B, SQ, H, d_v] bf16
    lse: [B, SQ, H] fp32, log2 domain (``max_log + log2(sum_exp)``)
    """
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    qf = q.float()
    kf = kv.float()
    b, sq, h, dim_q = qf.shape
    _, sk, g, _ = kf.shape
    assert kf.shape[-1] == 576, 'reference assumes DQK == 576'
    dim = d_v
    v = kf[..., :dim]
    if sm_scale is None:
        sm_scale = dim_q ** (-0.5)
    idx = indices.transpose(1, 2).long()
    scattered = torch.zeros((b, g, sq, sk + 1), dtype=torch.bool, device=q.device)
    scattered.scatter_(3, idx, True)
    valid = scattered[..., :sk]
    causal = torch.arange(sq, device=q.device)[:, None] >= torch.arange(sk, device=q.device)[None, :]
    valid = valid & causal.view(1, 1, sq, sk)
    q5 = qf.view(b, sq, g, h // g, dim_q)
    out = torch.empty((b, sq, h, dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, sq, h), dtype=torch.float32, device=q.device)
    log2e = 1.4426950408889634
    for m0 in range(0, sq, query_chunk):
        m1 = min(m0 + query_chunk, sq)
        qc = q5[:, m0:m1]
        qk = torch.einsum('bmghd,bngd->bghmn', qc, kf) * sm_scale
        vmask = valid[:, :, m0:m1, :].view(b, g, 1, m1 - m0, sk)
        qk = qk.masked_fill(~vmask, float('-inf'))
        p = torch.softmax(qk, dim=-1)
        o = torch.einsum('bghmn,bngd->bmghd', p, v)
        out[:, m0:m1] = o.reshape(b, m1 - m0, h, dim).to(torch.bfloat16)
        qk2 = qk * log2e
        mx = qk2.amax(dim=-1)
        se = torch.exp2(qk2 - mx[..., None]).sum(dim=-1)
        lse[:, m0:m1] = (mx + torch.log2(se)).permute(0, 3, 1, 2).reshape(b, m1 - m0, h)
    return (out, lse)


def glm_reference(q, kv, indices, sm_scale, d_v):
    """Bounded-memory reference attending to the selected cache IDs only."""
    out = torch.empty((*q.shape[:-1], d_v), device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    assert q.shape[0] == 1 and kv.shape[2] == 1
    for start in range(0, q.shape[1], 8):
        end = min(start + 8, q.shape[1])
        ids = indices[0, start:end, 0].long()
        valid = (ids >= 0) & (ids < kv.shape[1])
        selected = kv[0, ids.clamp(0, kv.shape[1] - 1), 0].float()
        logits = torch.einsum('chd,ckd->chk', q[0, start:end].float(), selected) * sm_scale
        logits.masked_fill_(~valid[:, None, :], float('-inf'))
        prob = torch.softmax(logits, -1)
        out[0, start:end] = torch.einsum('chk,ckd->chd', prob, selected[..., :d_v]).to(q.dtype)
        lse[0, start:end] = torch.logsumexp(logits, -1)
    return (out, lse)


def _tuple(result):
    if isinstance(result, tuple):
        return result[0], (result[1] if len(result) > 1 else None)
    return result, None


class UnsupportedCase(ValueError):
    """Known provider interface restriction, not a failed execution."""


def resolve_impl(name, case):
    """Return (fn() -> (out, lse), description). Inputs are pre-bound."""
    if name == "torch":
        q, kv, idx = case["q"], case["kv"], case["indices"]
        sm = case["sm_scale"]
        return (
            lambda: ref_sparse_mla_fwd(q, kv, idx, sm_scale=sm, d_v=512),
            "tests.reference (torch fp32)",
        )

    if name == "flag_gems":
        if (case["B"] != 1 or case["H"] not in (64, 128)
                or case["HKV"] != 1 or case["DQK"] not in (512, 576) or case["DV"] != 512):
            raise UnsupportedCase("requires B=1, H=64/128, HKV=1, DQK=512/576, DV=512")
        import importlib.util

        import flag_gems
        source = Path(flag_gems.__file__).resolve().parent / "fused/flashmla_sparse.py"
        spec = importlib.util.spec_from_file_location("gems_sparse_mla_triton_baseline", source)
        gems = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gems)
        # Select the explicit upstream Triton branch in this private module.
        gems._can_use_tle_flash_mla_sparse_fwd = lambda *a, **kw: False
        config = case.get("gems_config", "stock")
        if config == "c550":
            # The stock autotune is BH=64/BK=64, which needs 172032 B of smem
            # and cannot launch on C550 (64 KB).  BH=16/BK=16 (63488 B) is the
            # only tile that fits -- and it is 8-30x slower than TLE.
            gems.triton_flash_mla_sparse_fwd.configs = [
                triton.Config({"BK": 16, "BH": 16}, num_warps=4, num_stages=1)]
            gems.triton_flash_mla_sparse_fwd.cache.clear()
        q, kv, idx = case["q"][0].contiguous(), case["kv"][0].contiguous(), case["indices"][0].contiguous()
        sm = case["sm_scale"]
        def run_gems():
            out, _max, lse = gems.flash_mla_sparse_fwd(q, kv, idx, sm_scale=sm, d_v=512)
            return out.unsqueeze(0), lse.unsqueeze(0)
        return run_gems, f"{source}:triton_flash_mla_sparse_fwd (TLE disabled; config={config}; upstream allocations included)"

    if name == "sparse_mla_tle":
        q, kv, idx = case["q"], case["kv"], case["indices"]
        sm = case["sm_scale"]
        return (
            lambda: sparse_mla_fwd(q, kv, idx, sm_scale=sm, d_v=512,
                                   **({"is_causal": False} if case.get("causal") is False else {})),
            f"{Path(__file__).name} (TLE, shipped, in-file)",
        )

    raise ValueError(f"unknown impl: {name}")


def bench(fn, warmup_ms, rep_ms):
    import triton.testing

    return float(
        triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, return_mode="median"
        )
    )


def check(got_out, ref_out, atol, rtol):
    try:
        torch.testing.assert_close(
            got_out, ref_out.to(got_out.dtype), atol=atol, rtol=rtol, equal_nan=True
        )
        return True, None
    except AssertionError as exc:
        return False, str(exc).splitlines()[0]


def device_meta(device):
    meta = {
        "torch": torch.__version__,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import triton

        meta["triton"] = triton.__version__
        meta["triton_path"] = triton.__file__
    except Exception as exc:  # pragma: no cover
        meta["triton"] = f"<import failed: {exc}>"
    try:
        import flag_gems

        meta["flag_gems"] = getattr(flag_gems, "__version__", "unknown")
        meta["flag_gems_vendor"] = getattr(flag_gems, "vendor_name", "unknown")
    except Exception as exc:  # pragma: no cover
        meta["flag_gems"] = f"<import failed: {exc}>"
    try:
        props = torch.cuda.get_device_properties(device)
        for key in (
            "name",
            "total_memory",
            "multi_processor_count",
            "shared_memory_per_block",
            "shared_memory_per_multiprocessor",
            "regs_per_multiprocessor",
            "warp_size",
        ):
            if hasattr(props, key):
                meta[key] = getattr(props, key)
    except Exception as exc:  # pragma: no cover
        meta["device_props_error"] = str(exc)
    return meta


def run_isolated(args, names, cases):
    """Keep each shape's Gems autotuning independent and capture noisy imports."""
    columns = ("case", "SQ", "SKV", "H", "K", "tle_ms", "gems_ms",
               "speedup_gems_over_tle")
    print(f"timer=do_bench; gems_config={args.gems_config}; warmup={args.warmup_ms}ms rep={args.rep_ms}ms; isolated per case", flush=True)
    print("  ".join(columns), flush=True)
    combined = {"meta": {"timing": "triton.testing.do_bench", "isolated": True,
                         "gpu": os.environ.get("MACA_VISIBLE_DEVICES", "")},
                "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms, "cases": {}}
    handle = None
    failed = False
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        handle = open(args.csv, "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        handle.flush()
    try:
        for name in names:
            if name not in cases:
                raise ValueError(f"unknown case {name}")
            with tempfile.TemporaryDirectory(prefix="sparse-mla-bench-") as tmp:
                out = Path(tmp) / "result.json"
                logdir = Path(args.out).parent / "worker_logs"
                logdir.mkdir(parents=True, exist_ok=True)
                logfile = logdir / f"{name}.log"
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                       "--case", name, "--impls", args.impls, "--seed", str(args.seed),
                       "--warmup-ms", str(args.warmup_ms), "--rep-ms", str(args.rep_ms),
                       "--atol", str(args.atol), "--rtol", str(args.rtol), "--out", str(out),
                       "--gems-config", args.gems_config]
                if args.device:
                    cmd += ["--device", args.device]
                if args.no_check:
                    cmd += ["--no-check"]
                with logfile.open("w") as log:
                    process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"case {name} failed; see {logfile}")
                part = json.loads(out.read_text())
            entry = part["cases"][name]
            entry["meta"] = part["meta"]
            entry["worker_log"] = str(logfile)
            combined["cases"][name] = entry
            def timing(impl):
                data = entry["impls"].get(impl, {})
                return data.get("median_ms") or float("nan")
            tle, gems = timing("sparse_mla_tle"), timing("flag_gems")
            c = cases[name]
            row = dict(case=name, SQ=c["SQ"], SKV=c["SKV"], H=c["H"], K=c["topk"],
                       tle_ms=tle, gems_ms=gems,
                       speedup_gems_over_tle=gems/tle)
            print("  ".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k])
                             for k in columns), flush=True)
            for impl, data in entry["impls"].items():
                if data.get("status") == "SKIP":
                    print(f"  {impl}: SKIP ({data['skip_reason']})", flush=True)
                elif data.get("error") or data.get("correct") is False:
                    failed = True
                    print(f"  {impl}: FAIL/ERROR {str(data.get('error', ''))[:200]}", flush=True)
            if handle:
                writer.writerow(row)
                handle.flush()
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(combined, indent=2))
    finally:
        if handle:
            handle.close()
    return int(failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("dsa", "glm", "all"), default="all")
    ap.add_argument("--list-cases", action="store_true")
    ap.add_argument("--case", default="all",
                    help="comma-separated case names (s1-s4, glm_up00..12); "
                         "'all' runs every case in --suite")
    ap.add_argument("--impls", default="sparse_mla_tle,flag_gems")
    ap.add_argument("--csv", nargs="?", const=str(Path(__file__).resolve().parent / "results/compare.csv"))
    ap.add_argument("--gems-config", choices=("stock", "c550"), default="stock")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-ms", type=float, default=50.0)
    ap.add_argument("--rep-ms", type=float, default=100.0)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results/latest.json"))
    args = ap.parse_args()

    cases, spec = load_cases()
    benchmark_names = [c["name"] for c in spec["cases"] + spec["flashmla_cases"]]
    suite = {"dsa": spec["cases"], "glm": spec["flashmla_cases"]}.get(
        args.suite, spec["cases"] + spec["flashmla_cases"])
    suite_names = [c["name"] for c in suite]
    impl_names = [s.strip() for s in args.impls.split(",") if s.strip()]

    case_names = (suite_names if args.case == "all"
                  else [c.strip() for c in args.case.split(",") if c.strip()])
    invalid = [n for n in case_names if n not in benchmark_names]
    if invalid:
        ap.error(f"unknown case name(s): {invalid}; "
                 f"available: {benchmark_names}")
    if args.list_cases:
        for name in case_names:
            print(json.dumps(cases[name]))
        print(f"Total: {len(case_names)}")
        return 0
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.worker:
        return run_isolated(args, case_names, cases)
    result = {"meta": device_meta(device), "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms, "cases": {}}
    for name in case_names:
        if name not in cases:
            raise SystemExit(f"unknown case {name}; have {sorted(cases)}")
        case = dict(cases[name])
        q, kv, indices = build_case(case, device, seed=args.seed)
        case.update(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=case.get("sm_scale", case["DQK"] ** -0.5),
            gems_config=args.gems_config,
        )

        # --no-check skips the (slow) fp32 reference, which makes this the
        # latency-only mode: python 16-sparse_mla_tle.py --no-check
        ref_out = None
        if not args.no_check:
            ref_fn = ref_sparse_mla_fwd
            if case.get("input_kind") == "flashmla_glm":
                ref_fn = glm_reference
            ref_out, _ref_lse = ref_fn(q, kv, indices, sm_scale=case["sm_scale"], d_v=case["DV"])
        entry = {
            "benchmark_source": case.get("reference", spec["reference"]),
            "causal": case.get("causal", True),
            "sm_scale": case["sm_scale"],
            "params": {k: case[k] for k in
                       ("B", "SQ", "SKV", "H", "HKV", "DQK", "DV", "topk")},
            "impls": {},
        }

        for impl_name in impl_names:
            try:
                fn, desc = resolve_impl(impl_name, case)
                # correctness once, before timing
                got_out, _got_lse = _tuple(fn())
                torch.cuda.synchronize()
                ok, why = (True, None) if args.no_check else check(
                    got_out, ref_out, args.atol, args.rtol
                )
                ms = bench(fn, args.warmup_ms, args.rep_ms) if ok else None
                entry["impls"][impl_name] = {
                    "desc": desc,
                    "median_ms": ms,
                    "correct": ok,
                    "error": why,
                }
                print(f"[{name}] {impl_name}: ms={ms} correct={ok}")
            except UnsupportedCase as exc:
                entry["impls"][impl_name] = {
                    "status": "SKIP", "skip_reason": str(exc), "median_ms": None,
                    "correct": None,
                }
                print(f"[{name}] {impl_name}: SKIP {exc}")
            except Exception as exc:  # keep other impls measurable
                # A baseline that cannot launch on this shape (e.g. flag_gems
                # stock config exceeds the C550 64 KB smem wall on the DSA tiles)
                # is a SKIP, not a failure: TLE is still measured and checked.
                resource_limited = any(
                    tok in str(exc)
                    for tok in ("OutOfResources", "out of resource", "out of shared memory")
                )
                if resource_limited:
                    entry["impls"][impl_name] = {
                        "status": "SKIP", "skip_reason": str(exc), "median_ms": None,
                        "correct": None,
                    }
                    print(f"[{name}] {impl_name}: SKIP (resource: {str(exc)[:120]})")
                else:
                    entry["impls"][impl_name] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"[{name}] {impl_name:>12}: FAILED {type(exc).__name__}: {exc}")

        # The headline number: speedup = flag_gems / tle (>1 = TLE faster).
        # Other impls (torch) only serve as accuracy oracles, so ratios are
        # recorded for baselines against flag_gems when both ran.
        tle_ms = entry["impls"].get("sparse_mla_tle", {}).get("median_ms")
        gems_ms = entry["impls"].get("flag_gems", {}).get("median_ms")
        base = gems_ms or tle_ms
        for impl_name, row in entry["impls"].items():
            if impl_name != "sparse_mla_tle" and base and row.get("median_ms"):
                row["vs_baseline"] = row["median_ms"] / base
        if tle_ms and gems_ms:
            row = entry["impls"]["sparse_mla_tle"]
            row["gems_over_tle_speedup"] = gems_ms / tle_ms
            print(f"[{name}] sparse_mla_tle vs flag_gems: speedup(gems/tle)="
                  f"{gems_ms/tle_ms:.3f}x  ({100.0*gems_ms/tle_ms:.1f}% of gems)")

        result["cases"][name] = entry

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
