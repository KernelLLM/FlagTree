"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice
EPS = 1e-10

@triton.jit
def _mhc_bwd_kernel(out_ptr, dout_ptr, res_ptr, seqlen, n_stream, out_stride_s, out_stride_i, out_stride_j, dout_stride_s, dout_stride_i, dout_stride_j, res_stride_s, res_stride_i, res_stride_j, cg_iters: tl.constexpr, TILE_SIZE: tl.constexpr, N_STREAM: tl.constexpr):
    """Sinkhorn backward via implicit CG differentiation - one tile per program."""
    pid = tl.program_id(0)
    tile_start = pid * TILE_SIZE
    for t in range(TILE_SIZE):
        seq_idx = tile_start + t
        if seq_idx >= seqlen:
            continue
        base_out = seq_idx * out_stride_s
        base_dout = seq_idx * dout_stride_s
        for i in range(N_STREAM):
            for j in range(N_STREAM):
                r_val = tl.load(out_ptr + base_out + i * out_stride_i + j * out_stride_j)
                dr_val = tl.load(dout_ptr + base_dout + i * dout_stride_i + j * dout_stride_j)
        base_res = seq_idx * res_stride_s
        for i in range(N_STREAM):
            for j in range(N_STREAM):
                r_val = tl.load(out_ptr + base_out + i * out_stride_i + j * out_stride_j)
                dr_val = tl.load(dout_ptr + base_dout + i * dout_stride_i + j * dout_stride_j)
                tl.store(res_ptr + base_res + i * res_stride_i + j * res_stride_j, dr_val * r_val)

@triton.jit
def _mhc_bwd_kernel_n4(out_ptr, dout_ptr, res_ptr, seqlen, cg_iters: tl.constexpr, BLOCK_S: tl.constexpr):
    """Sinkhorn backward for n_stream=4, optimized with unrolled CG."""
    pid = tl.program_id(0)
    seq_start = pid * BLOCK_S
    seq_offsets = seq_start + tl.arange(0, BLOCK_S)
    mask = seq_offsets < seqlen
    base_out = seq_offsets * 16
    base_dout = seq_offsets * 16
    base_res = seq_offsets * 16
    R_00 = tl.load(out_ptr + base_out + 0, mask=mask, other=0.0)
    R_01 = tl.load(out_ptr + base_out + 1, mask=mask, other=0.0)
    R_02 = tl.load(out_ptr + base_out + 2, mask=mask, other=0.0)
    R_03 = tl.load(out_ptr + base_out + 3, mask=mask, other=0.0)
    R_10 = tl.load(out_ptr + base_out + 4, mask=mask, other=0.0)
    R_11 = tl.load(out_ptr + base_out + 5, mask=mask, other=0.0)
    R_12 = tl.load(out_ptr + base_out + 6, mask=mask, other=0.0)
    R_13 = tl.load(out_ptr + base_out + 7, mask=mask, other=0.0)
    R_20 = tl.load(out_ptr + base_out + 8, mask=mask, other=0.0)
    R_21 = tl.load(out_ptr + base_out + 9, mask=mask, other=0.0)
    R_22 = tl.load(out_ptr + base_out + 10, mask=mask, other=0.0)
    R_23 = tl.load(out_ptr + base_out + 11, mask=mask, other=0.0)
    R_30 = tl.load(out_ptr + base_out + 12, mask=mask, other=0.0)
    R_31 = tl.load(out_ptr + base_out + 13, mask=mask, other=0.0)
    R_32 = tl.load(out_ptr + base_out + 14, mask=mask, other=0.0)
    R_33 = tl.load(out_ptr + base_out + 15, mask=mask, other=0.0)
    dR_00 = tl.load(dout_ptr + base_dout + 0, mask=mask, other=0.0)
    dR_01 = tl.load(dout_ptr + base_dout + 1, mask=mask, other=0.0)
    dR_02 = tl.load(dout_ptr + base_dout + 2, mask=mask, other=0.0)
    dR_03 = tl.load(dout_ptr + base_dout + 3, mask=mask, other=0.0)
    dR_10 = tl.load(dout_ptr + base_dout + 4, mask=mask, other=0.0)
    dR_11 = tl.load(dout_ptr + base_dout + 5, mask=mask, other=0.0)
    dR_12 = tl.load(dout_ptr + base_dout + 6, mask=mask, other=0.0)
    dR_13 = tl.load(dout_ptr + base_dout + 7, mask=mask, other=0.0)
    dR_20 = tl.load(dout_ptr + base_dout + 8, mask=mask, other=0.0)
    dR_21 = tl.load(dout_ptr + base_dout + 9, mask=mask, other=0.0)
    dR_22 = tl.load(dout_ptr + base_dout + 10, mask=mask, other=0.0)
    dR_23 = tl.load(dout_ptr + base_dout + 11, mask=mask, other=0.0)
    dR_30 = tl.load(dout_ptr + base_dout + 12, mask=mask, other=0.0)
    dR_31 = tl.load(dout_ptr + base_dout + 13, mask=mask, other=0.0)
    dR_32 = tl.load(dout_ptr + base_dout + 14, mask=mask, other=0.0)
    dR_33 = tl.load(dout_ptr + base_dout + 15, mask=mask, other=0.0)
    RdR_00 = R_00 * dR_00
    RdR_01 = R_01 * dR_01
    RdR_02 = R_02 * dR_02
    RdR_03 = R_03 * dR_03
    RdR_10 = R_10 * dR_10
    RdR_11 = R_11 * dR_11
    RdR_12 = R_12 * dR_12
    RdR_13 = R_13 * dR_13
    RdR_20 = R_20 * dR_20
    RdR_21 = R_21 * dR_21
    RdR_22 = R_22 * dR_22
    RdR_23 = R_23 * dR_23
    RdR_30 = R_30 * dR_30
    RdR_31 = R_31 * dR_31
    RdR_32 = R_32 * dR_32
    RdR_33 = R_33 * dR_33
    b1_0 = RdR_00 + RdR_01 + RdR_02 + RdR_03
    b1_1 = RdR_10 + RdR_11 + RdR_12 + RdR_13
    b1_2 = RdR_20 + RdR_21 + RdR_22 + RdR_23
    b1_3 = RdR_30 + RdR_31 + RdR_32 + RdR_33
    b2_0 = RdR_00 + RdR_10 + RdR_20 + RdR_30
    b2_1 = RdR_01 + RdR_11 + RdR_21 + RdR_31
    b2_2 = RdR_02 + RdR_12 + RdR_22 + RdR_32
    b2_3 = RdR_03 + RdR_13 + RdR_23 + RdR_33
    x1_0 = tl.zeros_like(b1_0)
    x1_1 = tl.zeros_like(b1_1)
    x1_2 = tl.zeros_like(b1_2)
    x1_3 = tl.zeros_like(b1_3)
    x2_0 = tl.zeros_like(b2_0)
    x2_1 = tl.zeros_like(b2_1)
    x2_2 = tl.zeros_like(b2_2)
    x2_3 = tl.zeros_like(b2_3)
    r1_0 = b1_0
    r1_1 = b1_1
    r1_2 = b1_2
    r1_3 = b1_3
    r2_0 = b2_0
    r2_1 = b2_1
    r2_2 = b2_2
    r2_3 = b2_3
    p1_0 = r1_0
    p1_1 = r1_1
    p1_2 = r1_2
    p1_3 = r1_3
    p2_0 = r2_0
    p2_1 = r2_1
    p2_2 = r2_2
    p2_3 = r2_3
    r_normsq = r1_0 * r1_0 + r1_1 * r1_1 + r1_2 * r1_2 + r1_3 * r1_3 + r2_0 * r2_0 + r2_1 * r2_1 + r2_2 * r2_2 + r2_3 * r2_3
    for _ in range(cg_iters):
        Ap1_0 = R_00 * p2_0 + R_01 * p2_1 + R_02 * p2_2 + R_03 * p2_3 + p1_0
        Ap1_1 = R_10 * p2_0 + R_11 * p2_1 + R_12 * p2_2 + R_13 * p2_3 + p1_1
        Ap1_2 = R_20 * p2_0 + R_21 * p2_1 + R_22 * p2_2 + R_23 * p2_3 + p1_2
        Ap1_3 = R_30 * p2_0 + R_31 * p2_1 + R_32 * p2_2 + R_33 * p2_3 + p1_3
        Ap2_0 = R_00 * p1_0 + R_10 * p1_1 + R_20 * p1_2 + R_30 * p1_3 + p2_0
        Ap2_1 = R_01 * p1_0 + R_11 * p1_1 + R_21 * p1_2 + R_31 * p1_3 + p2_1
        Ap2_2 = R_02 * p1_0 + R_12 * p1_1 + R_22 * p1_2 + R_32 * p1_3 + p2_2
        Ap2_3 = R_03 * p1_0 + R_13 * p1_1 + R_23 * p1_2 + R_33 * p1_3 + p2_3
        pAp = p1_0 * Ap1_0 + p1_1 * Ap1_1 + p1_2 * Ap1_2 + p1_3 * Ap1_3 + p2_0 * Ap2_0 + p2_1 * Ap2_1 + p2_2 * Ap2_2 + p2_3 * Ap2_3
        alpha = r_normsq / (pAp + 1e-10)
        x1_0 = x1_0 + alpha * p1_0
        x1_1 = x1_1 + alpha * p1_1
        x1_2 = x1_2 + alpha * p1_2
        x1_3 = x1_3 + alpha * p1_3
        x2_0 = x2_0 + alpha * p2_0
        x2_1 = x2_1 + alpha * p2_1
        x2_2 = x2_2 + alpha * p2_2
        x2_3 = x2_3 + alpha * p2_3
        r1_0 = r1_0 - alpha * Ap1_0
        r1_1 = r1_1 - alpha * Ap1_1
        r1_2 = r1_2 - alpha * Ap1_2
        r1_3 = r1_3 - alpha * Ap1_3
        r2_0 = r2_0 - alpha * Ap2_0
        r2_1 = r2_1 - alpha * Ap2_1
        r2_2 = r2_2 - alpha * Ap2_2
        r2_3 = r2_3 - alpha * Ap2_3
        r_new_normsq = r1_0 * r1_0 + r1_1 * r1_1 + r1_2 * r1_2 + r1_3 * r1_3 + r2_0 * r2_0 + r2_1 * r2_1 + r2_2 * r2_2 + r2_3 * r2_3
        beta = r_new_normsq / (r_normsq + 1e-10)
        p1_0 = r1_0 + beta * p1_0
        p1_1 = r1_1 + beta * p1_1
        p1_2 = r1_2 + beta * p1_2
        p1_3 = r1_3 + beta * p1_3
        p2_0 = r2_0 + beta * p2_0
        p2_1 = r2_1 + beta * p2_1
        p2_2 = r2_2 + beta * p2_2
        p2_3 = r2_3 + beta * p2_3
        r_normsq = r_new_normsq
    res_00 = (dR_00 - x1_0 - x2_0) * R_00
    res_01 = (dR_01 - x1_0 - x2_1) * R_01
    res_02 = (dR_02 - x1_0 - x2_2) * R_02
    res_03 = (dR_03 - x1_0 - x2_3) * R_03
    res_10 = (dR_10 - x1_1 - x2_0) * R_10
    res_11 = (dR_11 - x1_1 - x2_1) * R_11
    res_12 = (dR_12 - x1_1 - x2_2) * R_12
    res_13 = (dR_13 - x1_1 - x2_3) * R_13
    res_20 = (dR_20 - x1_2 - x2_0) * R_20
    res_21 = (dR_21 - x1_2 - x2_1) * R_21
    res_22 = (dR_22 - x1_2 - x2_2) * R_22
    res_23 = (dR_23 - x1_2 - x2_3) * R_23
    res_30 = (dR_30 - x1_3 - x2_0) * R_30
    res_31 = (dR_31 - x1_3 - x2_1) * R_31
    res_32 = (dR_32 - x1_3 - x2_2) * R_32
    res_33 = (dR_33 - x1_3 - x2_3) * R_33
    tl.store(res_ptr + base_res + 0, res_00, mask=mask)
    tl.store(res_ptr + base_res + 1, res_01, mask=mask)
    tl.store(res_ptr + base_res + 2, res_02, mask=mask)
    tl.store(res_ptr + base_res + 3, res_03, mask=mask)
    tl.store(res_ptr + base_res + 4, res_10, mask=mask)
    tl.store(res_ptr + base_res + 5, res_11, mask=mask)
    tl.store(res_ptr + base_res + 6, res_12, mask=mask)
    tl.store(res_ptr + base_res + 7, res_13, mask=mask)
    tl.store(res_ptr + base_res + 8, res_20, mask=mask)
    tl.store(res_ptr + base_res + 9, res_21, mask=mask)
    tl.store(res_ptr + base_res + 10, res_22, mask=mask)
    tl.store(res_ptr + base_res + 11, res_23, mask=mask)
    tl.store(res_ptr + base_res + 12, res_30, mask=mask)
    tl.store(res_ptr + base_res + 13, res_31, mask=mask)
    tl.store(res_ptr + base_res + 14, res_32, mask=mask)
    tl.store(res_ptr + base_res + 15, res_33, mask=mask)
