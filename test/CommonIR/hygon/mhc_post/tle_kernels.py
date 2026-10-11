"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice
@triton.jit
def mhc_post_kernel_hc_mult_4(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, H: tl.constexpr, BLOCK_H: tl.constexpr):
    """
    Grid: (N, cdiv(H, BLOCK_H)).
    Each program handles one token × one h-tile × all 4 hc streams.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H
    a_base = pid_n * 16
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H
    c0 = tl.load(c_ptr + c_base + 0).to(tl.float32)
    c1 = tl.load(c_ptr + c_base + 1).to(tl.float32)
    c2 = tl.load(c_ptr + c_base + 2).to(tl.float32)
    c3 = tl.load(c_ptr + c_base + 3).to(tl.float32)
    a00 = tl.load(a_ptr + a_base + 0).to(tl.float32)
    a01 = tl.load(a_ptr + a_base + 1).to(tl.float32)
    a02 = tl.load(a_ptr + a_base + 2).to(tl.float32)
    a03 = tl.load(a_ptr + a_base + 3).to(tl.float32)
    a10 = tl.load(a_ptr + a_base + 4).to(tl.float32)
    a11 = tl.load(a_ptr + a_base + 5).to(tl.float32)
    a12 = tl.load(a_ptr + a_base + 6).to(tl.float32)
    a13 = tl.load(a_ptr + a_base + 7).to(tl.float32)
    a20 = tl.load(a_ptr + a_base + 8).to(tl.float32)
    a21 = tl.load(a_ptr + a_base + 9).to(tl.float32)
    a22 = tl.load(a_ptr + a_base + 10).to(tl.float32)
    a23 = tl.load(a_ptr + a_base + 11).to(tl.float32)
    a30 = tl.load(a_ptr + a_base + 12).to(tl.float32)
    a31 = tl.load(a_ptr + a_base + 13).to(tl.float32)
    a32 = tl.load(a_ptr + a_base + 14).to(tl.float32)
    a33 = tl.load(a_ptr + a_base + 15).to(tl.float32)
    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b0 = tl.load(b_ptr + b_base + 0 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + b_base + 1 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + b_base + 2 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    b3 = tl.load(b_ptr + b_base + 3 * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3
    tl.store(out_ptr + out_base + 0 * H + h_off, acc0.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 1 * H + h_off, acc1.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 2 * H + h_off, acc2.to(tl.bfloat16), mask=h_mask)
    tl.store(out_ptr + out_base + 3 * H + h_off, acc3.to(tl.bfloat16), mask=h_mask)

@triton.jit
def mhc_post_kernel_generic(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, H: tl.constexpr, HC: tl.constexpr, BLOCK_H: tl.constexpr):
    """Generic mHC post kernel for arbitrary HC.

    Grid: (N, HC, cdiv(H, BLOCK_H)).
    Each program handles one token × one output-stream(i) × one h-tile.
    """
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H
    a_base = pid_n * HC * HC
    b_base = pid_n * HC * H
    c_base = pid_n * HC
    d_base = pid_n * H
    out_base = pid_n * HC * H + pid_i * H
    d_vals = tl.load(d_ptr + d_base + h_off, mask=h_mask, other=0.0).to(tl.float32)
    c_i = tl.load(c_ptr + c_base + pid_i).to(tl.float32)
    acc = c_i * d_vals
    for j in tl.static_range(0, HC):
        a_ji = tl.load(a_ptr + a_base + j * HC + pid_i).to(tl.float32)
        b_j = tl.load(b_ptr + b_base + j * H + h_off, mask=h_mask, other=0.0).to(tl.float32)
        acc += a_ji * b_j
    tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)
