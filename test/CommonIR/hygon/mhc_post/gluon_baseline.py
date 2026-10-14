from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

@g.jit
def mhc_post_kernel_hc_mult_4(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, H: gl.constexpr, BLOCK_H: gl.constexpr, IO_LAYOUT: gl.constexpr):
    pid_n = gl.program_id(0)
    pid_h = gl.program_id(1)
    h_off = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=IO_LAYOUT)
    h_mask = h_off < H
    a_base = pid_n * 16
    c_base = pid_n * 4
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    out_base = pid_n * 4 * H
    c0 = gl.load(c_ptr + c_base + 0).to(gl.float32)
    c1 = gl.load(c_ptr + c_base + 1).to(gl.float32)
    c2 = gl.load(c_ptr + c_base + 2).to(gl.float32)
    c3 = gl.load(c_ptr + c_base + 3).to(gl.float32)
    a00 = gl.load(a_ptr + a_base + 0).to(gl.float32)
    a01 = gl.load(a_ptr + a_base + 1).to(gl.float32)
    a02 = gl.load(a_ptr + a_base + 2).to(gl.float32)
    a03 = gl.load(a_ptr + a_base + 3).to(gl.float32)
    a10 = gl.load(a_ptr + a_base + 4).to(gl.float32)
    a11 = gl.load(a_ptr + a_base + 5).to(gl.float32)
    a12 = gl.load(a_ptr + a_base + 6).to(gl.float32)
    a13 = gl.load(a_ptr + a_base + 7).to(gl.float32)
    a20 = gl.load(a_ptr + a_base + 8).to(gl.float32)
    a21 = gl.load(a_ptr + a_base + 9).to(gl.float32)
    a22 = gl.load(a_ptr + a_base + 10).to(gl.float32)
    a23 = gl.load(a_ptr + a_base + 11).to(gl.float32)
    a30 = gl.load(a_ptr + a_base + 12).to(gl.float32)
    a31 = gl.load(a_ptr + a_base + 13).to(gl.float32)
    a32 = gl.load(a_ptr + a_base + 14).to(gl.float32)
    a33 = gl.load(a_ptr + a_base + 15).to(gl.float32)
    d_vals = cdna3.buffer_load(d_ptr, (d_base + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
    b0 = cdna3.buffer_load(b_ptr, (b_base + 0 * H + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
    b1 = cdna3.buffer_load(b_ptr, (b_base + 1 * H + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
    b2 = cdna3.buffer_load(b_ptr, (b_base + 2 * H + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
    b3 = cdna3.buffer_load(b_ptr, (b_base + 3 * H + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
    acc0 = c0 * d_vals + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    acc1 = c1 * d_vals + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    acc2 = c2 * d_vals + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    acc3 = c3 * d_vals + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3
    cdna3.buffer_store(acc0.to(gl.bfloat16).to(out_ptr.dtype.element_ty), out_ptr, (out_base + 0 * H + h_off).to(gl.int32), mask=h_mask)
    cdna3.buffer_store(acc1.to(gl.bfloat16).to(out_ptr.dtype.element_ty), out_ptr, (out_base + 1 * H + h_off).to(gl.int32), mask=h_mask)
    cdna3.buffer_store(acc2.to(gl.bfloat16).to(out_ptr.dtype.element_ty), out_ptr, (out_base + 2 * H + h_off).to(gl.int32), mask=h_mask)
    cdna3.buffer_store(acc3.to(gl.bfloat16).to(out_ptr.dtype.element_ty), out_ptr, (out_base + 3 * H + h_off).to(gl.int32), mask=h_mask)

@g.jit
def mhc_post_kernel_generic(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, H: gl.constexpr, HC: gl.constexpr, BLOCK_H: gl.constexpr, IO_LAYOUT: gl.constexpr):
    pid_n = gl.program_id(0)
    pid_i = gl.program_id(1)
    pid_h = gl.program_id(2)
    h_off = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=IO_LAYOUT)
    h_mask = h_off < H
    a_base = pid_n * HC * HC
    b_base = pid_n * HC * H
    c_base = pid_n * HC
    d_base = pid_n * H
    out_base = pid_n * HC * H + pid_i * H
    d_vals = cdna3.buffer_load(d_ptr, (d_base + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
    c_i = gl.load(c_ptr + c_base + pid_i).to(gl.float32)
    acc = c_i * d_vals
    for j in gl.static_range(0, HC):
        a_ji = gl.load(a_ptr + a_base + j * HC + pid_i).to(gl.float32)
        b_j = cdna3.buffer_load(b_ptr, (b_base + j * H + h_off).to(gl.int32), mask=h_mask, other=0.0).to(gl.float32)
        acc += a_ji * b_j
    cdna3.buffer_store(acc.to(gl.bfloat16).to(out_ptr.dtype.element_ty), out_ptr, (out_base + h_off).to(gl.int32), mask=h_mask)
