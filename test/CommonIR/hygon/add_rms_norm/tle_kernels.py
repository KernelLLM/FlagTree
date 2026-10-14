"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
import tle_ext as tle
from triton.language.extra import libdevice
@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b

@triton.jit(do_not_specialize=['eps'])
def add_rms_norm_kernel(out_ptr, in_ptr1, in_ptr2, w_ptr, y_stride_r, y_stride_c, x1_stride_r, x1_stride_c, x2_stride_r, x2_stride_c, N, eps, BLOCK_SIZE: tl.constexpr):
    if tl.constexpr(in_ptr1.dtype.element_ty == tl.float16) or tl.constexpr(in_ptr1.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = in_ptr1.dtype.element_ty
    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr1 += pid * x1_stride_r
    in_ptr2 += pid * x2_stride_r
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(in_ptr1 + cols * x1_stride_c, mask, other=0.0).to(cdtype)
    x2 = tl.load(in_ptr2 + cols * x2_stride_c, mask, other=0.0).to(cdtype)
    x = x1 + x2
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)

@triton.jit(do_not_specialize=['eps'])
def add_rms_norm_loop_kernel(out_ptr, in_ptr1, in_ptr2, w_ptr, N, eps, TILE_N: tl.constexpr):
    if tl.constexpr(in_ptr1.dtype.element_ty == tl.float16) or tl.constexpr(in_ptr1.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = in_ptr1.dtype.element_ty
    pid = tle.program_id(0)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x1 = tl.load(in_ptr1 + pid * N + n_offsets).to(tl.float32)
        x2 = tl.load(in_ptr2 + pid * N + n_offsets).to(tl.float32)
        x = x1 + x2
        acc += x * x
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x1 = tl.load(in_ptr1 + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in_ptr2 + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    x = x1 + x2
    acc += x * x
    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)
    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = prev_multiple - start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x1 = tl.load(in_ptr1 + pid * N + n_offsets, mask=mask, other=0.0, eviction_policy='evict_first').to(cdtype)
        x2 = tl.load(in_ptr2 + pid * N + n_offsets, mask=mask, other=0.0, eviction_policy='evict_first').to(cdtype)
        x = x1 + x2
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (x * rrms * w).to(cdtype)
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = prev_multiple - start_n + tl.arange(0, TILE_N)
        x1 = tl.load(in_ptr1 + pid * N + n_offsets, eviction_policy='evict_first').to(cdtype)
        x2 = tl.load(in_ptr2 + pid * N + n_offsets, eviction_policy='evict_first').to(cdtype)
        x = x1 + x2
        w = tl.load(w_ptr + n_offsets)
        y = (x * rrms * w).to(cdtype)
        tl.store(out_ptr + pid * N + n_offsets, y)
