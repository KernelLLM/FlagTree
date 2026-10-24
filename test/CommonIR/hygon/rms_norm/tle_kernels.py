"""Exact original device AST; only host registration/autotuning is removed.
The complete upstream source and license are in tle/."""
import triton
import triton.language as tl
from rms_norm import tle_ext as ext

@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b

@triton.jit(do_not_specialize=['eps'])
def rms_norm_kernel(out_ptr, INV_RMS, in_ptr, w_ptr, y_stride_r, y_stride_c, x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: tl.constexpr):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(in_ptr.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty
    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
    y = x_normed * w
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)

@triton.jit(do_not_specialize=['eps'])
def rms_norm_loop_kernel(out_ptr, INV_RMS, in_ptr, w_ptr, N, eps, TILE_N: tl.constexpr):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(in_ptr.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty
    pid = ext.program_id(0)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x
    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)
    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = prev_multiple - start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0, eviction_policy='evict_first').to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = prev_multiple - start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy='evict_first').to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        tl.store(out_ptr + pid * N + n_offsets, y)

@triton.jit(do_not_specialize=['eps'])
def rms_norm_grad_dx_kernel(X, DY, INV_RMS, DX, W, dx_stride_r, dx_stride_c, x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    dy = dy * w
    normalized_buf = x * inv_rms
    row_sum_stats = tl.sum(normalized_buf * dy, axis=0)
    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms
    tl.store(DX + cols * dx_stride_c, dx, mask=mask)

@triton.jit
def rms_norm_grad_dw_kernel(X, DY, INV_RMS, DW, dx_stride_r, dx_stride_c, x_stride_r, x_stride_c, M, N, ROW_BLOCK_SIZE: tl.constexpr, COL_BLOCK_SIZE: tl.constexpr):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE
    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start
    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)
    row_mask = row_start + rows < M
    col_mask = col_start + cols < N
    x = tl.load(X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c, row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c, row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)
    tl.store(DW + row_pid * N + col_start + cols, partial_dweight_sum, mask=col_mask)
