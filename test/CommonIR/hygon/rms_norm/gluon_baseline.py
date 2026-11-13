from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3


@g.jit(do_not_specialize=['eps'])
def forward(out_ptr, INV_RMS, in_ptr, w_ptr, y_stride_r, y_stride_c,
            x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: gl.constexpr,
            SIZE_PER_THREAD: gl.constexpr, NUM_WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([SIZE_PER_THREAD], [64], [NUM_WARPS], [0])
    if gl.constexpr(in_ptr.dtype.element_ty == gl.float16) or gl.constexpr(in_ptr.dtype.element_ty == gl.bfloat16):
        cdtype: gl.constexpr = gl.float32
    else:
        cdtype: gl.constexpr = in_ptr.dtype.element_ty
    pid = gl.program_id(0)
    cols = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = cols < N
    x = cdna3.buffer_load(in_ptr, pid * x_stride_r + cols * x_stride_c, mask=mask, other=0.0).to(cdtype)
    var = gl.sum(x * x, axis=0) / N
    rrms = 1 / gl.sqrt(var + eps)
    w = cdna3.buffer_load(w_ptr, cols, mask=mask, other=0.0)
    # Keep the original intermediate rounding before weight multiplication.
    x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
    y = x_normed * w
    cdna3.buffer_store(y, out_ptr, pid * y_stride_r + cols * y_stride_c, mask=mask)
    gl.store(INV_RMS + pid, rrms)


@g.jit(do_not_specialize=['eps'])
def backward_dx(X, DY, INV_RMS, DX, W, dx_stride_r, dx_stride_c,
                x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: gl.constexpr,
                SIZE_PER_THREAD: gl.constexpr, NUM_WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([SIZE_PER_THREAD], [64], [NUM_WARPS], [0])
    pid = gl.program_id(0)
    cols = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = cols < N
    x = cdna3.buffer_load(X, pid * x_stride_r + cols * x_stride_c, mask=mask, other=0.0).to(gl.float32)
    inv_rms = gl.load(INV_RMS + pid).to(gl.float32)
    dy = cdna3.buffer_load(DY, pid * x_stride_r + cols * x_stride_c, mask=mask, other=0.0).to(gl.float32)
    w = cdna3.buffer_load(W, cols, mask=mask, other=0.0)
    dy = dy * w
    normalized_buf = x * inv_rms
    row_sum_stats = gl.sum(normalized_buf * dy, axis=0)
    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms
    cdna3.buffer_store(dx.to(DX.dtype.element_ty), DX, pid * dx_stride_r + cols * dx_stride_c, mask=mask)


@g.jit
def backward_dw(X, DY, INV_RMS, DW, dx_stride_r, dx_stride_c,
                x_stride_r, x_stride_c, M, N, ROW_BLOCK_SIZE: gl.constexpr,
                COL_BLOCK_SIZE: gl.constexpr, COL_SPT: gl.constexpr,
                THREADS_PER_ROW: gl.constexpr, NUM_WARPS: gl.constexpr,
                CONTIGUOUS_STORE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, COL_SPT], [THREADS_PER_ROW, 64 // THREADS_PER_ROW], [NUM_WARPS, 1], [1, 0])
    row_pid = gl.program_id(0)
    col_pid = gl.program_id(1)
    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE
    offset = row_start * x_stride_r + col_start * x_stride_c
    rows = gl.arange(0, ROW_BLOCK_SIZE, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, COL_BLOCK_SIZE, layout=gl.SliceLayout(0, layout))
    row_mask = row_start + rows < M
    col_mask = col_start + cols < N
    offsets = offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c
    mask = row_mask[:, None] & col_mask[None, :]
    x = cdna3.buffer_load(X, offsets, mask=mask, other=0.0).to(gl.float32)
    inv_rms = cdna3.buffer_load(INV_RMS, row_start + rows, mask=row_mask, other=0.0).to(gl.float32)
    dy = cdna3.buffer_load(DY, offsets, mask=mask, other=0.0).to(gl.float32)
    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = gl.sum(d_weight, axis=0)
    if CONTIGUOUS_STORE:
        # The TLE baseline moves the natural slice result to this store layout.
        store_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
        partial_dweight_sum = gl.convert_layout(partial_dweight_sum, store_layout)
        store_cols = gl.arange(0, COL_BLOCK_SIZE, layout=store_layout)
    else:
        # Historical optimized layout stores the natural reduction slice directly.
        store_cols = cols
    cdna3.buffer_store(partial_dweight_sum, DW, row_pid * N + col_start + store_cols, mask=col_start + store_cols < N)


@g.jit(do_not_specialize=['eps'])
def forward_loop(out_ptr, INV_RMS, in_ptr, w_ptr, N, eps, TILE_N: gl.constexpr,
                 LOAD_SPT: gl.constexpr, STORE_SPT: gl.constexpr, NUM_WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([LOAD_SPT], [64], [NUM_WARPS], [0])
    store_layout: gl.constexpr = gl.BlockedLayout([STORE_SPT], [64], [NUM_WARPS], [0])
    if gl.constexpr(in_ptr.dtype.element_ty == gl.float16) or gl.constexpr(in_ptr.dtype.element_ty == gl.bfloat16):
        cdtype: gl.constexpr = gl.float32
    else:
        cdtype: gl.constexpr = in_ptr.dtype.element_ty
    pid = gl.program_id(0)
    lane = gl.arange(0, TILE_N, layout=layout)
    acc = gl.full((TILE_N,), 0.0, gl.float32, layout=layout)
    num_steps = gl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        n_offsets = step * TILE_N + lane
        x = cdna3.buffer_load(in_ptr, pid * N + n_offsets).to(gl.float32)
        acc = acc + x * x
    n_offsets = (num_steps - 1) * TILE_N + lane
    mask = n_offsets < N
    x = cdna3.buffer_load(in_ptr, pid * N + n_offsets, mask=mask, other=0.0).to(gl.float32)
    acc = acc + x * x
    # TLE's same-shape allow_reorder view does not move the actual FP32 tuple.
    # Consume that tuple directly, retaining its effective load/acc ownership.
    var = gl.sum(acc, axis=0) / N
    rrms = 1 / gl.sqrt(var + eps)
    gl.store(INV_RMS + pid, rrms)
    prev_multiple = gl.cdiv(N, TILE_N) * TILE_N - TILE_N
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = prev_multiple - start_n + lane
        mask = n_offsets < N
        x = cdna3.buffer_load(in_ptr, pid * N + n_offsets, mask=mask, other=0.0).to(cdtype)
        w = cdna3.buffer_load(w_ptr, n_offsets, mask=mask, other=0.0)
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        if LOAD_SPT != STORE_SPT:
            y = gl.convert_layout(y, store_layout)
        store_cols = gl.arange(0, TILE_N, layout=store_layout)
        store_offsets = prev_multiple - start_n + store_cols
        cdna3.buffer_store(y, out_ptr, pid * N + store_offsets, mask=store_offsets < N)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = prev_multiple - start_n + lane
        x = cdna3.buffer_load(in_ptr, pid * N + n_offsets).to(cdtype)
        w = cdna3.buffer_load(w_ptr, n_offsets)
        x_normed = (x * rrms).to(in_ptr.dtype.element_ty)
        y = x_normed * w
        if LOAD_SPT != STORE_SPT:
            y = gl.convert_layout(y, store_layout)
        store_cols = gl.arange(0, TILE_N, layout=store_layout)
        store_offsets = prev_multiple - start_n + store_cols
        cdna3.buffer_store(y, out_ptr, pid * N + store_offsets)
