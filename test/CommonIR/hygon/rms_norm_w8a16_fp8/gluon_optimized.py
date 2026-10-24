# Native implementation template provenance is recorded in provenance.json.
# Hard20 revalidates all lowering and numeric claims independently.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

@g.jit(do_not_specialize=['eps'])
def rms_norm_fp8_w8a16_kernel(out_ptr, in_ptr, w_ptr, w_scale_ptr, N, eps, BLOCK_SIZE: gl.constexpr, GROUP_SIZE: gl.constexpr, IO_LAYOUT: gl.constexpr, SCALE_LAYOUT: gl.constexpr, STORE_LAYOUT: gl.constexpr):
    pid = gl.program_id(0)
    cols = gl.arange(0, BLOCK_SIZE, layout=IO_LAYOUT)
    mask = cols < N
    x = cdna3.buffer_load(in_ptr, (pid * N + cols).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    squares = x * x
    var = gl.sum(squares, axis=0) / N
    inv = 1 / gl.sqrt(var + eps)
    weight = cdna3.buffer_load(w_ptr, cols.to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    scale_cols = gl.convert_layout(cols, SCALE_LAYOUT)
    scale_mask = gl.convert_layout(mask, SCALE_LAYOUT)
    scale = cdna3.buffer_load(w_scale_ptr, (scale_cols // GROUP_SIZE).to(gl.int32), mask=scale_mask, other=0.0).to(gl.float32)
    scale = gl.convert_layout(scale, IO_LAYOUT)
    y = x * inv * weight * scale
    out_cols = gl.arange(0, BLOCK_SIZE, layout=STORE_LAYOUT)
    out_mask = out_cols < N
    cdna3.buffer_store(gl.convert_layout(y.to(out_ptr.dtype.element_ty), STORE_LAYOUT), out_ptr, (pid * N + out_cols).to(gl.int32), mask=out_mask)

@g.jit(do_not_specialize=['eps'])
def rms_norm_fp8_w8a16_grouped_kernel(out_ptr, in_ptr, w_ptr, w_scale_ptr, N, eps, GROUP_SIZE: gl.constexpr, NUM_GROUPS: gl.constexpr, IO_LAYOUT: gl.constexpr, SCALE_LAYOUT: gl.constexpr, STORE_LAYOUT: gl.constexpr):
    pid = gl.program_id(0)
    group_layout: gl.constexpr = gl.SliceLayout(1, IO_LAYOUT)
    col_layout: gl.constexpr = gl.SliceLayout(0, IO_LAYOUT)
    groups = gl.arange(0, NUM_GROUPS, layout=group_layout)
    cols = gl.arange(0, GROUP_SIZE, layout=col_layout)
    offsets = groups[:, None] * GROUP_SIZE + cols[None, :]
    mask = offsets < N
    x = cdna3.buffer_load(in_ptr, (pid * N + offsets).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    squares = x * x
    flat = gl.reshape(squares, (NUM_GROUPS * GROUP_SIZE,))
    var = gl.sum(flat, axis=0) / N
    inv = 1 / gl.sqrt(var + eps)
    weight = cdna3.buffer_load(w_ptr, offsets.to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    scale_groups = gl.convert_layout(groups, SCALE_LAYOUT)
    scales = cdna3.buffer_load(w_scale_ptr, scale_groups.to(gl.int32)).to(gl.float32)
    scales = gl.convert_layout(scales, group_layout)
    y = x * inv * weight * scales[:, None]
    store_groups = gl.arange(0, NUM_GROUPS, layout=gl.SliceLayout(1, STORE_LAYOUT))
    store_columns = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(0, STORE_LAYOUT))
    out_offsets = store_groups[:, None] * GROUP_SIZE + store_columns[None, :]
    out_mask = out_offsets < N
    cdna3.buffer_store(gl.convert_layout(y.to(out_ptr.dtype.element_ty), STORE_LAYOUT), out_ptr, (pid * N + out_offsets).to(gl.int32), mask=out_mask)

@g.jit(do_not_specialize=['eps'])
def rms_norm_fp8_w8a16_loop_kernel(out_ptr, in_ptr, w_ptr, w_scale_ptr, N, eps, TILE_N: gl.constexpr, GROUP_SIZE: gl.constexpr, IO_LAYOUT: gl.constexpr, SCALE_LAYOUT: gl.constexpr, STORE_LAYOUT: gl.constexpr):
    pid = gl.program_id(0).to(gl.int64)
    lane = gl.arange(0, TILE_N, layout=IO_LAYOUT)
    acc = gl.full((TILE_N,), 0, gl.float32, layout=IO_LAYOUT)
    steps = gl.cdiv(N, TILE_N)
    for step in range(0, steps - 1):
        cols = step * TILE_N + lane
        x = cdna3.buffer_load(in_ptr, (pid * N + cols).to(gl.int32)).to(gl.float32)
        acc = acc + x * x
    cols = (steps - 1) * TILE_N + lane
    mask = cols < N
    x = cdna3.buffer_load(in_ptr, (pid * N + cols).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    acc = acc + x * x
    var = gl.sum(acc, axis=0) / N
    inv = 1 / gl.sqrt(var + eps)
    previous = steps * TILE_N - TILE_N
    for start in range(0, TILE_N, TILE_N):
        cols = previous - start + lane
        mask = cols < N
        x = gl.load(in_ptr + pid * N + cols, mask=mask, other=0.0, eviction_policy='evict_first').to(gl.float32)
        weight = cdna3.buffer_load(w_ptr, cols.to(gl.int32), mask=mask, other=0.0).to(gl.float32)
        scols = gl.convert_layout(cols, SCALE_LAYOUT)
        smask = gl.convert_layout(mask, SCALE_LAYOUT)
        scales = cdna3.buffer_load(w_scale_ptr, (scols // GROUP_SIZE).to(gl.int32), mask=smask, other=0.0).to(gl.float32)
        y = x * inv * weight * gl.convert_layout(scales, IO_LAYOUT)
        out_cols = previous - start + gl.arange(0, TILE_N, layout=STORE_LAYOUT)
        out_mask = out_cols < N
        cdna3.buffer_store(gl.convert_layout(y.to(out_ptr.dtype.element_ty), STORE_LAYOUT), out_ptr, (pid * N + out_cols).to(gl.int32), mask=out_mask)
    for start in range(TILE_N, N, TILE_N):
        cols = previous - start + lane
        x = gl.load(in_ptr + pid * N + cols, eviction_policy='evict_first').to(gl.float32)
        weight = cdna3.buffer_load(w_ptr, cols.to(gl.int32)).to(gl.float32)
        scols = gl.convert_layout(cols, SCALE_LAYOUT)
        scales = cdna3.buffer_load(w_scale_ptr, (scols // GROUP_SIZE).to(gl.int32)).to(gl.float32)
        y = x * inv * weight * gl.convert_layout(scales, IO_LAYOUT)
        out_cols = previous - start + gl.arange(0, TILE_N, layout=STORE_LAYOUT)
        cdna3.buffer_store(gl.convert_layout(y.to(out_ptr.dtype.element_ty), STORE_LAYOUT), out_ptr, (pid * N + out_cols).to(gl.int32))
