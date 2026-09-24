# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems/src/flag_gems/ops/rms_norm.py
# Provenance: results/gluon/02/rms_norm_grad_dw_kernel_7efa9d58a9_9821b985d821.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def rms_norm_grad_dw_kernel_7efa9d58a9(X, DY, INV_RMS, DW, dx_stride_r, dx_stride_c, x_stride_r, x_stride_c, M, N, ROW_BLOCK_SIZE: gl.constexpr, COL_BLOCK_SIZE: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    row_pid = __h81.freeze(gl.program_id(0))
    col_pid = __h81.freeze(gl.program_id(1))
    row_start = __h81.freeze(__h81.to_auto(row_pid) * __h81.to_auto(ROW_BLOCK_SIZE))
    col_start = __h81.freeze(__h81.to_auto(col_pid) * __h81.to_auto(COL_BLOCK_SIZE))
    offset = __h81.freeze(__h81.to_auto(row_start) * __h81.to_auto(x_stride_r) + __h81.to_auto(col_start) * __h81.to_auto(x_stride_c))
    X = __h81.freeze(__h81.to_auto(X) + __h81.to_auto(offset), _layout=__h81.layout_of(X))
    DY = __h81.freeze(__h81.to_auto(DY) + __h81.to_auto(offset), _layout=__h81.layout_of(DY))
    INV_RMS = __h81.freeze(__h81.to_auto(INV_RMS) + __h81.to_auto(row_start), _layout=__h81.layout_of(INV_RMS))
    rows = __h81.freeze(__h81.arange(0, __h81.to_auto(ROW_BLOCK_SIZE), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0]))), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])))
    cols = __h81.freeze(__h81.arange(0, __h81.to_auto(COL_BLOCK_SIZE)), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])))
    row_mask = __h81.freeze(__h81.compare(__h81.to_auto(row_start) + __h81.to_auto(rows), __h81.to_auto(M), 'lt'), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])))
    col_mask = __h81.freeze(__h81.compare(__h81.to_auto(col_start) + __h81.to_auto(cols), __h81.to_auto(N), 'lt'), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])))
    x = __h81.freeze(__h81.load(__h81.to_auto(X) + __h81.expand_dims(__h81.to_auto(rows), 1) * __h81.to_auto(x_stride_r) + __h81.expand_dims(__h81.to_auto(cols), 0) * __h81.to_auto(x_stride_c), __h81.expand_dims(__h81.to_auto(row_mask), 1) & __h81.expand_dims(__h81.to_auto(col_mask), 0), other=0.0, _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])).to(gl.float32), _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0]))
    inv_rms = __h81.freeze(__h81.load(__h81.to_auto(INV_RMS) + __h81.to_auto(rows), __h81.to_auto(row_mask), other=0.0, _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0]))).to(gl.float32), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])))
    dy = __h81.freeze(__h81.load(__h81.to_auto(DY) + __h81.expand_dims(__h81.to_auto(rows), 1) * __h81.to_auto(x_stride_r) + __h81.expand_dims(__h81.to_auto(cols), 0) * __h81.to_auto(x_stride_c), __h81.expand_dims(__h81.to_auto(row_mask), 1) & __h81.expand_dims(__h81.to_auto(col_mask), 0), other=0.0, _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])).to(gl.float32), _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0]))
    d_weight = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(dy) * __h81.expand_dims(__h81.to_auto(inv_rms), 1), _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0]))
    partial_dweight_sum = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(d_weight), _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])), axis=0)), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])))
    __h81.store(__h81.to_auto(DW) + __h81.to_auto(row_pid) * __h81.to_auto(N) + __h81.to_auto(col_start) + __h81.to_auto(cols), __h81.to_auto(partial_dweight_sum), mask=__h81.to_auto(col_mask), _layout=gl.BlockedLayout([1], [64], [4], [0]))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    if dtype is not None:
        return dtype
    out_dtype = None
    if in_dtype.is_int_signed():
        out_dtype = gl.int32 if in_dtype.int_bitwidth < 32 else None
    elif in_dtype.is_int_unsigned():
        out_dtype = gl.uint32 if in_dtype.int_bitwidth < 32 else None
    return out_dtype

@g.jit
def _sum_combine_9669e8607d(a, b):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    return __h81.to_auto(a) + __h81.to_auto(b)

KERNEL = rms_norm_grad_dw_kernel_7efa9d58a9
