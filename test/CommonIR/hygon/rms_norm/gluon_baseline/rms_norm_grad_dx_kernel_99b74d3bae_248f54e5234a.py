# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems/src/flag_gems/ops/rms_norm.py
# Provenance: results/gluon/02/rms_norm_grad_dx_kernel_99b74d3bae_248f54e5234a.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit(do_not_specialize=['eps'])
def rms_norm_grad_dx_kernel_99b74d3bae(X, DY, INV_RMS, DX, W, dx_stride_r, dx_stride_c, x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    pid = __h81.freeze(program_id_f71308a253(__h81.pin(0)))
    DX = __h81.freeze(__h81.to_auto(DX) + __h81.to_auto(pid) * __h81.to_auto(dx_stride_r), _layout=__h81.layout_of(DX))
    X = __h81.freeze(__h81.to_auto(X) + __h81.to_auto(pid) * __h81.to_auto(x_stride_r), _layout=__h81.layout_of(X))
    DY = __h81.freeze(__h81.to_auto(DY) + __h81.to_auto(pid) * __h81.to_auto(x_stride_r), _layout=__h81.layout_of(DY))
    INV_RMS = __h81.freeze(__h81.to_auto(INV_RMS) + __h81.to_auto(pid), _layout=__h81.layout_of(INV_RMS))
    mask = __h81.freeze(__h81.compare(__h81.arange(0, __h81.to_auto(BLOCK_SIZE), layout=gl.BlockedLayout([1], [64], [4], [0])), __h81.to_auto(N), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    cols = __h81.freeze(__h81.arange(0, __h81.to_auto(BLOCK_SIZE)))
    x = __h81.freeze(__h81.load(__h81.to_auto(X) + __h81.to_auto(cols) * __h81.to_auto(x_stride_c), __h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    inv_rms = __h81.freeze(__h81.load(__h81.to_auto(INV_RMS)).to(gl.float32))
    dy = __h81.freeze(__h81.load(__h81.to_auto(DY) + __h81.to_auto(cols) * __h81.to_auto(x_stride_c), __h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    w = __h81.freeze(__h81.load(__h81.to_auto(W) + __h81.arange(0, __h81.to_auto(BLOCK_SIZE)), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    dy = __h81.freeze(__h81.to_auto(dy) * __h81.to_auto(w), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    normalized_buf = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(inv_rms), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    row_sum_stats = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(normalized_buf) * __h81.to_auto(dy), _layout=gl.BlockedLayout([1], [64], [4], [0])), axis=0)), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    norm_val = __h81.freeze(__h81.to_auto(normalized_buf) / __h81.to_auto(N), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    dx = __h81.freeze((__h81.to_auto(dy) - __h81.to_auto(norm_val) * __h81.to_auto(row_sum_stats)) * __h81.to_auto(inv_rms), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    __h81.store(__h81.to_auto(DX) + __h81.to_auto(cols) * __h81.to_auto(dx_stride_c), __h81.to_auto(dx), mask=__h81.to_auto(mask), _layout=gl.BlockedLayout([1], [64], [4], [0]))

@g.jit
def program_id_f71308a253(axis: int) -> gl.tensor:
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    return gl.program_id(__h81.to_auto(axis)).to(gl.int64)

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [4], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

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

KERNEL = rms_norm_grad_dx_kernel_99b74d3bae
