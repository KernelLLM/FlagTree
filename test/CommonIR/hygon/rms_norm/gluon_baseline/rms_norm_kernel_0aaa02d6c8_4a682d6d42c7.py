# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems/src/flag_gems/ops/rms_norm.py
# Provenance: results/gluon/02/rms_norm_kernel_0aaa02d6c8_4a682d6d42c7.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit(do_not_specialize=['eps'])
def rms_norm_kernel_0aaa02d6c8(out_ptr, INV_RMS, in_ptr, w_ptr, y_stride_r, y_stride_c, x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    if gl.constexpr(__h81.compare(__h81.pointer_element_type(__h81.to_auto(in_ptr), True), gl.float16, 'eq')) or gl.constexpr(__h81.compare(__h81.pointer_element_type(__h81.to_auto(in_ptr), True), gl.bfloat16, 'eq')):
        cdtype = __h81.freeze(gl.float32)
    else:
        cdtype = __h81.freeze(__h81.pointer_element_type(__h81.to_auto(in_ptr), True))
    pid = __h81.freeze(gl.program_id(0))
    out_ptr = __h81.freeze(__h81.to_auto(out_ptr) + __h81.to_auto(pid) * __h81.to_auto(y_stride_r), _layout=__h81.layout_of(out_ptr))
    in_ptr = __h81.freeze(__h81.to_auto(in_ptr) + __h81.to_auto(pid) * __h81.to_auto(x_stride_r), _layout=__h81.layout_of(in_ptr))
    mask = __h81.freeze(__h81.compare(__h81.arange(0, __h81.to_auto(BLOCK_SIZE), layout=gl.BlockedLayout([1], [64], [4], [0])), __h81.to_auto(N), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    cols = __h81.freeze(__h81.arange(0, __h81.to_auto(BLOCK_SIZE)))
    x = __h81.freeze(__h81.load(__h81.to_auto(in_ptr) + __h81.to_auto(cols) * __h81.to_auto(x_stride_c), __h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])).to(__h81.to_auto(cdtype)), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    var = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(x) * __h81.to_auto(x), _layout=gl.BlockedLayout([1], [64], [4], [0])), axis=0)) / __h81.to_auto(N), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    rrms = __h81.freeze(1 / gl.sqrt(__h81.to_auto(var) + __h81.to_auto(eps)))
    w = __h81.freeze(__h81.load(__h81.to_auto(w_ptr) + __h81.arange(0, __h81.to_auto(BLOCK_SIZE)), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    x_normed = __h81.freeze((__h81.to_auto(x) * __h81.to_auto(rrms)).to(__h81.pointer_element_type(__h81.to_auto(in_ptr), True)), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    y = __h81.freeze(__h81.to_auto(x_normed) * __h81.to_auto(w), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(cols) * __h81.to_auto(y_stride_c), __h81.to_auto(y), mask=__h81.to_auto(mask), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    __h81.store(__h81.to_auto(INV_RMS) + __h81.to_auto(pid), __h81.to_auto(rrms))

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

KERNEL = rms_norm_kernel_0aaa02d6c8
