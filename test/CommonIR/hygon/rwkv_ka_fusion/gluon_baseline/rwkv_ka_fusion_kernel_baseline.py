# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group9_20260921/rwkv_ka_fusion/remote_capture.py
# Provenance: results/gluon/baseline/rwkv_ka_fusion_kernel_baseline.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def rwkv_ka_fusion_kernel_e51288d289(k_ptr, kk_ptr, a_ptr, ka_ptr, o_k_ptr, o_kk_ptr, o_kka_ptr, T, C, H, N, N_size: gl.constexpr, block_size: gl.constexpr):
    """HCU helper SHA256: f2ca0831292f681ed665fd8fc4e7cd7a0eac68b42fbc6964b42c0e23d29d836d"""
    pid = __h81.freeze(gl.program_id(axis=0))
    k_start = __h81.freeze(__h81.to_auto(pid) * __h81.to_auto(block_size))
    for i in range(0, __h81.to_auto(H)):
        offs = __h81.freeze(__h81.to_auto(k_start) + __h81.to_auto(i) * __h81.to_auto(N) + __h81.arange(0, __h81.to_auto(N_size), layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        k = __h81.freeze(__h81.load(__h81.to_auto(k_ptr) + __h81.to_auto(offs), mask=__h81.compare(__h81.to_auto(offs), __h81.to_auto(T) * __h81.to_auto(C), 'lt'), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        a = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(offs), mask=__h81.compare(__h81.to_auto(offs), __h81.to_auto(T) * __h81.to_auto(C), 'lt'), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        c_offs = __h81.freeze(__h81.to_auto(i) * __h81.to_auto(N) + __h81.arange(0, __h81.to_auto(N_size)), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        ka = __h81.freeze(__h81.load(__h81.to_auto(ka_ptr) + __h81.to_auto(c_offs), mask=__h81.compare(__h81.to_auto(c_offs), __h81.to_auto(C), 'lt'), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        kk = __h81.freeze(__h81.load(__h81.to_auto(kk_ptr) + __h81.to_auto(c_offs), mask=__h81.compare(__h81.to_auto(c_offs), __h81.to_auto(C), 'lt'), other=0.0, _layout=gl.BlockedLayout([1], [64], [4], [0])), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        kt = __h81.freeze(__h81.to_auto(k) * __h81.to_auto(kk), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        kt2 = __h81.freeze(__h81.to_auto(kt) * __h81.to_auto(kt), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        norm_kt2 = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(kt2).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [4], [0])))), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        norm_kt = __h81.freeze(gl.sqrt(__h81.to_auto(norm_kt2) + 1e-12))
        okk = __h81.freeze(__h81.to_auto(kt) / __h81.to_auto(norm_kt), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        __h81.store(__h81.to_auto(o_kk_ptr) + __h81.to_auto(offs), __h81.to_auto(okk), mask=__h81.compare(__h81.to_auto(offs), __h81.to_auto(T) * __h81.to_auto(C), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        ok = __h81.freeze(__h81.to_auto(k) * (1 + (__h81.to_auto(a).to(gl.float32) - 1) * __h81.to_auto(ka)), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        okka = __h81.freeze(__h81.to_auto(okk) * __h81.to_auto(a), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        __h81.store(__h81.to_auto(o_k_ptr) + __h81.to_auto(offs), __h81.to_auto(ok), mask=__h81.compare(__h81.to_auto(offs), __h81.to_auto(T) * __h81.to_auto(C), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))
        __h81.store(__h81.to_auto(o_kka_ptr) + __h81.to_auto(offs), __h81.to_auto(okka), mask=__h81.compare(__h81.to_auto(offs), __h81.to_auto(T) * __h81.to_auto(C), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: f2ca0831292f681ed665fd8fc4e7cd7a0eac68b42fbc6964b42c0e23d29d836d"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [4], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: f2ca0831292f681ed665fd8fc4e7cd7a0eac68b42fbc6964b42c0e23d29d836d"""
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
    """HCU helper SHA256: f2ca0831292f681ed665fd8fc4e7cd7a0eac68b42fbc6964b42c0e23d29d836d"""
    return __h81.to_auto(a) + __h81.to_auto(b)

KERNEL = rwkv_ka_fusion_kernel_e51288d289
