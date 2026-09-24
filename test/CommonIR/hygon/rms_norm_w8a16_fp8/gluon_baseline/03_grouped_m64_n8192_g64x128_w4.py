# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260910_phase2/src/FlagGems/src/flag_gems/ops/rms_norm_w8a16_fp8.py
# Provenance: results/gluon/gluon_baseline/03_grouped_m64_n8192_g64x128_w4.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit(do_not_specialize=['eps'])
def rms_norm_fp8_w8a16_grouped_kernel_ddd51d22d5(out_ptr, in_ptr, w_ptr, w_scale_ptr, N, eps, GROUP_SIZE: gl.constexpr, NUM_GROUPS: gl.constexpr):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    pid = __h81.freeze(gl.program_id(0))
    groups = __h81.freeze(__h81.arange(0, __h81.to_auto(NUM_GROUPS), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])))
    cols = __h81.freeze(__h81.arange(0, __h81.to_auto(GROUP_SIZE)))
    offsets = __h81.freeze(__h81.expand_dims(__h81.to_auto(groups), 1) * __h81.to_auto(GROUP_SIZE) + __h81.expand_dims(__h81.to_auto(cols), 0), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
    mask = __h81.freeze(__h81.compare(__h81.to_auto(offsets), __h81.to_auto(N), 'lt'), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
    x = __h81.freeze(__h81.load(__h81.to_auto(in_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(offsets), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])).to(gl.float32), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
    var_flat = gl.reshape(__h81.to_auto(x) * __h81.to_auto(x), (__h81.to_auto(NUM_GROUPS) * __h81.to_auto(GROUP_SIZE),))
    var = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(var_flat), _layout=gl.BlockedLayout([1], [64], [4], [0])))) / __h81.to_auto(N), _layout=gl.BlockedLayout([1], [64], [4], [0]))
    rrms = __h81.freeze(1 / gl.sqrt(__h81.to_auto(var) + __h81.to_auto(eps)))
    w = __h81.freeze(__h81.load(__h81.to_auto(w_ptr) + __h81.to_auto(offsets), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])).to(gl.float32), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
    w_scale = __h81.freeze(__h81.expand_dims(__h81.load(__h81.to_auto(w_scale_ptr) + __h81.to_auto(groups), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))).to(gl.float32), 1), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])))
    y = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(rrms) * __h81.to_auto(w) * __h81.to_auto(w_scale), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(offsets), __h81.to_auto(y), mask=__h81.to_auto(mask), _layout=gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0]))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [4], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
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
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return __h81.to_auto(a) + __h81.to_auto(b)

KERNEL = rms_norm_fp8_w8a16_grouped_kernel_ddd51d22d5
