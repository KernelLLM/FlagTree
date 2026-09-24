# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260910_phase2/src/FlagGems/src/flag_gems/ops/rms_norm_w8a16_fp8.py
# Provenance: results/gluon/gluon_baseline/05_loop_m1_n32768_t8192_w16.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit(do_not_specialize=['eps'])
def rms_norm_fp8_w8a16_loop_kernel_99bac17ec0(out_ptr, in_ptr, w_ptr, w_scale_ptr, N, eps, TILE_N: gl.constexpr, GROUP_SIZE: gl.constexpr):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    pid = __h81.freeze(program_id_f71308a253(__h81.pin(0)))
    acc = __h81.freeze(__h81.zeros((__h81.to_auto(TILE_N),), dtype=gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    num_steps = __h81.freeze(gl.cdiv(__h81.to_auto(N), __h81.to_auto(TILE_N)))
    for step in range(0, __h81.to_auto(num_steps) - 1):
        start_n = __h81.freeze(__h81.to_auto(step) * __h81.to_auto(TILE_N))
        n_offsets = __h81.freeze(__h81.to_auto(start_n) + __h81.arange(0, __h81.to_auto(TILE_N), layout=gl.BlockedLayout([8], [64], [16], [0])), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        x = __h81.freeze(__h81.load(__h81.to_auto(in_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(n_offsets), _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(x) * __h81.to_auto(x), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    start_n = __h81.freeze((__h81.to_auto(num_steps) - 1) * __h81.to_auto(TILE_N))
    n_offsets = __h81.freeze(__h81.to_auto(start_n) + __h81.arange(0, __h81.to_auto(TILE_N)), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    mask = __h81.freeze(__h81.compare(__h81.to_auto(n_offsets), __h81.to_auto(N), 'lt'), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    x = __h81.freeze(__h81.load(__h81.to_auto(in_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(n_offsets), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(x) * __h81.to_auto(x), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    var = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(acc), _layout=gl.BlockedLayout([1], [64], [16], [0])))) / __h81.to_auto(N), _layout=gl.BlockedLayout([1], [64], [16], [0]))
    rrms = __h81.freeze(1 / gl.sqrt(__h81.to_auto(var) + __h81.to_auto(eps)))
    prev_multiple = __h81.freeze(prev_multiple_of_5649e089e8(N, TILE_N))
    for start_n in range(0, __h81.to_auto(TILE_N), __h81.to_auto(TILE_N)):
        n_offsets = __h81.freeze(__h81.to_auto(prev_multiple) - __h81.to_auto(start_n) + __h81.arange(0, __h81.to_auto(TILE_N)), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        mask = __h81.freeze(__h81.compare(__h81.to_auto(n_offsets), __h81.to_auto(N), 'lt'), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        x = __h81.freeze(__h81.load(__h81.to_auto(in_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(n_offsets), mask=__h81.to_auto(mask), other=0.0, eviction_policy='evict_first', _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        group_ids = __h81.freeze(__h81.to_auto(n_offsets) // __h81.to_auto(GROUP_SIZE), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        w = __h81.freeze(__h81.load(__h81.to_auto(w_ptr) + __h81.to_auto(n_offsets), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        w_scale = __h81.freeze(__h81.load(__h81.to_auto(w_scale_ptr) + __h81.to_auto(group_ids), mask=__h81.to_auto(mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        y = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(rrms) * __h81.to_auto(w) * __h81.to_auto(w_scale), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(n_offsets), __h81.to_auto(y), mask=__h81.to_auto(mask), _layout=gl.BlockedLayout([8], [64], [16], [0]))
    for start_n in range(__h81.to_auto(TILE_N), __h81.to_auto(N), __h81.to_auto(TILE_N)):
        n_offsets = __h81.freeze(__h81.to_auto(prev_multiple) - __h81.to_auto(start_n) + __h81.arange(0, __h81.to_auto(TILE_N)), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        x = __h81.freeze(__h81.load(__h81.to_auto(in_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(n_offsets), eviction_policy='evict_first', _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        group_ids = __h81.freeze(__h81.to_auto(n_offsets) // __h81.to_auto(GROUP_SIZE), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        w = __h81.freeze(__h81.load(__h81.to_auto(w_ptr) + __h81.to_auto(n_offsets), _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        w_scale = __h81.freeze(__h81.load(__h81.to_auto(w_scale_ptr) + __h81.to_auto(group_ids), _layout=gl.BlockedLayout([8], [64], [16], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        y = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(rrms) * __h81.to_auto(w) * __h81.to_auto(w_scale), _layout=gl.BlockedLayout([8], [64], [16], [0]))
        __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(pid) * __h81.to_auto(N) + __h81.to_auto(n_offsets), __h81.to_auto(y), _layout=gl.BlockedLayout([8], [64], [16], [0]))

@g.jit
def program_id_f71308a253(axis: int) -> gl.tensor:
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return gl.program_id(__h81.to_auto(axis)).to(gl.int64)

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

@g.jit
def cdiv_5f871e20f4(x, div):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    '\n    Computes the ceiling division of :code:`x` by :code:`div`\n\n    :param x: the input number\n    :type x: Block\n    :param div: the divisor\n    :type div: Block\n    '
    return (__h81.to_auto(x) + (__h81.to_auto(div) - 1)) // __h81.to_auto(div)

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [16], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def prev_multiple_of_5649e089e8(a, b):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return gl.cdiv(__h81.to_auto(a), __h81.to_auto(b)) * __h81.to_auto(b) - __h81.to_auto(b)

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

KERNEL = rms_norm_fp8_w8a16_loop_kernel_99bac17ec0
