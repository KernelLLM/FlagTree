# Layout candidate: sp=4, warps=2; source=h2560_hc4_b1024_w8_s2_weuNone.py
# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260910_phase2/src/FlagGems-vllm/src/flaggems_vllm/ops/mhc/hc_head_fused_kernel.py
# Provenance: results/gluon/gluon_baseline/h2560_hc4_b1024_w8_s2_weuNone.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def _hc_head_fused_kernel_5d609257b0(residual_ptr, fn_ptr, hc_scale_ptr, hc_base_ptr, out_ptr, T, H: gl.constexpr, rms_eps, hc_eps, residual_stride_t, fn_stride_m, out_stride_t, HC: gl.constexpr, BLOCK_H: gl.constexpr):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    pid_t = __h81.freeze(gl.program_id(0))
    if __h81.compare(__h81.to_auto(pid_t), __h81.to_auto(T), 'ge'):
        return
    x_base = __h81.freeze(__h81.to_auto(pid_t) * __h81.to_auto(residual_stride_t))
    sqr_acc = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_H)], dtype=gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
    mix_acc0 = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_H)], dtype=gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
    mix_acc1 = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_H)], dtype=gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
    mix_acc2 = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_H)], dtype=gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
    mix_acc3 = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_H)], dtype=gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
    for h_start in range(0, __h81.to_auto(H), __h81.to_auto(BLOCK_H)):
        h_off = __h81.freeze(__h81.to_auto(h_start) + __h81.arange(0, __h81.to_auto(BLOCK_H), layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_off), __h81.to_auto(H), 'lt'), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        r0 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        r1 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        sqr_acc = __h81.freeze(__h81.to_auto(sqr_acc) + (__h81.to_auto(r0) * __h81.to_auto(r0) + __h81.to_auto(r1) * __h81.to_auto(r1)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        fn00 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 0 * __h81.to_auto(fn_stride_m) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        fn01 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 0 * __h81.to_auto(fn_stride_m) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        mix_acc0 = __h81.freeze(__h81.to_auto(mix_acc0) + (__h81.to_auto(r0) * __h81.to_auto(fn00) + __h81.to_auto(r1) * __h81.to_auto(fn01)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        fn10 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 1 * __h81.to_auto(fn_stride_m) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        fn11 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 1 * __h81.to_auto(fn_stride_m) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        mix_acc1 = __h81.freeze(__h81.to_auto(mix_acc1) + (__h81.to_auto(r0) * __h81.to_auto(fn10) + __h81.to_auto(r1) * __h81.to_auto(fn11)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        if __h81.compare(__h81.to_auto(HC), 2, 'gt'):
            r2 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            r3 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            sqr_acc = __h81.freeze(__h81.to_auto(sqr_acc) + (__h81.to_auto(r2) * __h81.to_auto(r2) + __h81.to_auto(r3) * __h81.to_auto(r3)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            mix_acc0 = __h81.freeze(__h81.to_auto(mix_acc0) + __h81.to_auto(r2) * __h81.load(__h81.to_auto(fn_ptr) + 0 * __h81.to_auto(fn_stride_m) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            mix_acc0 = __h81.freeze(__h81.to_auto(mix_acc0) + __h81.to_auto(r3) * __h81.load(__h81.to_auto(fn_ptr) + 0 * __h81.to_auto(fn_stride_m) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            mix_acc1 = __h81.freeze(__h81.to_auto(mix_acc1) + __h81.to_auto(r2) * __h81.load(__h81.to_auto(fn_ptr) + 1 * __h81.to_auto(fn_stride_m) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            mix_acc1 = __h81.freeze(__h81.to_auto(mix_acc1) + __h81.to_auto(r3) * __h81.load(__h81.to_auto(fn_ptr) + 1 * __h81.to_auto(fn_stride_m) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn20 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 2 * __h81.to_auto(fn_stride_m) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn21 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 2 * __h81.to_auto(fn_stride_m) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn22 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 2 * __h81.to_auto(fn_stride_m) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn23 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 2 * __h81.to_auto(fn_stride_m) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            mix_acc2 = __h81.freeze(__h81.to_auto(mix_acc2) + (__h81.to_auto(r0) * __h81.to_auto(fn20) + __h81.to_auto(r1) * __h81.to_auto(fn21) + __h81.to_auto(r2) * __h81.to_auto(fn22) + __h81.to_auto(r3) * __h81.to_auto(fn23)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn30 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 3 * __h81.to_auto(fn_stride_m) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn31 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 3 * __h81.to_auto(fn_stride_m) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn32 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 3 * __h81.to_auto(fn_stride_m) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            fn33 = __h81.freeze(__h81.load(__h81.to_auto(fn_ptr) + 3 * __h81.to_auto(fn_stride_m) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            mix_acc3 = __h81.freeze(__h81.to_auto(mix_acc3) + (__h81.to_auto(r0) * __h81.to_auto(fn30) + __h81.to_auto(r1) * __h81.to_auto(fn31) + __h81.to_auto(r2) * __h81.to_auto(fn32) + __h81.to_auto(r3) * __h81.to_auto(fn33)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
    K = __h81.freeze(__h81.to_auto(HC) * __h81.to_auto(H))
    sqr_total = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(sqr_acc), _layout=gl.BlockedLayout([1], [64], [2], [0])))), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    rsqrt_val = __h81.freeze(gl.rsqrt(__h81.to_auto(sqr_total) / __h81.to_auto(K) + __h81.to_auto(rms_eps)))
    hc_scale = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr)))
    mix0 = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(mix_acc0), _layout=gl.BlockedLayout([1], [64], [2], [0])))), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    mix1 = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(mix_acc1), _layout=gl.BlockedLayout([1], [64], [2], [0])))), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    hc_base0 = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + 0))
    hc_base1 = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + 1))
    pre_mix0 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.to_auto(mix0) * __h81.to_auto(rsqrt_val) * __h81.to_auto(hc_scale) + __h81.to_auto(hc_base0))) + __h81.to_auto(hc_eps))
    pre_mix1 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.to_auto(mix1) * __h81.to_auto(rsqrt_val) * __h81.to_auto(hc_scale) + __h81.to_auto(hc_base1))) + __h81.to_auto(hc_eps))
    if __h81.compare(__h81.to_auto(HC), 2, 'gt'):
        mix2 = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(mix_acc2), _layout=gl.BlockedLayout([1], [64], [2], [0])))), _layout=gl.BlockedLayout([1], [64], [2], [0]))
        mix3 = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(mix_acc3), _layout=gl.BlockedLayout([1], [64], [2], [0])))), _layout=gl.BlockedLayout([1], [64], [2], [0]))
        hc_base2 = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + 2))
        hc_base3 = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + 3))
        pre_mix2 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.to_auto(mix2) * __h81.to_auto(rsqrt_val) * __h81.to_auto(hc_scale) + __h81.to_auto(hc_base2))) + __h81.to_auto(hc_eps))
        pre_mix3 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.to_auto(mix3) * __h81.to_auto(rsqrt_val) * __h81.to_auto(hc_scale) + __h81.to_auto(hc_base3))) + __h81.to_auto(hc_eps))
    out_base = __h81.freeze(__h81.to_auto(pid_t) * __h81.to_auto(out_stride_t))
    for h_start in range(0, __h81.to_auto(H), __h81.to_auto(BLOCK_H)):
        h_off = __h81.freeze(__h81.to_auto(h_start) + __h81.arange(0, __h81.to_auto(BLOCK_H)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_off), __h81.to_auto(H), 'lt'), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        r0 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        r1 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        acc = __h81.freeze(__h81.to_auto(pre_mix0) * __h81.to_auto(r0) + __h81.to_auto(pre_mix1) * __h81.to_auto(r1), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        if __h81.compare(__h81.to_auto(HC), 2, 'gt'):
            r2 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            r3 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(x_base) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [2], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [2], [0]))
            acc = __h81.freeze(__h81.to_auto(acc) + (__h81.to_auto(pre_mix2) * __h81.to_auto(r2) + __h81.to_auto(pre_mix3) * __h81.to_auto(r3)), _layout=gl.BlockedLayout([4], [64], [2], [0]))
        __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(out_base) + __h81.to_auto(h_off), __h81.to_auto(acc).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([4], [64], [2], [0]))

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [2], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def sigmoid_aba2984c1a(x):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return 1 / (1 + gl.exp(-__h81.to_auto(x)))

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

KERNEL = _hc_head_fused_kernel_5d609257b0
