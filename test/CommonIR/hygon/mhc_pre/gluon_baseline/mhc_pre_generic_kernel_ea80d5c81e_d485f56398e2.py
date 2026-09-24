# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems-vllm/src/flaggems_vllm/ops/mhc/mhc_pre.py
# Provenance: results/gluon/16/mhc_pre_generic_kernel_ea80d5c81e_d485f56398e2.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def mhc_pre_generic_kernel_ea80d5c81e(gemm_out_ptr, hc_scale_ptr, hc_base_ptr, residual_ptr, post_mix_ptr, comb_mix_ptr, layer_input_ptr, num_tokens, num_tokens_bucket, res_stride_n, res_stride_i, res_stride_h, li_stride_n, li_stride_h, hidden_size, hc_hidden_size, rms_eps: gl.constexpr, hc_pre_eps: gl.constexpr, hc_sinkhorn_eps: gl.constexpr, hc_post_mult_value: gl.constexpr, sinkhorn_repeat: gl.constexpr, HC: gl.constexpr, BLOCK_H: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    pid_n = __h81.freeze(gl.program_id(0))
    if __h81.compare(__h81.to_auto(pid_n), __h81.to_auto(num_tokens), 'ge'):
        return
    res_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(res_stride_n))
    go_base = __h81.freeze(__h81.to_auto(pid_n) * (__h81.to_auto(HC) * 2 + __h81.to_auto(HC) * __h81.to_auto(HC)))
    comb_base = __h81.freeze(__h81.to_auto(pid_n) * (__h81.to_auto(HC) * __h81.to_auto(HC)))
    sq = __h81.freeze(0.0, _layout=gl.BlockedLayout([1], [64], [8], [0]))
    for k in gl.static_range(__h81.to_auto(HC)):
        head_base = __h81.freeze(__h81.to_auto(res_base) + __h81.to_auto(k) * __h81.to_auto(res_stride_i))
        for h_start in range(0, __h81.to_auto(hidden_size), __h81.to_auto(BLOCK_H)):
            h_offsets = __h81.freeze(__h81.to_auto(h_start) + __h81.arange(0, __h81.to_auto(BLOCK_H), layout=gl.BlockedLayout([1], [64], [8], [0])), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_offsets), __h81.to_auto(hidden_size), 'lt'), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            v = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(head_base) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [8], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            sq = __h81.freeze(__h81.to_auto(sq) + __h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(v) * __h81.to_auto(v), _layout=gl.BlockedLayout([1], [64], [8], [0])))), _layout=gl.BlockedLayout([1], [64], [8], [0]))
    rms_inv = __h81.freeze(gl.rsqrt(__h81.to_auto(sq) / __h81.to_auto(hc_hidden_size) + __h81.to_auto(rms_eps)))
    scale_0 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 0))
    scale_1 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 1))
    scale_2 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 2))
    for i in gl.static_range(__h81.to_auto(HC)):
        post_i = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(HC) + __h81.to_auto(i)) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_1) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(HC) + __h81.to_auto(i)))) * __h81.to_auto(hc_post_mult_value))
        __h81.store(__h81.to_auto(post_mix_ptr) + __h81.to_auto(pid_n) * __h81.to_auto(HC) + __h81.to_auto(i), __h81.to_auto(post_i))
    cb = __h81.freeze(2 * __h81.to_auto(HC))
    for i in gl.static_range(__h81.to_auto(HC)):
        for j in gl.static_range(__h81.to_auto(HC)):
            idx = __h81.freeze(__h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j))
            v = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + __h81.to_auto(idx)) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + __h81.to_auto(idx)), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(idx), __h81.to_auto(v))
    for i in gl.static_range(__h81.to_auto(HC)):
        row_max = __h81.freeze(__h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + 0))
        for j in gl.static_range(1, __h81.to_auto(HC)):
            row_max = __h81.freeze(gl.maximum(__h81.to_auto(row_max), __h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j))))
        row_sum = __h81.freeze(0.0)
        for j in gl.static_range(__h81.to_auto(HC)):
            e = __h81.freeze(gl.exp(__h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)) - __h81.to_auto(row_max)))
            __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j), __h81.to_auto(e))
            row_sum = __h81.freeze(__h81.to_auto(row_sum) + __h81.to_auto(e))
        inv_row_sum = __h81.freeze(1.0 / __h81.to_auto(row_sum))
        for j in gl.static_range(__h81.to_auto(HC)):
            v = __h81.freeze(__h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j), __h81.to_auto(v) * __h81.to_auto(inv_row_sum) + __h81.to_auto(hc_sinkhorn_eps))
    for j in gl.static_range(__h81.to_auto(HC)):
        col_sum = __h81.freeze(0.0)
        for i in gl.static_range(__h81.to_auto(HC)):
            col_sum = __h81.freeze(__h81.to_auto(col_sum) + __h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)))
        inv_col_sum = __h81.freeze(1.0 / (__h81.to_auto(col_sum) + __h81.to_auto(hc_sinkhorn_eps)))
        for i in gl.static_range(__h81.to_auto(HC)):
            v = __h81.freeze(__h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j), __h81.to_auto(v) * __h81.to_auto(inv_col_sum))
    for _ in gl.static_range(__h81.to_auto(sinkhorn_repeat) - 1):
        for i in gl.static_range(__h81.to_auto(HC)):
            row_sum = __h81.freeze(0.0)
            for j in gl.static_range(__h81.to_auto(HC)):
                row_sum = __h81.freeze(__h81.to_auto(row_sum) + __h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)))
            inv_row_sum = __h81.freeze(1.0 / (__h81.to_auto(row_sum) + __h81.to_auto(hc_sinkhorn_eps)))
            for j in gl.static_range(__h81.to_auto(HC)):
                v = __h81.freeze(__h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)), _layout=gl.BlockedLayout([1], [64], [8], [0]))
                __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j), __h81.to_auto(v) * __h81.to_auto(inv_row_sum))
        for j in gl.static_range(__h81.to_auto(HC)):
            col_sum = __h81.freeze(0.0)
            for i in gl.static_range(__h81.to_auto(HC)):
                col_sum = __h81.freeze(__h81.to_auto(col_sum) + __h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)))
            inv_col_sum = __h81.freeze(1.0 / (__h81.to_auto(col_sum) + __h81.to_auto(hc_sinkhorn_eps)))
            for i in gl.static_range(__h81.to_auto(HC)):
                v = __h81.freeze(__h81.load(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j)), _layout=gl.BlockedLayout([1], [64], [8], [0]))
                __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(i) * __h81.to_auto(HC) + __h81.to_auto(j), __h81.to_auto(v) * __h81.to_auto(inv_col_sum))
    for h_start in range(0, __h81.to_auto(hidden_size), __h81.to_auto(BLOCK_H)):
        h_offsets = __h81.freeze(__h81.to_auto(h_start) + __h81.arange(0, __h81.to_auto(BLOCK_H)), _layout=gl.BlockedLayout([1], [64], [8], [0]))
        h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_offsets), __h81.to_auto(hidden_size), 'lt'), _layout=gl.BlockedLayout([1], [64], [8], [0]))
        acc = __h81.freeze(__h81.zeros([__h81.to_auto(BLOCK_H)], dtype=gl.float32), _layout=gl.BlockedLayout([1], [64], [8], [0]))
        for k in gl.static_range(__h81.to_auto(HC)):
            pre_k = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(k)) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_0) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(k)))) + __h81.to_auto(hc_pre_eps))
            rk = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(res_base) + __h81.to_auto(k) * __h81.to_auto(res_stride_i) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [8], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [8], [0]))
            acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(pre_k) * __h81.to_auto(rk), _layout=gl.BlockedLayout([1], [64], [8], [0]))
        __h81.store(__h81.to_auto(layer_input_ptr) + __h81.to_auto(pid_n) * __h81.to_auto(li_stride_n) + __h81.to_auto(h_offsets) * __h81.to_auto(li_stride_h), __h81.to_auto(acc).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([1], [64], [8], [0]))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [8], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def sigmoid_aba2984c1a(x):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    return 1 / (1 + gl.exp(-__h81.to_auto(x)))

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

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

KERNEL = mhc_pre_generic_kernel_ea80d5c81e
