# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems-vllm/src/flaggems_vllm/ops/fused_inv_rope_fp8_quant.py
# Provenance: results/gluon/59/_fused_inv_rope_fp8_quant_per_head_625a5a92b8_25b9a6019dc5.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module
_ext__fused_inv_rope_fp8_quant_per_head_625a5a92b8_tl = _import_module('triton.language')
_ext_max_bc5e2fe0cf_core = _import_module('triton.language.core')

@g.jit
def _fused_inv_rope_fp8_quant_per_head_625a5a92b8(o_ptr, positions_ptr, cos_sin_cache_ptr, fp8_ptr, scale_ptr, num_tokens, heads_per_group: gl.constexpr, o_stride_token, o_stride_head, cache_stride_pos, fp8_stride_group, fp8_stride_token, scale_stride_group, scale_stride_k, fp8_max: gl.constexpr, eps: gl.constexpr, QUANT_GROUP_SIZE: gl.constexpr, CHUNKS_PER_HEAD: gl.constexpr, ROPE_START: gl.constexpr, HALF_ROPE: gl.constexpr, TMA_ALIGNED_SCALES: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    pid_token = __h81.freeze(gl.program_id(0).to(gl.int64))
    pid_gh = __h81.freeze(gl.program_id(1).to(gl.int64))
    g = __h81.freeze(__h81.to_auto(pid_gh) // __h81.to_auto(heads_per_group))
    head_in_group = __h81.freeze(__h81.to_auto(pid_gh) % __h81.to_auto(heads_per_group))
    global_head = __h81.freeze(__h81.to_auto(pid_gh))
    qb_start = __h81.freeze(__h81.to_auto(head_in_group) * __h81.to_auto(CHUNKS_PER_HEAD))
    if __h81.compare(__h81.to_auto(pid_token), __h81.to_auto(num_tokens), 'ge'):
        if __h81.to_auto(TMA_ALIGNED_SCALES):
            scale_addr = __h81.freeze(__h81.to_auto(scale_ptr) + __h81.to_auto(g) * __h81.to_auto(scale_stride_group) + __h81.to_auto(pid_token) + __h81.to_auto(head_in_group) * __h81.to_auto(scale_stride_k))
            __h81.store(__h81.to_auto(scale_addr), __h81.zeros((), dtype=gl.int32))
        else:
            block_offsets = __h81.freeze(__h81.arange(0, __h81.to_auto(CHUNKS_PER_HEAD), layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            qb_indices = __h81.freeze(__h81.to_auto(qb_start) + __h81.to_auto(block_offsets), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            scale_addrs = __h81.freeze(__h81.to_auto(scale_ptr) + __h81.to_auto(g) * __h81.to_auto(scale_stride_group) + __h81.to_auto(pid_token) + __h81.to_auto(qb_indices) * __h81.to_auto(scale_stride_k), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            __h81.store(__h81.to_auto(scale_addrs), __h81.zeros((__h81.to_auto(CHUNKS_PER_HEAD),), dtype=gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        return
    input_base = __h81.freeze(__h81.to_auto(o_ptr) + __h81.to_auto(pid_token) * __h81.to_auto(o_stride_token) + __h81.to_auto(global_head) * __h81.to_auto(o_stride_head))
    HEAD_DIM: gl.constexpr = __h81.to_auto(CHUNKS_PER_HEAD) * __h81.to_auto(QUANT_GROUP_SIZE)
    offsets = __h81.freeze(__h81.arange(0, __h81.to_auto(HEAD_DIM), layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x = __h81.freeze(__h81.load(__h81.to_auto(input_base) + __h81.to_auto(offsets), _layout=gl.BlockedLayout([8], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    rope_abs_start: gl.constexpr = (__h81.to_auto(CHUNKS_PER_HEAD) - 1) * __h81.to_auto(QUANT_GROUP_SIZE) + __h81.to_auto(ROPE_START)
    pos = __h81.freeze(__h81.load(__h81.to_auto(positions_ptr) + __h81.to_auto(pid_token)))
    cache_base = __h81.freeze(__h81.to_auto(cos_sin_cache_ptr) + __h81.to_auto(pos) * __h81.to_auto(cache_stride_pos))
    is_rope = __h81.freeze(__h81.compare(__h81.to_auto(offsets), __h81.to_auto(rope_abs_start), 'ge'), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    rope_local = __h81.freeze(__h81.to_auto(offsets) - __h81.to_auto(rope_abs_start), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_partner = __h81.freeze(__h81.load(__h81.to_auto(input_base) + (__h81.to_auto(offsets) ^ 1), mask=__h81.to_auto(is_rope), other=0.0, _layout=gl.BlockedLayout([8], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    cs_idx = __h81.freeze(gl.maximum(__h81.to_auto(rope_local) >> 1, 0), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    cos_v = __h81.freeze(__h81.load(__h81.to_auto(cache_base) + __h81.to_auto(cs_idx), mask=__h81.to_auto(is_rope), other=1.0, _layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    sin_v = __h81.freeze(__h81.load(__h81.to_auto(cache_base) + __h81.to_auto(HALF_ROPE) + __h81.to_auto(cs_idx), mask=__h81.to_auto(is_rope), other=0.0, _layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_add = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(cos_v) + __h81.to_auto(x_partner) * __h81.to_auto(sin_v), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_sub = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(cos_v) - __h81.to_auto(x_partner) * __h81.to_auto(sin_v), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    is_even = __h81.freeze(__h81.compare(__h81.to_auto(rope_local) & 1, 0, 'eq'), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    rotated = __h81.freeze(gl.where(__h81.to_auto(is_even), __h81.to_auto(x_add), __h81.to_auto(x_sub)), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x = __h81.freeze(gl.where(__h81.to_auto(is_rope), __h81.to_auto(rotated), __h81.to_auto(x)), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_2d = __h81.freeze(gl.reshape(gl.abs(__h81.to_auto(x)), (__h81.to_auto(CHUNKS_PER_HEAD), __h81.to_auto(QUANT_GROUP_SIZE))), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    block_absmax = __h81.freeze(gl.maximum(__h81.to_auto(gl.max(__h81.concrete(__h81.to_auto(x_2d), _layout=gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])), axis=1)), __h81.to_auto(eps)), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])))
    scales = __h81.freeze(__h81.to_auto(block_absmax) * (1.0 / __h81.to_auto(fp8_max)), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])))
    if __h81.to_auto(TMA_ALIGNED_SCALES):
        scales = __h81.freeze(gl.exp2(gl.ceil(gl.log2(gl.maximum(gl.abs(__h81.to_auto(scales)), 1e-10)))), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])))
    scales_exp = __h81.freeze(gl.reshape(_ext__fused_inv_rope_fp8_quant_per_head_625a5a92b8_tl.broadcast_to(gl.reshape(__h81.to_auto(scales), (__h81.to_auto(CHUNKS_PER_HEAD), 1)), (__h81.to_auto(CHUNKS_PER_HEAD), __h81.to_auto(QUANT_GROUP_SIZE))), (__h81.to_auto(HEAD_DIM),)), _layout=gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0]))
    x_quant = __h81.freeze(_ext__fused_inv_rope_fp8_quant_per_head_625a5a92b8_tl.clamp(__h81.to_auto(x) / __h81.to_auto(scales_exp), -__h81.to_auto(fp8_max), __h81.to_auto(fp8_max)).to(gl.float8e4nv), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    fp8_base = __h81.freeze(__h81.to_auto(fp8_ptr) + __h81.to_auto(g) * __h81.to_auto(fp8_stride_group) + __h81.to_auto(pid_token) * __h81.to_auto(fp8_stride_token) + __h81.to_auto(qb_start) * __h81.to_auto(QUANT_GROUP_SIZE))
    __h81.store(__h81.to_auto(fp8_base) + __h81.to_auto(offsets), __h81.to_auto(x_quant), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    block_offsets = __h81.freeze(__h81.arange(0, __h81.to_auto(CHUNKS_PER_HEAD), layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    qb_indices = __h81.freeze(__h81.to_auto(qb_start) + __h81.to_auto(block_offsets), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    if __h81.to_auto(TMA_ALIGNED_SCALES):
        scale_bits = __h81.freeze(__h81.to_auto(scales).to(gl.int32, bitcast=True))
        ue8m0_bytes = __h81.freeze(__h81.to_auto(scale_bits) >> 23 & 255)
        packed_val = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(ue8m0_bytes) << __h81.to_auto(block_offsets) * 8))))
        scale_addr = __h81.freeze(__h81.to_auto(scale_ptr) + __h81.to_auto(g) * __h81.to_auto(scale_stride_group) + __h81.to_auto(pid_token) + __h81.to_auto(head_in_group) * __h81.to_auto(scale_stride_k))
        __h81.store(__h81.to_auto(scale_addr), __h81.to_auto(packed_val))
    else:
        scale_addrs = __h81.freeze(__h81.to_auto(scale_ptr) + __h81.to_auto(g) * __h81.to_auto(scale_stride_group) + __h81.to_auto(pid_token) + __h81.to_auto(qb_indices) * __h81.to_auto(scale_stride_k), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        __h81.store(__h81.to_auto(scale_addrs), __h81.to_auto(scales), _layout=gl.BlockedLayout([1], [64], [1], [0]))

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

@g.jit
def max_bc5e2fe0cf(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    input = __h81.freeze(_ext_max_bc5e2fe0cf_core._promote_bfloat16_to_float32(__h81.to_auto(input)), _layout=__h81.layout_of(input))
    if __h81.to_auto(return_indices):
        if __h81.to_auto(return_indices_tie_break_left):
            return _ext_max_bc5e2fe0cf_core._reduce_with_indices(__h81.to_auto(input), __h81.to_auto(axis), _argmax_combine_tie_break_left_9977b0154f, keep_dims=__h81.to_auto(keep_dims))
        else:
            return _ext_max_bc5e2fe0cf_core._reduce_with_indices(__h81.to_auto(input), __h81.to_auto(axis), _argmax_combine_tie_break_fast_43f1013d43, keep_dims=__h81.to_auto(keep_dims))
    else:
        if __h81.compare(gl.constexpr(__h81.to_auto(input).dtype.primitive_bitwidth), gl.constexpr(32), 'lt'):
            if gl.constexpr(__h81.to_auto(input).dtype.is_floating()):
                input = __h81.freeze(__h81.to_auto(input).to(gl.float32), _layout=__h81.layout_of(input))
            else:
                assert __h81.to_auto(input).dtype.is_int(), 'Expecting input to be integer type'
                input = __h81.freeze(__h81.to_auto(input).to(gl.int32), _layout=__h81.layout_of(input))
        return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])), __h81.to_auto(axis), _elementwise_max_dfab1c952e, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input)), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def _argmax_combine_tie_break_left_9977b0154f(value1, index1, value2, index2):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    return _argmax_combine_0ed820d30e(value1, index1, value2, index2, __h81.pin(True))

@g.jit
def _argmax_combine_tie_break_fast_43f1013d43(value1, index1, value2, index2):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    return _argmax_combine_0ed820d30e(value1, index1, value2, index2, __h81.pin(False))

@g.jit
def _elementwise_max_dfab1c952e(a, b):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    return gl.maximum(__h81.to_auto(a), __h81.to_auto(b))

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

@g.jit
def _argmax_combine_0ed820d30e(value1, index1, value2, index2, tie_break_left):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    if __h81.to_auto(tie_break_left):
        tie = __h81.freeze(__h81.compare(__h81.to_auto(value1), __h81.to_auto(value2), 'eq') and __h81.compare(__h81.to_auto(index1), __h81.to_auto(index2), 'lt'))
    else:
        tie = __h81.freeze(False)
    gt = __h81.freeze(__h81.compare(__h81.to_auto(value1), __h81.to_auto(value2), 'gt') or __h81.to_auto(tie))
    v_ret = __h81.freeze(gl.where(__h81.to_auto(gt), __h81.to_auto(value1), __h81.to_auto(value2)))
    i_ret = __h81.freeze(gl.where(__h81.to_auto(gt), __h81.to_auto(index1), __h81.to_auto(index2)))
    return (__h81.to_auto(v_ret), __h81.to_auto(i_ret))

KERNEL = _fused_inv_rope_fp8_quant_per_head_625a5a92b8
