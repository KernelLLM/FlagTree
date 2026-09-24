# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems-vllm/src/flaggems_vllm/ops/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py
# Provenance: results/gluon/31/fused_qnorm_rope_kv_insert_kernel_ba34c6d87e_48aa04dc7b81.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module
_ext_fused_qnorm_rope_kv_insert_kernel_ba34c6d87e_tl = _import_module('triton.language')
_ext_max_bc5e2fe0cf_core = _import_module('triton.language.core')

@g.jit
def quant2d_w1(q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, eps, cache_block_size: gl.constexpr, num_tokens: gl.constexpr, num_heads: gl.constexpr, kv_block_stride, num_tokens_insert: gl.constexpr):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    HEAD_DIM: gl.constexpr = 512
    NOPE_DIM: gl.constexpr = 448
    ROPE_DIM: gl.constexpr = 64
    HALF_ROPE_DIM: gl.constexpr = 32
    QUANT_BLOCK: gl.constexpr = 64
    NUM_QUANT_BLOCKS: gl.constexpr = __h81.to_auto(NOPE_DIM) // __h81.to_auto(QUANT_BLOCK)
    SCALE_BYTES_PER_TOKEN: gl.constexpr = __h81.to_auto(NUM_QUANT_BLOCKS) + 1
    TOKEN_DATA_BYTES: gl.constexpr = __h81.to_auto(NOPE_DIM) + 2 * __h81.to_auto(ROPE_DIM)
    FP8_MAX: gl.constexpr = 448.0
    pid = __h81.freeze(gl.program_id(0).to(gl.int64))
    blocks_per_token = __h81.freeze(__h81.to_auto(num_heads) + 1)
    token_idx = __h81.freeze(__h81.to_auto(pid) // __h81.to_auto(blocks_per_token))
    if __h81.compare(__h81.to_auto(token_idx), __h81.to_auto(num_tokens), 'ge'):
        return
    slot_idx = __h81.freeze(__h81.to_auto(pid) % __h81.to_auto(blocks_per_token))
    is_kv = __h81.freeze(__h81.compare(__h81.to_auto(slot_idx), __h81.to_auto(num_heads), 'eq'))
    if __h81.to_auto(is_kv) and __h81.compare(__h81.to_auto(token_idx), __h81.to_auto(num_tokens_insert), 'ge'):
        return
    q_base = __h81.freeze(__h81.to_auto(q) + (__h81.to_auto(token_idx) * __h81.to_auto(num_heads) + __h81.to_auto(slot_idx)) * __h81.to_auto(HEAD_DIM))
    kv_base = __h81.freeze(__h81.to_auto(kv) + __h81.to_auto(token_idx) * __h81.to_auto(HEAD_DIM))
    offset = __h81.freeze(__h81.arange(0, __h81.to_auto(HEAD_DIM), layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    mask_nope = __h81.freeze(__h81.compare(__h81.to_auto(offset), __h81.to_auto(NOPE_DIM), 'lt'), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    offset_rope = __h81.freeze(__h81.arange(0, __h81.to_auto(ROPE_DIM), layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    offset_half_rope = __h81.freeze(__h81.arange(0, __h81.to_auto(HALF_ROPE_DIM), layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    offset_quant = __h81.freeze(__h81.arange(0, __h81.to_auto(QUANT_BLOCK)))
    if not __h81.to_auto(is_kv):
        q_blk = __h81.freeze(__h81.load(__h81.to_auto(q_base) + __h81.to_auto(offset), _layout=gl.BlockedLayout([8], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [1], [0]))
        q_blk_rope = __h81.freeze(__h81.load(__h81.to_auto(q_base) + __h81.to_auto(NOPE_DIM) + __h81.to_auto(offset_rope), _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        variance = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(q_blk) * __h81.to_auto(q_blk), _layout=gl.BlockedLayout([8], [64], [1], [0])))) / __h81.to_auto(HEAD_DIM), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        rsqrt = __h81.freeze(gl.rsqrt(__h81.to_auto(variance) + __h81.to_auto(eps)))
        q_blk = __h81.freeze(__h81.to_auto(q_blk) * __h81.to_auto(rsqrt), _layout=gl.BlockedLayout([8], [64], [1], [0]))
        __h81.store(__h81.to_auto(q_base) + __h81.to_auto(offset), __h81.to_auto(q_blk).to(gl.bfloat16), mask=__h81.to_auto(mask_nope), _layout=gl.BlockedLayout([8], [64], [1], [0]))
        qkv_blk_rope = __h81.freeze(__h81.to_auto(q_blk_rope) * __h81.to_auto(rsqrt), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    else:
        qkv_blk_rope = __h81.freeze(__h81.load(__h81.to_auto(kv_base) + __h81.to_auto(NOPE_DIM) + __h81.to_auto(offset_rope), _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    position_id = __h81.freeze(__h81.load(__h81.to_auto(position_ids) + __h81.to_auto(token_idx)))
    cs_base = __h81.freeze(__h81.to_auto(cos_sin_cache) + __h81.to_auto(position_id) * __h81.to_auto(ROPE_DIM))
    cos_blk = __h81.freeze(__h81.load(__h81.to_auto(cs_base) + __h81.to_auto(offset_half_rope), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    sin_blk = __h81.freeze(__h81.load(__h81.to_auto(cs_base) + __h81.to_auto(offset_half_rope) + __h81.to_auto(HALF_ROPE_DIM), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    qkv_blk_rope = __h81.freeze(gl.reshape(__h81.to_auto(qkv_blk_rope), __h81.to_auto(HALF_ROPE_DIM), 2), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    (even_blk, odd_blk) = __h81.freeze(__h81.split(__h81.to_auto(qkv_blk_rope)))
    new_even_blk = __h81.freeze(__h81.to_auto(even_blk) * __h81.to_auto(cos_blk) - __h81.to_auto(odd_blk) * __h81.to_auto(sin_blk), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    new_odd_blk = __h81.freeze(__h81.to_auto(even_blk) * __h81.to_auto(sin_blk) + __h81.to_auto(odd_blk) * __h81.to_auto(cos_blk), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    qkv_blk_rope = __h81.freeze(gl.reshape(__h81.join(__h81.to_auto(new_even_blk), __h81.to_auto(new_odd_blk)), __h81.to_auto(ROPE_DIM)).to(gl.bfloat16), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    if not __h81.to_auto(is_kv):
        __h81.store(__h81.to_auto(q_base) + __h81.to_auto(NOPE_DIM) + __h81.to_auto(offset_rope), __h81.to_auto(qkv_blk_rope), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        return
    slot_id = __h81.freeze(__h81.load(__h81.to_auto(slot_mapping) + __h81.to_auto(token_idx)))
    if __h81.compare(__h81.to_auto(slot_id), 0, 'lt'):
        return
    block_idx = __h81.freeze(__h81.to_auto(slot_id) // __h81.to_auto(cache_block_size))
    pos_in_block = __h81.freeze(__h81.to_auto(slot_id) % __h81.to_auto(cache_block_size))
    block_base = __h81.freeze(__h81.to_auto(k_cache) + __h81.to_auto(block_idx) * __h81.to_auto(kv_block_stride))
    token_fp8_ptr = __h81.freeze(__h81.to_auto(block_base) + __h81.to_auto(pos_in_block) * __h81.to_auto(TOKEN_DATA_BYTES))
    token_bf16_ptr = __h81.freeze(__h81.to_auto(token_fp8_ptr) + __h81.to_auto(NOPE_DIM))
    token_bf16_ptr = __h81.freeze(__h81.to_auto(token_bf16_ptr).to(gl.pointer_type(gl.bfloat16)))
    token_scale_ptr = __h81.freeze(__h81.to_auto(block_base) + __h81.to_auto(cache_block_size) * __h81.to_auto(TOKEN_DATA_BYTES) + __h81.to_auto(pos_in_block) * __h81.to_auto(SCALE_BYTES_PER_TOKEN))
    __h81.store(__h81.to_auto(token_bf16_ptr) + __h81.to_auto(offset_rope), __h81.to_auto(qkv_blk_rope), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    quant_layout_b0: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])
    quant_offsets_b0 = __h81.freeze(__h81.arange(0, 512, layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    quant_absolute_b0 = __h81.freeze(__h81.to_auto(quant_offsets_b0) + 0, _layout=gl.BlockedLayout([8], [64], [1], [0]))
    quant_mask_b0 = __h81.freeze(__h81.compare(__h81.to_auto(quant_absolute_b0), __h81.to_auto(NOPE_DIM), 'lt'), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    kv_quant_b0 = __h81.freeze(__h81.load(__h81.to_auto(kv_base) + __h81.to_auto(quant_absolute_b0), mask=__h81.to_auto(quant_mask_b0), other=0.0, _layout=gl.BlockedLayout([8], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    kv_quant_2d_b0 = __h81.freeze(gl.reshape(gl.abs(__h81.to_auto(kv_quant_b0)), 8, 64), _layout=quant_layout_b0)
    block_max_b0 = __h81.freeze(gl.maximum(__h81.to_auto(gl.max(__h81.concrete(__h81.to_auto(kv_quant_2d_b0), _layout=quant_layout_b0), axis=1)), 0.0001), _layout=gl.SliceLayout(1, quant_layout_b0))
    raw_scale_b0 = __h81.freeze(__h81.to_auto(block_max_b0) / __h81.to_auto(FP8_MAX), _layout=gl.SliceLayout(1, quant_layout_b0))
    exponent_b0 = __h81.freeze(gl.ceil(gl.log2(__h81.to_auto(raw_scale_b0))), _layout=gl.SliceLayout(1, quant_layout_b0))
    scale_b0 = __h81.freeze(gl.exp2(__h81.to_auto(exponent_b0)), _layout=gl.SliceLayout(1, quant_layout_b0))
    scale_expanded_b0 = __h81.freeze(gl.reshape(_ext_fused_qnorm_rope_kv_insert_kernel_ba34c6d87e_tl.broadcast_to(gl.reshape(__h81.to_auto(scale_b0), 8, 1), (8, 64)), 512), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_scaled_b0 = __h81.freeze(__h81.to_auto(kv_quant_b0) / __h81.to_auto(scale_expanded_b0), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_clamped_b0 = __h81.freeze(_ext_fused_qnorm_rope_kv_insert_kernel_ba34c6d87e_tl.clamp(__h81.to_auto(x_scaled_b0), -__h81.to_auto(FP8_MAX), __h81.to_auto(FP8_MAX)), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    x_uint8_b0 = __h81.freeze(__h81.to_auto(x_clamped_b0).to(gl.float8e4nv).to(gl.uint8, bitcast=True), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    __h81.store(__h81.to_auto(token_fp8_ptr) + __h81.to_auto(quant_absolute_b0), __h81.to_auto(x_uint8_b0), mask=__h81.to_auto(quant_mask_b0), _layout=gl.BlockedLayout([8], [64], [1], [0]))
    scale_offsets_b0 = __h81.freeze(__h81.arange(0, 8, layout=gl.BlockedLayout([1], [64], [1], [0])) + 0, _layout=gl.BlockedLayout([1], [64], [1], [0]))
    scale_mask_b0 = __h81.freeze(__h81.compare(__h81.to_auto(scale_offsets_b0), __h81.to_auto(NUM_QUANT_BLOCKS), 'lt'), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    encoded_scale_b0 = __h81.freeze(gl.maximum(gl.minimum(__h81.to_auto(exponent_b0) + 127.0, 255.0), 0.0))
    encoded_scale_b0 = __h81.freeze(__h81.concrete(__h81.to_auto(encoded_scale_b0), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    __h81.store(__h81.to_auto(token_scale_ptr) + __h81.to_auto(scale_offsets_b0), __h81.to_auto(encoded_scale_b0).to(gl.uint8), mask=__h81.to_auto(scale_mask_b0), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    __h81.store(__h81.to_auto(token_scale_ptr) + __h81.to_auto(NUM_QUANT_BLOCKS), __h81.zeros((), dtype=gl.uint8))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [1], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def max_bc5e2fe0cf(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
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
        return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [1], [0])), __h81.to_auto(axis), _elementwise_max_dfab1c952e, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
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
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    return __h81.to_auto(a) + __h81.to_auto(b)

@g.jit
def _argmax_combine_tie_break_left_9977b0154f(value1, index1, value2, index2):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    return _argmax_combine_0ed820d30e(value1, index1, value2, index2, __h81.pin(True))

@g.jit
def _argmax_combine_tie_break_fast_43f1013d43(value1, index1, value2, index2):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    return _argmax_combine_0ed820d30e(value1, index1, value2, index2, __h81.pin(False))

@g.jit
def _elementwise_max_dfab1c952e(a, b):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    return gl.maximum(__h81.to_auto(a), __h81.to_auto(b))

@g.jit
def _argmax_combine_0ed820d30e(value1, index1, value2, index2, tie_break_left):
    """HCU helper SHA256: 72dc7b7a01144b36aca6e2d4e77cada4ac198f91f9a3057d1453fb5bbc2949fd"""
    if __h81.to_auto(tie_break_left):
        tie = __h81.freeze(__h81.compare(__h81.to_auto(value1), __h81.to_auto(value2), 'eq') and __h81.compare(__h81.to_auto(index1), __h81.to_auto(index2), 'lt'))
    else:
        tie = __h81.freeze(False)
    gt = __h81.freeze(__h81.compare(__h81.to_auto(value1), __h81.to_auto(value2), 'gt') or __h81.to_auto(tie))
    v_ret = __h81.freeze(gl.where(__h81.to_auto(gt), __h81.to_auto(value1), __h81.to_auto(value2)))
    i_ret = __h81.freeze(gl.where(__h81.to_auto(gt), __h81.to_auto(index1), __h81.to_auto(index2)))
    return (__h81.to_auto(v_ret), __h81.to_auto(i_ret))

KERNEL = quant2d_w1
