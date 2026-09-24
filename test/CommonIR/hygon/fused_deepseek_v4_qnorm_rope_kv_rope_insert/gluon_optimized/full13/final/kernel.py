# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems/src/flag_gems/fused/fused_deepseek_v4_qnorm_rope_kv_rope_insert.py
# Provenance: results/gluon/58/_fused_qkv_kernel_c82b519fc2_eb298838faa5.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def w1_g65536(q_ptr, kv_ptr, k_cache_ptr, slot_mapping_ptr, position_ids_ptr, cos_sin_cache_ptr, stride_q_tok, stride_q_head, stride_kv_tok, stride_cache_block, stride_cache_token, stride_cos_sin_pos, eps, num_q_items, total_items, NUM_HEADS: gl.constexpr, CACHE_BLOCK_SIZE: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    pid = __h81.freeze(gl.program_id(0))
    item_id = __h81.freeze(__h81.to_auto(pid))
    while __h81.compare(__h81.to_auto(item_id), __h81.to_auto(total_items), 'lt'):
        if __h81.compare(__h81.to_auto(item_id), __h81.to_auto(num_q_items), 'lt'):
            tok_idx = __h81.freeze(__h81.to_auto(item_id) // __h81.to_auto(NUM_HEADS))
            head_idx = __h81.freeze(__h81.to_auto(item_id) % __h81.to_auto(NUM_HEADS))
            base = __h81.freeze(__h81.to_auto(tok_idx) * __h81.to_auto(stride_q_tok) + __h81.to_auto(head_idx) * __h81.to_auto(stride_q_head))
            offs = __h81.freeze(__h81.arange(0, 512, layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
            x = __h81.freeze(__h81.load(__h81.to_auto(q_ptr) + __h81.to_auto(base) + __h81.to_auto(offs), _layout=gl.BlockedLayout([8], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([8], [64], [1], [0]))
            sq_sum = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(x) * __h81.to_auto(x), _layout=gl.BlockedLayout([8], [64], [1], [0])), axis=0)), _layout=gl.BlockedLayout([8], [64], [1], [0]))
            rsqrt_val = __h81.freeze(gl.rsqrt(__h81.to_auto(sq_sum) / 512.0 + __h81.to_auto(eps)))
            x_normed = __h81.freeze(__h81.to_auto(x) * __h81.to_auto(rsqrt_val), _layout=gl.BlockedLayout([8], [64], [1], [0]))
            pos = __h81.freeze(__h81.load(__h81.to_auto(position_ids_ptr) + __h81.to_auto(tok_idx)))
            half_offs = __h81.freeze(__h81.arange(0, 32, layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            cos = __h81.freeze(__h81.load(__h81.to_auto(cos_sin_cache_ptr) + __h81.to_auto(pos) * __h81.to_auto(stride_cos_sin_pos) + __h81.to_auto(half_offs), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            sin = __h81.freeze(__h81.load(__h81.to_auto(cos_sin_cache_ptr) + __h81.to_auto(pos) * __h81.to_auto(stride_cos_sin_pos) + 32 + __h81.to_auto(half_offs), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            rope_even = __h81.freeze(448 + __h81.to_auto(half_offs) * 2)
            rope_odd = __h81.freeze(__h81.to_auto(rope_even) + 1)
            x_re = __h81.freeze(__h81.load(__h81.to_auto(q_ptr) + __h81.to_auto(base) + __h81.to_auto(rope_even), _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32) * __h81.to_auto(rsqrt_val), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            x_ro = __h81.freeze(__h81.load(__h81.to_auto(q_ptr) + __h81.to_auto(base) + __h81.to_auto(rope_odd), _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32) * __h81.to_auto(rsqrt_val), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            q_out_e = __h81.freeze(__h81.to_auto(x_re) * __h81.to_auto(cos) - __h81.to_auto(x_ro) * __h81.to_auto(sin), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            q_out_o = __h81.freeze(__h81.to_auto(x_re) * __h81.to_auto(sin) + __h81.to_auto(x_ro) * __h81.to_auto(cos), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            nope_mask = __h81.freeze(__h81.compare(__h81.to_auto(offs), 448, 'lt'), _layout=gl.BlockedLayout([8], [64], [1], [0]))
            __h81.store(__h81.to_auto(q_ptr) + __h81.to_auto(base) + __h81.to_auto(offs), __h81.to_auto(x_normed).to(gl.bfloat16), mask=__h81.to_auto(nope_mask), _layout=gl.BlockedLayout([8], [64], [1], [0]))
            __h81.store(__h81.to_auto(q_ptr) + __h81.to_auto(base) + __h81.to_auto(rope_even), __h81.to_auto(q_out_e).to(gl.bfloat16), _layout=gl.BlockedLayout([1], [64], [1], [0]))
            __h81.store(__h81.to_auto(q_ptr) + __h81.to_auto(base) + __h81.to_auto(rope_odd), __h81.to_auto(q_out_o).to(gl.bfloat16), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        else:
            kv_idx = __h81.freeze(__h81.to_auto(item_id) - __h81.to_auto(num_q_items))
            slot_id = __h81.freeze(__h81.load(__h81.to_auto(slot_mapping_ptr) + __h81.to_auto(kv_idx)))
            if __h81.compare(__h81.to_auto(slot_id), 0, 'ge'):
                kv_base = __h81.freeze(__h81.to_auto(kv_idx) * __h81.to_auto(stride_kv_tok))
                offs = __h81.freeze(__h81.arange(0, 512, layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
                kv_data = __h81.freeze(__h81.load(__h81.to_auto(kv_ptr) + __h81.to_auto(kv_base) + __h81.to_auto(offs), _layout=gl.BlockedLayout([8], [64], [1], [0])), _layout=gl.BlockedLayout([8], [64], [1], [0]))
                pos = __h81.freeze(__h81.load(__h81.to_auto(position_ids_ptr) + __h81.to_auto(kv_idx)))
                half_offs = __h81.freeze(__h81.arange(0, 32, layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                cos = __h81.freeze(__h81.load(__h81.to_auto(cos_sin_cache_ptr) + __h81.to_auto(pos) * __h81.to_auto(stride_cos_sin_pos) + __h81.to_auto(half_offs), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                sin = __h81.freeze(__h81.load(__h81.to_auto(cos_sin_cache_ptr) + __h81.to_auto(pos) * __h81.to_auto(stride_cos_sin_pos) + 32 + __h81.to_auto(half_offs), _layout=gl.BlockedLayout([1], [64], [1], [0])), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                rope_even = __h81.freeze(448 + __h81.to_auto(half_offs) * 2)
                rope_odd = __h81.freeze(__h81.to_auto(rope_even) + 1)
                x_e = __h81.freeze(__h81.load(__h81.to_auto(kv_ptr) + __h81.to_auto(kv_base) + __h81.to_auto(rope_even), _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                x_o = __h81.freeze(__h81.load(__h81.to_auto(kv_ptr) + __h81.to_auto(kv_base) + __h81.to_auto(rope_odd), _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                out_e = __h81.freeze(__h81.to_auto(x_e) * __h81.to_auto(cos) - __h81.to_auto(x_o) * __h81.to_auto(sin), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                out_o = __h81.freeze(__h81.to_auto(x_e) * __h81.to_auto(sin) + __h81.to_auto(x_o) * __h81.to_auto(cos), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                block_idx = __h81.freeze(__h81.to_auto(slot_id) // __h81.to_auto(CACHE_BLOCK_SIZE))
                pos_in_block = __h81.freeze(__h81.to_auto(slot_id) % __h81.to_auto(CACHE_BLOCK_SIZE))
                cache_off = __h81.freeze(__h81.to_auto(block_idx) * __h81.to_auto(stride_cache_block) + __h81.to_auto(pos_in_block) * __h81.to_auto(stride_cache_token))
                nope_mask = __h81.freeze(__h81.compare(__h81.to_auto(offs), 448, 'lt'), _layout=gl.BlockedLayout([8], [64], [1], [0]))
                __h81.store(__h81.to_auto(k_cache_ptr) + __h81.to_auto(cache_off) + __h81.to_auto(offs), __h81.to_auto(kv_data), mask=__h81.to_auto(nope_mask), _layout=gl.BlockedLayout([8], [64], [1], [0]))
                __h81.store(__h81.to_auto(k_cache_ptr) + __h81.to_auto(cache_off) + __h81.to_auto(rope_even), __h81.to_auto(out_e).to(gl.bfloat16), _layout=gl.BlockedLayout([1], [64], [1], [0]))
                __h81.store(__h81.to_auto(k_cache_ptr) + __h81.to_auto(cache_off) + __h81.to_auto(rope_odd), __h81.to_auto(out_o).to(gl.bfloat16), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        item_id = __h81.freeze(__h81.to_auto(item_id) + gl.num_programs(0))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([8], [64], [1], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

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

KERNEL = w1_g65536
