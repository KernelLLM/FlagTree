"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice
@triton.jit
def _fused_qkv_kernel(q_ptr, kv_ptr, k_cache_ptr, slot_mapping_ptr, position_ids_ptr, cos_sin_cache_ptr, stride_q_tok, stride_q_head, stride_kv_tok, stride_cache_block, stride_cache_token, stride_cos_sin_pos, eps, num_q_items, total_items, NUM_HEADS: tl.constexpr, CACHE_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    item_id = pid
    while item_id < total_items:
        if item_id < num_q_items:
            tok_idx = item_id // NUM_HEADS
            head_idx = item_id % NUM_HEADS
            base = tok_idx * stride_q_tok + head_idx * stride_q_head
            offs = tl.arange(0, 512)
            x = tl.load(q_ptr + base + offs).to(tl.float32)
            sq_sum = tl.sum(x * x, axis=0)
            rsqrt_val = tl.math.rsqrt(sq_sum / 512.0 + eps)
            x_normed = x * rsqrt_val
            pos = tl.load(position_ids_ptr + tok_idx)
            half_offs = tl.arange(0, 32)
            cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
            sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)
            rope_even = 448 + half_offs * 2
            rope_odd = rope_even + 1
            x_re = tl.load(q_ptr + base + rope_even).to(tl.float32) * rsqrt_val
            x_ro = tl.load(q_ptr + base + rope_odd).to(tl.float32) * rsqrt_val
            q_out_e = x_re * cos - x_ro * sin
            q_out_o = x_re * sin + x_ro * cos
            nope_mask = offs < 448
            tl.store(q_ptr + base + offs, x_normed.to(tl.bfloat16), mask=nope_mask)
            tl.store(q_ptr + base + rope_even, q_out_e.to(tl.bfloat16))
            tl.store(q_ptr + base + rope_odd, q_out_o.to(tl.bfloat16))
        else:
            kv_idx = item_id - num_q_items
            slot_id = tl.load(slot_mapping_ptr + kv_idx)
            if slot_id >= 0:
                kv_base = kv_idx * stride_kv_tok
                offs = tl.arange(0, 512)
                kv_data = tl.load(kv_ptr + kv_base + offs)
                pos = tl.load(position_ids_ptr + kv_idx)
                half_offs = tl.arange(0, 32)
                cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
                sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)
                rope_even = 448 + half_offs * 2
                rope_odd = rope_even + 1
                x_e = tl.load(kv_ptr + kv_base + rope_even).to(tl.float32)
                x_o = tl.load(kv_ptr + kv_base + rope_odd).to(tl.float32)
                out_e = x_e * cos - x_o * sin
                out_o = x_e * sin + x_o * cos
                block_idx = slot_id // CACHE_BLOCK_SIZE
                pos_in_block = slot_id % CACHE_BLOCK_SIZE
                cache_off = block_idx * stride_cache_block + pos_in_block * stride_cache_token
                nope_mask = offs < 448
                tl.store(k_cache_ptr + cache_off + offs, kv_data, mask=nope_mask)
                tl.store(k_cache_ptr + cache_off + rope_even, out_e.to(tl.bfloat16))
                tl.store(k_cache_ptr + cache_off + rope_odd, out_o.to(tl.bfloat16))
        item_id += tl.num_programs(0)
