"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice
@triton.jit
def _mqa_logits_kernel(Q_ptr, KV_data_ptr, KV_scales_ptr, Weights_ptr, Block_tables_ptr, Output_ptr, Ctx_lens_ptr, total_rows, max_ctx, num_heads: tl.constexpr, head_dim: tl.constexpr, max_model_len, block_size: tl.constexpr, max_blocks_per_seq, num_phys_blocks, stride_q_row, stride_kv_flat, stride_bt_row, stride_out_row, stride_w_row, BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    """Per-tile kernel: each program processes one BLOCK_KV tile for one row."""
    kv_block = tl.program_id(0)
    row_idx = tl.program_id(1)
    if row_idx >= total_rows:
        return
    ctx_len = tl.load(Ctx_lens_ptr + row_idx)
    kv_start = kv_block * BLOCK_KV
    if kv_start >= ctx_len:
        return
    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    bt_row_base = Block_tables_ptr + row_idx * stride_bt_row
    out_row_base = Output_ptr + row_idx * stride_out_row
    h_ids = tl.arange(0, num_heads)
    d_ids = tl.arange(0, BLOCK_D)
    q_offsets = h_ids[:, None] * head_dim + d_ids[None, :]
    q_u8 = tl.load(q_row_base + q_offsets)
    q_fp8 = q_u8.to(tl.float8e4nv, bitcast=True)
    w_all = tl.load(w_row_base + tl.arange(0, num_heads))
    end_pos = tl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size
    p_ids = tl.arange(0, block_size)
    for blk_idx in range(NUM_BLOCKS):
        lb = first_lb + blk_idx
        logical_base = lb * block_size
        if logical_base < end_pos:
            phys_block = tl.load(bt_row_base + lb)
            phys_block = tl.maximum(phys_block, 0)
            phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
            flat_base = phys_block * block_size
            kv_offsets = (flat_base + p_ids[:, None]) * stride_kv_flat + d_ids[None, :]
            kv_u8 = tl.load(KV_data_ptr + kv_offsets)
            kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)
            dots = tl.dot(q_fp8, tl.trans(kv_fp8))
            scale_tile = tl.load(KV_scales_ptr + flat_base + p_ids)
            scores = tl.maximum(dots * scale_tile[None, :], 0.0)
            weighted = scores * w_all[:, None]
            output_tile = tl.sum(weighted, axis=0)
            pos_ids = logical_base + p_ids
            valid_mask = pos_ids < end_pos
            tl.store(out_row_base + pos_ids, output_tile, mask=valid_mask)
