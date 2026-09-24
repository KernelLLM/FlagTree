from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

import hcu_ops as hcu


@g.constexpr_function
def load_layout():
    return gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])


@g.constexpr_function
def mma_layout():
    return hcu.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[1, 4],
        element_bitwidth=32,
        tiles_per_warp=None,
    )


@g.jit
def mqa_logits_global_layout_optimized(
    Q_ptr,
    KV_data_ptr,
    KV_scales_ptr,
    Weights_ptr,
    Block_tables_ptr,
    Output_ptr,
    Ctx_lens_ptr,
    total_rows,
    max_ctx,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    max_model_len,
    block_size: gl.constexpr,
    max_blocks_per_seq,
    num_phys_blocks,
    stride_q_row,
    stride_kv_flat,
    stride_bt_row,
    stride_out_row,
    stride_w_row,
    BLOCK_KV: gl.constexpr,
    BLOCK_D: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
):
    kv_block = gl.program_id(0)
    row_idx = gl.program_id(1)
    if row_idx >= total_rows:
        return
    ctx_len = gl.load(Ctx_lens_ptr + row_idx)
    kv_start = kv_block * BLOCK_KV
    if kv_start >= ctx_len:
        return

    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    bt_row_base = Block_tables_ptr + row_idx * stride_bt_row
    out_row_base = Output_ptr + row_idx * stride_out_row

    # Producers keep the coalesced [4,1] blocked ownership.  Cheap index
    # vectors for broadcasts/stores are born in the downstream MFMA slices,
    # avoiding conversion of the large Q/K and score tiles.
    h_load = gl.arange(0, num_heads, layout=gl.SliceLayout(1, load_layout()))
    d_load = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, load_layout()))
    p_load = gl.arange(0, block_size, layout=gl.SliceLayout(1, load_layout()))
    h_mma = gl.arange(0, num_heads, layout=gl.SliceLayout(1, mma_layout()))
    p_mma = gl.arange(0, block_size, layout=gl.SliceLayout(0, mma_layout()))

    q_u8 = gl.load(q_row_base + h_load[:, None] * head_dim + d_load[None, :])
    q_fp8 = q_u8.to(gl.float8e4nv, bitcast=True)
    weights = gl.load(w_row_base + h_mma)
    end_pos = gl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size

    for blk_idx in range(NUM_BLOCKS):
        lb = first_lb + blk_idx
        logical_base = lb * block_size
        if logical_base < end_pos:
            phys_block = gl.load(bt_row_base + lb)
            phys_block = gl.maximum(phys_block, 0)
            phys_block = gl.minimum(phys_block, num_phys_blocks - 1)
            flat_base = phys_block * block_size
            kv_u8 = gl.load(
                KV_data_ptr
                + (flat_base + p_load[:, None]) * stride_kv_flat
                + d_load[None, :]
            )
            kv_fp8 = kv_u8.to(gl.float8e4nv, bitcast=True)
            dots = hcu.dot(
                q_fp8,
                kv_fp8.trans(),
                out_dtype=gl.float32,
                _layout=mma_layout(),
                _k_width=[4, 4],
                _acc_dtype=gl.float32,
                _operand_dtypes=(gl.float16, gl.float16),
            )
            scales = gl.load(KV_scales_ptr + flat_base + p_mma)
            output_tile = gl.sum(
                gl.maximum(dots * scales[None, :], 0.0) * weights[:, None],
                axis=0,
            )
            pos = logical_base + p_mma
            gl.store(out_row_base + pos, output_tile, mask=pos < end_pos)


KERNEL = mqa_logits_global_layout_optimized
