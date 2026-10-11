from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna3

NUM_WARPS = 4
NUM_STAGES = 2
Q_LAYOUT = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
KV_LAYOUT = Q_LAYOUT
MMA_LAYOUT = AMDMFMALayout(
    version=3,
    instr_shape=[16, 16, 16],
    transposed=False,
    warps_per_cta=[1, 4],
    element_bitwidth=32,
)
WEIGHT_LAYOUT = gl.SliceLayout(1, MMA_LAYOUT)
SCALE_LAYOUT = gl.SliceLayout(0, MMA_LAYOUT)
REDUCE_LAYOUT = MMA_LAYOUT
STORE_LAYOUT = gl.SliceLayout(0, MMA_LAYOUT)


def layout_parameters():
    return (
        Q_LAYOUT,
        KV_LAYOUT,
        WEIGHT_LAYOUT,
        SCALE_LAYOUT,
        MMA_LAYOUT,
        REDUCE_LAYOUT,
        STORE_LAYOUT,
    )


@gluon.jit
def mqa_logits(
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
    Q_LOAD: gl.constexpr,
    KV_LOAD: gl.constexpr,
    WEIGHT: gl.constexpr,
    SCALE: gl.constexpr,
    MMA: gl.constexpr,
    REDUCE: gl.constexpr,
    STORE: gl.constexpr,
):
    kv_block = gl.program_id(0)
    row = gl.program_id(1)
    if row >= total_rows:
        return
    length = gl.load(Ctx_lens_ptr + row)
    kv_start = kv_block * BLOCK_KV
    if kv_start >= length:
        return
    q_base = Q_ptr + row * stride_q_row
    w_base = Weights_ptr + row * stride_w_row
    table_base = Block_tables_ptr + row * stride_bt_row
    output_base = Output_ptr + row * stride_out_row

    q_heads = gl.arange(0, num_heads, layout=gl.SliceLayout(1, Q_LOAD))
    q_dims = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, Q_LOAD))
    q_bytes = gl.load(q_base + q_heads[:, None] * head_dim + q_dims[None, :])
    weight_heads = gl.arange(0, num_heads, layout=WEIGHT)
    weights = gl.load(w_base + weight_heads)
    # Match the historical byte-stage Q staging, before FP8 -> FP16.
    # Converting FP16 first doubles LDS and is NOT equivalent to this contract.
    q_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, [1, 0])
    q_shared = gl.allocate_shared_memory(
        gl.uint8, (num_heads, BLOCK_D), q_shared_layout, q_bytes
    )
    q_operand_bytes = q_shared.load(gl.DotOperandLayout(0, MMA, 4))
    q_fp8 = q_operand_bytes.to(gl.float8e4nv, bitcast=True)

    end = gl.minimum(kv_start + BLOCK_KV, length)
    first_logical_block = kv_start // block_size
    kv_positions = gl.arange(0, block_size, layout=gl.SliceLayout(1, KV_LOAD))
    kv_dims = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, KV_LOAD))
    for block_index in range(NUM_BLOCKS):
        logical_block = first_logical_block + block_index
        logical_start = logical_block * block_size
        if logical_start < end:
            physical_block = gl.load(table_base + logical_block)
            physical_block = gl.minimum(
                gl.maximum(physical_block, 0), num_phys_blocks - 1
            )
            flat_start = physical_block * block_size
            kv_bytes = gl.load(
                KV_data_ptr
                + (flat_start + kv_positions[:, None]) * stride_kv_flat
                + kv_dims[None, :]
            )
            # Transpose is a coordinate view; transfer raw bytes into B's
            # exact MFMA packing before decoding, as in the historical IR.
            a = q_fp8.to(gl.float16)
            b_bytes = gl.convert_layout(
                kv_bytes.trans(), gl.DotOperandLayout(1, MMA, 4)
            )
            b = b_bytes.to(gl.float8e4nv, bitcast=True).to(gl.float16)
            dots = cdna3.mfma(
                a, b, gl.full((num_heads, block_size), 0, gl.float32, MMA)
            )
            scale_positions = gl.arange(0, block_size, layout=SCALE)
            scales = gl.load(KV_scales_ptr + flat_start + scale_positions)
            scales = gl.convert_layout(scales, gl.SliceLayout(0, MMA))
            weights_mma = gl.convert_layout(weights, gl.SliceLayout(1, MMA))
            weighted = gl.maximum(dots * scales[None, :], 0.0) * weights_mma[:, None]
            logits = gl.sum(gl.convert_layout(weighted, REDUCE), axis=0)
            output_positions = logical_start + gl.arange(0, block_size, layout=STORE)
            gl.store(
                output_base + output_positions,
                gl.convert_layout(logits, STORE),
                mask=output_positions < end,
            )


KERNEL = mqa_logits
