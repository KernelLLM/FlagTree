# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems-vllm/src/flaggems_vllm/ops/fp8_fp4_paged_mqa_logits.py
# Provenance: results/gluon/29/_mqa_logits_kernel_3b76c6aa80_6947810008b7.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module
_ext__mqa_logits_kernel_3b76c6aa80_tl = _import_module('triton.language')

@g.jit
def _mqa_logits_kernel_3b76c6aa80(Q_ptr, KV_data_ptr, KV_scales_ptr, Weights_ptr, Block_tables_ptr, Output_ptr, Ctx_lens_ptr, total_rows, max_ctx, num_heads: gl.constexpr, head_dim: gl.constexpr, max_model_len, block_size: gl.constexpr, max_blocks_per_seq, num_phys_blocks, stride_q_row, stride_kv_flat, stride_bt_row, stride_out_row, stride_w_row, BLOCK_KV: gl.constexpr, BLOCK_D: gl.constexpr, NUM_BLOCKS: gl.constexpr):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    'Per-tile kernel: each program processes one BLOCK_KV tile for one row.'
    kv_block = __h81.freeze(gl.program_id(0))
    row_idx = __h81.freeze(gl.program_id(1))
    if __h81.compare(__h81.to_auto(row_idx), __h81.to_auto(total_rows), 'ge'):
        return
    ctx_len = __h81.freeze(__h81.load(__h81.to_auto(Ctx_lens_ptr) + __h81.to_auto(row_idx)))
    kv_start = __h81.freeze(__h81.to_auto(kv_block) * __h81.to_auto(BLOCK_KV))
    if __h81.compare(__h81.to_auto(kv_start), __h81.to_auto(ctx_len), 'ge'):
        return
    q_row_base = __h81.freeze(__h81.to_auto(Q_ptr) + __h81.to_auto(row_idx) * __h81.to_auto(stride_q_row))
    w_row_base = __h81.freeze(__h81.to_auto(Weights_ptr) + __h81.to_auto(row_idx) * __h81.to_auto(stride_w_row))
    bt_row_base = __h81.freeze(__h81.to_auto(Block_tables_ptr) + __h81.to_auto(row_idx) * __h81.to_auto(stride_bt_row))
    out_row_base = __h81.freeze(__h81.to_auto(Output_ptr) + __h81.to_auto(row_idx) * __h81.to_auto(stride_out_row))
    h_ids = __h81.freeze(__h81.arange(0, __h81.to_auto(num_heads)), _layout=gl.SliceLayout(0, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None)))
    d_ids = __h81.freeze(__h81.arange(0, __h81.to_auto(BLOCK_D)))
    q_offsets = __h81.freeze(__h81.expand_dims(__h81.to_auto(h_ids), 1) * __h81.to_auto(head_dim) + __h81.expand_dims(__h81.to_auto(d_ids), 0), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])))
    q_u8 = __h81.freeze(__h81.load(__h81.to_auto(q_row_base) + __h81.to_auto(q_offsets), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
    q_fp8 = __h81.freeze(__h81.to_auto(q_u8).to(gl.float8e4nv, bitcast=True), _layout=gl.DotOperandLayout(0, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None), 4))
    w_all = __h81.freeze(__h81.load(__h81.to_auto(w_row_base) + __h81.arange(0, __h81.to_auto(num_heads)), _layout=gl.SliceLayout(1, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None))), _layout=gl.SliceLayout(1, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None)))
    end_pos = __h81.freeze(gl.minimum(__h81.to_auto(kv_start) + __h81.to_auto(BLOCK_KV), __h81.to_auto(ctx_len)))
    first_lb = __h81.freeze(__h81.to_auto(kv_start) // __h81.to_auto(block_size))
    p_ids = __h81.freeze(__h81.arange(0, __h81.to_auto(block_size)))
    for blk_idx in range(__h81.to_auto(NUM_BLOCKS)):
        lb = __h81.freeze(__h81.to_auto(first_lb) + __h81.to_auto(blk_idx))
        logical_base = __h81.freeze(__h81.to_auto(lb) * __h81.to_auto(block_size))
        if __h81.compare(__h81.to_auto(logical_base), __h81.to_auto(end_pos), 'lt'):
            phys_block = __h81.freeze(__h81.load(__h81.to_auto(bt_row_base) + __h81.to_auto(lb)))
            phys_block = __h81.freeze(gl.maximum(__h81.to_auto(phys_block), 0))
            phys_block = __h81.freeze(gl.minimum(__h81.to_auto(phys_block), __h81.to_auto(num_phys_blocks) - 1))
            flat_base = __h81.freeze(__h81.to_auto(phys_block) * __h81.to_auto(block_size))
            kv_offsets = __h81.freeze((__h81.to_auto(flat_base) + __h81.expand_dims(__h81.to_auto(p_ids), 1)) * __h81.to_auto(stride_kv_flat) + __h81.expand_dims(__h81.to_auto(d_ids), 0))
            kv_u8 = __h81.freeze(__h81.load(__h81.to_auto(KV_data_ptr) + __h81.to_auto(kv_offsets), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])), _layout=gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0]))
            kv_fp8 = __h81.freeze(__h81.to_auto(kv_u8).to(gl.float8e4nv, bitcast=True), _layout=gl.DistributedLinearLayout([[0, 1], [0, 2], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], [[0, 0], [0, 0]], [], [64, 128]))
            dots = __h81.freeze(__h81.dot(__h81.to_auto(q_fp8), _ext__mqa_logits_kernel_3b76c6aa80_tl.trans(__h81.to_auto(kv_fp8)), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None), _k_width=[4, 4], _acc_dtype=gl.float32, _operand_dtypes=(gl.float16, gl.float16)), _layout=gl.DotOperandLayout(1, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None), 4))
            scale_tile = __h81.freeze(__h81.load(__h81.to_auto(KV_scales_ptr) + __h81.to_auto(flat_base) + __h81.to_auto(p_ids), _layout=gl.SliceLayout(0, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None))), _layout=gl.SliceLayout(0, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None)))
            scores = __h81.freeze(gl.maximum(__h81.to_auto(dots) * __h81.expand_dims(__h81.to_auto(scale_tile), 0), 0.0), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None))
            weighted = __h81.freeze(__h81.to_auto(scores) * __h81.expand_dims(__h81.to_auto(w_all), 1), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None))
            output_tile = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(weighted), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None)), axis=0)), _layout=gl.SliceLayout(0, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None)))
            pos_ids = __h81.freeze(__h81.to_auto(logical_base) + __h81.to_auto(p_ids), _layout=gl.BlockedLayout([1], [64], [4], [0]))
            valid_mask = __h81.freeze(__h81.compare(__h81.to_auto(pos_ids), __h81.to_auto(end_pos), 'lt'), _layout=gl.BlockedLayout([1], [64], [4], [0]))
            __h81.store(__h81.to_auto(out_row_base) + __h81.to_auto(pos_ids), __h81.to_auto(output_tile), mask=__h81.to_auto(valid_mask), _layout=gl.BlockedLayout([1], [64], [4], [0]))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 093efe07010b4be7513e9b5f8120e860da84667d307f44ea453d22cd1046bfde"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[4, 1], element_bitwidth=32, tiles_per_warp=None)), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

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

KERNEL = _mqa_logits_kernel_3b76c6aa80
