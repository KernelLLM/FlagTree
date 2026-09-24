# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems-vllm/src/flaggems_vllm/ops/DSA/indexer_k_tiled.py
# Provenance: results/gluon/65/triton_lighting_indexer_k_tiled_4015fb1c4b_e5_mma_1x4_store_4x1_f5faccdad68e.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def triton_lighting_indexer_k_tiled_4015fb1c4b_e5_mma_1x4_store_4x1(q_index, k_index, cu_bg_seqlens, cu_ed_seqlens, weights, logits, stride_qh, stride_qd, stride_kn, stride_kd, stride_wh, stride_lm, stride_ln, Q: gl.constexpr, H: gl.constexpr, K: gl.constexpr, TK: gl.constexpr, D: gl.constexpr, CU: gl.constexpr, BQ: gl.constexpr, BK: gl.constexpr):
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    (i_sh, i_k) = __h81.freeze((gl.program_id(0), gl.program_id(1)))
    offs_cu = __h81.freeze(__h81.arange(0, __h81.to_auto(BQ)) + __h81.to_auto(i_sh) * __h81.to_auto(BQ))
    mask_cu = __h81.freeze(__h81.compare(__h81.to_auto(offs_cu), __h81.to_auto(CU), 'lt'), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])))
    (bos_vec, eos_vec) = __h81.freeze((__h81.load(__h81.to_auto(cu_bg_seqlens) + __h81.to_auto(offs_cu), __h81.to_auto(mask_cu), 1000000000) + __h81.to_auto(i_k) * __h81.to_auto(TK), __h81.load(__h81.to_auto(cu_ed_seqlens) + __h81.to_auto(offs_cu), __h81.to_auto(mask_cu), -1000000000)))
    eos_vec = __h81.freeze(gl.minimum(__h81.to_auto(eos_vec), __h81.to_auto(bos_vec) + (__h81.to_auto(i_k) + 1) * __h81.to_auto(TK)), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])))
    (bos, eos) = __h81.freeze((max(__h81.to_auto(gl.min(__h81.concrete(__h81.to_auto(bos_vec)), 0)), 0), min(__h81.to_auto(gl.max(__h81.concrete(__h81.to_auto(eos_vec)), 0)), __h81.to_auto(K))))
    CK = __h81.freeze(__h81.to_auto(eos) - __h81.to_auto(bos))
    if __h81.compare(__h81.to_auto(CK), 0, 'gt'):
        q_base = __h81.freeze(__h81.to_auto(q_index))
        k_base = __h81.freeze(__h81.to_auto(k_index) + __h81.to_auto(bos) * __h81.to_auto(stride_kn))
        w_base = __h81.freeze(__h81.to_auto(weights))
        o_base = __h81.freeze(__h81.to_auto(logits) + __h81.to_auto(bos) * __h81.to_auto(stride_ln))
        offs_bq = __h81.freeze(__h81.arange(0, __h81.to_auto(BQ) * __h81.to_auto(H)) + __h81.to_auto(i_sh) * (__h81.to_auto(BQ) * __h81.to_auto(H)), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0])))
        offs_boq = __h81.freeze(__h81.arange(0, __h81.to_auto(BQ)) + __h81.to_auto(i_sh) * __h81.to_auto(BQ))
        offs_d = __h81.freeze(__h81.arange(0, __h81.to_auto(D)), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0])))
        offs_w = __h81.freeze(__h81.to_auto(offs_bq))
        mask_bq = __h81.freeze(__h81.compare(__h81.to_auto(offs_bq), __h81.to_auto(Q) * __h81.to_auto(H), 'lt'), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0])))
        mask_d = __h81.freeze(__h81.compare(__h81.to_auto(offs_d), __h81.to_auto(D), 'lt'), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0])))
        mask_boq = __h81.freeze(__h81.compare(__h81.to_auto(offs_boq), __h81.to_auto(Q), 'lt'))
        q_ptr = __h81.freeze(__h81.to_auto(q_base) + __h81.expand_dims(__h81.to_auto(offs_bq), 1) * __h81.to_auto(stride_qh) + __h81.expand_dims(__h81.to_auto(offs_d), 0) * __h81.to_auto(stride_qd), _layout=gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0]))
        q_msk = __h81.freeze(__h81.expand_dims(__h81.to_auto(mask_bq), 1) & __h81.expand_dims(__h81.to_auto(mask_d), 0), _layout=gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0]))
        q_blk = __h81.freeze(__h81.load(__h81.to_auto(q_ptr), __h81.to_auto(q_msk), 0.0, _layout=gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0])).to(gl.float16), _layout=gl.BlockedLayout([1, 2], [4, 16], [1, 4], [1, 0]))
        w_ptr = __h81.freeze(__h81.to_auto(w_base) + __h81.to_auto(offs_w) * __h81.to_auto(stride_wh))
        w_msk = __h81.freeze(__h81.to_auto(mask_bq))
        w_blk = __h81.freeze(__h81.load(__h81.to_auto(w_ptr), __h81.to_auto(w_msk), 0.0, _layout=gl.SliceLayout(1, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4], element_bitwidth=32, tiles_per_warp=None))).to(gl.float16), _layout=gl.SliceLayout(1, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4], element_bitwidth=32, tiles_per_warp=None)))
        CK = __h81.freeze(gl.cdiv(__h81.to_auto(CK), __h81.to_auto(BK)))
        for ck in range(__h81.to_auto(CK), warp_specialize=True):
            offs_bk = __h81.freeze(__h81.to_auto(ck) * __h81.to_auto(BK) + __h81.arange(0, __h81.to_auto(BK)), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])))
            mask_bk = __h81.freeze(__h81.compare(__h81.to_auto(bos) + __h81.to_auto(offs_bk), __h81.to_auto(eos), 'lt'), _layout=gl.SliceLayout(0, gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1])))
            k_ptr = __h81.freeze(__h81.to_auto(k_base) + __h81.expand_dims(__h81.to_auto(offs_d), 1) * __h81.to_auto(stride_kd) + __h81.expand_dims(__h81.to_auto(offs_bk), 0) * __h81.to_auto(stride_kn), _layout=gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1]))
            k_msk = __h81.freeze(__h81.expand_dims(__h81.to_auto(mask_d), 1) & __h81.expand_dims(__h81.to_auto(mask_bk), 0), _layout=gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1]))
            k_blk = __h81.freeze(__h81.load(__h81.to_auto(k_ptr), __h81.to_auto(k_msk), 0.0, _layout=gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1])).to(gl.float16), _layout=gl.DotOperandLayout(1, __h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4], element_bitwidth=32, tiles_per_warp=None), 4))
            acc = __h81.freeze(__h81.dot(__h81.to_auto(q_blk), __h81.to_auto(k_blk), out_dtype=gl.float16, _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4], element_bitwidth=32, tiles_per_warp=None), _k_width=[4, 4], _acc_dtype=gl.float32, _operand_dtypes=(gl.float16, gl.float16)), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4], element_bitwidth=32, tiles_per_warp=None))
            acc = __h81.freeze(gl.maximum(__h81.to_auto(acc), 0.0) * __h81.expand_dims(__h81.to_auto(w_blk), 1), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4], element_bitwidth=32, tiles_per_warp=None))
            out_blk = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(acc).trans().reshape(__h81.to_auto(BK), __h81.to_auto(BQ), __h81.to_auto(H))), -1)).trans(), _layout=gl.DistributedLinearLayout([[4, 0], [8, 0], [16, 0], [32, 0]], [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0]], [[0, 0], [0, 0]], [], [64, 16]))
            offs_out_bk = __h81.freeze(__h81.to_auto(ck) * __h81.to_auto(BK) + __h81.arange(0, __h81.to_auto(BK)), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])))
            mask_out_bk = __h81.freeze(__h81.compare(__h81.to_auto(bos) + __h81.to_auto(offs_out_bk), __h81.to_auto(eos), 'lt'), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])))
            out_ptr = __h81.freeze(__h81.to_auto(o_base) + __h81.expand_dims(__h81.to_auto(offs_boq), 1) * __h81.to_auto(stride_lm) + __h81.expand_dims(__h81.to_auto(offs_out_bk), 0) * __h81.to_auto(stride_ln), _layout=gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0]))
            out_msk = __h81.freeze(__h81.expand_dims(__h81.to_auto(mask_boq), 1) & __h81.expand_dims(__h81.to_auto(mask_out_bk), 0) & __h81.compare(__h81.expand_dims(__h81.to_auto(bos_vec), 1), __h81.expand_dims(__h81.to_auto(offs_out_bk), 0) + __h81.to_auto(bos), 'le') & __h81.compare(__h81.expand_dims(__h81.to_auto(eos_vec), 1), __h81.expand_dims(__h81.to_auto(offs_out_bk), 0) + __h81.to_auto(bos), 'gt'), _layout=gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0]))
            __h81.store(__h81.to_auto(out_ptr), __h81.to_auto(out_blk).to(gl.float16), __h81.to_auto(out_msk), _layout=gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0]))

@g.jit
def cdiv_5f871e20f4(x, div):
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    '\n    Computes the ceiling division of :code:`x` by :code:`div`\n\n    :param x: the input number\n    :type x: Block\n    :param div: the divisor\n    :type div: Block\n    '
    return (__h81.to_auto(x) + (__h81.to_auto(div) - 1)) // __h81.to_auto(div)

KERNEL = triton_lighting_indexer_k_tiled_4015fb1c4b_e5_mma_1x4_store_4x1
