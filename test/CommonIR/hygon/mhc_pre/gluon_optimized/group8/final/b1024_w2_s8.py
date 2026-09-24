# Layout candidate b1024_w2_s8; generated from the equivalent Gluon source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group8_20260914/src/FlagGems-vllm/src/flaggems_vllm/ops/mhc/mhc_pre.py
# Provenance: results/gluon/official/h1280_b1024_w4_s1_weuNone.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def mhc_pre_fused_kernel_hc_mult_4_63ca2f6011(gemm_out_ptr, hc_scale_ptr, hc_base_ptr, residual_ptr, post_mix_ptr, comb_mix_ptr, layer_input_ptr, num_tokens, num_tokens_bucket, res_stride_n, res_stride_i, res_stride_h, li_stride_n, li_stride_h, hidden_size, hc_hidden_size, rms_eps: gl.constexpr, hc_pre_eps: gl.constexpr, hc_sinkhorn_eps: gl.constexpr, hc_post_mult_value: gl.constexpr, sinkhorn_repeat: gl.constexpr, HC_MULT3: gl.constexpr, BLOCK_H: gl.constexpr):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    _mhc_pre_fused_kernel_hc_mult_4_impl_2903e7b1dd(gemm_out_ptr, hc_scale_ptr, hc_base_ptr, residual_ptr, post_mix_ptr, comb_mix_ptr, layer_input_ptr, num_tokens, num_tokens_bucket, res_stride_n, res_stride_i, res_stride_h, li_stride_n, li_stride_h, hidden_size, hc_hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, HC_MULT3, BLOCK_H)

@g.jit
def _mhc_pre_fused_kernel_hc_mult_4_impl_2903e7b1dd(gemm_out_ptr, hc_scale_ptr, hc_base_ptr, residual_ptr, post_mix_ptr, comb_mix_ptr, layer_input_ptr, num_tokens, num_tokens_bucket, res_stride_n, res_stride_i, res_stride_h, li_stride_n, li_stride_h, hidden_size, hc_hidden_size, rms_eps: gl.constexpr, hc_pre_eps: gl.constexpr, hc_sinkhorn_eps: gl.constexpr, hc_post_mult_value: gl.constexpr, sinkhorn_repeat: gl.constexpr, HC_MULT3: gl.constexpr, BLOCK_H: gl.constexpr):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    'Fully fused: sqrsum + RMS norm + sigmoid + Sinkhorn + weighted sum. One token per program.'
    pid_n = __h81.freeze(gl.program_id(0))
    if __h81.compare(__h81.to_auto(pid_n), __h81.to_auto(num_tokens), 'ge'):
        return
    sq = __h81.freeze(0.0)
    res_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(res_stride_n))
    for k in gl.static_range(4):
        head_base = __h81.freeze(__h81.to_auto(res_base) + __h81.to_auto(k) * __h81.to_auto(res_stride_i))
        for h_start in range(0, __h81.to_auto(hidden_size), __h81.to_auto(BLOCK_H)):
            h_offsets = __h81.freeze(__h81.to_auto(h_start) + __h81.arange(0, __h81.to_auto(BLOCK_H), layout=gl.BlockedLayout([8], [64], [2], [0])))
            h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_offsets), __h81.to_auto(hidden_size), 'lt'))
            v = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(head_base) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [2], [0])).to(gl.float32))
            sq = __h81.freeze(__h81.to_auto(sq) + __h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(v) * __h81.to_auto(v), _layout=gl.BlockedLayout([1], [64], [2], [0])))))
    rms_inv = __h81.freeze(gl.rsqrt(__h81.to_auto(sq) / __h81.to_auto(hc_hidden_size) + __h81.to_auto(rms_eps)))
    scale_0 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 0))
    scale_1 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 1))
    scale_2 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 2))
    go_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC_MULT3))
    pre_mix_0 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 0) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_0) + __h81.load(__h81.to_auto(hc_base_ptr) + 0))) + __h81.to_auto(hc_pre_eps))
    pre_mix_1 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 1) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_0) + __h81.load(__h81.to_auto(hc_base_ptr) + 1))) + __h81.to_auto(hc_pre_eps))
    pre_mix_2 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 2) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_0) + __h81.load(__h81.to_auto(hc_base_ptr) + 2))) + __h81.to_auto(hc_pre_eps))
    pre_mix_3 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 3) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_0) + __h81.load(__h81.to_auto(hc_base_ptr) + 3))) + __h81.to_auto(hc_pre_eps))
    post_0 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 4) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_1) + __h81.load(__h81.to_auto(hc_base_ptr) + 4))) * __h81.to_auto(hc_post_mult_value))
    __h81.store(__h81.to_auto(post_mix_ptr) + __h81.to_auto(pid_n) * 4 + 0, __h81.to_auto(post_0))
    post_1 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 5) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_1) + __h81.load(__h81.to_auto(hc_base_ptr) + 5))) * __h81.to_auto(hc_post_mult_value))
    __h81.store(__h81.to_auto(post_mix_ptr) + __h81.to_auto(pid_n) * 4 + 1, __h81.to_auto(post_1))
    post_2 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 6) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_1) + __h81.load(__h81.to_auto(hc_base_ptr) + 6))) * __h81.to_auto(hc_post_mult_value))
    __h81.store(__h81.to_auto(post_mix_ptr) + __h81.to_auto(pid_n) * 4 + 2, __h81.to_auto(post_2))
    post_3 = __h81.freeze(sigmoid_aba2984c1a(__h81.pin(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + 7) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_1) + __h81.load(__h81.to_auto(hc_base_ptr) + 7))) * __h81.to_auto(hc_post_mult_value))
    __h81.store(__h81.to_auto(post_mix_ptr) + __h81.to_auto(pid_n) * 4 + 3, __h81.to_auto(post_3))
    cb = __h81.freeze(8)
    cm_00 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 0) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 0))
    cm_01 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 1) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 1))
    cm_02 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 2) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 2))
    cm_03 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 3) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 3))
    cm_10 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 4) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 4))
    cm_11 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 5) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 5))
    cm_12 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 6) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 6))
    cm_13 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 7) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 7))
    cm_20 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 8) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 8))
    cm_21 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 9) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 9))
    cm_22 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 10) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 10))
    cm_23 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 11) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 11))
    cm_30 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 12) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 12))
    cm_31 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 13) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 13))
    cm_32 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 14) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 14))
    cm_33 = __h81.freeze(__h81.load(__h81.to_auto(gemm_out_ptr) + __h81.to_auto(go_base) + __h81.to_auto(cb) + 15) * __h81.to_auto(rms_inv) * __h81.to_auto(scale_2) + __h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(cb) + 15))
    rm = __h81.freeze(gl.maximum(gl.maximum(__h81.to_auto(cm_00), __h81.to_auto(cm_01)), gl.maximum(__h81.to_auto(cm_02), __h81.to_auto(cm_03))))
    cm_00 = __h81.freeze(gl.exp(__h81.to_auto(cm_00) - __h81.to_auto(rm)))
    cm_01 = __h81.freeze(gl.exp(__h81.to_auto(cm_01) - __h81.to_auto(rm)))
    cm_02 = __h81.freeze(gl.exp(__h81.to_auto(cm_02) - __h81.to_auto(rm)))
    cm_03 = __h81.freeze(gl.exp(__h81.to_auto(cm_03) - __h81.to_auto(rm)))
    rs = __h81.freeze(__h81.to_auto(cm_00) + __h81.to_auto(cm_01) + __h81.to_auto(cm_02) + __h81.to_auto(cm_03))
    inv_rs = __h81.freeze(1.0 / __h81.to_auto(rs))
    cm_00 = __h81.freeze(__h81.to_auto(cm_00) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_01 = __h81.freeze(__h81.to_auto(cm_01) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_02 = __h81.freeze(__h81.to_auto(cm_02) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_03 = __h81.freeze(__h81.to_auto(cm_03) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    rm = __h81.freeze(gl.maximum(gl.maximum(__h81.to_auto(cm_10), __h81.to_auto(cm_11)), gl.maximum(__h81.to_auto(cm_12), __h81.to_auto(cm_13))))
    cm_10 = __h81.freeze(gl.exp(__h81.to_auto(cm_10) - __h81.to_auto(rm)))
    cm_11 = __h81.freeze(gl.exp(__h81.to_auto(cm_11) - __h81.to_auto(rm)))
    cm_12 = __h81.freeze(gl.exp(__h81.to_auto(cm_12) - __h81.to_auto(rm)))
    cm_13 = __h81.freeze(gl.exp(__h81.to_auto(cm_13) - __h81.to_auto(rm)))
    rs = __h81.freeze(__h81.to_auto(cm_10) + __h81.to_auto(cm_11) + __h81.to_auto(cm_12) + __h81.to_auto(cm_13))
    inv_rs = __h81.freeze(1.0 / __h81.to_auto(rs))
    cm_10 = __h81.freeze(__h81.to_auto(cm_10) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_11 = __h81.freeze(__h81.to_auto(cm_11) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_12 = __h81.freeze(__h81.to_auto(cm_12) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_13 = __h81.freeze(__h81.to_auto(cm_13) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    rm = __h81.freeze(gl.maximum(gl.maximum(__h81.to_auto(cm_20), __h81.to_auto(cm_21)), gl.maximum(__h81.to_auto(cm_22), __h81.to_auto(cm_23))))
    cm_20 = __h81.freeze(gl.exp(__h81.to_auto(cm_20) - __h81.to_auto(rm)))
    cm_21 = __h81.freeze(gl.exp(__h81.to_auto(cm_21) - __h81.to_auto(rm)))
    cm_22 = __h81.freeze(gl.exp(__h81.to_auto(cm_22) - __h81.to_auto(rm)))
    cm_23 = __h81.freeze(gl.exp(__h81.to_auto(cm_23) - __h81.to_auto(rm)))
    rs = __h81.freeze(__h81.to_auto(cm_20) + __h81.to_auto(cm_21) + __h81.to_auto(cm_22) + __h81.to_auto(cm_23))
    inv_rs = __h81.freeze(1.0 / __h81.to_auto(rs))
    cm_20 = __h81.freeze(__h81.to_auto(cm_20) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_21 = __h81.freeze(__h81.to_auto(cm_21) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_22 = __h81.freeze(__h81.to_auto(cm_22) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_23 = __h81.freeze(__h81.to_auto(cm_23) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    rm = __h81.freeze(gl.maximum(gl.maximum(__h81.to_auto(cm_30), __h81.to_auto(cm_31)), gl.maximum(__h81.to_auto(cm_32), __h81.to_auto(cm_33))))
    cm_30 = __h81.freeze(gl.exp(__h81.to_auto(cm_30) - __h81.to_auto(rm)))
    cm_31 = __h81.freeze(gl.exp(__h81.to_auto(cm_31) - __h81.to_auto(rm)))
    cm_32 = __h81.freeze(gl.exp(__h81.to_auto(cm_32) - __h81.to_auto(rm)))
    cm_33 = __h81.freeze(gl.exp(__h81.to_auto(cm_33) - __h81.to_auto(rm)))
    rs = __h81.freeze(__h81.to_auto(cm_30) + __h81.to_auto(cm_31) + __h81.to_auto(cm_32) + __h81.to_auto(cm_33))
    inv_rs = __h81.freeze(1.0 / __h81.to_auto(rs))
    cm_30 = __h81.freeze(__h81.to_auto(cm_30) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_31 = __h81.freeze(__h81.to_auto(cm_31) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_32 = __h81.freeze(__h81.to_auto(cm_32) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cm_33 = __h81.freeze(__h81.to_auto(cm_33) * __h81.to_auto(inv_rs) + __h81.to_auto(hc_sinkhorn_eps))
    cs0 = __h81.freeze(__h81.to_auto(cm_00) + __h81.to_auto(cm_10) + __h81.to_auto(cm_20) + __h81.to_auto(cm_30))
    cs1 = __h81.freeze(__h81.to_auto(cm_01) + __h81.to_auto(cm_11) + __h81.to_auto(cm_21) + __h81.to_auto(cm_31))
    cs2 = __h81.freeze(__h81.to_auto(cm_02) + __h81.to_auto(cm_12) + __h81.to_auto(cm_22) + __h81.to_auto(cm_32))
    cs3 = __h81.freeze(__h81.to_auto(cm_03) + __h81.to_auto(cm_13) + __h81.to_auto(cm_23) + __h81.to_auto(cm_33))
    inv_cs0 = __h81.freeze(1.0 / (__h81.to_auto(cs0) + __h81.to_auto(hc_sinkhorn_eps)))
    inv_cs1 = __h81.freeze(1.0 / (__h81.to_auto(cs1) + __h81.to_auto(hc_sinkhorn_eps)))
    inv_cs2 = __h81.freeze(1.0 / (__h81.to_auto(cs2) + __h81.to_auto(hc_sinkhorn_eps)))
    inv_cs3 = __h81.freeze(1.0 / (__h81.to_auto(cs3) + __h81.to_auto(hc_sinkhorn_eps)))
    cm_00 = __h81.freeze(__h81.to_auto(cm_00) * __h81.to_auto(inv_cs0))
    cm_10 = __h81.freeze(__h81.to_auto(cm_10) * __h81.to_auto(inv_cs0))
    cm_20 = __h81.freeze(__h81.to_auto(cm_20) * __h81.to_auto(inv_cs0))
    cm_30 = __h81.freeze(__h81.to_auto(cm_30) * __h81.to_auto(inv_cs0))
    cm_01 = __h81.freeze(__h81.to_auto(cm_01) * __h81.to_auto(inv_cs1))
    cm_11 = __h81.freeze(__h81.to_auto(cm_11) * __h81.to_auto(inv_cs1))
    cm_21 = __h81.freeze(__h81.to_auto(cm_21) * __h81.to_auto(inv_cs1))
    cm_31 = __h81.freeze(__h81.to_auto(cm_31) * __h81.to_auto(inv_cs1))
    cm_02 = __h81.freeze(__h81.to_auto(cm_02) * __h81.to_auto(inv_cs2))
    cm_12 = __h81.freeze(__h81.to_auto(cm_12) * __h81.to_auto(inv_cs2))
    cm_22 = __h81.freeze(__h81.to_auto(cm_22) * __h81.to_auto(inv_cs2))
    cm_32 = __h81.freeze(__h81.to_auto(cm_32) * __h81.to_auto(inv_cs2))
    cm_03 = __h81.freeze(__h81.to_auto(cm_03) * __h81.to_auto(inv_cs3))
    cm_13 = __h81.freeze(__h81.to_auto(cm_13) * __h81.to_auto(inv_cs3))
    cm_23 = __h81.freeze(__h81.to_auto(cm_23) * __h81.to_auto(inv_cs3))
    cm_33 = __h81.freeze(__h81.to_auto(cm_33) * __h81.to_auto(inv_cs3))
    for _ in gl.static_range(__h81.to_auto(sinkhorn_repeat) - 1):
        rs0 = __h81.freeze(__h81.to_auto(cm_00) + __h81.to_auto(cm_01) + __h81.to_auto(cm_02) + __h81.to_auto(cm_03))
        rs1 = __h81.freeze(__h81.to_auto(cm_10) + __h81.to_auto(cm_11) + __h81.to_auto(cm_12) + __h81.to_auto(cm_13))
        rs2 = __h81.freeze(__h81.to_auto(cm_20) + __h81.to_auto(cm_21) + __h81.to_auto(cm_22) + __h81.to_auto(cm_23))
        rs3 = __h81.freeze(__h81.to_auto(cm_30) + __h81.to_auto(cm_31) + __h81.to_auto(cm_32) + __h81.to_auto(cm_33))
        inv_rs0 = __h81.freeze(1.0 / (__h81.to_auto(rs0) + __h81.to_auto(hc_sinkhorn_eps)))
        inv_rs1 = __h81.freeze(1.0 / (__h81.to_auto(rs1) + __h81.to_auto(hc_sinkhorn_eps)))
        inv_rs2 = __h81.freeze(1.0 / (__h81.to_auto(rs2) + __h81.to_auto(hc_sinkhorn_eps)))
        inv_rs3 = __h81.freeze(1.0 / (__h81.to_auto(rs3) + __h81.to_auto(hc_sinkhorn_eps)))
        cm_00 = __h81.freeze(__h81.to_auto(cm_00) * __h81.to_auto(inv_rs0))
        cm_01 = __h81.freeze(__h81.to_auto(cm_01) * __h81.to_auto(inv_rs0))
        cm_02 = __h81.freeze(__h81.to_auto(cm_02) * __h81.to_auto(inv_rs0))
        cm_03 = __h81.freeze(__h81.to_auto(cm_03) * __h81.to_auto(inv_rs0))
        cm_10 = __h81.freeze(__h81.to_auto(cm_10) * __h81.to_auto(inv_rs1))
        cm_11 = __h81.freeze(__h81.to_auto(cm_11) * __h81.to_auto(inv_rs1))
        cm_12 = __h81.freeze(__h81.to_auto(cm_12) * __h81.to_auto(inv_rs1))
        cm_13 = __h81.freeze(__h81.to_auto(cm_13) * __h81.to_auto(inv_rs1))
        cm_20 = __h81.freeze(__h81.to_auto(cm_20) * __h81.to_auto(inv_rs2))
        cm_21 = __h81.freeze(__h81.to_auto(cm_21) * __h81.to_auto(inv_rs2))
        cm_22 = __h81.freeze(__h81.to_auto(cm_22) * __h81.to_auto(inv_rs2))
        cm_23 = __h81.freeze(__h81.to_auto(cm_23) * __h81.to_auto(inv_rs2))
        cm_30 = __h81.freeze(__h81.to_auto(cm_30) * __h81.to_auto(inv_rs3))
        cm_31 = __h81.freeze(__h81.to_auto(cm_31) * __h81.to_auto(inv_rs3))
        cm_32 = __h81.freeze(__h81.to_auto(cm_32) * __h81.to_auto(inv_rs3))
        cm_33 = __h81.freeze(__h81.to_auto(cm_33) * __h81.to_auto(inv_rs3))
        cs0 = __h81.freeze(__h81.to_auto(cm_00) + __h81.to_auto(cm_10) + __h81.to_auto(cm_20) + __h81.to_auto(cm_30))
        cs1 = __h81.freeze(__h81.to_auto(cm_01) + __h81.to_auto(cm_11) + __h81.to_auto(cm_21) + __h81.to_auto(cm_31))
        cs2 = __h81.freeze(__h81.to_auto(cm_02) + __h81.to_auto(cm_12) + __h81.to_auto(cm_22) + __h81.to_auto(cm_32))
        cs3 = __h81.freeze(__h81.to_auto(cm_03) + __h81.to_auto(cm_13) + __h81.to_auto(cm_23) + __h81.to_auto(cm_33))
        inv_cs0 = __h81.freeze(1.0 / (__h81.to_auto(cs0) + __h81.to_auto(hc_sinkhorn_eps)))
        inv_cs1 = __h81.freeze(1.0 / (__h81.to_auto(cs1) + __h81.to_auto(hc_sinkhorn_eps)))
        inv_cs2 = __h81.freeze(1.0 / (__h81.to_auto(cs2) + __h81.to_auto(hc_sinkhorn_eps)))
        inv_cs3 = __h81.freeze(1.0 / (__h81.to_auto(cs3) + __h81.to_auto(hc_sinkhorn_eps)))
        cm_00 = __h81.freeze(__h81.to_auto(cm_00) * __h81.to_auto(inv_cs0))
        cm_01 = __h81.freeze(__h81.to_auto(cm_01) * __h81.to_auto(inv_cs1))
        cm_02 = __h81.freeze(__h81.to_auto(cm_02) * __h81.to_auto(inv_cs2))
        cm_03 = __h81.freeze(__h81.to_auto(cm_03) * __h81.to_auto(inv_cs3))
        cm_10 = __h81.freeze(__h81.to_auto(cm_10) * __h81.to_auto(inv_cs0))
        cm_11 = __h81.freeze(__h81.to_auto(cm_11) * __h81.to_auto(inv_cs1))
        cm_12 = __h81.freeze(__h81.to_auto(cm_12) * __h81.to_auto(inv_cs2))
        cm_13 = __h81.freeze(__h81.to_auto(cm_13) * __h81.to_auto(inv_cs3))
        cm_20 = __h81.freeze(__h81.to_auto(cm_20) * __h81.to_auto(inv_cs0))
        cm_21 = __h81.freeze(__h81.to_auto(cm_21) * __h81.to_auto(inv_cs1))
        cm_22 = __h81.freeze(__h81.to_auto(cm_22) * __h81.to_auto(inv_cs2))
        cm_23 = __h81.freeze(__h81.to_auto(cm_23) * __h81.to_auto(inv_cs3))
        cm_30 = __h81.freeze(__h81.to_auto(cm_30) * __h81.to_auto(inv_cs0))
        cm_31 = __h81.freeze(__h81.to_auto(cm_31) * __h81.to_auto(inv_cs1))
        cm_32 = __h81.freeze(__h81.to_auto(cm_32) * __h81.to_auto(inv_cs2))
        cm_33 = __h81.freeze(__h81.to_auto(cm_33) * __h81.to_auto(inv_cs3))
    co = __h81.freeze(__h81.to_auto(pid_n) * 16)
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 0, __h81.to_auto(cm_00))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 1, __h81.to_auto(cm_01))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 2, __h81.to_auto(cm_02))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 3, __h81.to_auto(cm_03))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 4, __h81.to_auto(cm_10))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 5, __h81.to_auto(cm_11))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 6, __h81.to_auto(cm_12))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 7, __h81.to_auto(cm_13))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 8, __h81.to_auto(cm_20))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 9, __h81.to_auto(cm_21))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 10, __h81.to_auto(cm_22))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 11, __h81.to_auto(cm_23))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 12, __h81.to_auto(cm_30))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 13, __h81.to_auto(cm_31))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 14, __h81.to_auto(cm_32))
    __h81.store(__h81.to_auto(comb_mix_ptr) + __h81.to_auto(co) + 15, __h81.to_auto(cm_33))
    for h_start in range(0, __h81.to_auto(hidden_size), __h81.to_auto(BLOCK_H)):
        h_offsets = __h81.freeze(__h81.to_auto(h_start) + __h81.arange(0, __h81.to_auto(BLOCK_H)))
        h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_offsets), __h81.to_auto(hidden_size), 'lt'))
        r0 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(res_base) + 0 * __h81.to_auto(res_stride_i) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [2], [0])).to(gl.float32))
        r1 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(res_base) + 1 * __h81.to_auto(res_stride_i) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [2], [0])).to(gl.float32))
        acc = __h81.freeze(__h81.to_auto(pre_mix_0) * __h81.to_auto(r0) + __h81.to_auto(pre_mix_1) * __h81.to_auto(r1))
        r2 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(res_base) + 2 * __h81.to_auto(res_stride_i) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [2], [0])).to(gl.float32))
        r3 = __h81.freeze(__h81.load(__h81.to_auto(residual_ptr) + __h81.to_auto(res_base) + 3 * __h81.to_auto(res_stride_i) + __h81.to_auto(h_offsets) * __h81.to_auto(res_stride_h), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([8], [64], [2], [0])).to(gl.float32))
        acc = __h81.freeze(__h81.to_auto(acc) + (__h81.to_auto(pre_mix_2) * __h81.to_auto(r2) + __h81.to_auto(pre_mix_3) * __h81.to_auto(r3)))
        __h81.store(__h81.to_auto(layer_input_ptr) + __h81.to_auto(pid_n) * __h81.to_auto(li_stride_n) + __h81.to_auto(h_offsets) * __h81.to_auto(li_stride_h), __h81.to_auto(acc).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([8], [64], [2], [0]))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1], [64], [2], [0])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.jit
def sigmoid_aba2984c1a(x):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return 1 / (1 + gl.exp(-__h81.to_auto(x)))

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
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
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return __h81.to_auto(a) + __h81.to_auto(b)

KERNEL = mhc_pre_fused_kernel_hc_mult_4_63ca2f6011
