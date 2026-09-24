# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260910_phase2/src/FlagGems-vllm/src/flaggems_vllm/ops/mhc/hc_split_sinkhorn.py
# Provenance: results/gluon/gluon_baseline/generic_t65536_hc2_b1_w1.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def mhc_split_sinkhorn_kernel_generic_0fabb64301(mixes_ptr, hc_scale_ptr, hc_base_ptr, pre_ptr, post_ptr, comb_ptr, num_tokens, SINKHORN_ITERS: gl.constexpr, HC_MULT: gl.constexpr, MIX_HC: gl.constexpr):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    'Generic split+sinkhorn kernel for arbitrary HC_MULT (one token per program).'
    pid_n = __h81.freeze(gl.program_id(0))
    if __h81.compare(__h81.to_auto(pid_n), __h81.to_auto(num_tokens), 'ge'):
        return
    base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(MIX_HC))
    pre_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC_MULT))
    post_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC_MULT))
    comb_base = __h81.freeze(__h81.to_auto(pid_n) * (__h81.to_auto(HC_MULT) * __h81.to_auto(HC_MULT)))
    scale_0 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 0))
    scale_1 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 1))
    scale_2 = __h81.freeze(__h81.load(__h81.to_auto(hc_scale_ptr) + 2))
    for j in gl.static_range(__h81.to_auto(HC_MULT)):
        pre_idx = __h81.freeze(__h81.to_auto(j))
        post_idx = __h81.freeze(__h81.to_auto(HC_MULT) + __h81.to_auto(j))
        pre_m = __h81.freeze(__h81.load(__h81.to_auto(mixes_ptr) + __h81.to_auto(base) + __h81.to_auto(pre_idx)))
        post_m = __h81.freeze(__h81.load(__h81.to_auto(mixes_ptr) + __h81.to_auto(base) + __h81.to_auto(post_idx)))
        pre_b = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(pre_idx)))
        post_b = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(post_idx)))
        __h81.store(__h81.to_auto(pre_ptr) + __h81.to_auto(pre_base) + __h81.to_auto(j), sigmoid_aba2984c1a(__h81.pin(__h81.to_auto(pre_m) * __h81.to_auto(scale_0) + __h81.to_auto(pre_b))) + 1e-06)
        __h81.store(__h81.to_auto(post_ptr) + __h81.to_auto(post_base) + __h81.to_auto(j), 2.0 * sigmoid_aba2984c1a(__h81.pin(__h81.to_auto(post_m) * __h81.to_auto(scale_1) + __h81.to_auto(post_b))))
    comb_offset = __h81.freeze(2 * __h81.to_auto(HC_MULT))
    for row in gl.static_range(__h81.to_auto(HC_MULT)):
        for col in gl.static_range(__h81.to_auto(HC_MULT)):
            idx = __h81.freeze(__h81.to_auto(comb_offset) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            out_idx = __h81.freeze(__h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            m = __h81.freeze(__h81.load(__h81.to_auto(mixes_ptr) + __h81.to_auto(base) + __h81.to_auto(idx)))
            b = __h81.freeze(__h81.load(__h81.to_auto(hc_base_ptr) + __h81.to_auto(idx)))
            __h81.store(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(out_idx), __h81.to_auto(m) * __h81.to_auto(scale_2) + __h81.to_auto(b))
    for row in gl.static_range(__h81.to_auto(HC_MULT)):
        row_ptr0 = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT))
        row_max = __h81.freeze(__h81.load(__h81.to_auto(row_ptr0)))
        for col in gl.static_range(__h81.to_auto(HC_MULT)):
            row_ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            row_max = __h81.freeze(gl.maximum(__h81.to_auto(row_max), __h81.load(__h81.to_auto(row_ptr))))
        row_sum = __h81.freeze(0.0)
        for col in gl.static_range(__h81.to_auto(HC_MULT)):
            row_ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            v = __h81.freeze(gl.exp(__h81.load(__h81.to_auto(row_ptr)) - __h81.to_auto(row_max)))
            row_sum = __h81.freeze(__h81.to_auto(row_sum) + __h81.to_auto(v))
            __h81.store(__h81.to_auto(row_ptr), __h81.to_auto(v))
        inv_row_sum = __h81.freeze(1.0 / __h81.to_auto(row_sum))
        for col in gl.static_range(__h81.to_auto(HC_MULT)):
            row_ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            v = __h81.freeze(__h81.load(__h81.to_auto(row_ptr)) * __h81.to_auto(inv_row_sum) + 1e-06)
            __h81.store(__h81.to_auto(row_ptr), __h81.to_auto(v))
    for col in gl.static_range(__h81.to_auto(HC_MULT)):
        col_sum = __h81.freeze(0.0)
        for row in gl.static_range(__h81.to_auto(HC_MULT)):
            ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            col_sum = __h81.freeze(__h81.to_auto(col_sum) + __h81.load(__h81.to_auto(ptr)))
        inv_col_sum = __h81.freeze(1.0 / (__h81.to_auto(col_sum) + 1e-06))
        for row in gl.static_range(__h81.to_auto(HC_MULT)):
            ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
            __h81.store(__h81.to_auto(ptr), __h81.load(__h81.to_auto(ptr)) * __h81.to_auto(inv_col_sum))
    for _ in range(__h81.to_auto(SINKHORN_ITERS) - 1):
        for row in gl.static_range(__h81.to_auto(HC_MULT)):
            row_sum = __h81.freeze(0.0)
            for col in gl.static_range(__h81.to_auto(HC_MULT)):
                ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
                row_sum = __h81.freeze(__h81.to_auto(row_sum) + __h81.load(__h81.to_auto(ptr)))
            inv_row_sum = __h81.freeze(1.0 / (__h81.to_auto(row_sum) + 1e-06))
            for col in gl.static_range(__h81.to_auto(HC_MULT)):
                ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
                __h81.store(__h81.to_auto(ptr), __h81.load(__h81.to_auto(ptr)) * __h81.to_auto(inv_row_sum))
        for col in gl.static_range(__h81.to_auto(HC_MULT)):
            col_sum = __h81.freeze(0.0)
            for row in gl.static_range(__h81.to_auto(HC_MULT)):
                ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
                col_sum = __h81.freeze(__h81.to_auto(col_sum) + __h81.load(__h81.to_auto(ptr)))
            inv_col_sum = __h81.freeze(1.0 / (__h81.to_auto(col_sum) + 1e-06))
            for row in gl.static_range(__h81.to_auto(HC_MULT)):
                ptr = __h81.freeze(__h81.to_auto(comb_ptr) + __h81.to_auto(comb_base) + __h81.to_auto(row) * __h81.to_auto(HC_MULT) + __h81.to_auto(col))
                __h81.store(__h81.to_auto(ptr), __h81.load(__h81.to_auto(ptr)) * __h81.to_auto(inv_col_sum))

@g.jit
def sigmoid_aba2984c1a(x):
    """HCU helper SHA256: 0a24f7f7d698bdb8c366a4bbdf390f5a323c1b66691e8385072f7080ddc45c2c"""
    return 1 / (1 + gl.exp(-__h81.to_auto(x)))

KERNEL = mhc_split_sinkhorn_kernel_generic_0fabb64301
