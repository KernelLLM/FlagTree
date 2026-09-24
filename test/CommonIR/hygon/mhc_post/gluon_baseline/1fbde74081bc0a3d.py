# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group1_20260908/src/FlagGems-vllm/src/flaggems_vllm/ops/mhc/mhc_post.py
# Provenance: results/gluon/gluon_baseline/1fbde74081bc0a3d.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def mhc_post_kernel_hc_mult_4_ea58f0f0f4(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, H: gl.constexpr, BLOCK_H: gl.constexpr):
    """HCU helper SHA256: e734cf7b7fc2993746b4db3870d6e735826605eca768fd519d565a8db38a3f4a"""
    '\n    Grid: (N, cdiv(H, BLOCK_H)).\n    Each program handles one token × one h-tile × all 4 hc streams.\n    '
    pid_n = __h81.freeze(gl.program_id(0))
    pid_h = __h81.freeze(gl.program_id(1))
    h_off = __h81.freeze(__h81.to_auto(pid_h) * __h81.to_auto(BLOCK_H) + __h81.arange(0, __h81.to_auto(BLOCK_H), layout=gl.BlockedLayout([2], [64], [4], [0])), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_off), __h81.to_auto(H), 'lt'), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    a_base = __h81.freeze(__h81.to_auto(pid_n) * 16)
    c_base = __h81.freeze(__h81.to_auto(pid_n) * 4)
    b_base = __h81.freeze(__h81.to_auto(pid_n) * 4 * __h81.to_auto(H))
    d_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(H))
    out_base = __h81.freeze(__h81.to_auto(pid_n) * 4 * __h81.to_auto(H))
    c0 = __h81.freeze(__h81.load(__h81.to_auto(c_ptr) + __h81.to_auto(c_base) + 0).to(gl.float32))
    c1 = __h81.freeze(__h81.load(__h81.to_auto(c_ptr) + __h81.to_auto(c_base) + 1).to(gl.float32))
    c2 = __h81.freeze(__h81.load(__h81.to_auto(c_ptr) + __h81.to_auto(c_base) + 2).to(gl.float32))
    c3 = __h81.freeze(__h81.load(__h81.to_auto(c_ptr) + __h81.to_auto(c_base) + 3).to(gl.float32))
    a00 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 0).to(gl.float32))
    a01 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 1).to(gl.float32))
    a02 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 2).to(gl.float32))
    a03 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 3).to(gl.float32))
    a10 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 4).to(gl.float32))
    a11 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 5).to(gl.float32))
    a12 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 6).to(gl.float32))
    a13 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 7).to(gl.float32))
    a20 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 8).to(gl.float32))
    a21 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 9).to(gl.float32))
    a22 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 10).to(gl.float32))
    a23 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 11).to(gl.float32))
    a30 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 12).to(gl.float32))
    a31 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 13).to(gl.float32))
    a32 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 14).to(gl.float32))
    a33 = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + 15).to(gl.float32))
    d_vals = __h81.freeze(__h81.load(__h81.to_auto(d_ptr) + __h81.to_auto(d_base) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([2], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    b0 = __h81.freeze(__h81.load(__h81.to_auto(b_ptr) + __h81.to_auto(b_base) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([2], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    b1 = __h81.freeze(__h81.load(__h81.to_auto(b_ptr) + __h81.to_auto(b_base) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([2], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    b2 = __h81.freeze(__h81.load(__h81.to_auto(b_ptr) + __h81.to_auto(b_base) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([2], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    b3 = __h81.freeze(__h81.load(__h81.to_auto(b_ptr) + __h81.to_auto(b_base) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([2], [64], [4], [0])).to(gl.float32), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    acc0 = __h81.freeze(__h81.to_auto(c0) * __h81.to_auto(d_vals) + __h81.to_auto(a00) * __h81.to_auto(b0) + __h81.to_auto(a10) * __h81.to_auto(b1) + __h81.to_auto(a20) * __h81.to_auto(b2) + __h81.to_auto(a30) * __h81.to_auto(b3), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    acc1 = __h81.freeze(__h81.to_auto(c1) * __h81.to_auto(d_vals) + __h81.to_auto(a01) * __h81.to_auto(b0) + __h81.to_auto(a11) * __h81.to_auto(b1) + __h81.to_auto(a21) * __h81.to_auto(b2) + __h81.to_auto(a31) * __h81.to_auto(b3), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    acc2 = __h81.freeze(__h81.to_auto(c2) * __h81.to_auto(d_vals) + __h81.to_auto(a02) * __h81.to_auto(b0) + __h81.to_auto(a12) * __h81.to_auto(b1) + __h81.to_auto(a22) * __h81.to_auto(b2) + __h81.to_auto(a32) * __h81.to_auto(b3), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    acc3 = __h81.freeze(__h81.to_auto(c3) * __h81.to_auto(d_vals) + __h81.to_auto(a03) * __h81.to_auto(b0) + __h81.to_auto(a13) * __h81.to_auto(b1) + __h81.to_auto(a23) * __h81.to_auto(b2) + __h81.to_auto(a33) * __h81.to_auto(b3), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(out_base) + 0 * __h81.to_auto(H) + __h81.to_auto(h_off), __h81.to_auto(acc0).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(out_base) + 1 * __h81.to_auto(H) + __h81.to_auto(h_off), __h81.to_auto(acc1).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(out_base) + 2 * __h81.to_auto(H) + __h81.to_auto(h_off), __h81.to_auto(acc2).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([2], [64], [4], [0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(out_base) + 3 * __h81.to_auto(H) + __h81.to_auto(h_off), __h81.to_auto(acc3).to(gl.bfloat16), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([2], [64], [4], [0]))

KERNEL = mhc_post_kernel_hc_mult_4_ea58f0f0f4
