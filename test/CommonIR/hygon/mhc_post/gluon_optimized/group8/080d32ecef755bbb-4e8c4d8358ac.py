from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81
from importlib import import_module as _import_module

@g.jit
def mhc_post_kernel_generic_aa7d23b7f7(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, H: gl.constexpr, HC: gl.constexpr, BLOCK_H: gl.constexpr):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    'Generic mHC post kernel for arbitrary HC.\n\n    Grid: (N, HC, cdiv(H, BLOCK_H)).\n    Each program handles one token × one output-stream(i) × one h-tile.\n    '
    pid_n = __h81.freeze(gl.program_id(0))
    pid_i = __h81.freeze(gl.program_id(1))
    pid_h = __h81.freeze(gl.program_id(2))
    h_off = __h81.freeze(__h81.to_auto(pid_h) * __h81.to_auto(BLOCK_H) + __h81.arange(0, __h81.to_auto(BLOCK_H), layout=gl.BlockedLayout([4], [64], [8], [0])), _layout=gl.BlockedLayout([4], [64], [8], [0]))
    h_mask = __h81.freeze(__h81.compare(__h81.to_auto(h_off), __h81.to_auto(H), 'lt'), _layout=gl.BlockedLayout([4], [64], [8], [0]))
    a_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC) * __h81.to_auto(HC))
    b_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC) * __h81.to_auto(H))
    c_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC))
    d_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(H))
    out_base = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(HC) * __h81.to_auto(H) + __h81.to_auto(pid_i) * __h81.to_auto(H))
    d_vals = __h81.freeze(__h81.load(__h81.to_auto(d_ptr) + __h81.to_auto(d_base) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [8], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [8], [0]))
    c_i = __h81.freeze(__h81.load(__h81.to_auto(c_ptr) + __h81.to_auto(c_base) + __h81.to_auto(pid_i)).to(gl.float32))
    acc = __h81.freeze(__h81.to_auto(c_i) * __h81.to_auto(d_vals), _layout=gl.BlockedLayout([4], [64], [8], [0]))
    for j in gl.static_range(0, __h81.to_auto(HC)):
        a_ji = __h81.freeze(__h81.load(__h81.to_auto(a_ptr) + __h81.to_auto(a_base) + __h81.to_auto(j) * __h81.to_auto(HC) + __h81.to_auto(pid_i)).to(gl.float32))
        b_j = __h81.freeze(__h81.load(__h81.to_auto(b_ptr) + __h81.to_auto(b_base) + __h81.to_auto(j) * __h81.to_auto(H) + __h81.to_auto(h_off), mask=__h81.to_auto(h_mask), other=0.0, _layout=gl.BlockedLayout([4], [64], [8], [0])).to(gl.float32), _layout=gl.BlockedLayout([4], [64], [8], [0]))
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(a_ji) * __h81.to_auto(b_j), _layout=gl.BlockedLayout([4], [64], [8], [0]))
    __h81.store(__h81.to_auto(out_ptr) + __h81.to_auto(out_base) + __h81.to_auto(h_off), __h81.cast(acc, gl.bfloat16, _layout=gl.BlockedLayout([4], [64], [8], [0])), mask=__h81.to_auto(h_mask), _layout=gl.BlockedLayout([4], [64], [8], [0]))
KERNEL = mhc_post_kernel_generic_aa7d23b7f7
