from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

@g.jit
def mv_bn1_col_w1(A, B, C, N, M, stride_an, stride_am, stride_bm, stride_cn,
             BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr):
    pid = __h81.freeze(gl.program_id(0))
    offset_m = __h81.freeze(__h81.arange(0, __h81.to_auto(BLOCK_M)), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    a_ptrs = __h81.freeze(__h81.to_auto(A) + __h81.to_auto(pid) * __h81.to_auto(stride_an)
                             + __h81.to_auto(offset_m) * __h81.to_auto(stride_am))
    b_ptrs = __h81.freeze(__h81.to_auto(B) + __h81.to_auto(offset_m) * __h81.to_auto(stride_bm))
    acc = __h81.freeze(__h81.zeros((__h81.to_auto(BLOCK_M),), dtype=gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    for m in range(0, __h81.to_auto(M), __h81.to_auto(BLOCK_M)):
        mask = __h81.freeze(__h81.compare(__h81.to_auto(m) + __h81.to_auto(offset_m), __h81.to_auto(M), 'lt'))
        a = __h81.freeze(__h81.load(__h81.to_auto(a_ptrs), mask=__h81.to_auto(mask), other=0.0,
                                    _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        b = __h81.freeze(__h81.load(__h81.to_auto(b_ptrs), mask=__h81.to_auto(mask), other=0.0,
                                    _layout=gl.BlockedLayout([1], [64], [1], [0])).to(gl.float32), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(a) * __h81.to_auto(b), _layout=gl.BlockedLayout([1], [64], [1], [0]))
        a_ptrs = __h81.freeze(__h81.to_auto(a_ptrs) + __h81.to_auto(BLOCK_M) * __h81.to_auto(stride_am))
        b_ptrs = __h81.freeze(__h81.to_auto(b_ptrs) + __h81.to_auto(BLOCK_M) * __h81.to_auto(stride_bm))
    value = __h81.freeze(gl.sum(__h81.concrete(__h81.to_auto(acc), _layout=gl.BlockedLayout([1], [64], [1], [0])), axis=0), _layout=gl.BlockedLayout([1], [64], [1], [0]))
    __h81.store(__h81.to_auto(C) + __h81.to_auto(pid) * __h81.to_auto(stride_cn), __h81.to_auto(value))

KERNEL = mv_bn1_col_w1
