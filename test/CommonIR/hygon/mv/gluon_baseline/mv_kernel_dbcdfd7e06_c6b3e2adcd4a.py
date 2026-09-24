# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_20260907/src/FlagGems/src/flag_gems/ops/mv.py
# Provenance: results/gluon/06/mv_kernel_dbcdfd7e06_c6b3e2adcd4a.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def mv_kernel_dbcdfd7e06(A, B, C, N, M, stride_an, stride_am, stride_bm, stride_cn, BLOCK_N: gl.constexpr, BLOCK_M: gl.constexpr):
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    pid = __h81.freeze(program_id_f71308a253(__h81.pin(0)))
    offset_n = __h81.freeze(__h81.to_auto(pid) * __h81.to_auto(BLOCK_N) + __h81.expand_dims(__h81.arange(0, __h81.to_auto(BLOCK_N)), 1))
    offset_m = __h81.freeze(__h81.expand_dims(__h81.arange(0, __h81.to_auto(BLOCK_M)), 0))
    n_mask = __h81.freeze(__h81.compare(__h81.to_auto(offset_n), __h81.to_auto(N), 'lt'))
    A_ptrs = __h81.freeze(__h81.to_auto(A) + __h81.to_auto(offset_n) * __h81.to_auto(stride_an) + __h81.to_auto(offset_m) * __h81.to_auto(stride_am))
    B_ptrs = __h81.freeze(__h81.to_auto(B) + __h81.to_auto(offset_m) * __h81.to_auto(stride_bm))
    acc = __h81.freeze(__h81.zeros((__h81.to_auto(BLOCK_N), __h81.to_auto(BLOCK_M)), dtype=gl.float32), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1]))
    for m in range(0, __h81.to_auto(M), __h81.to_auto(BLOCK_M)):
        m_mask = __h81.freeze(__h81.compare(__h81.to_auto(m) + __h81.to_auto(offset_m), __h81.to_auto(M), 'lt'))
        a = __h81.freeze(__h81.load(__h81.to_auto(A_ptrs), mask=__h81.to_auto(n_mask) & __h81.to_auto(m_mask), other=0.0, _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1])).to(gl.float32), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1]))
        b = __h81.freeze(__h81.load(__h81.to_auto(B_ptrs), mask=__h81.to_auto(m_mask), other=0.0, _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1])).to(gl.float32), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1]))
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(a) * __h81.to_auto(b), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1]))
        A_ptrs = __h81.freeze(__h81.to_auto(A_ptrs) + __h81.to_auto(BLOCK_M) * __h81.to_auto(stride_am))
        B_ptrs = __h81.freeze(__h81.to_auto(B_ptrs) + __h81.to_auto(BLOCK_M) * __h81.to_auto(stride_bm))
    acc = __h81.freeze(__h81.to_auto(gl.sum(__h81.concrete(__h81.to_auto(acc), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1])), axis=1)), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1]))
    C_ptrs = __h81.freeze(__h81.to_auto(C) + __h81.to_auto(offset_n) * __h81.to_auto(stride_cn))
    __h81.store(__h81.to_auto(C_ptrs), __h81.expand_dims(__h81.to_auto(acc), 1), mask=__h81.to_auto(n_mask), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1]))

@g.jit
def program_id_f71308a253(axis: int) -> gl.tensor:
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    return gl.program_id(__h81.to_auto(axis)).to(gl.int64)

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

@g.jit
def sum_f3120590e3(input, axis=None, keep_dims=False, dtype: gl.constexpr=None):
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    out_dtype: gl.constexpr = _pick_sum_dtype_768abbebc3(__h81.pin(__h81.to_auto(input).dtype), dtype)
    if __h81.to_auto(out_dtype) is not None:
        input = __h81.freeze(__h81.to_auto(input).to(__h81.to_auto(out_dtype)), _layout=__h81.layout_of(input))
    return __h81.to_auto(gl.reduce(__h81.concrete(__h81.to_auto(input), _layout=gl.BlockedLayout([1, 1], [1, 64], [1, 8], [0, 1])), __h81.to_auto(axis), _sum_combine_9669e8607d, keep_dims=__h81.to_auto(keep_dims)))

@g.constexpr_function
def _pick_sum_dtype_768abbebc3(in_dtype, dtype):
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
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
    """HCU helper SHA256: 94eb0d787f57425d81cc224fc836910da6e7e6b0121ae3b6a22658badb459d9e"""
    return __h81.to_auto(a) + __h81.to_auto(b)

KERNEL = mv_kernel_dbcdfd7e06
