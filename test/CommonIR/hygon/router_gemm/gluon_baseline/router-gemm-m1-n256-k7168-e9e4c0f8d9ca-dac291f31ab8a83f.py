# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group2_20260911/src/FlagGems/src/flag_gems/ops/mm.py
# Provenance: results/gluon/gluon_baseline/router-gemm-m1-n256-k7168-e9e4c0f8d9ca-dac291f31ab8a83f.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def mm_kernel_general_9368d87839(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_M: gl.constexpr, IS_FP64: gl.constexpr=False):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    pid = __h81.freeze(program_id_f71308a253(__h81.pin(0)))
    grid_m = __h81.freeze(gl.cdiv(__h81.to_auto(M), __h81.to_auto(BLOCK_M)))
    grid_n = __h81.freeze(gl.cdiv(__h81.to_auto(N), __h81.to_auto(BLOCK_N)))
    width = __h81.freeze(__h81.to_auto(GROUP_M) * __h81.to_auto(grid_n))
    group_id = __h81.freeze(__h81.to_auto(pid) // __h81.to_auto(width))
    group_size = __h81.freeze(min(__h81.to_auto(grid_m) - __h81.to_auto(group_id) * __h81.to_auto(GROUP_M), __h81.to_auto(GROUP_M)))
    pid_m = __h81.freeze(__h81.to_auto(group_id) * __h81.to_auto(GROUP_M) + __h81.to_auto(pid) % __h81.to_auto(group_size))
    pid_n = __h81.freeze(__h81.to_auto(pid) % __h81.to_auto(width) // __h81.to_auto(group_size))
    rm = __h81.freeze(__h81.to_auto(pid_m) * __h81.to_auto(BLOCK_M) + __h81.arange(0, __h81.to_auto(BLOCK_M), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0]))), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])))
    rn = __h81.freeze(__h81.to_auto(pid_n) * __h81.to_auto(BLOCK_N) + __h81.arange(0, __h81.to_auto(BLOCK_N), layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0]))), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))
    ram = __h81.freeze(gl.max_contiguous(gl.multiple_of(__h81.to_auto(rm) % __h81.to_auto(M), __h81.to_auto(BLOCK_M)), __h81.to_auto(BLOCK_M)).to(gl.int64))
    rbn = __h81.freeze(gl.max_contiguous(gl.multiple_of(__h81.to_auto(rn) % __h81.to_auto(N), __h81.to_auto(BLOCK_N)), __h81.to_auto(BLOCK_N)).to(gl.int64), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))
    rm = __h81.freeze(__h81.cast(rm, gl.int64), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])))
    rn = __h81.freeze(__h81.cast(rn, gl.int64), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))
    prev_multiple = __h81.freeze(prev_multiple_of_f634c2a333(K, BLOCK_K))
    if __h81.to_auto(IS_FP64):
        acc = __h81.freeze(__h81.zeros((__h81.to_auto(BLOCK_M), __h81.to_auto(BLOCK_N)), dtype=gl.float64), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
    else:
        acc = __h81.freeze(__h81.zeros((__h81.to_auto(BLOCK_M), __h81.to_auto(BLOCK_N)), dtype=gl.float32), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
    for start_k in range(0, __h81.to_auto(prev_multiple), __h81.to_auto(BLOCK_K)):
        rk_a = __h81.freeze((__h81.to_auto(start_k) + __h81.arange(0, __h81.to_auto(BLOCK_K), layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])))).to(gl.int64), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])))
        a = __h81.freeze(__h81.load(__h81.to_auto(A) + (__h81.expand_dims(__h81.to_auto(ram), 1) * __h81.to_auto(stride_am) + __h81.expand_dims(__h81.to_auto(rk_a), 0) * __h81.to_auto(stride_ak)), _layout=gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])), _layout=gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0]))
        rk_b = __h81.freeze((__h81.to_auto(start_k) + __h81.arange(0, __h81.to_auto(BLOCK_K), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))).to(gl.int64), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))
        b = __h81.freeze(__h81.load(__h81.to_auto(B) + (__h81.expand_dims(__h81.to_auto(rk_b), 1) * __h81.to_auto(stride_bk) + __h81.expand_dims(__h81.to_auto(rbn), 0) * __h81.to_auto(stride_bn)), _layout=gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])), _layout=gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0]))
        if __h81.compare(__h81.to_auto(a).dtype, __h81.to_auto(b).dtype, 'ne'):
            a = __h81.freeze(__h81.cast(a, __h81.pointer_element_type(__h81.to_auto(C), True)), _layout=gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0]))
            b = __h81.freeze(__h81.cast(b, __h81.pointer_element_type(__h81.to_auto(C), True)), _layout=gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0]))
        if __h81.to_auto(IS_FP64):
            acc = __h81.freeze(__h81.to_auto(acc) + __h81.dot(__h81.to_auto(a), __h81.to_auto(b), allow_tf32=False), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
        else:
            acc = __h81.freeze(__h81.to_auto(acc) + __h81.dot(__h81.to_auto(a), __h81.to_auto(b), out_dtype=gl.float32, allow_tf32=False, _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None), _k_width=[4, 4], _acc_dtype=gl.float32, _operand_dtypes=(gl.bfloat16, gl.bfloat16)), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
    rk_a = __h81.freeze((__h81.to_auto(prev_multiple) + __h81.arange(0, __h81.to_auto(BLOCK_K), layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])))).to(gl.int64), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])))
    mask_k_a = __h81.freeze(__h81.compare(__h81.to_auto(rk_a), __h81.to_auto(K), 'lt'), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])))
    a = __h81.freeze(__h81.load(__h81.to_auto(A) + (__h81.expand_dims(__h81.to_auto(ram), 1) * __h81.to_auto(stride_am) + __h81.expand_dims(__h81.to_auto(rk_a), 0) * __h81.to_auto(stride_ak)), mask=__h81.expand_dims(__h81.to_auto(mask_k_a), 0), other=0.0, _layout=gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])), _layout=gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0]))
    rk_b = __h81.freeze((__h81.to_auto(prev_multiple) + __h81.arange(0, __h81.to_auto(BLOCK_K), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))).to(gl.int64), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))
    mask_k_b = __h81.freeze(__h81.compare(__h81.to_auto(rk_b), __h81.to_auto(K), 'lt'), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])))
    b = __h81.freeze(__h81.load(__h81.to_auto(B) + (__h81.expand_dims(__h81.to_auto(rk_b), 1) * __h81.to_auto(stride_bk) + __h81.expand_dims(__h81.to_auto(rbn), 0) * __h81.to_auto(stride_bn)), mask=__h81.expand_dims(__h81.to_auto(mask_k_b), 1), other=0.0, _layout=gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])), _layout=gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0]))
    if __h81.compare(__h81.to_auto(a).dtype, __h81.to_auto(b).dtype, 'ne'):
        a = __h81.freeze(__h81.cast(a, __h81.pointer_element_type(__h81.to_auto(C), True)), _layout=gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0]))
        b = __h81.freeze(__h81.cast(b, __h81.pointer_element_type(__h81.to_auto(C), True)), _layout=gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0]))
    if __h81.to_auto(IS_FP64):
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.dot(__h81.to_auto(a), __h81.to_auto(b), allow_tf32=False), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
    else:
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.dot(__h81.to_auto(a), __h81.to_auto(b), out_dtype=gl.float32, allow_tf32=False, _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None), _k_width=[4, 4], _acc_dtype=gl.float32, _operand_dtypes=(gl.bfloat16, gl.bfloat16)), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
    acc = __h81.freeze(__h81.cast(acc, __h81.pointer_element_type(__h81.to_auto(C), True)), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 2], element_bitwidth=32, tiles_per_warp=None))
    rm = __h81.freeze((__h81.to_auto(pid_m) * __h81.to_auto(BLOCK_M) + __h81.arange(0, __h81.to_auto(BLOCK_M), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])))).to(gl.int64), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])))
    rn = __h81.freeze((__h81.to_auto(pid_n) * __h81.to_auto(BLOCK_N) + __h81.arange(0, __h81.to_auto(BLOCK_N), layout=gl.SliceLayout(0, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])))).to(gl.int64), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])))
    C = __h81.freeze(__h81.to_auto(C) + (__h81.expand_dims(__h81.to_auto(rm), 1) * __h81.to_auto(stride_cm) + __h81.expand_dims(__h81.to_auto(rn), 0) * __h81.to_auto(stride_cn)), _layout=gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0]))
    mask = __h81.freeze(__h81.expand_dims(__h81.compare(__h81.to_auto(rm), __h81.to_auto(M), 'lt'), 1) & __h81.expand_dims(__h81.compare(__h81.to_auto(rn), __h81.to_auto(N), 'lt'), 0), _layout=gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0]))
    __h81.store(__h81.to_auto(C), __h81.to_auto(acc), mask=__h81.to_auto(mask), _layout=gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0]))

@g.jit
def program_id_f71308a253(axis: int) -> gl.tensor:
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    return gl.program_id(__h81.to_auto(axis)).to(gl.int64)

@g.jit
def cdiv_5f871e20f4(x, div):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    '\n    Computes the ceiling division of :code:`x` by :code:`div`\n\n    :param x: the input number\n    :type x: Block\n    :param div: the divisor\n    :type div: Block\n    '
    return (__h81.to_auto(x) + (__h81.to_auto(div) - 1)) // __h81.to_auto(div)

@g.jit
def prev_multiple_of_f634c2a333(a, b):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    return gl.cdiv(__h81.to_auto(a), __h81.to_auto(b)) * __h81.to_auto(b) - __h81.to_auto(b)

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

KERNEL = mm_kernel_general_9368d87839
