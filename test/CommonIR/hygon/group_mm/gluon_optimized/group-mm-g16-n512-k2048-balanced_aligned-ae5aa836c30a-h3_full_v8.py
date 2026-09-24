# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group2_20260911/src/FlagGems/src/flag_gems/ops/group_gemm.py
# Provenance: results/gluon/gluon_baseline/group-mm-g16-n512-k2048-balanced_aligned-ae5aa836c30a-ba4cd8ef8d750d01.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module
_ext_grouped_mm_tma_kernel_2107124f16_tl = _import_module('triton.language')

@g.jit
def grouped_mm_tma_kernel_2107124f16(a_desc, b_desc, c_desc, C, offs, num_groups: gl.constexpr, M, N: gl.constexpr, K: gl.constexpr, stride_cm: gl.constexpr, stride_cn: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, GROUP_M: gl.constexpr):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    total_grid = __h81.freeze(gl.num_programs(axis=0))
    tile_idx = __h81.freeze(gl.program_id(axis=0))
    num_n_tiles = __h81.freeze(gl.cdiv(__h81.to_auto(N), __h81.to_auto(BLOCK_N)))
    last_problem_end = __h81.freeze(0)
    group_start = __h81.freeze(0)
    group_end = __h81.freeze(0)
    for group_idx in _ext_grouped_mm_tma_kernel_2107124f16_tl.range(__h81.to_auto(num_groups)):
        group_end = __h81.freeze(__h81.load(__h81.to_auto(offs) + __h81.to_auto(group_idx)).to(gl.int32))
        m = __h81.freeze(__h81.to_auto(group_end) - __h81.to_auto(group_start))
        num_m_tiles = __h81.freeze(gl.cdiv(__h81.to_auto(m), __h81.to_auto(BLOCK_M)))
        num_tiles = __h81.freeze(__h81.to_auto(num_m_tiles) * __h81.to_auto(num_n_tiles))
        current_problem_end = __h81.freeze(__h81.to_auto(last_problem_end) + __h81.to_auto(num_tiles))
        if __h81.compare(__h81.to_auto(tile_idx), __h81.to_auto(last_problem_end), 'ge') and __h81.compare(__h81.to_auto(tile_idx), __h81.to_auto(current_problem_end), 'lt'):
            loop_count = __h81.freeze((__h81.to_auto(current_problem_end) - __h81.to_auto(tile_idx) + __h81.to_auto(total_grid) - 1) // __h81.to_auto(total_grid))
            for _ in _ext_grouped_mm_tma_kernel_2107124f16_tl.range(__h81.to_auto(loop_count)):
                tile_idx_in_gemm = __h81.freeze(__h81.to_auto(tile_idx) - __h81.to_auto(last_problem_end))
                (tile_m_idx, tile_n_idx) = __h81.freeze(grouped_launch_43e33799c4(tile_idx_in_gemm, m, N, BLOCK_M, BLOCK_N, GROUP_M))
                offs_am = __h81.freeze(__h81.to_auto(group_start) + __h81.to_auto(tile_m_idx) * __h81.to_auto(BLOCK_M))
                offs_bn = __h81.freeze(__h81.to_auto(tile_n_idx) * __h81.to_auto(BLOCK_N))
                offs_bk = __h81.freeze(__h81.to_auto(group_idx) * __h81.to_auto(K))
                accumulator = __h81.freeze(__h81.zeros((__h81.to_auto(BLOCK_M), __h81.to_auto(BLOCK_N)), dtype=gl.float32), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[2, 4], element_bitwidth=32, tiles_per_warp=None))
                for k in _ext_grouped_mm_tma_kernel_2107124f16_tl.range(0, gl.cdiv(__h81.to_auto(K), __h81.to_auto(BLOCK_K))):
                    a = __h81.freeze(__h81.descriptor_load(__h81.to_auto(a_desc), [__h81.to_auto(offs_am), __h81.to_auto(k) * __h81.to_auto(BLOCK_K)], _layout=gl.BlockedLayout([1, 1], [64, 1], [2, 4], [0, 1])), _layout=gl.BlockedLayout([1, 1], [64, 1], [2, 4], [0, 1]))
                    b = __h81.freeze(__h81.descriptor_load(__h81.to_auto(b_desc), [__h81.to_auto(offs_bk) + __h81.to_auto(k) * __h81.to_auto(BLOCK_K), __h81.to_auto(offs_bn)], _layout=gl.BlockedLayout([1, 1], [64, 1], [1, 8], [0, 1])), _layout=gl.BlockedLayout([1, 1], [64, 1], [1, 8], [0, 1]))
                    accumulator = __h81.freeze(__h81.dot(__h81.to_auto(a), __h81.to_auto(b), acc=__h81.to_auto(accumulator), allow_tf32=False, _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[2, 4], element_bitwidth=32, tiles_per_warp=None), _k_width=[4, 4], _acc_dtype=gl.float32, _operand_dtypes=(gl.bfloat16, gl.bfloat16)), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[2, 4], element_bitwidth=32, tiles_per_warp=None))
                c = __h81.freeze(__h81.cast(accumulator, __h81.descriptor_dtype(__h81.to_auto(c_desc))), _layout=__h81.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[2, 4], element_bitwidth=32, tiles_per_warp=None))
                if __h81.compare(__h81.to_auto(offs_am) + __h81.to_auto(BLOCK_M), __h81.to_auto(group_end), 'le'):
                    __h81.descriptor_store(__h81.to_auto(c_desc), [__h81.to_auto(offs_am), __h81.to_auto(offs_bn)], __h81.to_auto(c), _layout=gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0]))
                else:
                    offs_cm = __h81.freeze(__h81.to_auto(offs_am) + __h81.arange(0, __h81.to_auto(BLOCK_M), layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0]))), _layout=gl.SliceLayout(1, gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0])))
                    offs_cn = __h81.freeze(__h81.to_auto(offs_bn) + __h81.arange(0, __h81.to_auto(BLOCK_N), layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0]))), _layout=gl.SliceLayout(0, gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0])))
                    c_ptrs = __h81.freeze(__h81.to_auto(C) + __h81.to_auto(stride_cm) * __h81.expand_dims(__h81.to_auto(offs_cm), 1) + __h81.to_auto(stride_cn) * __h81.expand_dims(__h81.to_auto(offs_cn), 0), _layout=gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0]))
                    c_mask = __h81.freeze(__h81.compare(__h81.expand_dims(__h81.to_auto(offs_cm), 1), __h81.to_auto(group_end), 'lt') & __h81.compare(__h81.expand_dims(__h81.to_auto(offs_cn), 0), __h81.to_auto(N), 'lt'), _layout=gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0]))
                    __h81.store(__h81.to_auto(c_ptrs), __h81.to_auto(c), mask=__h81.to_auto(c_mask), _layout=gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0]))
                tile_idx = __h81.freeze(__h81.to_auto(tile_idx) + __h81.to_auto(total_grid))
        last_problem_end = __h81.freeze(__h81.to_auto(current_problem_end))
        group_start = __h81.freeze(__h81.to_auto(group_end))

@g.jit
def cdiv_5f871e20f4(x, div):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    '\n    Computes the ceiling division of :code:`x` by :code:`div`\n\n    :param x: the input number\n    :type x: Block\n    :param div: the divisor\n    :type div: Block\n    '
    return (__h81.to_auto(x) + (__h81.to_auto(div) - 1)) // __h81.to_auto(div)

@g.jit
def grouped_launch_43e33799c4(pid, m, n, block_m: gl.constexpr, block_n: gl.constexpr, group_m: gl.constexpr):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    grid_m = __h81.freeze(gl.cdiv(__h81.to_auto(m), __h81.to_auto(block_m)))
    grid_n = __h81.freeze(gl.cdiv(__h81.to_auto(n), __h81.to_auto(block_n)))
    width = __h81.freeze(__h81.to_auto(group_m) * __h81.to_auto(grid_n))
    group_id = __h81.freeze(__h81.to_auto(pid) // __h81.to_auto(width))
    group_size = __h81.freeze(gl.minimum(__h81.to_auto(grid_m) - __h81.to_auto(group_id) * __h81.to_auto(group_m), __h81.to_auto(group_m)))
    pid_m = __h81.freeze(__h81.to_auto(group_id) * __h81.to_auto(group_m) + __h81.to_auto(pid) % __h81.to_auto(group_size))
    pid_n = __h81.freeze(__h81.to_auto(pid) % __h81.to_auto(width) // __h81.to_auto(group_size))
    return (pid_m, pid_n)

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

KERNEL = grouped_mm_tma_kernel_2107124f16
