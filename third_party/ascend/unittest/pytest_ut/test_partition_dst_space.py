# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Validation for the tile.load/tile.store PartitionView path with an explicit
# dst_space/src_space argument (tile.load/tile.store -> CommonIRToHIVM ->
# hivm.load/hivm.store).

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa import tile_load, tile_store


@triton.jit
def partition_dst_space_ub_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Reference the space inline (like native_matmul) so the JIT does not treat
    # it as a mutable module global.
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_view = tl.make_partition_view(src, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])
    dst_view = tl.make_partition_view(dst, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])

    value = tile_load(src_view, index=(row_block, col_block),
                      dst_space=tle.language.dsa.ascend.UB)
    tile_store(dst_view, value=value, index=(row_block, col_block),
               src_space=tle.language.dsa.ascend.UB)


@triton.jit
def partition_default_space_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # dst_space/src_space omitted -> defaults to UB.
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_view = tl.make_partition_view(src, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])
    dst_view = tl.make_partition_view(dst, shape=[rows, cols], strides=[cols, 1], tile=[BLOCK_M, BLOCK_N])

    value = tile_load(src_view, index=(row_block, col_block))
    tile_store(dst_view, value=value, index=(row_block, col_block))


def _run(kernel):
    rows, cols = 128, 128
    block_m, block_n = 64, 64
    src = torch.rand((rows, cols), device="npu")
    actual = torch.zeros_like(src)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    kernel[grid](src, actual, rows, cols, BLOCK_M=block_m, BLOCK_N=block_n)
    torch.testing.assert_close(actual, src)


def test_partition_dst_space_ub_explicit():
    _run(partition_dst_space_ub_kernel)


def test_partition_dst_space_default_is_ub():
    _run(partition_default_space_kernel)
