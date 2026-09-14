import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.experimental.tle.language.dsa import tile_load, tile_store

@triton.jit
def partition_view_load_store_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_tiles = tl.make_partition_view(
        src,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )
    dst_tiles = tl.make_partition_view(
        dst,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )

    value = tile_load(src_tiles, index=(row_block, col_block))
    tile_store(dst_tiles, value, index=(row_block, col_block))


@triton.jit
def partition_view_padding_kernel(src, dst, src_elements, dst_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    src_tiles = tl.make_partition_view(
        src,
        shape=[src_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        padding_value="-inf",
    )
    dst_tiles = tl.make_partition_view(
        dst,
        shape=[dst_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tile_load(src_tiles, index=(block, ))
    tile_store(dst_tiles, value, index=(block, ))


def test_tensor_view_partition_load_store():
    rows = 128
    cols = 128
    block_m = 64
    block_n = 64
    src = torch.rand((rows, cols), device="npu")
    actual = torch.zeros_like(src)

    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    partition_view_load_store_kernel[grid](src, actual, rows, cols, BLOCK_M=block_m, BLOCK_N=block_n)

    torch.testing.assert_close(actual, src)


def test_tensor_view_partition_padding():
    src_elements = 10
    block_size = 8
    dst_elements = 16
    src = torch.arange(src_elements, dtype=torch.float32, device="npu")
    actual = torch.empty(dst_elements, dtype=torch.float32, device="npu")

    partition_view_padding_kernel[(2, )](src, actual, src_elements, dst_elements, BLOCK_SIZE=block_size)

    expected = torch.full_like(actual, -torch.inf)
    expected[:src_elements] = src
    torch.testing.assert_close(actual, expected)
