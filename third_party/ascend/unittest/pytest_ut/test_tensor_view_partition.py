# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def partition_view_load_store_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_view = tl.make_tensor_view(src, (rows, cols), (cols, 1))
    dst_view = tl.make_tensor_view(dst, (rows, cols), (cols, 1))

    value = tl.load(src_view, index=(row_block, col_block), tile=(BLOCK_M, BLOCK_N))
    tl.store(dst_view, value, index=(row_block, col_block), tile=(BLOCK_M, BLOCK_N))


def test_tensor_view_partition_load_store():
    rows = 100
    cols = 100
    block_m = 64
    block_n = 64
    src = torch.rand((rows, cols), device="npu")
    actual = torch.zeros_like(src)

    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    partition_view_load_store_kernel[grid](src, actual, rows, cols, BLOCK_M=block_m, BLOCK_N=block_n)

    torch.testing.assert_close(actual, src)
