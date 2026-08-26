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
def strided_view_load_store_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr,
                                   TRAVERSAL_STRIDE: tl.constexpr):
    block = tl.program_id(0)
    src_view = tl.make_tensor_view(src, (n_elements,), (1,))
    dst_view = tl.make_tensor_view(dst, (n_elements,), (1,))

    value = tl.load(src_view, index=(block,), tile=(BLOCK_SIZE,), traversal_strides=(TRAVERSAL_STRIDE,))
    tl.store(dst_view, value, index=(block,), tile=(BLOCK_SIZE,), traversal_strides=(TRAVERSAL_STRIDE,))


def test_tensor_view_strided_load_store():
    grid_size = 4
    block_size = 1024
    traversal_stride = 1536
    n_elements = (grid_size - 1) * traversal_stride + block_size
    src = torch.rand(n_elements, device="npu")
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    for block in range(grid_size):
        start = block * traversal_stride
        expected[start:start + block_size] = src[start:start + block_size]

    strided_view_load_store_kernel[(grid_size, )](
        src, actual, n_elements, BLOCK_SIZE=block_size, TRAVERSAL_STRIDE=traversal_stride
    )

    torch.testing.assert_close(actual, expected)
