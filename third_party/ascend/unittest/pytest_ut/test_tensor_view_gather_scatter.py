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
def gather_view_load_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data_indices = tl.load(indices + offsets, mask=mask, other=0)
    src_view = tl.make_tensor_view(src, (n_elements,), (1,))
    dst_view = tl.make_tensor_view(dst, (n_elements,), (1,))

    value = tl.load(src_view, index=(data_indices,), tile=(BLOCK_SIZE,), sparse_dim=0, mask=mask)
    tl.store(dst_view, value, index=(block,), tile=(BLOCK_SIZE,))


@triton.jit
def scatter_view_store_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data_indices = tl.load(indices + offsets, mask=mask, other=0)
    src_view = tl.make_tensor_view(src, (n_elements,), (1,))
    dst_view = tl.make_tensor_view(dst, (n_elements,), (1,))

    value = tl.load(src_view, index=(block,), tile=(BLOCK_SIZE,))
    tl.store(dst_view, value, index=(data_indices,), tile=(BLOCK_SIZE,), sparse_dim=0, mask=mask)


def test_tensor_view_gather_load():
    n_elements = 1000
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randint(0, n_elements, (n_elements,), dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    gather_view_load_kernel[(triton.cdiv(n_elements, block_size), )](
        src, indices, actual, n_elements, BLOCK_SIZE=block_size
    )

    torch.testing.assert_close(actual, src[indices.long()])


def test_tensor_view_scatter_store():
    n_elements = 1000
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randperm(n_elements, device="npu").to(torch.int32)
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    expected[indices.long()] = src

    scatter_view_store_kernel[(triton.cdiv(n_elements, block_size), )](
        src, indices, actual, n_elements, BLOCK_SIZE=block_size
    )

    torch.testing.assert_close(actual, expected)
