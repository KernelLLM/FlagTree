# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

"""Hardware E2E for the TensorView gather/scatter lowering route."""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


ROWS = 12
COLS = 4
TILE = 4


@triton.jit
def tensor_view_gather_scatter_kernel(src_ptr, index_ptr, gather_out_ptr,
                                      scatter_out_ptr,
                                      COLS: tl.constexpr,
                                      TILE: tl.constexpr):
    lanes = tl.arange(0, TILE)
    sparse_rows = tl.load(index_ptr + lanes)
    row_offsets = sparse_rows[:, None] * COLS
    col_offsets = lanes[None, :]

    # Data-dependent row addresses are recognized as sparse_dim=0 by Pass A.
    values = tl.load(src_ptr + row_offsets + col_offsets)

    # Keep an ordinary dense result for direct gather validation.
    dense_rows = lanes[:, None] * COLS
    tl.store(gather_out_ptr + dense_rows + col_offsets, values)

    # The same data-dependent addresses exercise tv.view_store -> scatter_store.
    tl.store(scatter_out_ptr + row_offsets + col_offsets, values)


def test_tensor_view_gather_scatter_hardware_e2e():
    device = torch.device("npu")
    indices_cpu = torch.tensor([9, 2, 7, 0], dtype=torch.int32)
    src_cpu = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    indices = indices_cpu.to(device)
    src = src_cpu.to(device)

    gather_out = torch.full((TILE, COLS), -1.0, dtype=src.dtype, device=device)
    scatter_out = torch.full_like(src, -1.0)

    # This option is essential: without it the kernel uses the pre-existing
    # native access lowering instead of Triton -> TensorView -> HIVM.
    tensor_view_gather_scatter_kernel[(1, )](
        src,
        indices,
        gather_out,
        scatter_out,
        COLS=COLS,
        TILE=TILE,
        use_tensor_view=True,
    )
    torch.npu.synchronize()

    index64 = indices_cpu.to(torch.int64)
    gather_ref = torch.index_select(src_cpu, 0, index64)
    scatter_ref = torch.full_like(src_cpu, -1.0)
    scatter_ref[index64, :] = gather_ref

    torch.testing.assert_close(gather_out.cpu(), gather_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scatter_out.cpu(), scatter_ref, rtol=0.0, atol=0.0)
