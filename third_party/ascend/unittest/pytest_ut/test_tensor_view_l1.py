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

# Exercise the FLIR TensorViewLowering path.

import re
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle as tle

from triton.runtime.cache import get_dump_manager


@triton.jit
def tensor_view_l1_matmul_kernel(a, b, c, M, N, K, STRIDED: tl.constexpr):
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    if STRIDED:
        a_view = tle.make_strided_view(a, shape=[M, K], strides=[K, 1], tile=[32, 32], traversal_strides=[48, 32],
                                       padding_value="zero")
        b_view = tle.make_strided_view(b, shape=[K, N], strides=[N, 1], tile=[32, 32], traversal_strides=[32, 48],
                                       padding_value="zero")
        step: tl.constexpr = 48
    else:
        a_view = tle.make_partition_view(a, shape=[M, K], strides=[K, 1], tile=[32, 32], padding_value="zero")
        b_view = tle.make_partition_view(b, shape=[K, N], strides=[N, 1], tile=[32, 32], padding_value="zero")
        step: tl.constexpr = 32
    acc = tl.zeros((32, 32), dtype=tl.float32)
    for k in range(tl.cdiv(K, 32)):
        lhs = tl.load(a_view, index=(row_block, k))
        rhs = tl.load(b_view, index=(k, column_block))
        acc = tl.dot(lhs, rhs, acc)
    rows = row_block * step + tl.arange(0, 32)
    columns = column_block * step + tl.arange(0, 32)
    tl.store(c + rows[:, None] * N + columns[None, :], acc.to(tl.float16),
             mask=(rows[:, None] < M) & (columns[None, :] < N))


@pytest.mark.parametrize("strided", [False, True], ids=["partition", "strided"])
@pytest.mark.parametrize("boundary", [False, True], ids=["full", "boundary"])
def test_tensor_view_l1(strided, boundary, tmp_path, monkeypatch):
    # Capture IR from this compilation even when the default disk cache is warm.
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path / "dump"))

    step = 48 if strided else 32
    m, n, k = (step + 19, step + 23, 45) if boundary else (step + 32, step + 32, 64)
    a = ((torch.arange(m * k).reshape(m, k) % 17 - 8).float() / 8).half()
    b = ((torch.arange(k * n).reshape(k, n) % 13 - 6).float() / 8).half()
    a_npu, b_npu = a.to("npu"), b.to("npu")
    actual = torch.zeros((m, n), dtype=torch.float16, device="npu")
    grid = (triton.cdiv(m, step), triton.cdiv(n, step))
    kernel = tensor_view_l1_matmul_kernel[grid](a_npu, b_npu, actual, m, n, k, STRIDED=strided,
                                                use_commonir_tensor_view=False, debug=True)

    reference = (a.float() @ b.float()).half()
    expected = torch.zeros_like(reference)
    # Strided tiles leave gaps. They must remain untouched in the output.
    for row in range(0, m, step):
        for column in range(0, n, step):
            expected[row:row + 32, column:column + 32] = reference[row:row + 32, column:column + 32]
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)

    dump = get_dump_manager(kernel.hash).get_file("kernel.npuir.mlir")
    assert dump is not None, "No backend debug IR was captured"
    ir = Path(dump).read_text()
    # This backend spells L1 as cbuf. Accept either spelling for other versions.
    transfer = re.search(
        r"hivm\.(?:hir\.)?(?:load|nd2nz)\b[^\n]*"
        r"ins\([^\n]*#hivm\.address_space<gm>[^\n]*"
        r"outs\([^\n]*#hivm\.address_space<(?:cbuf|l1)>", ir)
    assert transfer is not None, f"No GM-to-L1 load/nd2nz found; inspect {dump}"
