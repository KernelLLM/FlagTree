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
"""Shared ``codegen_fns`` for tests that drive ``ast_to_ttir`` directly.

``CodeGenerator`` builds a plain ``ir.builder``; the ascend and buffer builder
methods (``get_target_attribute``, ``is_910_95``, ``_ascend_builder``, ...) are
bridged onto it by the backend's ``initialize_builder`` codegen function. Tests
that bypass ``AscendBackend`` and call ``ast_to_ttir`` themselves must supply
that function too, otherwise codegen fails with::

    'triton._C.libtriton.ir.builder' object has no attribute 'get_target_attribute'

This module keeps that single source of truth. It deliberately does not import
``test_common``, which pulls in ``torch_npu``: these callers compile to MLIR
only and need no NPU runtime.
"""

from triton.backends.ascend.compiler import _initialize_ascend_builder

__all__ = ["ascend_codegen_fns"]


def ascend_codegen_fns(**extra):
    """Return ``codegen_fns`` wired for the ascend backend.

    Keyword arguments are merged in, so a caller needing the address-space
    helper writes::

        ascend_codegen_fns(create_address_space=al.semantic.create_address_space)
    """
    fns = {"initialize_builder": _initialize_ascend_builder}
    fns.update(extra)
    return fns
