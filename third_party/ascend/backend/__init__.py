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
__version__ = '3.6.0'

import logging
from triton._C.libtriton.ascend import ir as ascend_ir

from .testing import do_bench_npu


def _apply_ascend_patch():
    from triton.compiler.code_generator import CodeGenerator

    if not getattr(CodeGenerator, "_ascend_patch_applied", False):
        _original_cg_init = CodeGenerator.__init__

        def _patched_cg_init(self, *args, **kwargs):
            """
            Monkey Patch for Ascend:
            Injects 'hacc.target' attribute into the module after initilization of module.
            """
            _original_cg_init(self, *args, **kwargs)
            options = self.builder.options
            context = self.context
            if hasattr(options, "arch") and options.arch:
                try:
                    builder = ascend_ir.ascendnpu_ir_builder(context, options.arch)

                    target_attr_str = f'#hacc.target<"{options.arch}">'
                    self.module.set_attr("hacc.target", builder.parse_attr(target_attr_str))
                except Exception as e:
                    logging.warning(f"[Ascend Patch] Failed to set hacc.target: {e}")

        CodeGenerator.__init__ = _patched_cg_init
        CodeGenerator._ascend_patch_applied = True

    # Patch compiler.parse to support Ascend-specific IR extensions
    # for ir_override and TRITON_KERNEL_OVERRIDE features.
    from triton.compiler import compiler as _compiler_module

    if not getattr(_compiler_module, "_ascend_parse_patch_applied", False):
        from pathlib import Path as _Path
        from triton._C.libtriton import ir as _ir

        # Keep the original community logic intact, adding Ascend extensions
        # to the existing text/binary branches.
        def _patched_parse(full_name, ext, context):
            if ext == "ttir" or ext == "ttgir":
                module = _ir.parse_mlir_module(full_name, context)
                module.context = context
                return module
            if ext in ("llir", "ptx", "amdgcn", "ttadapter", "bcmlir"):
                return _Path(full_name).read_text()
            if ext in ("cubin", "hsaco", "mlirbc", "npubin"):
                return _Path(full_name).read_bytes()

        _compiler_module.parse = _patched_parse
        _compiler_module._ascend_parse_patch_applied = True

    # Patch TritonSemantic.dot for Ascend-specific HF32 guard and
    # max_num_imprecise_acc warning.
    from triton.language.semantic import TritonSemantic

    if not getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
        _original_dot = TritonSemantic.dot

        def _patched_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
            """
            Monkey Patch for Ascend:
            - HF32 precision only works for fp32 x fp32.
              When either input is not fp32, silently fall back
              to default precision (ieee).
            - Warn when max_num_imprecise_acc is explicitly set, since
              Ascend NPU does not support imprecise accumulation.
            """
            # Ascend does not support tf32, tf32 is converted to hf32. Avoid tf32 verification interception
            if input_precision == "tf32":
                input_precision = "hf32"
            # Ascend NPU does not support imprecise accumulation.
            # Force max_num_imprecise_acc to None so the upstream None
            # branch handles it (via max_num_imprecise_acc_default = 0),
            # avoiding the fp8 ValueError path which is NVIDIA-only.
            if max_num_imprecise_acc is not None:
                print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. "
                      "Thus it is ignored.")
                max_num_imprecise_acc = None

            return _original_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype)

        TritonSemantic.dot = _patched_dot
        TritonSemantic._ascend_dot_patch_applied = True

    # Install the unified, non-intrusive dtype interception for operators
    # whose dtypes are unsupported on Ascend. The rules live in
    # triton.backends.ascend.op_dtype_config, next to this backend.
    if not getattr(CodeGenerator, "_ascend_dtype_guard_applied", False):
        try:
            from triton.backends.ascend.dtype_guard import install_dtype_guard

            install_dtype_guard()
            CodeGenerator._ascend_dtype_guard_applied = True
        except Exception as e:
            logging.warning(f"[Ascend Patch] Failed to install dtype guard: {e}")

    # Hand the math builtins' dtype admission over to the Ascend rules.
    #
    # ``triton.language.math`` wraps these builtins in ``_check_dtype``, whose
    # whitelist describes libdevice on GPUs. It is both redundant and wrong
    # here: ``exp`` is listed as ``["fp32", "fp64"]`` while Ascend supports
    # fp16 and rejects fp64 (see the ``exp`` rule in op_dtype_config), so
    # ``tl.exp`` on an fp16 tensor is refused by the core whitelist before the
    # backend ever sees it. The core check is also bypassed for the member-call
    # form: ``_tensor_member_fn`` binds the undecorated function onto
    # ``tensor``, so ``x.exp()`` already compiles on fp16 while ``tl.exp(x)``
    # does not.
    #
    # Unwrapping ``_check_dtype`` removes that inconsistent second gate and
    # leaves dtype admission to the single Ascend source of truth. dtype_guard
    # intercepts both the named-builtin and the member-call form, and keeps
    # rejecting every dtype listed as unsupported (fp64 included). Note that
    # its decision is a blacklist: ``validate`` only rejects dtypes present in
    # ``unsupported``, and the ``supported`` list merely formats the message.
    # So for ops whose rule omits fp16 from both lists (``log``, ``sqrt``,
    # ``floor``, ``erf``), unwrapping does admit fp16. Add fp16 to those rules'
    # ``unsupported`` list if the hardware cannot run them, rather than relying
    # on the core whitelist to reject them as a side effect.
    #
    # Only builtins that have an Ascend rule are unwrapped, so every op keeps
    # at least one gate; ``div_rn`` has no rule and therefore keeps the core
    # check.
    if not getattr(CodeGenerator, "_ascend_math_dtype_unwrapped", False):
        try:
            import functools
            from triton import language as _language
            from triton.language import core as _core
            from triton.language import math as _math
            from triton.backends.ascend.op_dtype_config import RULES_BY_NAME

            def _unwrap_check_dtype(fn):
                """Return the builtin with the ``_check_dtype`` layer removed.

                ``math`` builds each op as ``core.builtin(_check_dtype(...)(
                _add_math_*_docstr(...)(fn)))``. Only the ``_check_dtype`` frame
                is dropped; the ``@core.builtin`` contract (the ``_semantic``
                keyword requirement and the ``TRITON_BUILTIN`` marker the AST
                tracer dispatches on) is rebuilt around the inner function.
                Builtins without both wrapper layers (``fma``) are left alone.
                """
                inner = getattr(fn, "__wrapped__", None)
                if inner is None:
                    return None
                inner = getattr(inner, "__wrapped__", None)
                if inner is None:
                    return None

                @functools.wraps(inner)
                def unchecked(*args, **kwargs):
                    if kwargs.get("_semantic") is None:
                        raise ValueError("Did you forget to add @triton.jit ? "
                                         "(`_semantic` argument must be provided "
                                         "outside of JIT functions.)")
                    return inner(*args, **kwargs)

                setattr(unchecked, _core.TRITON_BUILTIN, True)
                return unchecked

            for _name in sorted(set(dir(_math))):
                if f"triton.language.{_name}" not in RULES_BY_NAME:
                    continue
                _original = getattr(_math, _name, None)
                if not callable(_original) or not getattr(_original, _core.TRITON_BUILTIN, False):
                    continue
                _unchecked = _unwrap_check_dtype(_original)
                if _unchecked is None:
                    continue
                # ``triton.language`` re-exports the same object; rebind both so
                # ``tl.exp`` and ``triton.language.math.exp`` stay identical.
                setattr(_math, _name, _unchecked)
                if getattr(_language, _name, None) is _original:
                    setattr(_language, _name, _unchecked)

            CodeGenerator._ascend_math_dtype_unwrapped = True
        except Exception as e:
            logging.warning(f"[Ascend Patch] Failed to defer math dtype checks: {e}")


__all__ = ["do_bench_npu"]
