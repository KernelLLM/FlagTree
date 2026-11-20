# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import triton
import triton.compiler

from triton.backends.ascend import compiler as ascend_compiler
from triton.runtime import driver
from triton.backends.ascend.compiler import (
    AscendBackend,
    _apply_o0_pipeline_options,
    _compile_linalg_to_npu_bin,
)


def _make_backend():
    backend = object.__new__(AscendBackend)
    backend.target = SimpleNamespace(backend="npu", arch="test-arch")
    return backend


@pytest.mark.backend("mock")
@pytest.mark.parametrize("enable_o0_pipeline", [False, True])
def test_o0_pipeline_from_kernel_launch_reaches_compilation(monkeypatch, enable_o0_pipeline):
    monkeypatch.setattr(ascend_compiler, "_ENABLE_O0_PIPELINE", None)
    backend = _make_backend()
    fake_driver = SimpleNamespace(
        get_current_device=lambda: 0,
        get_current_stream=lambda device: 0,
        get_current_target=lambda: backend.target,
    )
    monkeypatch.setattr(driver, "_active", fake_driver)
    monkeypatch.setattr(triton.compiler, "make_backend", lambda target: backend)

    @triton.jit
    def kernel():
        pass

    captured_options = []

    def capture_compile(key, signature, device, constexprs, options, attrs, warmup):
        captured_options.append(options)
        # Stop before compilation and device execution; JIT.run handles None.
        return None

    monkeypatch.setattr(kernel, "_do_compile", capture_compile)

    kernel[(1,)](enable_o0_pipeline=enable_o0_pipeline)

    assert len(captured_options) == 1
    assert captured_options[0].enable_o0_pipeline is enable_o0_pipeline


def test_o0_pipeline_from_environment_reaches_npu_options(monkeypatch):
    monkeypatch.setattr(ascend_compiler, "_ENABLE_O0_PIPELINE", "1")
    options = _make_backend().parse_options({})

    assert options.enable_o0_pipeline is True
    # Per-pass switches stay unset until the pipeline expands them.
    assert options.disable_cv_pipelining is None
    assert options.disable_reduce_rank_subview is None
    assert options.disable_inline_load_copy is None
    assert options.disable_multi_buffer is None


def test_o0_pipeline_expands_to_per_pass_switches():
    metadata = {}
    _apply_o0_pipeline_options(metadata)

    assert metadata == {
        "disable_cv_pipelining": True,
        "disable_reduce_rank_subview": True,
        "disable_inline_load_copy": True,
        "disable_multi_buffer": True,
        "enable_preload": False,
        "enable_ubuf_saving": False,
    }


def test_o0_pipeline_rejects_910_95_path():
    options = SimpleNamespace(
        compile_on_910_95=True,
        enable_o0_pipeline=True,
    )

    with pytest.raises(ValueError, match="not supported on the 910_95"):
        _compile_linalg_to_npu_bin("", {}, options)


def test_o0_pipeline_applies_disables_in_compile(monkeypatch):
    options = SimpleNamespace(compile_on_910_95=False, enable_o0_pipeline=True)

    captured = {}

    def fake_A2_A3(linalg, metadata, opt):
        captured.update(metadata)
        return None

    monkeypatch.setattr(ascend_compiler, "linalg_to_bin_enable_npu_compile_A2_A3", fake_A2_A3)

    _compile_linalg_to_npu_bin("", {}, options)

    assert captured["disable_cv_pipelining"] is True
    assert captured["disable_reduce_rank_subview"] is True
    assert captured["disable_inline_load_copy"] is True
    assert captured["disable_multi_buffer"] is True
    assert captured["enable_preload"] is False
    assert captured["enable_ubuf_saving"] is False
