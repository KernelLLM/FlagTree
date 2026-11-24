# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the MIT License.
#
# This test validates that distributed operations are assigned the correct core
# types after conversion to hivm.custom, including explicit attributes and
# kernel-level fallback inference.
# Pure IR-level test, does not require NPU hardware.
#
# Note: This test need NPU driver, beccause libshmem_device will check npu-smi

import os
import re
from typing import Any, Callable, Optional

import pytest
from triton._C import libtriton
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
import triton.language as tl
from triton_dist.language.extra import libshmem_device


def _make_context():
    """Construct an MLIRContext with all relevant dialects loaded."""
    ir = libtriton.ir
    distributed = libtriton.distributed
    ascend = libtriton.ascend
    ctx = ir.context()
    disable_multithreading = getattr(ctx, "disable_multithreading", None)
    if disable_multithreading is not None:
        disable_multithreading()
    ir.load_dialects(ctx)
    distributed.ir.load_dialects(ctx)
    ascend.ir.load_dialects(ctx)
    ascend.load_dialects(ctx)
    return ctx


g_ctx = _make_context()


def _make_ir(kernel, param_types=None):
    target = GPUTarget("npu", "Ascend910B1", 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    source = ASTSource(kernel, param_types or {}, {})
    return source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        g_ctx,
    )


def _collect_extern_calls(module):
    extern_calls = []

    def collect(op):
        if op.get_name() == "distributed.extern_call":
            extern_calls.append(
                (op.get_str_attr("symbol"), op.get_str_attr("distributed.core_type"))
            )

    module.walk(collect)
    return extern_calls


def _run_pass(module):
    """Run the distributed -> hivm conversion pass on the module."""
    pm = libtriton.ir.pass_manager(g_ctx)
    libtriton.distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(
        pm
    )
    previous = os.environ.get("MLIR_DISABLE_MULTITHREADING")
    os.environ["MLIR_DISABLE_MULTITHREADING"] = "1"
    try:
        pm.run(module, "")
    finally:
        if previous is None:
            os.environ.pop("MLIR_DISABLE_MULTITHREADING", None)
        else:
            os.environ["MLIR_DISABLE_MULTITHREADING"] = previous


def _extract_core_types(module_str):
    """Extract symbol and core type from every hivm.hir.custom op."""
    results = []
    for line in module_str.splitlines():
        if "hivm.hir.custom" not in line:
            continue
        sym_match = re.search(r'symbol = "([^"]+)"', line)
        core_match = re.search(r"hivm\.tcore_type = #hivm\.tcore_type<(\w+)>", line)
        symbol = sym_match.group(1) if sym_match else "?"
        core_type = core_match.group(1) if core_match else None
        results.append((symbol, core_type))
    return results


def _assert_coretypes_in_ir(expected_types, module_str):
    core_types = _extract_core_types(module_str)
    assert core_types, "No hivm.custom operators found after conversion"
    by_symbol = {symbol: core_type for symbol, core_type in core_types}

    for symbol, expected in expected_types.items():
        assert symbol in by_symbol, f"{symbol} not found in converted IR: {by_symbol}"
        assert by_symbol[symbol] == expected, (
            f"{symbol} should have {expected} core type, "
            f"actual: {by_symbol[symbol]}"
        )


def _assert_two_stages(
    kernel: Callable[..., Any],
    param_types: dict[str, str],
    expected_extern_calls: list[tuple[str, Optional[str]]],
    expected_types: dict[str, str],
) -> None:
    """Assert frontend core-type attributes and converted HIVM core types.

    Args:
        kernel: JIT-compiled Triton kernel to lower.
        param_types: Mapping from kernel parameter names to Triton type strings.
        expected_extern_calls: Expected ``(symbol, distributed.core_type)`` pairs in
            the generated distributed IR.
        expected_types: Expected ``(symbol, core_type)`` pairs after
            distributed-to-HIVM conversion.
    """
    module = _make_ir(kernel, param_types)
    ttir = module.str()
    assert _collect_extern_calls(module) == expected_extern_calls, ttir

    _run_pass(module)
    _assert_coretypes_in_ir(expected_types, module.str())


@triton.jit
def _barrier_kernel():
    libshmem_device.barrier_all()
    libshmem_device.barrier(0)


@pytest.mark.ci
@pytest.mark.dist
def test_barrier_core_type():
    expected_symbols = [
        "aclshmem_barrier_all",
        "aclshmem_barrier",
    ]
    _assert_two_stages(
        _barrier_kernel,
        {},
        [(symbol, "cube_and_vector") for symbol in expected_symbols],
        {symbol: "CUBE_AND_VECTOR" for symbol in expected_symbols},
    )


@triton.jit
def _barrier_vec_kernel():
    libshmem_device.barrier_all_vec()
    libshmem_device.barrier_vec(0)


@pytest.mark.ci
@pytest.mark.dist
def test_barrier_vec_core_type():
    expected_symbols = [
        "aclshmemx_barrier_all_vec",
        "aclshmemx_barrier_vec",
    ]
    _assert_two_stages(
        _barrier_vec_kernel,
        {},
        [(symbol, "vector") for symbol in expected_symbols],
        {symbol: "VECTOR" for symbol in expected_symbols},
    )


@triton.jit
def _rma_ops_kernel(dest, source, sig_addr):
    libshmem_device.putmem(dest, source, 4, 0)
    libshmem_device.getmem(dest, source, 4, 0)
    libshmem_device.putmem_nbi(dest, source, 4, 0)
    libshmem_device.getmem_nbi(dest, source, 4, 0)
    libshmem_device.putmem_signal(
        dest,
        source,
        4,
        sig_addr,
        1,
        libshmem_device.ACLSHMEM_SIGNAL_SET,
        0,
    )
    libshmem_device.putmem_signal_nbi(
        dest,
        source,
        4,
        sig_addr,
        1,
        libshmem_device.ACLSHMEM_SIGNAL_SET,
        0,
    )


@pytest.mark.ci
@pytest.mark.dist
def test_rma_ops_core_types():
    expected_symbols = [
        "aclshmem_putmem_float",
        "aclshmem_getmem_float",
        "aclshmem_putmem_nbi_float",
        "aclshmem_getmem_nbi_float",
        "aclshmem_putmem_signal_float",
        "aclshmem_putmem_signal_nbi_float",
    ]
    _assert_two_stages(
        _rma_ops_kernel,
        {
            "dest": "*fp32",
            "source": "*fp32",
            "sig_addr": "*i32",
        },
        [(symbol, "vector") for symbol in expected_symbols],
        {symbol: "VECTOR" for symbol in expected_symbols},
    )


@triton.jit
def _downstream_infer_ops_kernel(int_dest):
    libshmem_device.int_p(int_dest, 1, 0)
    libshmem_device.signal_op(
        int_dest,
        1,
        libshmem_device.ACLSHMEM_SIGNAL_SET,
        0,
    )
    libshmem_device.signal_wait_until(
        int_dest,
        libshmem_device.ACLSHMEM_CMP_EQ,
        1,
    )


@pytest.mark.ci
@pytest.mark.dist
def test_downstream_infer_ops_core_types():
    expected_symbols = [
        "aclshmem_int32_p",
        "aclshmemx_signal_op",
        "aclshmem_signal_wait_until",
    ]
    _assert_two_stages(
        _downstream_infer_ops_kernel,
        {"int_dest": "*i32"},
        [(symbol, "cube_or_vector") for symbol in expected_symbols],
        {symbol: "VECTOR" for symbol in expected_symbols},
    )


@triton.jit
def _cube_and_vector_op_in_pure_vector_kernel(dest, source):
    offsets = tl.arange(0, 16)
    values = tl.load(source + offsets)
    tl.store(dest + offsets, values)
    libshmem_device.barrier_all()


@pytest.mark.ci
@pytest.mark.dist
def test_cube_and_vector_infer_override():
    """Verify explicit CUBE_AND_VECTOR overrides pure-vector inference."""
    _assert_two_stages(
        _cube_and_vector_op_in_pure_vector_kernel,
        {"dest": "*fp32", "source": "*fp32"},
        [("aclshmem_barrier_all", "cube_and_vector")],
        {"aclshmem_barrier_all": "CUBE_AND_VECTOR"},
    )


@triton.jit
def _vector_op_in_cube_and_vector_kernel(dest, source):
    offsets = tl.arange(0, 16)
    matrix_offsets = offsets[:, None] * 16 + offsets[None, :]
    a = tl.load(source + matrix_offsets).to(tl.float16)
    b = tl.load(source + matrix_offsets).to(tl.float16)
    result = tl.dot(a, b)
    tl.store(dest + matrix_offsets, result.to(tl.float32))
    libshmem_device.barrier_all_vec()


@pytest.mark.ci
@pytest.mark.dist
def test_vector_infer_override():
    """Verify explicit VECTOR overrides mixed-kernel inference."""
    _assert_two_stages(
        _vector_op_in_cube_and_vector_kernel,
        {"dest": "*fp32", "source": "*fp32"},
        [("aclshmemx_barrier_all_vec", "vector")],
        {"aclshmemx_barrier_all_vec": "VECTOR"},
    )


@triton.jit
def _cube_or_vector_op_in_pure_vector_kernel(dest, source):
    offsets = tl.arange(0, 16)
    values = tl.load(source + offsets)
    tl.store(dest + offsets, values)
    libshmem_device.quiet()


@pytest.mark.ci
@pytest.mark.dist
def test_cube_or_vector_infer_fallback_to_vector():
    """Verify cube_or_vector falls back to VECTOR without ``tl.dot``."""
    _assert_two_stages(
        _cube_or_vector_op_in_pure_vector_kernel,
        {"dest": "*fp32", "source": "*fp32"},
        [("aclshmem_quiet", "cube_or_vector")],
        {"aclshmem_quiet": "VECTOR"},
    )


@triton.jit
def _cube_or_vector_op_in_cube_and_vector_kernel(dest, source):
    offsets = tl.arange(0, 16)
    matrix_offsets = offsets[:, None] * 16 + offsets[None, :]
    a = tl.load(source + matrix_offsets).to(tl.float16)
    b = tl.load(source + matrix_offsets).to(tl.float16)
    result = tl.dot(a, b)
    tl.store(dest + matrix_offsets, result.to(tl.float32))
    libshmem_device.quiet()


@pytest.mark.ci
@pytest.mark.dist
def test_cube_or_vector_infer_fallback_to_cube_and_vector():
    """Verify cube_or_vector falls back to CUBE_AND_VECTOR with ``tl.dot``."""
    _assert_two_stages(
        _cube_or_vector_op_in_cube_and_vector_kernel,
        {"dest": "*fp32", "source": "*fp32"},
        [("aclshmem_quiet", "cube_or_vector")],
        {"aclshmem_quiet": "CUBE_AND_VECTOR"},
    )
