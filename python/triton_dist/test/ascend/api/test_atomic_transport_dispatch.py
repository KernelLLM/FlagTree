# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from types import SimpleNamespace

import pytest
import triton
import triton.language as tl

from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend import libaclshmem_device
from triton_dist.language.extra.ascend.aclshmem_constants import ACLSHMEMTransport
from shmem_api_test_utils import capture_extern
from test_atomic_ops import ATOMIC_OPS, INT_TYPES, FLOAT_TYPE, _atomic_api

pytestmark = [
    pytest.mark.ci,
    pytest.mark.dist,
    pytest.mark.usefixtures("ascend950_atomic_target"),
]

OPS = ATOMIC_OPS + [("add", 1, False)]
TRANSPORTS = [
    libshmem_device.ACLSHMEM_TRANSPORT_AUTO,
    libshmem_device.ACLSHMEM_TRANSPORT_UDMA,
    libshmem_device.ACLSHMEM_TRANSPORT_ROCE,
]


@pytest.mark.parametrize("op,num_values,returns_value", OPS)
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("dtype,suffix,ir_type", INT_TYPES + [FLOAT_TYPE])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_atomic_transport_dispatch(
    monkeypatch,
    op,
    num_values,
    returns_value,
    transport,
    dtype,
    suffix,
    ir_type,
    public,
    wrapped,
):
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    module = libshmem_device if public else libaclshmem_device
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(dtype))
    values = [object() for _ in range(num_values)]
    selector = tl.constexpr(transport) if wrapped else transport
    api, atomic_selector = _atomic_api(op)
    operands = values if num_values else [None]
    args = (
        [ptr, *values, 1]
        if atomic_selector is None
        else [ptr, *operands, 1, atomic_selector]
    )
    result = getattr(module, api)(
        *args,
        transport=selector,
        _semantic=semantic,
    )
    assert result == "extern_result"
    assert result.attrs == {"distributed.core_type": "vector"}
    assert len(calls) == 1
    args, signatures = calls[0]
    assert args == [ptr] + [("cast", v, dtype) for v in values] + [
        ("cast", 1, tl.int32)
    ]
    prefix = {
        ACLSHMEMTransport.AUTO: "aclshmem",
        ACLSHMEMTransport.UDMA: "aclshmemx_udma",
        ACLSHMEMTransport.ROCE: "aclshmemx_roce",
    }[transport]
    signature = (tl.pointer_type(dtype),) + (dtype,) * num_values + (tl.int32,)
    float_ops = {
        ACLSHMEMTransport.AUTO: {"set", "swap", "add"},
        ACLSHMEMTransport.UDMA: {"add"},
        ACLSHMEMTransport.ROCE: {"fetch", "set", "swap"},
    }
    if dtype == tl.float32 and op not in float_ops[transport]:
        assert signature not in signatures
    else:
        assert signatures[signature] == (
            f"{prefix}_atomic_{op}_{suffix}",
            dtype if returns_value else (),
        )


@pytest.mark.parametrize("transport", [-1, 4, "udma", None, True, object()])
def test_invalid_atomic_transport_emits_no_call(monkeypatch, transport):
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(tl.int32))
    with pytest.raises(ValueError, match="transport"):
        libaclshmem_device.atomic_fetch_op(
            ptr,
            1,
            0,
            libshmem_device.ACLSHMEM_AMO_ADD,
            transport=transport,
            _semantic=semantic,
        )
    assert not calls


def test_runtime_tensor_transport_rejected(monkeypatch):
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(tl.int32))
    with pytest.raises(ValueError, match="compile-time"):
        libaclshmem_device.atomic_fetch_op(
            ptr,
            1,
            0,
            libshmem_device.ACLSHMEM_AMO_ADD,
            transport=tl.tensor(None, tl.int32),
            _semantic=semantic,
        )
    assert not calls


@triton.jit
def _atomic_kernel(
    dest,
    result,
    pe,
    OP: tl.constexpr,
    SELECTOR: tl.constexpr,
    N: tl.constexpr,
    RETURNS: tl.constexpr,
    TRANSPORT: tl.constexpr,
):
    if N == 0:
        value = OP(dest, None, pe, SELECTOR, transport=TRANSPORT)
    elif N == 1:
        value = OP(dest, 1, pe, SELECTOR, transport=TRANSPORT)
    else:
        value = OP(dest, 0, 1, pe, transport=TRANSPORT)
    if RETURNS:
        tl.store(result, value)


@pytest.mark.parametrize("op,num_values,returns_value", OPS)
@pytest.mark.parametrize("transport", TRANSPORTS)
def test_atomic_transport_jit_symbol(op, num_values, returns_value, transport):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.ascend.compiler import AscendBackend
    from triton.compiler.compiler import ASTSource
    from test_gm_addr_args_indices import _run_pass

    target = GPUTarget("npu", "Ascend950DT_9582", 32)
    backend = AscendBackend(target)
    options = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(
        _atomic_kernel,
        signature={"dest": "*i32", "result": "*i32", "pe": "i32"},
        constexprs={
            "OP": getattr(libshmem_device, _atomic_api(op)[0]),
            "SELECTOR": _atomic_api(op)[1],
            "N": num_values,
            "RETURNS": returns_value,
            "TRANSPORT": transport,
        },
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        ctx,
    )
    prefix = {
        ACLSHMEMTransport.AUTO: "aclshmem",
        ACLSHMEMTransport.UDMA: "aclshmemx_udma",
        ACLSHMEMTransport.ROCE: "aclshmemx_roce",
    }[transport]
    lines = [
        line for line in module.str().splitlines() if "distributed.extern_call" in line
    ]
    assert len(lines) == 1
    assert 'distributed.core_type = "vector"' in lines[0]
    assert f'symbol = "{prefix}_atomic_{op}_int32"' in lines[0]
    # Selector is resolved in Python: only pointer, operands and PE reach the ABI.
    expected_args = "(!tt.ptr<i32>, " + ", ".join(["i32"] * (num_values + 1)) + ")"
    assert expected_args in lines[0]

    _run_pass(module, ctx)
    custom_calls = [line for line in module.str().splitlines() if "hir.custom" in line]
    assert len(custom_calls) == 1
    assert f'symbol = "{prefix}_atomic_{op}_int32"' in custom_calls[0]
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in custom_calls[0]
    assert "gm_addr_args_indices = array<i32: 0>" in custom_calls[0]


@pytest.mark.parametrize(
    "op,num_values,returns_value",
    [
        ("fetch", 0, True),
        ("set", 1, False),
        ("swap", 1, True),
        ("compare_swap", 2, True),
        ("inc", 0, False),
        ("add", 1, False),
        ("fetch_inc", 0, True),
        ("fetch_add", 1, True),
    ],
)
@pytest.mark.parametrize("dtype", [tl.uint32, tl.uint64])
def test_mte_atomic_dispatch(monkeypatch, op, num_values, returns_value, dtype):
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(dtype))
    api, selector = _atomic_api(op)
    values = [1] * num_values
    args = (
        [ptr, *values, 2]
        if selector is None
        else [ptr, *(values or [None]), 2, selector]
    )
    getattr(libaclshmem_device, api)(
        *args, transport=ACLSHMEMTransport.MTE, _semantic=semantic
    )
    signature = (ptr.dtype,) + (dtype,) * num_values + (tl.int32,)
    assert calls[0][1][signature] == (
        f"aclshmemx_mte_atomic_{op}_{dtype.name}",
        dtype if returns_value else (),
    )


@pytest.mark.parametrize(
    "op",
    [
        libshmem_device.ACLSHMEM_AMO_AND,
        libshmem_device.ACLSHMEM_AMO_OR,
        libshmem_device.ACLSHMEM_AMO_XOR,
    ],
)
@pytest.mark.parametrize("fetch", [False, True])
def test_mte_bitwise_atomic_rejected(monkeypatch, op, fetch):
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(tl.uint32))
    with pytest.raises(ValueError, match="MTE transport is not supported"):
        api = (
            libaclshmem_device.atomic_fetch_op
            if fetch
            else libaclshmem_device.atomic_op
        )
        api(ptr, 1, 2, op, transport=ACLSHMEMTransport.MTE, _semantic=semantic)
    assert not calls


@pytest.mark.parametrize("is_950", [False, True])
@pytest.mark.parametrize("transport", list(ACLSHMEMTransport))
@pytest.mark.parametrize(
    "dtype",
    [
        tl.int8,
        tl.int16,
        tl.int32,
        tl.uint32,
        tl.int64,
        tl.uint64,
        tl.float16,
        tl.bfloat16,
        tl.float32,
    ],
)
@pytest.mark.parametrize("op", ["add", "fetch_add", "swap", "inc"])
def test_atomic_architecture_validation(monkeypatch, is_950, transport, dtype, op):
    from triton.backends.ascend import utils

    monkeypatch.setattr(utils, "is_compile_on_910_95", lambda: is_950)
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(dtype))
    api, selector = _atomic_api(op)
    value = None if op == "inc" else 1
    supported_before_950 = (
        op == "add"
        and transport in (ACLSHMEMTransport.AUTO, ACLSHMEMTransport.MTE)
        and dtype in (tl.int8, tl.int16, tl.int32, tl.float16, tl.bfloat16, tl.float32)
    )
    if not is_950 and not supported_before_950:
        with pytest.raises(ValueError, match="requires Ascend950"):
            getattr(libaclshmem_device, api)(
                ptr, value, 1, selector, transport=transport, _semantic=semantic
            )
        assert not calls
    else:
        # capture_extern checks architecture selection; the existing dtype ABI
        # tests separately check which types each op/transport admits on 950.
        getattr(libaclshmem_device, api)(
            ptr, value, 1, selector, transport=transport, _semantic=semantic
        )
        assert len(calls) == 1
