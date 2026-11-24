# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import triton.language as tl

from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend import libaclshmem_device

from shmem_api_test_utils import (
    check_extern_lowering,
    check_scalar_extern_abi,
    check_unsupported_dtype,
)
from test_atomic_ops import ATOMIC_OPS, FLOAT_TYPE, INT_TYPES, _atomic_api

pytestmark = [
    pytest.mark.ci,
    pytest.mark.dist,
    pytest.mark.usefixtures("ascend950_atomic_target"),
]

TRANSPORT_OPS = ATOMIC_OPS + [("add", 1, False)]


def _transport_types(transport, op):
    float_ops = {"roce": ("fetch", "set", "swap"), "udma": ("add",)}
    return INT_TYPES + ([FLOAT_TYPE] if op in float_ops[transport] else [])


TRANSPORT_CASES = [
    (transport, op, num_values, returns_value, dtype, suffix, ir_type)
    for transport in ("roce", "udma")
    for op, num_values, returns_value in TRANSPORT_OPS
    for dtype, suffix, ir_type in _transport_types(transport, op)
]


@pytest.mark.parametrize(
    "transport,op,num_values,returns_value,dtype,suffix,ir_type", TRANSPORT_CASES
)
@pytest.mark.parametrize("public", [False, True])
def test_transport_atomic_extern_abi(
    monkeypatch,
    transport,
    op,
    num_values,
    returns_value,
    dtype,
    suffix,
    ir_type,
    public,
):
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    module = libshmem_device if public else libaclshmem_device
    type_map = {
        (tl.pointer_type(ty),)
        + (ty,) * num_values
        + (tl.int32,): (
            f"aclshmemx_{transport}_atomic_{op}_{name}",
            ty if returns_value else (),
        )
        for ty, name, _ in _transport_types(transport, op)
    }
    check_scalar_extern_abi(
        monkeypatch,
        module,
        _atomic_api(op)[0],
        f"aclshmemx_{transport}_atomic_{op}_{suffix}",
        dtype,
        num_values,
        type_map,
        op=_atomic_api(op)[1],
        transport=getattr(libshmem_device, f"ACLSHMEM_TRANSPORT_{transport.upper()}"),
    )


@pytest.mark.parametrize(
    "transport,op,num_values,dtype",
    [
        (transport, op, num_values, dtype)
        for transport in ("roce", "udma")
        for op, num_values, _ in TRANSPORT_OPS
        for dtype in (tl.float16, tl.float32, tl.float64, tl.int8, tl.uint16)
        if dtype not in [ty for ty, _, _ in _transport_types(transport, op)]
    ],
)
def test_transport_atomic_rejects_unsupported_dtype(
    monkeypatch, transport, op, num_values, dtype
):
    check_unsupported_dtype(
        monkeypatch,
        _atomic_api(op)[0],
        dtype,
        num_values,
        op=_atomic_api(op)[1],
        transport=getattr(libshmem_device, f"ACLSHMEM_TRANSPORT_{transport.upper()}"),
    )


@pytest.mark.parametrize(
    "transport,op,num_values,returns_value,dtype,suffix,ir_type", TRANSPORT_CASES
)
@pytest.mark.parametrize("mixed", [False, True])
def test_transport_atomic_lowering(
    transport, op, num_values, returns_value, dtype, suffix, ir_type, mixed
):
    check_extern_lowering(
        f"aclshmemx_{transport}_atomic_{op}_{suffix}",
        [f"!tt.ptr<{ir_type}>"] + [ir_type] * num_values + ["i32"],
        ir_type if returns_value else None,
        "VECTOR",
        False,
        mixed,
    )
