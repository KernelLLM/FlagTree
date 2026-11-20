# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Relay frontend tests.

The default NPU-IR package does not include relay support.
Execution of the emitted symbols requires relay-enabled templates and SHMEM.
These tests validate dispatch and lowering without executing device code.
"""

import inspect
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
from shmem_api_test_utils import capture_extern

from triton_dist.language.extra import libshmem_device as shmem
from triton_dist.language.extra.ascend import libaclshmem_device as backend

pytestmark = [pytest.mark.ci, pytest.mark.dist]


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("dtype", [tl.uint8, tl.float32])
def test_shared_relay_abi(monkeypatch, op, dtype):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    semantic, calls = capture_extern(monkeypatch)
    dst = SimpleNamespace(dtype=tl.pointer_type(dtype))
    src = SimpleNamespace(dtype=tl.pointer_type(dtype))
    fn = getattr(shmem, f"{op}mem_nbi_relay")
    assert not {"qp_id", "transport", "scratch", "sync_id"} & set(
        inspect.signature(fn).parameters
    )
    result = fn(dst, src, 7, 2, 3, _semantic=semantic)
    ptr = tl.pointer_type(dtype)
    suffix = backend.DTYPE_TO_KERNEL_SUFFIX[dtype.name]
    assert len(calls) == 1
    args, signatures = calls[0]
    assert args == [
        dst,
        src,
        ("cast", 7, tl.uint32),
        ("cast", 2, tl.int32),
        ("cast", 3, tl.int32),
    ]
    assert signatures == {
        (ptr, ptr, tl.uint32, tl.int32, tl.int32): (
            f"aclshmemx_udma_relay_{op}_nbi_{suffix}",
            (),
        )
    }
    assert result.attrs == {"distributed.core_type": "vector"}


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize(
    "kwargs,error", [({"qp_id": 0}, TypeError), ({"relay_pe": None}, ValueError)]
)
def test_shared_relay_rejects_invalid_arguments(monkeypatch, op, kwargs, error):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    semantic, calls = capture_extern(monkeypatch)
    ptr = SimpleNamespace(dtype=tl.pointer_type(tl.uint8))
    args = {"relay_pe": 3, **kwargs}
    with pytest.raises(error):
        getattr(shmem, f"{op}mem_nbi_relay")(ptr, ptr, 7, 2, **args, _semantic=semantic)
    assert not calls


@triton.jit
def _relay_kernel(dst, src, size, peer, relay_peer, IS_GET: tl.constexpr):
    if IS_GET:
        shmem.getmem_nbi_relay(dst, src, size, peer, relay_peer)
    else:
        shmem.putmem_nbi_relay(dst, src, size, peer, relay_peer)


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("payload_type", ["*u8", "*fp32"])
def test_shared_relay_jit_lowering(monkeypatch, op, payload_type):
    from test_gm_addr_args_indices import _run_pass
    from triton._C.libtriton import ir
    from triton.backends.ascend.compiler import AscendBackend
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    target = GPUTarget("npu", "Ascend950DT_9582", 32)
    compiler = AscendBackend(target)
    options = compiler.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    compiler.load_dialects(ctx)
    source = ASTSource(
        _relay_kernel,
        signature={
            "dst": payload_type,
            "src": payload_type,
            "size": "u32",
            "peer": "i32",
            "relay_peer": "i32",
        },
        constexprs={"IS_GET": op == "get"},
    )
    module = source.make_ir(
        target,
        options,
        compiler.get_codegen_implementation(options),
        compiler.get_module_map(),
        ctx,
    )
    _run_pass(module, ctx)
    calls = [line for line in module.str().splitlines() if "hir.custom" in line]
    assert len(calls) == 1
    suffix = "uint8" if payload_type == "*u8" else "float"
    assert f'symbol = "aclshmemx_udma_relay_{op}_nbi_{suffix}"' in calls[0]
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in calls[0]
    assert "gm_addr_args_indices = array<i32: 0, 1>" in calls[0]
