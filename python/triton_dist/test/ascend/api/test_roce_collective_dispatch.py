# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""RoCE frontend dispatch and lowering coverage; no hardware required."""

import inspect

import pytest
import triton
import triton.language as tl
from shmem_api_test_utils import capture_extern
from triton_dist.language.extra import libshmem_device as shmem
from triton_dist.language.extra.ascend import libaclshmem_device as backend

pytestmark = [pytest.mark.ci, pytest.mark.dist]

CASES = [
    ("sync", "team_sync", True),
    ("sync_all", "sync_all", False),
    ("barrier", "barrier", True),
    ("barrier_all", "barrier_all", False),
    ("barrier_vec", "barrier", True),
    ("barrier_all_vec", "barrier_all", False),
]


def _transport_api(name):
    return name + "_roce" if name.startswith("barrier") else name


@pytest.mark.parametrize("name,symbol,has_team", CASES)
@pytest.mark.parametrize("public", [False, True])
def test_roce_collective_abi(monkeypatch, name, symbol, has_team, public):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    semantic, calls = capture_extern(monkeypatch)
    module = shmem if public else backend
    kwargs = (
        {}
        if name.startswith("barrier")
        else {"transport": shmem.ACLSHMEM_TRANSPORT_ROCE}
    )
    result = getattr(module, _transport_api(name))(
        *([0] if has_team else []),
        **kwargs,
        _semantic=semantic,
    )
    assert len(calls) == 1
    args, signatures = calls[0]
    assert args == ([("cast", 0, tl.int32)] if has_team else [])
    assert signatures == {
        (tl.int32,) if has_team else (): (
            f"aclshmemx_roce_{symbol}",
            tl.int32 if has_team else (),
        )
    }
    assert result.attrs == {"distributed.core_type": "vector"}


@pytest.mark.parametrize("name,symbol,has_team", CASES)
def test_generic_collective_default(monkeypatch, name, symbol, has_team):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    semantic, calls = capture_extern(monkeypatch)
    result = getattr(shmem, name)(*([0] if has_team else []), _semantic=semantic)
    generic = "sync" if symbol == "team_sync" else symbol
    is_vec = name.endswith("_vec")
    expected_symbol = f"aclshmemx_{name}" if is_vec else f"aclshmem_{generic}"
    assert list(calls[0][1].values()) == [(expected_symbol, ())]
    assert result.attrs == {
        "distributed.core_type": "vector" if is_vec else "cube_and_vector"
    }


@pytest.mark.parametrize("name,symbol,has_team", CASES[:2])
@pytest.mark.parametrize(
    "transport", [shmem.ACLSHMEM_TRANSPORT_UDMA, shmem.ACLSHMEM_TRANSPORT_MTE]
)
def test_unsupported_collective_rejected(
    monkeypatch, name, symbol, has_team, transport
):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match=f"{transport.name} collective"):
        getattr(backend, _transport_api(name))(
            *([0] if has_team else []),
            transport=transport,
            _semantic=semantic,
        )
    assert not calls


@triton.jit
def _collective_kernel(
    team, OP: tl.constexpr, HAS_TEAM: tl.constexpr, IS_BARRIER: tl.constexpr
):
    if IS_BARRIER:
        if HAS_TEAM:
            OP(team)
        else:
            OP()
    elif HAS_TEAM:
        OP(team, transport=shmem.ACLSHMEM_TRANSPORT_ROCE)
    else:
        OP(transport=shmem.ACLSHMEM_TRANSPORT_ROCE)


@pytest.mark.parametrize("name,symbol,has_team", CASES)
def test_roce_collective_lowering(monkeypatch, name, symbol, has_team):
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
        _collective_kernel,
        signature={"team": "i32"},
        constexprs={
            "OP": getattr(shmem, _transport_api(name)),
            "HAS_TEAM": has_team,
            "IS_BARRIER": name.startswith("barrier"),
        },
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
    assert f'symbol = "aclshmemx_roce_{symbol}"' in calls[0]
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in calls[0]


@pytest.mark.parametrize(
    "name,has_team",
    [
        ("barrier", True),
        ("barrier_all", False),
        ("barrier_vec", True),
        ("barrier_all_vec", False),
    ],
)
@pytest.mark.parametrize("public", [False, True])
def test_barrier_transport_api_split(monkeypatch, name, has_team, public):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    module = shmem if public else backend
    assert "transport" not in inspect.signature(getattr(module, name)).parameters
    assert not hasattr(module, name + "_lowlevel")
    low = getattr(module, name + "_roce")
    assert "transport" not in inspect.signature(low).parameters
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(TypeError, match="transport"):
        low(
            *([0] if has_team else []),
            transport=shmem.ACLSHMEM_TRANSPORT_AUTO,
            _semantic=semantic,
        )
    assert not calls
