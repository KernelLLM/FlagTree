# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import inspect
import warnings
from types import SimpleNamespace

import pytest
from shmem_api_test_utils import capture_extern

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device as shmem
from triton_dist.language.extra.ascend import libaclshmem_device as backend
from triton_dist.language.extra.utils import ModuleProxy

pytestmark = [pytest.mark.ci, pytest.mark.dist]

AUTO, UDMA, ROCE = (
    shmem.ACLSHMEM_TRANSPORT_AUTO,
    shmem.ACLSHMEM_TRANSPORT_UDMA,
    shmem.ACLSHMEM_TRANSPORT_ROCE,
)
DTYPES = [
    tl.float16,
    tl.float32,
    tl.bfloat16,
    tl.int8,
    tl.int16,
    tl.int32,
    tl.int64,
    tl.uint8,
    tl.uint16,
    tl.uint32,
    tl.uint64,
]


def pointer(dtype):
    return SimpleNamespace(dtype=tl.pointer_type(dtype))


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("qp", [None, 0, 3, tl.constexpr(0), tl.constexpr(None)])
@pytest.mark.parametrize(
    "transport,engine",
    [(UDMA, "udma"), (ROCE, "roce")],
)
def test_transport_rma_abi(monkeypatch, op, dtype, qp, transport, engine):
    semantic, calls = capture_extern(monkeypatch)
    dst, src = pointer(dtype), pointer(dtype)
    result = getattr(backend, f"{op}mem_nbi_lowlevel")(
        dst, src, 7, 2, qp_id=qp, transport=tl.constexpr(transport), _semantic=semantic
    )
    is_qp = tl.core._unwrap_if_constexpr(qp) is not None
    variant = "qp_" if is_qp else ""
    ptr = tl.pointer_type(dtype)
    suffix = backend.DTYPE_TO_KERNEL_SUFFIX[dtype.name]
    args, signatures = calls[0]
    # Seven bytes remain seven bytes even for 16/32/64-bit payloads.
    assert args == [
        dst,
        src,
        ("cast", 7, tl.uint32),
        ("cast", 2, tl.int32),
    ] + ([("cast", tl.core._unwrap_if_constexpr(qp), tl.uint32)] if is_qp else [])
    assert signatures[
        (ptr, ptr, tl.uint32, tl.int32) + ((tl.uint32,) if is_qp else ())
    ] == (
        f"aclshmemx_{engine}_{variant}{op}_nbi_{suffix}",
        (),
    )
    assert result.attrs == {"distributed.core_type": "vector"}


@pytest.mark.parametrize(
    "kwargs,symbol,qp_args",
    [
        pytest.param(
            {}, "aclshmem_putmem_nbi_uint8", [], id="omitted-transport-and-qp"
        ),
        pytest.param(
            {"transport": UDMA}, "aclshmemx_udma_put_nbi_uint8", [], id="omitted-qp"
        ),
        pytest.param(
            {"transport": UDMA, "qp_id": None},
            "aclshmemx_udma_put_nbi_uint8",
            [],
            id="explicit-none",
        ),
        pytest.param(
            {"transport": UDMA, "qp_id": 0},
            "aclshmemx_udma_qp_put_nbi_uint8",
            [("cast", 0, tl.uint32)],
            id="explicit-zero",
        ),
    ],
)
def test_shared_putmem_nbi_preserves_backend_defaults(
    monkeypatch, kwargs, symbol, qp_args
):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    # The displayed shared default must not override Ascend's default.
    signature = inspect.signature(shmem.putmem_nbi)
    assert signature.parameters["qp_id"].default == 0
    assert "transport" not in signature.parameters
    assert inspect.signature(backend.putmem_nbi).parameters["qp_id"].default is None
    assert (
        inspect.signature(shmem.putmem_nbi_lowlevel).parameters["qp_id"].default is None
    )

    semantic, calls = capture_extern(monkeypatch)
    fn = shmem.putmem_nbi_lowlevel if "transport" in kwargs else shmem.putmem_nbi
    fn(pointer(tl.uint8), pointer(tl.uint8), 7, 2, **kwargs, _semantic=semantic)

    assert len(calls) == 1
    args, signatures = calls[0]
    signature = (tl.pointer_type(tl.uint8),) * 2 + (tl.uint32, tl.int32)
    if qp_args:
        signature += (tl.uint32,)
    assert signatures[signature] == (symbol, ())
    assert args[4:] == qp_args


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("public", [False, True])
def test_auto_retains_typed_generic_abi(monkeypatch, op, public):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    module = shmem if public else backend
    semantic, calls = capture_extern(monkeypatch)
    dst, src = pointer(tl.float32), pointer(tl.float32)
    getattr(module, f"{op}mem_nbi")(dst, src, 7, 2, _semantic=semantic)
    assert calls[0][0] == [dst, src, ("cast", 7, tl.uint32), ("cast", 2, tl.int32)]
    sig = (tl.pointer_type(tl.float32),) * 2 + (tl.uint32, tl.int32)
    assert calls[0][1][sig] == (f"aclshmem_{op}mem_nbi_float", ())


@pytest.mark.parametrize("qp", [None, 0, 3])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize(
    "selection", [{}, {"transport": UDMA}, {"transport": tl.constexpr(UDMA)}]
)
def test_udma_signal_abi(monkeypatch, qp, public, selection):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    module = shmem if public else backend
    semantic, calls = capture_extern(monkeypatch)
    signal = pointer(tl.uint64)
    module.putmem_signal_nbi_lowlevel(
        pointer(tl.float32),
        pointer(tl.float32),
        7,
        signal,
        0x100000001,
        shmem.ACLSHMEM_SIGNAL_SET,
        2,
        qp_id=qp,
        **selection,
        _semantic=semantic,
    )
    args, signatures = calls[0]
    assert args[2:6] == [
        ("cast", 7, tl.uint32),
        signal,
        ("cast", 0x100000001, tl.uint64),
        ("cast", 2, tl.int32),
    ]
    assert args[6:] == ([] if qp is None else [("cast", qp, tl.uint32)])
    assert list(signatures.values()) == [
        (
            f"aclshmemx_udma_{'qp_put_signal' if qp is not None else 'putmem_signal'}_nbi_float",
            (),
        )
    ]


@pytest.mark.parametrize(
    "transport,pe,qp,symbol",
    [
        (AUTO, None, None, "aclshmem_quiet"),
        (UDMA, 2, None, "aclshmemx_udma_quiet"),
        (UDMA, 2, 0, "aclshmemx_udma_qp_quiet"),
        (ROCE, 2, None, "aclshmemx_roce_quiet_uint8"),
        (ROCE, 2, 0, "aclshmemx_roce_qp_quiet_uint8"),
    ],
)
def test_quiet_matches_submission(monkeypatch, transport, pe, qp, symbol):
    semantic, calls = capture_extern(monkeypatch)
    if transport == AUTO:
        backend.quiet(_semantic=semantic)
    else:
        backend.quiet_lowlevel(pe, transport=transport, qp_id=qp, _semantic=semantic)
    assert list(calls[0][1].values()) == [(symbol, ())]
    assert calls[0][0] == ([] if pe is None else [("cast", pe, tl.int32)]) + (
        [] if qp is None else [("cast", qp, tl.uint32)]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transport": AUTO},
        {"transport": AUTO, "qp_id": 0},
        {"transport": UDMA, "qp_id": -1},
        {"transport": UDMA, "qp_id": 2**32},
        {"transport": UDMA, "qp_id": True},
        {"transport": UDMA, "qp_id": 1.5},
        {"transport": True},
        {"transport": 4},
        {"transport": "udma"},
        {"transport": tl.tensor(None, tl.int32)},
        {"transport": UDMA, "qp_id": tl.tensor(None, tl.float32)},
        {"transport": UDMA, "qp_id": tl.tensor(None, tl.block_type(tl.int32, [4]))},
    ],
)
def test_invalid_selection_emits_no_call(monkeypatch, kwargs):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError):
        backend.putmem_nbi_lowlevel(
            pointer(tl.int32), pointer(tl.int32), 4, 1, **kwargs, _semantic=semantic
        )
    assert not calls


@pytest.mark.parametrize("size,pe", [(-1, 0), (2**32, 0), (4, -1), (4, 2**31)])
def test_invalid_scalar_ranges(monkeypatch, size, pe):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError):
        backend.getmem_nbi_lowlevel(
            pointer(tl.int32),
            pointer(tl.int32),
            size,
            pe,
            transport=UDMA,
            _semantic=semantic,
        )
    assert not calls


@pytest.mark.parametrize(
    "dtype,op",
    [(tl.int32, shmem.ACLSHMEM_SIGNAL_SET), (tl.uint64, shmem.ACLSHMEM_SIGNAL_ADD)],
)
def test_invalid_signal_emits_no_call(monkeypatch, dtype, op):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError):
        backend.putmem_signal_nbi_lowlevel(
            pointer(tl.int32),
            pointer(tl.int32),
            4,
            pointer(dtype),
            1,
            op,
            1,
            _semantic=semantic,
        )
    assert not calls


@pytest.mark.parametrize(
    "pe,transport,qp",
    [(None, UDMA, None), (1, AUTO, None), (None, AUTO, 0)],
)
def test_invalid_quiet_emits_no_call(monkeypatch, pe, transport, qp):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError):
        backend.quiet_lowlevel(pe, transport=transport, qp_id=qp, _semantic=semantic)
    assert not calls


def test_no_public_resource_arguments():
    for name in (
        "getmem_nbi",
        "putmem_nbi",
        "putmem_signal_nbi",
        "quiet",
        "getmem_nbi_lowlevel",
        "putmem_nbi_lowlevel",
        "putmem_signal_nbi_lowlevel",
        "quiet_lowlevel",
    ):
        for module in (shmem, backend):
            assert not {
                "scratch",
                "sync_id",
                "relay_pe" if not name.endswith("_relay") else "qp_id",
            } & set(inspect.signature(getattr(module, name)).parameters)


def test_shared_signatures_have_no_transport():
    for name, fn in vars(shmem).items():
        if (
            callable(fn)
            and hasattr(fn, "__signature__")
            and not name.endswith("_lowlevel")
        ):
            assert "transport" not in inspect.signature(fn).parameters, name


def test_legacy_backend_default_forwarding(monkeypatch):
    calls = []

    def putmem_nbi(dest, source, nbytes, pe, _semantic=None):
        calls.append((dest, source, nbytes, pe))

    monkeypatch.setattr(
        shmem._shmem_module, "_module", SimpleNamespace(putmem_nbi=putmem_nbi)
    )
    shmem.putmem_nbi(1, 2, 3, 4, _semantic=object())
    assert calls == [(1, 2, 3, 4)]


@pytest.mark.parametrize(
    "fetch,op,suffix,values",
    [
        (False, shmem.ACLSHMEM_AMO_ADD, "add", 1),
        (False, shmem.ACLSHMEM_AMO_SWAP, "swap", 1),
        (False, shmem.ACLSHMEM_AMO_INC, "inc", 0),
        (True, shmem.ACLSHMEM_AMO_NONE, "fetch", 0),
        (True, shmem.ACLSHMEM_AMO_INC, "fetch_inc", 0),
        (True, shmem.ACLSHMEM_AMO_AND, "fetch_and", 1),
    ],
)
@pytest.mark.parametrize(
    "transport,prefix",
    [
        (AUTO, "aclshmem"),
        (UDMA, "aclshmemx_udma"),
        (ROCE, "aclshmemx_roce"),
    ],
)
def test_grouped_atomic_transport(
    monkeypatch, fetch, op, suffix, values, transport, prefix
):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    semantic, calls = capture_extern(monkeypatch)
    fn = backend.atomic_fetch_op if fetch else backend.atomic_op
    fn(
        pointer(tl.int32),
        1 if values else None,
        2,
        op,
        transport=transport,
        _semantic=semantic,
    )
    signature = (tl.pointer_type(tl.int32),) + (tl.int32,) * (values + 1)
    assert calls[0][1][signature][0] == f"{prefix}_atomic_{suffix}_int32"
    assert len(calls[0][0]) == values + 2


@triton.jit
def _rma_kernel(
    dst,
    src,
    sig,
    size,
    peer,
    qp,
    OP: tl.constexpr,
    TRANSPORT: tl.constexpr,
    KIND: tl.constexpr,
):
    if KIND == "signal_plain":
        OP(dst, src, size, sig, 1, shmem.ACLSHMEM_SIGNAL_SET, peer)
    elif KIND == "signal":
        OP(
            dst,
            src,
            size,
            sig,
            1,
            shmem.ACLSHMEM_SIGNAL_SET,
            peer,
            qp_id=qp,
        )
    elif KIND == "quiet_plain":
        if TRANSPORT == shmem.ACLSHMEM_TRANSPORT_MTE:
            OP(transport=TRANSPORT)
        else:
            OP(peer, transport=TRANSPORT)
    elif KIND == "quiet":
        OP(peer, qp_id=qp, transport=TRANSPORT)
    elif KIND == "plain":
        OP(dst, src, size, peer, transport=TRANSPORT)
    else:
        OP(dst, src, size, peer, qp_id=qp, transport=TRANSPORT)


@pytest.mark.parametrize(
    "name,kind,symbol,gm",
    [
        ("getmem_nbi", "plain", "aclshmemx_udma_get_nbi_uint8", "0, 1"),
        ("putmem_nbi", "plain", "aclshmemx_udma_put_nbi_uint8", "0, 1"),
        ("getmem_nbi", "qp", "aclshmemx_udma_qp_get_nbi_uint8", "0, 1"),
        ("putmem_nbi", "qp", "aclshmemx_udma_qp_put_nbi_uint8", "0, 1"),
        (
            "putmem_signal_nbi",
            "signal",
            "aclshmemx_udma_qp_put_signal_nbi_uint8",
            "0, 1, 3",
        ),
        (
            "putmem_signal_nbi",
            "signal_plain",
            "aclshmemx_udma_putmem_signal_nbi_uint8",
            "0, 1, 3",
        ),
        ("quiet", "quiet", "aclshmemx_udma_qp_quiet", ""),
        ("quiet", "quiet_plain", "aclshmemx_udma_quiet", ""),
    ],
)
@pytest.mark.parametrize("payload_type", ["*u8", "*fp32"])
@pytest.mark.parametrize(
    "transport",
    [
        UDMA,
        ROCE,
        shmem.ACLSHMEM_TRANSPORT_MTE,
    ],
)
def test_rma_jit_lowering(monkeypatch, name, kind, symbol, gm, payload_type, transport):
    if transport == shmem.ACLSHMEM_TRANSPORT_MTE:
        if name == "putmem_signal_nbi":
            pytest.skip("MTE signaling is not supported")
        if kind not in ("plain", "quiet_plain"):
            pytest.skip("MTE has no QPs")
        symbol = symbol.replace("udma", "mte")
    if transport == ROCE:
        if name == "putmem_signal_nbi":
            pytest.skip("RoCE signaling is not supported")
        symbol = symbol.replace("udma", "roce")
        if name == "quiet":
            symbol += "_uint8"
    if payload_type == "*fp32" and name != "quiet":
        symbol = symbol.removesuffix("_uint8") + "_float"
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    from test_gm_addr_args_indices import _run_pass

    from triton._C.libtriton import ir
    from triton.backends.ascend.compiler import AscendBackend
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    target = GPUTarget("npu", "Ascend950DT_9582", 32)
    compiler_backend = AscendBackend(target)
    options = compiler_backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    compiler_backend.load_dialects(ctx)
    source = ASTSource(
        _rma_kernel,
        signature={
            "dst": payload_type,
            "src": payload_type,
            "sig": "*u64",
            "size": "u32",
            "peer": "i32",
            "qp": "u32",
        },
        constexprs={
            "OP": getattr(
                shmem, name + "_lowlevel"
            ),
            "TRANSPORT": transport,
            "KIND": kind,
        },
    )
    module = source.make_ir(
        target,
        options,
        compiler_backend.get_codegen_implementation(options),
        compiler_backend.get_module_map(),
        ctx,
    )
    assert f'symbol = "{symbol}"' in module.str()
    _run_pass(module, ctx)
    lines = [line for line in module.str().splitlines() if "hir.custom" in line]
    assert len(lines) == 1
    assert f'symbol = "{symbol}"' in lines[0]
    assert "hivm.tcore_type = #hivm.tcore_type<VECTOR>" in lines[0]
    # Payload pointers retain their types and remain entry-block operands.
    assert f"gm_addr_args_indices = array<i32{': ' + gm if gm else ''}>" in lines[0]


def test_legacy_backend_native_qp_default():
    calls = []

    def putmem_nbi(dest, source, nbytes, pe, qp_id=7, _semantic=None):
        calls.append(qp_id)

    proxy = ModuleProxy([(lambda: True, SimpleNamespace(putmem_nbi=putmem_nbi))])

    @proxy.dispatch
    def putmem_nbi(dest, source, bytes, pe, qp_id=0): ...

    putmem_nbi(1, 2, 3, 4, _semantic=object())
    putmem_nbi(1, 2, 3, 4, qp_id=0, _semantic=object())
    assert calls == [7, 0]


@pytest.mark.parametrize(
    "dst_type,src_type", [(tl.float32, tl.int32), (tl.float64, tl.float64)]
)
def test_invalid_payload_types(monkeypatch, dst_type, src_type):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match="pointer type"):
        backend.putmem_nbi_lowlevel(
            pointer(dst_type),
            pointer(src_type),
            4,
            1,
            transport=UDMA,
            _semantic=semantic,
        )
    assert not calls


@pytest.mark.parametrize(
    "transport",
    [
        AUTO,
        ROCE,
        shmem.ACLSHMEM_TRANSPORT_MTE,
        True,
        "udma",
        tl.tensor(None, tl.int32),
    ],
)
@pytest.mark.parametrize("public", [False, True])
def test_lowlevel_signal_rejects_other_transports(monkeypatch, transport, public):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    module = shmem if public else backend
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match="transport"):
        module.putmem_signal_nbi_lowlevel(
            pointer(tl.uint8),
            pointer(tl.uint8),
            4,
            pointer(tl.uint64),
            1,
            shmem.ACLSHMEM_SIGNAL_SET,
            1,
            transport=transport,
            _semantic=semantic,
        )
    assert not calls


@pytest.mark.parametrize(
    "name", ["getmem_nbi", "putmem_nbi", "putmem_signal_nbi", "quiet"]
)
def test_rma_api_separates_transport_selection(name):
    high = inspect.signature(getattr(backend, name))
    assert "transport" not in high.parameters
    if name in ("getmem_nbi", "putmem_nbi"):
        assert high.parameters["qp_id"].default is None
    else:
        assert "qp_id" not in high.parameters
    for module in (shmem, backend):
        low = inspect.signature(getattr(module, name + "_lowlevel"))
        if name == "putmem_signal_nbi":
            assert low.parameters["transport"].default == UDMA
            assert low.parameters["transport"].kind is inspect.Parameter.KEYWORD_ONLY
            assert not hasattr(module, "putmem_signal_nbi_udma")
        else:
            assert low.parameters["transport"].default is inspect.Parameter.empty
            assert low.parameters["transport"].kind is inspect.Parameter.KEYWORD_ONLY
        assert low.parameters["qp_id"].default is None


@pytest.mark.parametrize("name", ["getmem_nbi", "putmem_nbi", "quiet"])
def test_lowlevel_rejects_auto(monkeypatch, name):
    semantic, calls = capture_extern(monkeypatch)
    args = [pointer(tl.int32), pointer(tl.int32), 4]
    if name == "quiet":
        args = []
    with pytest.raises(ValueError, match="transport must be UDMA, ROCE or MTE"):
        getattr(backend, name + "_lowlevel")(
            *args, 1, transport=AUTO, _semantic=semantic
        )
    assert not calls


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("qp", [0, 3, tl.constexpr(0), tl.tensor(None, tl.int32)])
def test_highlevel_qp_warns_and_preserves_generic_abi(
    monkeypatch, op, public, positional, qp
):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    module = shmem if public else backend
    semantic, calls = capture_extern(monkeypatch)
    dst, src = pointer(tl.float32), pointer(tl.float32)
    args = [dst, src, 7, 2]
    kwargs = {} if positional else {"qp_id": qp}
    if positional:
        args.append(qp)
    with pytest.warns(UserWarning, match=f"qp_id is ignored.*{op}mem_nbi_lowlevel"):
        getattr(module, f"{op}mem_nbi")(*args, **kwargs, _semantic=semantic)
    assert len(calls) == 1
    assert calls[0][0] == [dst, src, ("cast", 7, tl.uint32), ("cast", 2, tl.int32)]
    sig = (tl.pointer_type(tl.float32),) * 2 + (tl.uint32, tl.int32)
    assert calls[0][1][sig] == (f"aclshmem_{op}mem_nbi_float", ())


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"qp_id": None}, {"qp_id": tl.constexpr(None)}])
def test_highlevel_without_qp_does_not_warn(monkeypatch, op, public, kwargs):
    monkeypatch.setattr(shmem._shmem_module, "_module", backend)
    module = shmem if public else backend
    semantic, calls = capture_extern(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(module, f"{op}mem_nbi")(
            pointer(tl.int32), pointer(tl.int32), 4, 1, **kwargs, _semantic=semantic
        )
    assert not caught
    assert len(calls) == 1


@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_mte_rma_abi(monkeypatch, op, dtype):
    test_transport_rma_abi(
        monkeypatch, op, dtype, None, shmem.ACLSHMEM_TRANSPORT_MTE, "mte"
    )


def test_mte_quiet_core_wide(monkeypatch):
    semantic, calls = capture_extern(monkeypatch)
    result = backend.quiet_lowlevel(
        transport=shmem.ACLSHMEM_TRANSPORT_MTE, _semantic=semantic
    )
    assert calls == [([], {(): ("aclshmemx_mte_quiet", ())})]
    assert result.attrs == {"distributed.core_type": "cube_or_vector"}


@pytest.mark.parametrize(
    "api",
    [
        "getmem_nbi_lowlevel",
        "putmem_nbi_lowlevel",
        "quiet_lowlevel",
    ],
)
def test_mte_rejects_qp(monkeypatch, api):
    semantic, calls = capture_extern(monkeypatch)
    args = [pointer(tl.int32), pointer(tl.int32), 7]
    args = [] if api == "quiet_lowlevel" else args + [2]
    with pytest.raises(ValueError, match="MTE does not support qp_id"):
        getattr(backend, api)(
            *args, qp_id=0, transport=shmem.ACLSHMEM_TRANSPORT_MTE, _semantic=semantic
        )
    assert not calls


def test_mte_rejects_peer_quiet(monkeypatch):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match="core-wide"):
        backend.quiet_lowlevel(
            2, transport=shmem.ACLSHMEM_TRANSPORT_MTE, _semantic=semantic
        )
    assert not calls
