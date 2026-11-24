# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import warnings

from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call
from triton_dist.language.extra.ascend.aclshmem_constants import (
    ACLSHMEMAtomicOp,
    ACLSHMEMCmpOp,
    ACLSHMEMSignalOp,
    ACLSHMEMTransport,
)

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
pi_i32_t = tl.core.pointer_type(tl.core.dtype("int32"))

void_ptr = core.pointer_type(core.void)


def _check_enum_value(value, enum_cls: type, param_name: str, op_name: str) -> None:
    # Triton's `constexpr` wraps the actual Python value;
    # dict lookup would miss it. if constexpr, unwrap first.
    raw = value.value if isinstance(value, core.constexpr) else value
    try:
        enum_cls(raw)
    except ValueError:
        valid = ", ".join(f"{m.name}={m.value}" for m in enum_cls)
        raise ValueError(
            f"{op_name}: parameter '{param_name}' got invalid value {raw!r} "
            f"(type {type(raw).__name__}). Valid values are: {valid}."
        ) from None


def _transport_selector(transport, api):
    raw = core._unwrap_if_constexpr(transport)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{api}: transport must be a compile-time ACLSHMEMTransport")
    _check_enum_value(raw, ACLSHMEMTransport, "transport", api)
    return ACLSHMEMTransport(raw)


# The subset of dtypes supported by Ascend SHMEM RMA ops.
RMA_DTYPES = [
    "fp16",
    "fp32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bf16",
]

# Integer types common to the scalar atomic template wrappers.
ATOMIC_DTYPES = ("int32", "uint32", "int64", "uint64")
# Float-capable Ascend950 atomic variants; availability depends on op/transport.
ATOMIC_950_DTYPES = ATOMIC_DTYPES + ("fp32",)
ATOMIC_PRE_950_ADD_DTYPES = ("int8", "int16", "int32", "fp16", "bf16", "fp32")
# Generic atomic_add also exposes the MTE types available on Ascend910B/910C.
ATOMIC_ADD_DTYPES = ATOMIC_DTYPES + ("int8", "int16", "fp16", "bf16", "fp32")

DTYPE_TO_KERNEL_SUFFIX = {
    "fp16": "half",
    "fp32": "float",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bf16": "bfloat16",
}


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    elem_dtype_list = (
        tl.core.dtype.SINT_TYPES
        + tl.core.dtype.UINT_TYPES
        + tl.core.dtype.FP_TYPES
        + tl.core.dtype.OTHER_TYPES
    )
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


# Keep pointer_type hashable for Triton extern signature dict keys.
patch_hash_method_for_pointer_type()


def _set_attr(call, attrs: dict[str, str], _semantic=None):
    for key, value in attrs.items():
        handle_attr = _semantic.builder.get_string_attr(value)
        call.handle.set_attr(key, handle_attr)
    return call


def _mark_core_type(call, core_type: str, _semantic=None):
    return _set_attr(call, {"distributed.core_type": core_type}, _semantic)


@core.extern
def my_pe(_semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def n_pes(_semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def team_my_pe(team, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32,): ("aclshmem_team_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def team_n_pes(team, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32,): ("aclshmem_team_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def team_translate_pe(src_team, pe_in_src_team, dest_team, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [
            tl.cast(src_team, tl.int32, _semantic=_semantic),
            tl.cast(pe_in_src_team, tl.int32, _semantic=_semantic),
            tl.cast(dest_team, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.int32, tl.int32, tl.int32): (
                "aclshmem_team_translate_pe",
                core.dtype("int32"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def team_pe_mapping(team, pe, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [
            tl.cast(team, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.int32, tl.int32): (
                "aclshmem_team_pe_mapping",
                core.dtype("int32"),
            ),
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def int_p(dest, value, pe, _semantic=None):
    # force have a return value, even not used.
    call = extern_call(
        "libshmem_device",
        "",
        [dest, value, pe],
        {
            (
                core.pointer_type(core.dtype("int32")),
                core.dtype("int32"),
                core.dtype("int32"),
            ): (
                "aclshmem_int32_p",
                (),
            ),  # void return type
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def g(source, pe, _semantic=None):
    """Read a scalar from symmetric memory on the source PE."""
    call = extern_call(
        "libshmem_device",
        "",
        [source, tl.cast(pe, tl.int32, _semantic=_semantic)],
        {
            (core.pointer_type(core.dtype(dtype)), tl.int32): (
                f"aclshmem_g_{DTYPE_TO_KERNEL_SUFFIX[dtype]}",
                core.dtype(dtype),
            )
            for dtype in RMA_DTYPES
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


def _atomic_impl(
    dest, values, pe, op, dtypes=ATOMIC_DTYPES, returns_value=True, _semantic=None
):
    # Cast scalar operands to the pointee type so Python literals also select
    # the correct signedness and width. The signature map rejects unsupported
    # pointee types before emitting a call to a nonexistent wrapper.
    dtype = dest.dtype.element_ty
    call = extern_call(
        "libshmem_device",
        "",
        [dest]
        + [tl.cast(value, dtype, _semantic=_semantic) for value in values]
        + [tl.cast(pe, tl.int32, _semantic=_semantic)],
        {
            (core.pointer_type(core.dtype(dtype)),)
            + (core.dtype(dtype),) * len(values)
            + (tl.int32,): (
                f"{op}_{DTYPE_TO_KERNEL_SUFFIX[dtype]}",
                core.dtype(dtype) if returns_value else (),
            )
            for dtype in dtypes
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


def _atomic_selector(op, allowed, api):
    op = core._unwrap_if_constexpr(op)
    if not isinstance(op, ACLSHMEMAtomicOp) or op.name.lower() not in allowed:
        raise ValueError(
            f"{api}: op must be a compile-time ACLSHMEMAtomicOp in {allowed}, got {op!r}"
        )
    return op.name.lower()


def _atomic_values(value, op, api):
    if op in ("inc", "none"):
        if core._unwrap_if_constexpr(value) is not None:
            raise ValueError(f"{api}: value must be None for op={op!r}")
        return ()
    return (value,)


@core.extern
def atomic_op(dest, value, pe, op, transport=ACLSHMEMTransport.AUTO, _semantic=None):
    """Scalar remote update selected by a compile-time ACLSHMEMAtomicOp.

    Supported selectors: ADD, AND, OR, XOR, INC, SET, SWAP. Only swap
    returns the old value. Pass None as value for inc. AUTO uses generic
    runtime routing; MTE/UDMA/ROCE select the corresponding transport API.

    Low-level MTE and RoCE transports are experimental and not officially
    supported.

    Explicit ROCE atomics require the XSCALE or HNS_1825 backend.
    HNS_1825 uses software atomics for 32-bit or 64-bit integers and requires
    a remote PE and QP 0. AUTO routing does not support HNS_1825 atomics
    when the selected path is ROCE.
    """
    op = _atomic_selector(
        op, ("add", "and", "or", "xor", "inc", "set", "swap"), "atomic_op"
    )
    return _atomic_dispatch(
        dest,
        _atomic_values(value, op, "atomic_op"),
        pe,
        op,
        transport,
        returns_value=op == "swap",
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_op(
    dest, value, pe, op, transport=ACLSHMEMTransport.AUTO, _semantic=None
):
    """Return the old scalar value, optionally updating it atomically.

    Supported ACLSHMEMAtomicOp selectors: ADD, AND, OR, XOR, INC, NONE.
    Pass None as value for inc and none; none performs an atomic read.
    AUTO uses generic routing; MTE/UDMA/ROCE select the transport API.

    Low-level MTE and RoCE transports are experimental and not officially
    supported.

    Explicit ROCE atomics require the XSCALE or HNS_1825 backend.
    HNS_1825 uses software atomics for 32-bit or 64-bit integers and requires
    a remote PE and QP 0. AUTO routing does not support HNS_1825 atomics
    when the selected path is ROCE.
    """
    op = _atomic_selector(
        op, ("add", "and", "or", "xor", "inc", "none"), "atomic_fetch_op"
    )
    suffix = "fetch" if op == "none" else f"fetch_{op}"
    return _atomic_dispatch(
        dest,
        _atomic_values(value, op, "atomic_fetch_op"),
        pe,
        suffix,
        transport,
        _semantic=_semantic,
    )


def _atomic_dispatch(
    dest, values, pe, op, transport, returns_value=True, _semantic=None
):
    from triton.backends.ascend.utils import is_compile_on_910_95

    transport = _transport_selector(transport, f"atomic_{op}")
    if not is_compile_on_910_95():
        # SHMEM's only pre-950 AMO is MTE add for these element types.
        if (
            op != "add"
            or transport not in (ACLSHMEMTransport.AUTO, ACLSHMEMTransport.MTE)
            or dest.dtype.element_ty.name not in ATOMIC_PRE_950_ADD_DTYPES
        ):
            raise ValueError(
                f"atomic_{op}: {dest.dtype.element_ty} with {transport.name} "
                "transport requires Ascend950 or later"
            )
    dtypes = ATOMIC_DTYPES
    if transport == ACLSHMEMTransport.AUTO:
        symbol = f"aclshmem_atomic_{op}"
        if op in ("set", "swap"):
            dtypes = ATOMIC_950_DTYPES
        elif op == "add":
            dtypes = ATOMIC_ADD_DTYPES
    elif transport == ACLSHMEMTransport.MTE:
        if op not in (
            "fetch",
            "set",
            "swap",
            "compare_swap",
            "inc",
            "add",
            "fetch_inc",
            "fetch_add",
        ):
            raise ValueError(f"atomic_{op}: MTE transport is not supported")
        symbol = f"aclshmemx_mte_atomic_{op}"
        dtypes = (
            ("uint32", "uint64")
            if op in ("set", "swap", "compare_swap")
            else ATOMIC_ADD_DTYPES if op == "add" else ATOMIC_950_DTYPES
        )
    else:
        name = "udma" if transport == ACLSHMEMTransport.UDMA else "roce"
        symbol = f"aclshmemx_{name}_atomic_{op}"
        if (name == "udma" and op == "add") or (
            name == "roce" and op in ("fetch", "set", "swap")
        ):
            dtypes = ATOMIC_950_DTYPES
    return _atomic_impl(
        dest,
        values,
        pe,
        symbol,
        dtypes=dtypes,
        returns_value=returns_value,
        _semantic=_semantic,
    )


@core.extern
def atomic_compare_swap(
    dest, cond, value, pe, transport=ACLSHMEMTransport.AUTO, _semantic=None
):
    """Select AUTO runtime routing or an explicit MTE/UDMA/ROCE API at compile time.

    Low-level MTE and RoCE transports are experimental and not officially
    supported.

    Explicit ROCE atomics require the XSCALE or HNS_1825 backend.
    HNS_1825 uses software atomics for 32-bit or 64-bit integers and requires
    a remote PE and QP 0. AUTO routing does not support HNS_1825 atomics
    when the selected path is ROCE.
    """
    return _atomic_dispatch(
        dest,
        (cond, value),
        pe,
        "compare_swap",
        transport,
        returns_value=True,
        _semantic=_semantic,
    )


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [local_ptr, pe],
        {
            (core.pointer_type(core.dtype(core_dtype)), core.dtype(pe_dtype)): (
                "aclshmem_ptr",
                core.pointer_type(core.dtype(core_dtype)),  # of the same dtype
            )
            for core_dtype in core.dtype.SINT_TYPES
            + core.dtype.UINT_TYPES
            + core.dtype.FP_TYPES
            + core.dtype.OTHER_TYPES
            for pe_dtype in ["int32", "uint32"]
        },
        is_pure=True,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


def _collective_transport(transport, api):
    transport = _transport_selector(transport, api)
    if transport in (ACLSHMEMTransport.UDMA, ACLSHMEMTransport.MTE):
        raise ValueError(
            f"{api}: {transport.name} collective transport is not supported"
        )
    return transport


def _roce_collective(op, team, _semantic):
    has_team = op in ("team_sync", "barrier")
    args = [_rma_index(team, "team", op, tl.int32, _semantic)] if has_team else []
    types = (tl.int32,) if has_team else ()
    result_type = tl.int32 if has_team else ()
    call = extern_call(
        "libshmem_device",
        "",
        args,
        {types: (f"aclshmemx_roce_{op}", result_type)},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def barrier_all(_semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_barrier_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_and_vector", _semantic)


@core.extern
def barrier_all_roce(_semantic=None):
    """Vector-core barrier using RoCE transport.

    The low-level RoCE transport is experimental and not officially supported.
    """
    return _roce_collective("barrier_all", None, _semantic)


@core.extern
def barrier_all_vec(_semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmemx_barrier_all_vec", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def barrier_all_vec_roce(_semantic=None):
    """Vector-core barrier using RoCE transport.

    The low-level RoCE transport is experimental and not officially supported.
    """
    return _roce_collective("barrier_all", None, _semantic)


@core.extern
def barrier(team, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32,): ("aclshmem_barrier", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_and_vector", _semantic)


@core.extern
def barrier_roce(team, _semantic=None):
    """Vector-core barrier using RoCE transport.

    The low-level RoCE transport is experimental and not officially supported.
    """
    return _roce_collective("barrier", team, _semantic)


@core.extern
def barrier_vec(team, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32,): ("aclshmemx_barrier_vec", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def barrier_vec_roce(team, _semantic=None):
    """Vector-core barrier using RoCE transport.

    The low-level RoCE transport is experimental and not officially supported.
    """
    return _roce_collective("barrier", team, _semantic)


@core.extern
def sync(team, *, transport=ACLSHMEMTransport.AUTO, _semantic=None):
    """Synchronize a team, including with UDMA initialized.

    AUTO uses shared-memory signaling, not UDMA queues. Complete outstanding
    UDMA operations with quiet_lowlevel before calling this collective.

    The low-level RoCE transport is experimental and not officially supported.
    """
    transport = _collective_transport(transport, "sync")
    if transport == ACLSHMEMTransport.ROCE:
        return _roce_collective("team_sync", team, _semantic)
    call = extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {(tl.int32,): ("aclshmem_sync", ())},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_and_vector", _semantic)


@core.extern
def sync_all(*, transport=ACLSHMEMTransport.AUTO, _semantic=None):
    """Synchronize all PEs; AUTO also works with UDMA initialized.

    This does not drain UDMA queues. Use quiet_lowlevel for completion first;
    explicit transport=UDMA is unsupported (there is no UDMA sync entry point).

    The low-level RoCE transport is experimental and not officially supported.
    """
    transport = _collective_transport(transport, "sync_all")
    if transport == ACLSHMEMTransport.ROCE:
        return _roce_collective("sync_all", None, _semantic)
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {(): ("aclshmem_sync_all", ())},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_and_vector", _semantic)


@core.extern
def sync_vec(team, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {(tl.int32,): ("aclshmemx_sync_vec", ())},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def sync_vec_all(_semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {(): ("aclshmemx_sync_vec_all", ())},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


def _rma_transport(transport, qp_id, api):
    transport = _transport_selector(transport, api)
    qp_id = core._unwrap_if_constexpr(qp_id)
    if transport == ACLSHMEMTransport.AUTO:
        raise ValueError(
            f"{api}: transport must be UDMA, ROCE or MTE; use the high-level API for AUTO"
        )
    if transport == ACLSHMEMTransport.MTE and qp_id is not None:
        raise ValueError(f"{api}: MTE does not support qp_id")
    return transport, qp_id


def _rma_index(value, name, api, dtype, _semantic):
    raw = core._unwrap_if_constexpr(value)
    maximum = 0xFFFFFFFF if dtype == tl.uint32 else 0x7FFFFFFF
    if isinstance(raw, int) and not isinstance(raw, bool):
        if not 0 <= raw <= maximum:
            raise ValueError(f"{api}: {name} must be in [0, {maximum}]")
    elif not (
        isinstance(raw, core.tensor)
        and not raw.type.is_block()
        and raw.dtype.is_int()
        and raw.dtype != tl.int1
    ):
        raise ValueError(f"{api}: {name} must be a scalar integer")
    return tl.cast(value, dtype, _semantic=_semantic)


def _rma_pointer_type(dest, source, api):
    # C++ wrappers apply typed pointer offsets before converting to byte pointers.
    for ptr in (dest, source):
        if (
            not isinstance(ptr.dtype, core.pointer_type)
            or ptr.dtype.element_ty.name not in RMA_DTYPES
        ):
            raise ValueError(f"{api}: unsupported RMA pointer type {ptr.dtype}")
    if dest.dtype != source.dtype:
        raise ValueError(
            f"{api}: source and destination must have matching pointer types"
        )
    return dest.dtype


def _transport_mem_call(
    dest,
    source,
    nbytes,
    pe,
    operation,
    qp_id=None,
    relay_pe=None,
    transport=ACLSHMEMTransport.UDMA,
    _semantic=None,
):
    engine = transport.name.lower()
    api = f"{operation}mem_nbi"
    size = _rma_index(nbytes, "nbytes", api, tl.uint32, _semantic)
    peer = _rma_index(pe, "pe", api, tl.int32, _semantic)
    extra, extra_types = [], ()
    variant = ""
    if qp_id is not None:
        extra = [_rma_index(qp_id, "qp_id", api, tl.uint32, _semantic)]
        extra_types = (tl.uint32,)
        variant = "qp_"
    elif relay_pe is not None:
        extra = [_rma_index(relay_pe, "relay_pe", api, tl.int32, _semantic)]
        extra_types = (tl.int32,)
        variant = "relay_"
    ptr = _rma_pointer_type(dest, source, api)
    suffix = DTYPE_TO_KERNEL_SUFFIX[ptr.element_ty.name]
    call = extern_call(
        "libshmem_device",
        "",
        [dest, source, size, peer] + extra,
        {
            (ptr, ptr, tl.uint32, tl.int32)
            + extra_types: (f"aclshmemx_{engine}_{variant}{operation}_nbi_{suffix}", ())
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def _rma_mem_impl(dest, source, elem_size, pe, op: core.constexpr, _semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [
            dest,
            source,
            tl.cast(elem_size, tl.uint32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype(core_dtype)),
                core.pointer_type(core.dtype(core_dtype)),
                tl.uint32,
                tl.int32,
            ): (
                f"aclshmem_{op.value}_{DTYPE_TO_KERNEL_SUFFIX[core_dtype]}",
                (),
            )
            for core_dtype in RMA_DTYPES
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def getmem(dest, source, bytes, pe, _semantic=None):
    return _rma_mem_impl(
        dest, source, bytes, pe, core.constexpr("getmem"), _semantic=_semantic
    )


@core.extern
def putmem(dest, source, bytes, pe, _semantic=None):
    return _rma_mem_impl(
        dest, source, bytes, pe, core.constexpr("putmem"), _semantic=_semantic
    )


@core.extern
def getmem_nbi(dest, source, bytes, pe, qp_id=None, _semantic=None):
    """Transfer bytes using generic runtime routing; a supplied QP is ignored."""
    if core._unwrap_if_constexpr(qp_id) is not None:
        warnings.warn(
            "getmem_nbi: qp_id is ignored by the high-level Ascend API; "
            "use getmem_nbi_lowlevel with an explicit transport to select a QP.",
            UserWarning,
            stacklevel=2,
        )
    return _rma_mem_impl(
        dest, source, bytes, pe, core.constexpr("getmem_nbi"), _semantic=_semantic
    )


@core.extern
def getmem_nbi_lowlevel(
    dest, source, bytes, pe, qp_id=None, *, transport, _semantic=None
):
    """Transfer bytes via explicit MTE/UDMA/RoCE; None selects the non-QP API.

    Low-level MTE and RoCE transports are experimental and not officially
    supported.

    With transport=ROCE, specifying qp_id requires the XSCALE
    or HNS_1825 backend; the IN_DIE backend does not support this QP-specific
    interface.
    """
    transport, qp_id = _rma_transport(transport, qp_id, "getmem_nbi_lowlevel")
    return _transport_mem_call(
        dest,
        source,
        bytes,
        pe,
        "get",
        qp_id=qp_id,
        transport=transport,
        _semantic=_semantic,
    )


@core.extern
def putmem_nbi(dest, source, bytes, pe, qp_id=None, _semantic=None):
    """Transfer bytes using generic runtime routing; a supplied QP is ignored."""
    if core._unwrap_if_constexpr(qp_id) is not None:
        warnings.warn(
            "putmem_nbi: qp_id is ignored by the high-level Ascend API; "
            "use putmem_nbi_lowlevel with an explicit transport to select a QP.",
            UserWarning,
            stacklevel=2,
        )
    return _rma_mem_impl(
        dest, source, bytes, pe, core.constexpr("putmem_nbi"), _semantic=_semantic
    )


@core.extern
def putmem_nbi_lowlevel(
    dest, source, bytes, pe, qp_id=None, *, transport, _semantic=None
):
    """Transfer bytes via explicit MTE/UDMA/RoCE; None selects the non-QP API.

    Low-level MTE and RoCE transports are experimental and not officially
    supported.

    With transport=ROCE, specifying qp_id requires the XSCALE
    or HNS_1825 backend; the IN_DIE backend does not support this QP-specific
    interface.
    """
    transport, qp_id = _rma_transport(transport, qp_id, "putmem_nbi_lowlevel")
    return _transport_mem_call(
        dest,
        source,
        bytes,
        pe,
        "put",
        qp_id=qp_id,
        transport=transport,
        _semantic=_semantic,
    )


@core.extern
def getmem_nbi_relay(dest, source, bytes, pe, relay_pe, _semantic=None):
    """UDMA relay read; requires relay-enabled templates/runtime.

    Complete with quiet_lowlevel(pe, transport=ACLSHMEMTransport.UDMA).
    """
    if core._unwrap_if_constexpr(relay_pe) is None:
        raise ValueError("getmem_nbi_relay: relay_pe is required")
    return _transport_mem_call(
        dest, source, bytes, pe, "get", relay_pe=relay_pe, _semantic=_semantic
    )


@core.extern
def putmem_nbi_relay(dest, source, bytes, pe, relay_pe, _semantic=None):
    """UDMA relay write; no QP selection. Requires relay-enabled templates/runtime."""
    if core._unwrap_if_constexpr(relay_pe) is None:
        raise ValueError("putmem_nbi_relay: relay_pe is required")
    return _transport_mem_call(
        dest, source, bytes, pe, "put", relay_pe=relay_pe, _semantic=_semantic
    )


@core.extern
def _putmem_signal_impl(
    dest,
    source,
    nbytes,
    sig_addr,
    signal,
    sig_op,
    pe,
    op: core.constexpr,
    _semantic=None,
):
    _check_enum_value(sig_op, ACLSHMEMSignalOp, "sig_op", op.value)
    call = extern_call(
        "libshmem_device",
        "",
        [
            dest,
            source,
            tl.cast(nbytes, tl.uint32, _semantic=_semantic),
            sig_addr,
            tl.cast(signal, tl.int32, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype(core_dtype)),
                core.pointer_type(core.dtype(core_dtype)),
                tl.uint32,
                core.pointer_type(core.dtype("int32")),
                tl.int32,
                tl.int32,
                tl.int32,
            ): (
                f"aclshmem_{op.value}_{DTYPE_TO_KERNEL_SUFFIX[core_dtype]}",
                (),
            )
            for core_dtype in RMA_DTYPES
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def putmem_signal(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("putmem_signal"),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi(
    dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None
):
    """Write bytes and an int32 signal using generic runtime routing."""
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("putmem_signal_nbi"),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi_lowlevel(
    dest,
    source,
    nbytes,
    sig_addr,
    signal,
    sig_op,
    pe,
    qp_id=None,
    *,
    transport=ACLSHMEMTransport.UDMA,
    _semantic=None,
):
    """Write bytes via UDMA with a uint64 SET signal; RoCE/MTE are unsupported."""
    transport = _transport_selector(transport, "putmem_signal_nbi_lowlevel")
    if transport != ACLSHMEMTransport.UDMA:
        raise ValueError("putmem_signal_nbi_lowlevel: only UDMA transport is supported")
    qp_id = core._unwrap_if_constexpr(qp_id)
    op = core._unwrap_if_constexpr(sig_op)
    if not isinstance(op, int) or isinstance(op, bool) or op != ACLSHMEMSignalOp.SET:
        raise ValueError(
            "putmem_signal_nbi_lowlevel: UDMA supports only compile-time ACLSHMEM_SIGNAL_SET"
        )
    if sig_addr.dtype != pi_u64_t:
        raise ValueError(
            "putmem_signal_nbi_lowlevel: UDMA requires a uint64 signal pointer"
        )
    size = _rma_index(nbytes, "nbytes", "putmem_signal_nbi_lowlevel", tl.uint32, _semantic)
    peer = _rma_index(pe, "pe", "putmem_signal_nbi_lowlevel", tl.int32, _semantic)
    extra = (
        []
        if qp_id is None
        else [
            _rma_index(qp_id, "qp_id", "putmem_signal_nbi_lowlevel", tl.uint32, _semantic)
        ]
    )
    ptr = _rma_pointer_type(dest, source, "putmem_signal_nbi_lowlevel")
    suffix = DTYPE_TO_KERNEL_SUFFIX[ptr.element_ty.name]
    operation = "putmem_signal_nbi" if qp_id is None else "qp_put_signal_nbi"
    call = extern_call(
        "libshmem_device",
        "",
        [
            dest,
            source,
            size,
            sig_addr,
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            peer,
        ]
        + extra,
        {
            (ptr, ptr, tl.uint32, pi_u64_t, tl.uint64, tl.int32)
            + (() if qp_id is None else (tl.uint32,)): (
                f"aclshmemx_udma_{operation}_{suffix}",
                (),
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    _check_enum_value(sig_op, ACLSHMEMSignalOp, "sig_op", "signal_op")
    call = extern_call(
        "libshmem_device",
        "",
        [
            sig_addr,
            tl.cast(signal, tl.int32, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_i32_t, tl.int32, tl.int32, tl.int32): (
                "aclshmemx_signal_op",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    _check_enum_value(cmp_, ACLSHMEMCmpOp, "cmp_", "signal_wait_until")
    call = extern_call(
        "libshmem_device",
        "",
        [
            sig_addr,
            tl.cast(cmp_, tl.int32, _semantic=_semantic),
            tl.cast(cmp_val, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_i32_t, tl.int32, tl.int32): (
                "aclshmem_signal_wait_until",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def quiet(_semantic=None):
    """Complete operations issued through generic runtime routing.

    For multi-QP completion, use quiet_lowlevel with an explicit transport, PE,
    and qp_id to complete operations on each specific QP.
    """
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {(): ("aclshmem_quiet", ())},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)


@core.extern
def quiet_lowlevel(pe=None, *, transport, qp_id=None, _semantic=None):
    """Complete MTE work on this core, or UDMA/RoCE work for a PE/QP.

    Low-level MTE and RoCE transports are experimental and not officially
    supported.

    With transport=ROCE, specifying qp_id requires the XSCALE
    or HNS_1825 backend; the IN_DIE backend does not support this QP-specific
    interface.
    """
    transport, qp_id = _rma_transport(transport, qp_id, "quiet_lowlevel")
    if transport == ACLSHMEMTransport.MTE:
        if core._unwrap_if_constexpr(pe) is not None:
            raise ValueError("quiet_lowlevel: MTE quiet is core-wide; omit pe")
        call = extern_call(
            "libshmem_device",
            "",
            [],
            {(): ("aclshmemx_mte_quiet", ())},
            is_pure=False,
            _semantic=_semantic,
        )
        return _mark_core_type(call, "cube_or_vector", _semantic)
    pe = core._unwrap_if_constexpr(pe)
    args = [_rma_index(pe, "pe", "quiet_lowlevel", tl.int32, _semantic)]
    types, symbol = (tl.int32,), "aclshmemx_udma_quiet"
    if qp_id is not None:
        args.append(_rma_index(qp_id, "qp_id", "quiet_lowlevel", tl.uint32, _semantic))
        types, symbol = types + (tl.uint32,), "aclshmemx_udma_qp_quiet"
    if transport == ACLSHMEMTransport.ROCE:
        symbol = (
            "aclshmemx_roce_" + ("qp_" if qp_id is not None else "") + "quiet_uint8"
        )
    call = extern_call(
        "libshmem_device",
        "",
        args,
        {types: (symbol, ())},
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "vector", _semantic)


@core.extern
def fence(_semantic=None):
    call = extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_fence", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    return _mark_core_type(call, "cube_or_vector", _semantic)
