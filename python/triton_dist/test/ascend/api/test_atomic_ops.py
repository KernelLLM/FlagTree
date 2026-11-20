# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.


import pytest
import shmem as ash
import torch

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend import libaclshmem_device

from shmem_api_test_utils import (
    capture_extern,
    check_extern_lowering,
    check_scalar_extern_abi,
    check_unsupported_dtype,
)
from utils import init_aclshmem

# Keep these ABI expectations independent of the implementation's dtype lists.
INT_TYPES = [
    (tl.int32, "int32", "i32"),
    (tl.uint32, "uint32", "i32"),
    (tl.int64, "int64", "i64"),
    (tl.uint64, "uint64", "i64"),
]
FLOAT_TYPE = (tl.float32, "float", "f32")
RMA_TYPES = INT_TYPES + [
    (tl.float16, "half", "f16"),
    FLOAT_TYPE,
    (tl.bfloat16, "bfloat16", "bf16"),
    (tl.int8, "int8", "i8"),
    (tl.uint8, "uint8", "i8"),
    (tl.int16, "int16", "i16"),
    (tl.uint16, "uint16", "i16"),
]
ATOMIC_OPS = [
    ("add", 1, False),
    ("fetch_add", 1, True),
    ("inc", 0, False),
    ("fetch_inc", 0, True),
    ("and", 1, False),
    ("or", 1, False),
    ("xor", 1, False),
    ("fetch_and", 1, True),
    ("fetch_or", 1, True),
    ("fetch_xor", 1, True),
    ("fetch", 0, True),
    ("set", 1, False),
    ("swap", 1, True),
    ("compare_swap", 2, True),
]

# atomic_add's MTE types are supported on Ascend910B/910C.
CI_ATOMIC_ADD_TYPES = [
    (torch.int8, "int8"),
    (torch.int16, "int16"),
    (torch.int32, "int32"),
    (torch.float16, "half"),
    (torch.bfloat16, "bfloat16"),
    (torch.float32, "float"),
]

# Ascend950-only types for atomic_add, atomic_fetch_add, atomic_inc,
# atomic_fetch_inc, and atomic_fetch.
DIST_ATOMIC_950_TYPES = [
    (torch.uint32, "uint32"),
    (torch.uint64, "uint64"),
    (torch.int64, "int64"),
]

# Ascend950-only logic operation types
DIST_ATOMIC_LOGIC_TYPES = [
    (torch.int32, "int32"),
    (torch.uint32, "uint32"),
    (torch.int64, "int64"),
    (torch.uint64, "uint64"),
]

# Ascend950-only swap operation types (compare_swap does not support float)
DIST_ATOMIC_SWAP_TYPES = [
    (torch.int32, "int32"),
    (torch.uint32, "uint32"),
    (torch.int64, "int64"),
    (torch.uint64, "uint64"),
    (torch.float32, "float"),
]

DIST_ATOMIC_COMPARE_SWAP_TYPES = [
    (torch.int32, "int32"),
    (torch.uint32, "uint32"),
    (torch.int64, "int64"),
    (torch.uint64, "uint64"),
]

G_ASH_SIZE = 1024 * 1024 * 1024


def _init_aclshmem(rank, world_size, engine=ash.OpEngineType.MTE):
    init_aclshmem(rank, world_size, G_ASH_SIZE, engine)


def _atomic_types(op):
    if op == "add":
        return [
            entry
            for entry in RMA_TYPES
            if entry[0]
            in (
                tl.int8,
                tl.int16,
                tl.bfloat16,
                tl.float16,
                tl.int32,
                tl.uint32,
                tl.int64,
                tl.uint64,
                tl.float32,
            )
        ]
    return INT_TYPES + ([FLOAT_TYPE] if op in ("set", "swap") else [])


def _atomic_api(op):
    if op == "compare_swap":
        return "atomic_compare_swap", None
    if op == "fetch":
        return "atomic_fetch_op", libshmem_device.ACLSHMEM_AMO_NONE
    if op.startswith("fetch_"):
        return "atomic_fetch_op", getattr(
            libshmem_device, "ACLSHMEM_AMO_" + op.removeprefix("fetch_").upper()
        )
    return "atomic_op", getattr(libshmem_device, "ACLSHMEM_AMO_" + op.upper())


ATOMIC_CASES = [
    (op, num_values, returns_value, dtype, suffix, ir_type)
    for op, num_values, returns_value in ATOMIC_OPS
    for dtype, suffix, ir_type in _atomic_types(op)
]
pytestmark = [pytest.mark.dist, pytest.mark.usefixtures("ascend950_atomic_target")]


@triton.jit
def sync_all_kernel():
    if tl.program_id(axis=0) == 0:
        libshmem_device.sync_all()


@triton.jit
def barrier_all_kernel():
    if tl.program_id(axis=0) == 0:
        libshmem_device.barrier_all()


@pytest.mark.parametrize("dtype,suffix,ir_type", RMA_TYPES)
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.ci
def test_scalar_get_abi(monkeypatch, dtype, suffix, ir_type, public):
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    module = libshmem_device if public else libaclshmem_device
    type_map = {
        (tl.pointer_type(ty), tl.int32): (f"aclshmem_g_{name}", ty)
        for ty, name, _ in RMA_TYPES
    }
    check_scalar_extern_abi(
        monkeypatch, module, "g", f"aclshmem_g_{suffix}", dtype, 0, type_map
    )


@pytest.mark.parametrize(
    "op,num_values,returns_value,dtype,suffix,ir_type", ATOMIC_CASES
)
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.ci
def test_atomic_extern_abi(
    monkeypatch, op, num_values, returns_value, dtype, suffix, ir_type, public
):
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    module = libshmem_device if public else libaclshmem_device
    type_map = {
        (tl.pointer_type(ty),)
        + (ty,) * num_values
        + (tl.int32,): (f"aclshmem_atomic_{op}_{name}", ty if returns_value else ())
        for ty, name, _ in _atomic_types(op)
    }
    check_scalar_extern_abi(
        monkeypatch,
        module,
        _atomic_api(op)[0],
        f"aclshmem_atomic_{op}_{suffix}",
        dtype,
        num_values,
        type_map,
        op=_atomic_api(op)[1],
    )


@pytest.mark.parametrize("dtype", [tl.float64, tl.int1])
@pytest.mark.ci
def test_scalar_get_rejects_unsupported_dtype(monkeypatch, dtype):
    check_unsupported_dtype(monkeypatch, "g", dtype, 0)


@pytest.mark.parametrize(
    "op,num_values,dtype",
    [
        (op, num_values, dtype)
        for op, num_values, _ in ATOMIC_OPS
        for dtype in [ty for ty, _, _ in RMA_TYPES] + [tl.float64, tl.int1]
        if dtype not in [ty for ty, _, _ in _atomic_types(op)]
    ],
)
@pytest.mark.ci
def test_atomic_rejects_unsupported_dtype(monkeypatch, op, num_values, dtype):
    check_unsupported_dtype(
        monkeypatch, _atomic_api(op)[0], dtype, num_values, op=_atomic_api(op)[1]
    )


@pytest.mark.parametrize("dtype,suffix,ir_type", RMA_TYPES)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.ci
def test_scalar_get_lowering(dtype, suffix, ir_type, mixed):
    check_extern_lowering(
        f"aclshmem_g_{suffix}",
        [f"!tt.ptr<{ir_type}>", "i32"],
        ir_type,
        "VECTOR",
        False,
        mixed,
    )


@pytest.mark.parametrize(
    "op,num_values,returns_value,dtype,suffix,ir_type", ATOMIC_CASES
)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.ci
def test_atomic_lowering(op, num_values, returns_value, dtype, suffix, ir_type, mixed):
    check_extern_lowering(
        f"aclshmem_atomic_{op}_{suffix}",
        [f"!tt.ptr<{ir_type}>"] + [ir_type] * num_values + ["i32"],
        ir_type if returns_value else None,
        "VECTOR",
        False,
        mixed,
    )


@pytest.mark.parametrize(
    "name", ["NONE", "INC", "ADD", "AND", "OR", "XOR", "SET", "SWAP"]
)
@pytest.mark.ci
def test_atomic_constants(name):
    from enum import IntEnum
    from triton_dist.language.extra.ascend.aclshmem_constants import ACLSHMEMAtomicOp

    constant = getattr(libshmem_device, "ACLSHMEM_AMO_" + name)
    assert isinstance(constant, IntEnum)
    assert constant is ACLSHMEMAtomicOp[name]


@pytest.mark.parametrize("api", ["atomic_op", "atomic_fetch_op"])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "op",
    [
        True,
        False,
        0,
        1,
        99,
        None,
        "add",
        "ADD",
        "fetch_add",
        "invalid",
        libshmem_device.ACLSHMEM_SIGNAL_ADD,
    ],
)
@pytest.mark.ci
def test_atomic_rejects_invalid_selector(monkeypatch, api, public, wrapped, op):
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    module = libshmem_device if public else libaclshmem_device
    semantic, calls = capture_extern(monkeypatch)
    selector = tl.constexpr(op) if wrapped else op
    with pytest.raises(ValueError, match="compile-time ACLSHMEMAtomicOp"):
        getattr(module, api)(None, 1, 0, selector, _semantic=semantic)
    assert not calls


@pytest.mark.parametrize(
    "api,op",
    [("atomic_op", "none"), ("atomic_fetch_op", "set"), ("atomic_fetch_op", "swap")],
)
@pytest.mark.ci
def test_atomic_rejects_selector_from_other_group(monkeypatch, api, op):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match="compile-time ACLSHMEMAtomicOp"):
        getattr(libaclshmem_device, api)(
            None,
            1,
            0,
            getattr(libshmem_device, "ACLSHMEM_AMO_" + op.upper()),
            _semantic=semantic,
        )
    assert not calls


@pytest.mark.parametrize("api", ["atomic_op", "atomic_fetch_op"])
@pytest.mark.ci
def test_atomic_rejects_runtime_selector(monkeypatch, api):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match="compile-time ACLSHMEMAtomicOp"):
        getattr(libaclshmem_device, api)(
            None, 1, 0, tl.tensor(None, tl.int32), _semantic=semantic
        )
    assert not calls


@pytest.mark.parametrize(
    "api,op",
    [("atomic_op", "inc"), ("atomic_fetch_op", "inc"), ("atomic_fetch_op", "none")],
)
@pytest.mark.parametrize("value", [0, 1, tl.constexpr(0), tl.tensor(None, tl.int32)])
@pytest.mark.ci
def test_atomic_requires_none_placeholder(monkeypatch, api, op, value):
    semantic, calls = capture_extern(monkeypatch)
    with pytest.raises(ValueError, match="value must be None"):
        getattr(libaclshmem_device, api)(
            None,
            value,
            0,
            getattr(libshmem_device, "ACLSHMEM_AMO_" + op.upper()),
            _semantic=semantic,
        )
    assert not calls


@triton.jit
def grouped_atomic_kernel(dest, result, OP: tl.constexpr, FETCH: tl.constexpr):
    if FETCH:
        if (
            OP == libshmem_device.ACLSHMEM_AMO_INC
            or OP == libshmem_device.ACLSHMEM_AMO_NONE
        ):
            old = libshmem_device.atomic_fetch_op(dest, None, 1, OP)
        else:
            old = libshmem_device.atomic_fetch_op(dest, 7, 1, OP)
        tl.store(result, old)
    elif OP == "compare_swap":
        old = libshmem_device.atomic_compare_swap(dest, 3, 7, 1)
        tl.store(result, old)
    elif OP == libshmem_device.ACLSHMEM_AMO_SWAP:
        old = libshmem_device.atomic_op(dest, 7, 1, OP)
        tl.store(result, old)
    elif OP == libshmem_device.ACLSHMEM_AMO_INC:
        libshmem_device.atomic_op(dest, None, 1, OP)
    else:
        libshmem_device.atomic_op(dest, 7, 1, OP)


@pytest.mark.parametrize(
    "op,num_values,returns_value,dtype,suffix,ir_type", ATOMIC_CASES
)
@pytest.mark.ci
def test_grouped_atomic_jit(
    monkeypatch, op, num_values, returns_value, dtype, suffix, ir_type
):
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.ascend.compiler import AscendBackend
    from triton.compiler.compiler import ASTSource
    from test_gm_addr_args_indices import _run_pass

    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    target = GPUTarget("npu", "Ascend950DT_9582", 32)
    backend = AscendBackend(target)
    options = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    type_name = dtype.name.replace("uint", "u").replace("int", "i")
    api, selector = _atomic_api(op)
    source = ASTSource(
        grouped_atomic_kernel,
        signature={"dest": "*" + type_name, "result": "*" + type_name},
        constexprs={
            "OP": selector if selector is not None else "compare_swap",
            "FETCH": api == "atomic_fetch_op",
        },
    )
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        ctx,
    )
    symbol = f"aclshmem_atomic_{op}_{suffix}"
    before = module.str()
    assert f'symbol = "{symbol}"' in before
    assert ("tt.store" in before) == returns_value
    _run_pass(module, ctx)
    custom = [line for line in module.str().splitlines() if "hir.custom" in line]
    assert len(custom) == 1
    assert f'symbol = "{symbol}"' in custom[0]
    assert "#hivm.tcore_type<VECTOR>" in custom[0]
    assert "gm_addr_args_indices = array<i32: 0>" in custom[0]


# Kernel-based runtime tests


@triton.jit
def scalar_get_kernel(source_ptr, result_ptr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        val = libshmem_device.g(source_ptr, peer_rank)
        tl.store(result_ptr, val)


@triton.jit
def atomic_add_kernel(counter_ptr, value, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            counter_ptr, value, peer_rank, libshmem_device.ACLSHMEM_AMO_ADD
        )
        libshmem_device.quiet()


@triton.jit
def atomic_fetch_add_kernel(counter_ptr, value, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_fetch_op(
            counter_ptr, value, peer_rank, libshmem_device.ACLSHMEM_AMO_ADD
        )


@triton.jit
def atomic_inc_kernel(counter_ptr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            counter_ptr, None, peer_rank, libshmem_device.ACLSHMEM_AMO_INC
        )
        libshmem_device.quiet()


@triton.jit
def atomic_fetch_inc_kernel(counter_ptr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_fetch_op(
            counter_ptr, None, peer_rank, libshmem_device.ACLSHMEM_AMO_INC
        )


@triton.jit
def atomic_swap_kernel(target_ptr, new_val, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            target_ptr, new_val, peer_rank, libshmem_device.ACLSHMEM_AMO_SWAP
        )


@triton.jit
def atomic_compare_swap_kernel(target_ptr, cond, new_val, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_compare_swap(target_ptr, cond, new_val, peer_rank)


@triton.jit
def atomic_fetch_kernel(source_ptr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_fetch_op(
            source_ptr, None, peer_rank, libshmem_device.ACLSHMEM_AMO_NONE
        )


@triton.jit
def atomic_set_kernel(target_ptr, new_val, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            target_ptr, new_val, peer_rank, libshmem_device.ACLSHMEM_AMO_SET
        )
        libshmem_device.quiet()


def run_test_atomic_add_ci(rank, world_size):
    _init_aclshmem(rank, world_size)

    next_peer = (rank + 1) % world_size
    for torch_dtype, suffix in CI_ATOMIC_ADD_TYPES:
        counter = ash.aclshmem_create_tensor([1], dtype=torch_dtype, device_id=rank)
        _fill(counter, 0)

        for _ in range(5):
            atomic_add_kernel[1, 1, 1](counter, 1, next_peer)
        barrier_all_kernel[1, 1, 1]()

        expected = 5
        actual = counter[0].cpu()
        if torch_dtype.is_floating_point:
            assert torch.isclose(
                actual, torch.tensor(expected, dtype=torch_dtype)
            ), f"Rank {rank}: atomic_add {suffix} got {actual}, expected {expected}"
        else:
            assert (
                actual == expected
            ), f"Rank {rank}: atomic_add {suffix} got {actual}, expected {expected}"

        barrier_all_kernel[1, 1, 1]()
        ash.aclshmem_free_tensor(counter)

    print(f"Rank {rank}: atomic_add CI types OK")
    _ = ash.aclshmem_finalize()


@pytest.mark.ci
def test_atomic_add_ci(dist_test):
    dist_test(run_test_atomic_add_ci, world_size=2)


def _fill(tensor, value):
    """Fill a tensor avoiding NPU dtype promotion errors for unsigned types."""
    cpu_tensor = torch.full_like(tensor, value, dtype=tensor.dtype, device="cpu")
    tensor.copy_(cpu_tensor)


def run_test_atomic_950_ops(rank, world_size):
    _init_aclshmem(rank, world_size)

    for torch_dtype, suffix in DIST_ATOMIC_950_TYPES:
        counter = ash.aclshmem_create_tensor([1], dtype=torch_dtype, device_id=rank)
        _fill(counter, 0)

        next_peer = (rank + 1) % world_size

        sync_all_kernel[1, 1, 1]()
        for _ in range(5):
            atomic_add_kernel[1, 1, 1](counter, 2, next_peer)
        barrier_all_kernel[1, 1, 1]()

        expected = 10  # 5 iterations × value 2, one writer per rank
        assert (
            counter[0].cpu() == expected
        ), f"Rank {rank}: atomic_add {suffix} mismatch, got {counter[0].cpu()}"

        _fill(counter, 0)
        sync_all_kernel[1, 1, 1]()
        for i in range(5):
            atomic_fetch_add_kernel[1, 1, 1](counter, 2, next_peer)
            sync_all_kernel[1, 1, 1]()

        expected = 10  # 5 iterations × value 2, one writer per rank
        assert (
            counter[0].cpu() == expected
        ), f"Rank {rank}: atomic_fetch_add {suffix} mismatch, got {counter[0].cpu()}"

        _fill(counter, 0)
        for i in range(5):
            atomic_inc_kernel[1, 1, 1](counter, next_peer)
            sync_all_kernel[1, 1, 1]()

        expected = 5  # 5 increments, one writer per rank
        assert (
            counter[0].cpu() == expected
        ), f"Rank {rank}: atomic_inc {suffix} mismatch, got {counter[0].cpu()}"

        _fill(counter, 0)
        for i in range(5):
            atomic_fetch_inc_kernel[1, 1, 1](counter, next_peer)
            sync_all_kernel[1, 1, 1]()

        expected = 5  # 5 increments, one writer per rank
        assert (
            counter[0].cpu() == expected
        ), f"Rank {rank}: atomic_fetch_inc {suffix} mismatch, got {counter[0].cpu()}"

        _fill(counter, 42)
        sync_all_kernel[1, 1, 1]()
        atomic_fetch_kernel[1, 1, 1](counter, next_peer)
        sync_all_kernel[1, 1, 1]()
        assert counter[0].cpu() == 42, f"Rank {rank}: atomic_fetch {suffix} mismatch"

        sync_all_kernel[1, 1, 1]()
        ash.aclshmem_free_tensor(counter)

    print(f"Rank {rank}: atomic 950 ops OK")
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_atomic_950_ops(dist_test):
    dist_test(run_test_atomic_950_ops, world_size=2)


def run_test_atomic_swap_ops(rank, world_size):
    _init_aclshmem(rank, world_size)

    for torch_dtype, suffix in DIST_ATOMIC_SWAP_TYPES:
        target = ash.aclshmem_create_tensor([1], dtype=torch_dtype, device_id=rank)
        _fill(target, 100)

        next_peer = (rank + 1) % world_size

        sync_all_kernel[1, 1, 1]()
        new_val = 200 + rank
        atomic_swap_kernel[1, 1, 1](target, new_val, next_peer)
        sync_all_kernel[1, 1, 1]()

        expected = 200 + ((rank - 1 + world_size) % world_size)
        if torch_dtype.is_floating_point:
            assert torch.isclose(
                target[0].cpu(), torch.tensor(expected, dtype=torch_dtype)
            ), f"Rank {rank}: atomic_swap {suffix} mismatch"
        else:
            assert (
                target[0].cpu() == expected
            ), f"Rank {rank}: atomic_swap {suffix} mismatch"

        _fill(target, 50)
        sync_all_kernel[1, 1, 1]()
        atomic_set_kernel[1, 1, 1](target, 77, next_peer)
        sync_all_kernel[1, 1, 1]()

        assert target[0].cpu() == 77, f"Rank {rank}: atomic_set {suffix} mismatch"

        sync_all_kernel[1, 1, 1]()
        ash.aclshmem_free_tensor(target)

    for torch_dtype, suffix in DIST_ATOMIC_COMPARE_SWAP_TYPES:
        target = ash.aclshmem_create_tensor([1], dtype=torch_dtype, device_id=rank)
        next_peer = (rank + 1) % world_size

        _fill(target, 99)
        sync_all_kernel[1, 1, 1]()
        atomic_compare_swap_kernel[1, 1, 1](target, 99, 123, next_peer)
        sync_all_kernel[1, 1, 1]()

        assert (
            target[0].cpu() == 123
        ), f"Rank {rank}: atomic_compare_swap {suffix} match case mismatch"

        sync_all_kernel[1, 1, 1]()
        atomic_compare_swap_kernel[1, 1, 1](target, 999, 456, next_peer)
        sync_all_kernel[1, 1, 1]()

        assert (
            target[0].cpu() == 123
        ), f"Rank {rank}: atomic_compare_swap {suffix} no-match case mismatch"

        sync_all_kernel[1, 1, 1]()
        ash.aclshmem_free_tensor(target)

    print(f"Rank {rank}: atomic swap ops OK")
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_atomic_swap_ops(dist_test):
    dist_test(run_test_atomic_swap_ops, world_size=2)


@triton.jit
def atomic_and_kernel(target_ptr, mask, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            target_ptr, mask, peer_rank, libshmem_device.ACLSHMEM_AMO_AND
        )
        libshmem_device.quiet()


@triton.jit
def atomic_or_kernel(target_ptr, mask, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            target_ptr, mask, peer_rank, libshmem_device.ACLSHMEM_AMO_OR
        )
        libshmem_device.quiet()


@triton.jit
def atomic_xor_kernel(target_ptr, mask, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_op(
            target_ptr, mask, peer_rank, libshmem_device.ACLSHMEM_AMO_XOR
        )
        libshmem_device.quiet()


@triton.jit
def atomic_fetch_and_kernel(target_ptr, mask, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_fetch_op(
            target_ptr, mask, peer_rank, libshmem_device.ACLSHMEM_AMO_AND
        )


@triton.jit
def atomic_fetch_or_kernel(target_ptr, mask, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_fetch_op(
            target_ptr, mask, peer_rank, libshmem_device.ACLSHMEM_AMO_OR
        )


@triton.jit
def atomic_fetch_xor_kernel(target_ptr, mask, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.atomic_fetch_op(
            target_ptr, mask, peer_rank, libshmem_device.ACLSHMEM_AMO_XOR
        )


def run_test_atomic_logic_ops(rank, world_size):
    logic_engine = ash.OpEngineType(ash.OpEngineType.UDMA.value)
    _init_aclshmem(rank, world_size, logic_engine)

    for torch_dtype, suffix in DIST_ATOMIC_LOGIC_TYPES:
        target = ash.aclshmem_create_tensor([1], dtype=torch_dtype, device_id=rank)
        next_peer = (rank + 1) % world_size

        # Each rank writes to next_peer, so check what prev_peer wrote locally
        # AND: prev writes 0x0F to local 0xFF -> 0x0F
        _fill(target, 0xFF)
        barrier_all_kernel[1, 1, 1]()
        atomic_and_kernel[1, 1, 1](target, 0x0F, next_peer)
        barrier_all_kernel[1, 1, 1]()
        expected = 0x0F
        assert (
            target[0].cpu() == expected
        ), f"Rank {rank}: atomic_and {suffix} got {target[0].cpu():x}, expected {expected:x}"

        # OR: prev writes 0xF0 to local 0x0F -> 0xFF
        _fill(target, 0x0F)
        barrier_all_kernel[1, 1, 1]()
        atomic_or_kernel[1, 1, 1](target, 0xF0, next_peer)
        barrier_all_kernel[1, 1, 1]()
        expected = 0xFF
        assert (
            target[0].cpu() == expected
        ), f"Rank {rank}: atomic_or {suffix} got {target[0].cpu():x}, expected {expected:x}"

        # XOR: prev writes 0x0F to local 0xFF -> 0xF0
        _fill(target, 0xFF)
        barrier_all_kernel[1, 1, 1]()
        atomic_xor_kernel[1, 1, 1](target, 0x0F, next_peer)
        barrier_all_kernel[1, 1, 1]()
        expected = 0xF0
        assert (
            target[0].cpu() == expected
        ), f"Rank {rank}: atomic_xor {suffix} got {target[0].cpu():x}, expected {expected:x}"

        # fetch_and: prev writes 0x0F to local 0xFF -> 0x0F
        _fill(target, 0xFF)
        barrier_all_kernel[1, 1, 1]()
        atomic_fetch_and_kernel[1, 1, 1](target, 0x0F, next_peer)
        barrier_all_kernel[1, 1, 1]()
        expected = 0x0F
        assert (
            target[0].cpu() == expected
        ), f"Rank {rank}: atomic_fetch_and {suffix} got {target[0].cpu():x}, expected {expected:x}"

        # fetch_or: prev writes 0xF0 to local 0x0F -> 0xFF
        _fill(target, 0x0F)
        barrier_all_kernel[1, 1, 1]()
        atomic_fetch_or_kernel[1, 1, 1](target, 0xF0, next_peer)
        barrier_all_kernel[1, 1, 1]()
        expected = 0xFF
        assert (
            target[0].cpu() == expected
        ), f"Rank {rank}: atomic_fetch_or {suffix} got {target[0].cpu():x}, expected {expected:x}"

        # fetch_xor: prev writes 0x0F to local 0xFF -> 0xF0
        _fill(target, 0xFF)
        barrier_all_kernel[1, 1, 1]()
        atomic_fetch_xor_kernel[1, 1, 1](target, 0x0F, next_peer)
        barrier_all_kernel[1, 1, 1]()
        expected = 0xF0
        assert (
            target[0].cpu() == expected
        ), f"Rank {rank}: atomic_fetch_xor {suffix} got {target[0].cpu():x}, expected {expected:x}"

        barrier_all_kernel[1, 1, 1]()
        ash.aclshmem_free_tensor(target)

    print(f"Rank {rank}: atomic logic ops OK")
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_atomic_logic_ops(dist_test):
    dist_test(run_test_atomic_logic_ops, world_size=2)
