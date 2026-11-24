# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import shutil

import pytest
import torch
import torch.distributed as dist

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from test_atomic_ops import _atomic_api

# Runtime coverage adapted from feat-1230-atomic-hl. Explicit transport modes
# exercise the unified Python API rather than separate engine-specific names.
CI_ATOMIC_ADD_TYPES = [
    (torch.int8, "int8"),
    (torch.int16, "int16"),
    (torch.int32, "int32"),
    (torch.float16, "half"),
    (torch.bfloat16, "bfloat16"),
    (torch.float32, "float"),
]
DIST_ATOMIC_950_TYPES = [
    (torch.int32, "int32"),
    (torch.uint32, "uint32"),
    (torch.int64, "int64"),
    (torch.uint64, "uint64"),
]
DIST_ATOMIC_LOGIC_TYPES = DIST_ATOMIC_950_TYPES
DIST_ATOMIC_COMPARE_SWAP_TYPES = DIST_ATOMIC_950_TYPES
DIST_ATOMIC_SWAP_TYPES = DIST_ATOMIC_950_TYPES + [(torch.float32, "float")]
RUNTIME_TRANSPORTS = [
    pytest.param(
        libshmem_device.ACLSHMEM_TRANSPORT_AUTO, marks=pytest.mark.dist, id="auto"
    ),
    pytest.param(
        libshmem_device.ACLSHMEM_TRANSPORT_UDMA, marks=pytest.mark.dist, id="udma"
    ),
    pytest.param(
        libshmem_device.ACLSHMEM_TRANSPORT_MTE, marks=pytest.mark.dist, id="mte"
    ),
    # No dist mark: RoCE runtime coverage is pending official enablement.
    pytest.param(libshmem_device.ACLSHMEM_TRANSPORT_ROCE, id="roce"),
]
G_ASH_SIZE = 1024 * 1024 * 1024


def _require_atomic_runtime(ascend950=False, devices=2):
    # Keep ABI/lowering tests collectable without loading the SHMEM runtime.
    if shutil.which("npu-smi") is None:
        pytest.skip("atomic runtime tests require Ascend NPUs")
    pytest.importorskip("shmem")
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() < devices:
        pytest.skip(f"SHMEM runtime tests require {devices} local NPUs")
    if ascend950:
        acl = pytest.importorskip("acl")
        if not acl.get_soc_name().startswith("Ascend950"):
            pytest.skip("these atomic operations require Ascend950")


def _init_aclshmem(rank, world_size, transport, logic=False, qp_count=None):
    import shmem as ash

    from utils import init_aclshmem

    completion_transport = transport
    engine = ash.OpEngineType.MTE.value
    if transport == libshmem_device.ACLSHMEM_TRANSPORT_ROCE:
        engine |= ash.OpEngineType.ROCE.value
    elif transport == libshmem_device.ACLSHMEM_TRANSPORT_UDMA or logic:
        # Generic bitwise atomics cannot use MTE. AUTO therefore needs UDMA
        # initialized too; keep API selection AUTO while draining UDMA below.
        engine |= ash.OpEngineType.UDMA.value
        completion_transport = libshmem_device.ACLSHMEM_TRANSPORT_UDMA
    if qp_count is not None:
        qp_engine = (
            ash.OpEngineType.ROCE
            if transport == libshmem_device.ACLSHMEM_TRANSPORT_ROCE
            else ash.OpEngineType.UDMA
        )
        assert ash.aclshmemx_set_qp_num(qp_engine, qp_count) == 0
    init_aclshmem(rank, world_size, G_ASH_SIZE, ash.OpEngineType(engine))
    return ash, completion_transport


@triton.jit
def scalar_get_kernel(source_ptr, result_ptr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        tl.store(result_ptr, libshmem_device.g(source_ptr, peer_rank))
        libshmem_device.quiet()


@triton.jit
def atomic_runtime_kernel(
    counter_ptr,
    result_ptr,
    value,
    cond,
    peer_rank: tl.constexpr,
    OP: tl.constexpr,
    SELECTOR: tl.constexpr,
    NUM_VALUES: tl.constexpr,
    RETURNS_VALUE: tl.constexpr,
    TRANSPORT: tl.constexpr,
    COMPLETION_TRANSPORT: tl.constexpr,
):
    # UDMA's per-peer atomic scratch/QP must have just one issuing core.
    if tl.program_id(axis=0) == 0:
        if NUM_VALUES == 0:
            old = OP(counter_ptr, None, peer_rank, SELECTOR, transport=TRANSPORT)
        elif NUM_VALUES == 1:
            old = OP(counter_ptr, value, peer_rank, SELECTOR, transport=TRANSPORT)
        else:
            old = OP(counter_ptr, cond, value, peer_rank, transport=TRANSPORT)
        if RETURNS_VALUE:
            tl.store(result_ptr, old)
        if COMPLETION_TRANSPORT == libshmem_device.ACLSHMEM_TRANSPORT_UDMA:
            libshmem_device.quiet_lowlevel(
                peer_rank, transport=libshmem_device.ACLSHMEM_TRANSPORT_UDMA
            )
        elif COMPLETION_TRANSPORT == libshmem_device.ACLSHMEM_TRANSPORT_ROCE:
            if not RETURNS_VALUE:
                # Generic quiet only drains instruction pipes/caches. A RoCE
                # fetch also waits on this PE/QP, completing preceding AMOs.
                libshmem_device.atomic_fetch_op(
                    counter_ptr,
                    None,
                    peer_rank,
                    libshmem_device.ACLSHMEM_AMO_NONE,
                    transport=libshmem_device.ACLSHMEM_TRANSPORT_ROCE,
                )
        if COMPLETION_TRANSPORT == libshmem_device.ACLSHMEM_TRANSPORT_MTE:
            libshmem_device.quiet_lowlevel(transport=COMPLETION_TRANSPORT)
        else:
            libshmem_device.quiet()


def _fill(tensor, value):
    """Avoid NPU dtype promotion for unsigned tensors."""
    tensor.copy_(torch.full_like(tensor, value, dtype=tensor.dtype, device="cpu"))


def _synchronize_ranks():
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.synchronize()


def _runtime_case(
    ash,
    rank,
    world_size,
    dtype,
    op,
    initial,
    value,
    expected,
    transport,
    completion_transport,
    returns_value=False,
    cond=0,
    repetitions=1,
    expected_old=None,
    step=0,
):
    """Return observations; assert after all ranks finish collective cleanup."""
    counter = ash.aclshmem_create_tensor([1], dtype=dtype, device_id=rank)
    result = torch.empty([1], dtype=dtype, device=f"npu:{rank}")
    checks = []
    try:
        _fill(counter, initial)
        _synchronize_ranks()
        num_values = (
            0
            if op in ("inc", "fetch_inc", "fetch")
            else 2 if op == "compare_swap" else 1
        )
        for iteration in range(repetitions):
            atomic_runtime_kernel[1, 1, 1](
                counter,
                result,
                value,
                cond,
                (rank + 1) % world_size,
                getattr(libshmem_device, _atomic_api(op)[0]),
                _atomic_api(op)[1],
                num_values,
                returns_value,
                transport,
                completion_transport,
            )
            _synchronize_ranks()
            if returns_value:
                checks.append(
                    (
                        f"{op} {dtype} returned value at iteration {iteration}",
                        result.cpu().item(),
                        expected_old + iteration * step,
                    )
                )
        checks.append((f"{op} {dtype} destination", counter.cpu().item(), expected))
        _synchronize_ranks()
    finally:
        torch.npu.synchronize()
        ash.aclshmem_free_tensor(counter)
    return checks


def _assert_runtime_checks(rank, checks):
    for description, actual, expected in checks:
        assert (
            actual == expected
        ), f"Rank {rank}: {description}: got {actual}, expected {expected}"


def run_test_atomic_add_ci(rank, world_size, transport):
    ash, completion = _init_aclshmem(rank, world_size, transport)
    checks = []
    try:
        for dtype, suffix in CI_ATOMIC_ADD_TYPES:
            checks.extend(
                _runtime_case(
                    ash,
                    rank,
                    world_size,
                    dtype,
                    "add",
                    0,
                    rank + 1,
                    5 * ((rank - 1) % world_size + 1),
                    transport,
                    completion,
                    repetitions=5,
                )
            )
    finally:
        assert ash.aclshmem_finalize() == 0
    _assert_runtime_checks(rank, checks)


@pytest.mark.ci
@pytest.mark.dist
@pytest.mark.parametrize(
    "transport",
    [
        libshmem_device.ACLSHMEM_TRANSPORT_AUTO,
        libshmem_device.ACLSHMEM_TRANSPORT_MTE,
    ],
)
def test_atomic_add_ci(dist_test, transport):
    _require_atomic_runtime()
    dist_test(run_test_atomic_add_ci, world_size=2, args=(transport,))


def run_test_atomic_950_ops(rank, world_size, transport):
    ash, completion = _init_aclshmem(rank, world_size, transport)
    checks = []
    try:
        types = DIST_ATOMIC_950_TYPES
        if transport == libshmem_device.ACLSHMEM_TRANSPORT_MTE:
            types = types + [(torch.float32, "float")]
        for dtype, suffix in types:
            for op, value, delta, returns in (
                ("add", 2, 2, False),
                ("fetch_add", 2, 2, True),
                ("inc", 0, 1, False),
                ("fetch_inc", 0, 1, True),
            ):
                checks.extend(
                    _runtime_case(
                        ash,
                        rank,
                        world_size,
                        dtype,
                        op,
                        0,
                        value,
                        5 * delta,
                        transport,
                        completion,
                        returns_value=returns,
                        repetitions=5,
                        expected_old=0,
                        step=delta,
                    )
                )
            checks.extend(
                _runtime_case(
                    ash,
                    rank,
                    world_size,
                    dtype,
                    "fetch",
                    42,
                    0,
                    42,
                    transport,
                    completion,
                    returns_value=True,
                    expected_old=42,
                )
            )
        if transport == libshmem_device.ACLSHMEM_TRANSPORT_UDMA:
            checks.extend(
                _runtime_case(
                    ash,
                    rank,
                    world_size,
                    torch.float32,
                    "add",
                    0,
                    1.5,
                    7.5,
                    transport,
                    completion,
                    repetitions=5,
                )
            )
    finally:
        assert ash.aclshmem_finalize() == 0
    _assert_runtime_checks(rank, checks)


@pytest.mark.parametrize("transport", RUNTIME_TRANSPORTS)
def test_atomic_950_ops(dist_test, transport):
    _require_atomic_runtime(ascend950=True)
    dist_test(run_test_atomic_950_ops, world_size=2, args=(transport,))


def run_test_atomic_swap_ops(rank, world_size, transport):
    ash, completion = _init_aclshmem(rank, world_size, transport)
    checks = []
    try:
        types = DIST_ATOMIC_SWAP_TYPES
        if transport == libshmem_device.ACLSHMEM_TRANSPORT_UDMA:
            types = DIST_ATOMIC_950_TYPES  # UDMA set/swap do not support float.
        if transport == libshmem_device.ACLSHMEM_TRANSPORT_MTE:
            types = [(torch.uint32, "uint32"), (torch.uint64, "uint64")]
        for dtype, suffix in types:
            checks.extend(
                _runtime_case(
                    ash,
                    rank,
                    world_size,
                    dtype,
                    "swap",
                    100,
                    200 + rank,
                    200 + (rank - 1) % world_size,
                    transport,
                    completion,
                    returns_value=True,
                    expected_old=100,
                )
            )
            checks.extend(
                _runtime_case(
                    ash,
                    rank,
                    world_size,
                    dtype,
                    "set",
                    50,
                    77,
                    77,
                    transport,
                    completion,
                )
            )
            if (
                dtype == torch.float32
                and transport == libshmem_device.ACLSHMEM_TRANSPORT_ROCE
            ):
                checks.extend(
                    _runtime_case(
                        ash,
                        rank,
                        world_size,
                        dtype,
                        "fetch",
                        42.5,
                        0,
                        42.5,
                        transport,
                        completion,
                        returns_value=True,
                        expected_old=42.5,
                    )
                )
        compare_types = DIST_ATOMIC_COMPARE_SWAP_TYPES
        if transport == libshmem_device.ACLSHMEM_TRANSPORT_MTE:
            compare_types = [(torch.uint32, "uint32"), (torch.uint64, "uint64")]
        for dtype, suffix in compare_types:
            for cond, expected in ((99, 123), (999, 99)):
                checks.extend(
                    _runtime_case(
                        ash,
                        rank,
                        world_size,
                        dtype,
                        "compare_swap",
                        99,
                        123,
                        expected,
                        transport,
                        completion,
                        returns_value=True,
                        cond=cond,
                        expected_old=99,
                    )
                )
    finally:
        assert ash.aclshmem_finalize() == 0
    _assert_runtime_checks(rank, checks)


@pytest.mark.parametrize("transport", RUNTIME_TRANSPORTS)
def test_atomic_swap_ops(dist_test, transport):
    _require_atomic_runtime(ascend950=True)
    dist_test(run_test_atomic_swap_ops, world_size=2, args=(transport,))


def run_test_atomic_logic_ops(rank, world_size, transport):
    ash, completion = _init_aclshmem(rank, world_size, transport, logic=True)
    checks = []
    try:
        for dtype, suffix in DIST_ATOMIC_LOGIC_TYPES:
            for op, initial, mask, expected in (
                ("and", 0xFF, 0x0F, 0x0F),
                ("or", 0x0F, 0xF0, 0xFF),
                ("xor", 0xFF, 0x0F, 0xF0),
            ):
                for fetch in (False, True):
                    name = f"fetch_{op}" if fetch else op
                    checks.extend(
                        _runtime_case(
                            ash,
                            rank,
                            world_size,
                            dtype,
                            name,
                            initial,
                            mask,
                            expected,
                            transport,
                            completion,
                            returns_value=fetch,
                            expected_old=initial,
                        )
                    )
    finally:
        assert ash.aclshmem_finalize() == 0
    _assert_runtime_checks(rank, checks)


# MTE does not implement bitwise atomics.
@pytest.mark.parametrize(
    "transport",
    [
        case
        for case in RUNTIME_TRANSPORTS
        if case.values[0] != libshmem_device.ACLSHMEM_TRANSPORT_MTE
    ],
)
def test_atomic_logic_ops(dist_test, transport):
    _require_atomic_runtime(ascend950=True)
    dist_test(run_test_atomic_logic_ops, world_size=2, args=(transport,))


def run_test_scalar_get(rank, world_size):
    ash, _ = _init_aclshmem(rank, world_size, libshmem_device.ACLSHMEM_TRANSPORT_AUTO)
    checks = []
    try:
        # Exercise every wrapper dtype, including unsigned widths.
        for dtype in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.uint16,
            torch.int32,
            torch.uint32,
            torch.int64,
            torch.uint64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            source = ash.aclshmem_create_tensor([1], dtype=dtype, device_id=rank)
            result = torch.empty([1], dtype=dtype, device=f"npu:{rank}")
            try:
                _fill(source, 42 + rank)
                _synchronize_ranks()
                scalar_get_kernel[1, 1, 1](source, result, (rank + 1) % world_size)
                _synchronize_ranks()
                checks.append(
                    (f"g {dtype}", result.cpu().item(), 42 + (rank + 1) % world_size)
                )
                _synchronize_ranks()
            finally:
                torch.npu.synchronize()
                ash.aclshmem_free_tensor(source)
    finally:
        assert ash.aclshmem_finalize() == 0
    _assert_runtime_checks(rank, checks)


@pytest.mark.ci
@pytest.mark.dist
def test_scalar_get_runtime(dist_test):
    _require_atomic_runtime()
    dist_test(run_test_scalar_get, world_size=2)
