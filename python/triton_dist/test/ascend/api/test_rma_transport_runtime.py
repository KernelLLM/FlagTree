# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Device correctness coverage for explicit transport RMA and completion."""

import os

import pytest
import torch
import triton
import triton.language as tl

from triton_dist.language.extra import libshmem_device as shmem
from test_atomic_transport_runtime import (
    _init_aclshmem,
    _require_atomic_runtime,
    _synchronize_ranks,
)

MTE = shmem.ACLSHMEM_TRANSPORT_MTE
UDMA = shmem.ACLSHMEM_TRANSPORT_UDMA
ROCE = shmem.ACLSHMEM_TRANSPORT_ROCE
TRANSPORT_QPS = [
    pytest.param(MTE, None, marks=pytest.mark.dist, id="mte"),
    pytest.param(UDMA, None, marks=pytest.mark.dist, id="udma"),
    pytest.param(UDMA, 0, marks=pytest.mark.dist, id="udma-qp0"),
    pytest.param(UDMA, 1, marks=pytest.mark.dist, id="udma-qp1"),
    # No dist mark: RoCE is pending official enablement.
    pytest.param(ROCE, None, id="roce"),
    pytest.param(ROCE, 0, id="roce-qp0"),
    pytest.param(ROCE, 1, id="roce-qp1"),
]


def _require_rma_runtime(transport, qp, relay=False):
    _require_atomic_runtime(ascend950=transport != MTE, devices=4 if relay else 2)
    # SHMEM has no Python query for the compile-time relay configuration.
    # The runtime and template library must both match this setting.
    relay_build = os.environ.get("TRITON_DIST_TEST_UDMA_RELAY", "0") == "1"
    if relay and not relay_build:
        pytest.skip(
            "requires relay-enabled SHMEM/templates and TRITON_DIST_TEST_UDMA_RELAY=1"
        )
    if transport == UDMA and qp is not None and relay_build:
        pytest.skip(
            "UDMA QP APIs require a direct-mode build; relay builds do not support them"
        )


@triton.jit
def _transfer_kernel(
    dst,
    src,
    signal,
    nbytes,
    peer,
    relay_peer,
    qp,
    OP: tl.constexpr,
    TRANSPORT: tl.constexpr,
    RELAY: tl.constexpr,
    SIGNAL_VALUE: tl.constexpr,
):
    # A single issuing core per PE/QP, as required by UDMA.
    if tl.program_id(0) == 0:
        if RELAY:
            if OP == "put":
                shmem.putmem_nbi_relay(dst + 16, src + 8, nbytes, peer, relay_peer)
            else:
                shmem.getmem_nbi_relay(dst + 16, src + 8, nbytes, peer, relay_peer)
        elif OP == "signal":
            shmem.putmem_signal_nbi_lowlevel(
                dst + 16,
                src + 8,
                nbytes,
                signal,
                SIGNAL_VALUE,
                shmem.ACLSHMEM_SIGNAL_SET,
                peer,
                qp_id=qp,
            )
        elif OP == "put":
            shmem.putmem_nbi_lowlevel(
                dst + 16,
                src + 8,
                nbytes,
                peer,
                qp_id=qp,
                transport=TRANSPORT,
            )
        else:
            shmem.getmem_nbi_lowlevel(
                dst + 16,
                src + 8,
                nbytes,
                peer,
                qp_id=qp,
                transport=TRANSPORT,
            )
        if TRANSPORT == shmem.ACLSHMEM_TRANSPORT_MTE:
            shmem.quiet_lowlevel(transport=TRANSPORT)
        else:
            shmem.quiet_lowlevel(peer, transport=TRANSPORT, qp_id=qp)


def _run_transfer(rank, world_size, transport, qp, op, dtype, relay=False):
    ash, _ = _init_aclshmem(
        rank,
        world_size,
        transport,
        qp_count=2 if qp is not None else None,
    )
    tensors = []
    checks = []
    try:
        for tensor_dtype in (dtype, dtype, torch.uint64):
            tensors.append(
                ash.aclshmem_create_tensor([128], dtype=tensor_dtype, device_id=rank)
            )
        src, dst, signal = tensors
        itemsize = src.element_size()
        peer = (rank + 1) % world_size
        source_rank = peer if op == "get" else (rank - 1) % world_size
        # Compare raw bytes to catch element-count/byte-count mistakes, typed
        # pointer offsets, partial elements, and writes outside the destination.
        for epoch, nbytes in enumerate((64, 67), start=1):
            payload = ((torch.arange(128 * itemsize) + rank * 17) % 251).to(torch.uint8)
            src.copy_(payload.view(dtype))
            dst.copy_(
                torch.full((128 * itemsize,), 0xA5, dtype=torch.uint8).view(dtype)
            )
            signal.copy_(torch.zeros(128, dtype=torch.uint64))
            _synchronize_ranks()
            signal_value = 2**32 + epoch  # Verify the full uint64 notification ABI.
            _transfer_kernel[(1,)](
                dst,
                src,
                signal,
                nbytes,
                peer,
                (rank + 2) % world_size,
                qp,
                op,
                transport,
                relay,
                signal_value,
            )
            _synchronize_ranks()
            expected = torch.full((128 * itemsize,), 0xA5, dtype=torch.uint8)
            remote = ((torch.arange(128 * itemsize) + source_rank * 17) % 251).to(
                torch.uint8
            )
            expected[16 * itemsize : 16 * itemsize + nbytes] = remote[
                8 * itemsize : 8 * itemsize + nbytes
            ]
            checks.append(
                (
                    f"{op} {dtype} {nbytes} bytes",
                    torch.equal(dst.cpu().view(torch.uint8), expected),
                )
            )
            expected_signal = torch.zeros(128, dtype=torch.uint64)
            if op == "signal":
                expected_signal[0] = signal_value
            checks.append(
                (
                    f"{op} signal {nbytes} bytes",
                    torch.equal(signal.cpu(), expected_signal),
                )
            )
            _synchronize_ranks()
    finally:
        torch.npu.synchronize()
        for tensor in reversed(tensors):
            ash.aclshmem_free_tensor(tensor)
        assert ash.aclshmem_finalize() == 0
    # Assert after collective cleanup so a mismatching rank cannot strand peers.
    for description, passed in checks:
        assert passed, f"rank {rank}: {description} mismatch"


@pytest.mark.parametrize("transport,qp", TRANSPORT_QPS)
@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_lowlevel_rma_runtime(dist_test, transport, qp, op, dtype):
    _require_rma_runtime(transport, qp)
    dist_test(_run_transfer, world_size=2, args=(transport, qp, op, dtype))


@pytest.mark.dist
@pytest.mark.parametrize("qp", [None, 0, 1])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_udma_signal_runtime(dist_test, qp, dtype):
    _require_rma_runtime(UDMA, qp)
    dist_test(_run_transfer, world_size=2, args=(UDMA, qp, "signal", dtype))


# No ci/dist marks: the default NPU-IR package is built without relay support.
@pytest.mark.parametrize("op", ["get", "put"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_udma_relay_runtime(dist_test, op, dtype):
    _require_rma_runtime(UDMA, None, relay=True)
    # Four distinct ranks give each caller a separate target and relay PE.
    dist_test(_run_transfer, world_size=4, args=(UDMA, None, op, dtype, True))
