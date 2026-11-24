# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.


import pytest
import shmem as ash
import torch
import torch.distributed as dist

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend import libaclshmem_device

from shmem_api_test_utils import capture_extern, check_extern_lowering
from utils import init_aclshmem

SYNC_OPS = [
    ("sync", "aclshmem_sync", True, "CUBE_AND_VECTOR"),
    ("sync_all", "aclshmem_sync_all", False, "CUBE_AND_VECTOR"),
    ("sync_vec", "aclshmemx_sync_vec", True, "VECTOR"),
    ("sync_vec_all", "aclshmemx_sync_vec_all", False, "VECTOR"),
]

pytestmark = [pytest.mark.dist]

G_ASH_SIZE = 1024 * 1024 * 1024


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, G_ASH_SIZE, ash.OpEngineType.MTE)


@pytest.mark.ci
@pytest.mark.parametrize("name,symbol,has_team,core_type", SYNC_OPS)
@pytest.mark.parametrize("public", [False, True])
def test_sync_extern_abi(monkeypatch, name, symbol, has_team, core_type, public):
    semantic, calls = capture_extern(monkeypatch)
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    module = libshmem_device if public else libaclshmem_device
    team = tl.constexpr(0)
    args = [team] if has_team else []
    result = getattr(module, name)(*args, _semantic=semantic)
    assert result == "extern_result"
    assert result.attrs == {"distributed.core_type": core_type.lower()}
    assert calls == [
        (
            [("cast", team, tl.int32)] if has_team else [],
            {(tl.int32,) if has_team else (): (symbol, ())},
        )
    ]


@pytest.mark.ci
@pytest.mark.parametrize("name,symbol,has_team,core_type", SYNC_OPS)
@pytest.mark.parametrize("mixed", [False, True])
def test_sync_extern_lowering(name, symbol, has_team, core_type, mixed):
    check_extern_lowering(
        symbol, ["i32"] if has_team else [], None, core_type, True, mixed
    )


@triton.jit
def sync_all_kernel():
    if tl.program_id(axis=0) == 0:
        libshmem_device.sync_all()


@triton.jit
def sync_kernel(team: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.sync(team)


@triton.jit
def sync_vec_all_kernel():
    if tl.program_id(axis=0) == 0:
        libshmem_device.sync_vec_all()


@triton.jit
def sync_vec_kernel(team: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        libshmem_device.sync_vec(team)


@triton.jit
def counter_increment_kernel(counter_ptr, peer_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0:
        val = libshmem_device.g(counter_ptr, peer_rank)
        libshmem_device.int_p(counter_ptr, val + 1, peer_rank)
        libshmem_device.quiet()


def run_test_sync_ops(rank, world_size):
    _init_aclshmem(rank, world_size)

    counter = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
    counter.copy_(torch.zeros_like(counter))

    next_peer = (rank + 1) % world_size

    sync_all_kernel[1, 1, 1]()
    print(f"Rank {rank}: sync_all OK")

    team = tl.constexpr(0)
    sync_kernel[1, 1, 1](team)
    print(f"Rank {rank}: sync OK")

    sync_vec_all_kernel[1, 1, 1]()
    print(f"Rank {rank}: sync_vec_all OK")

    sync_vec_kernel[1, 1, 1](team)
    print(f"Rank {rank}: sync_vec OK")

    for i in range(5):
        counter_increment_kernel[1, 1, 1](counter, next_peer)
        sync_all_kernel[1, 1, 1]()

    dist.barrier()
    assert counter[0].cpu() == 5, f"Rank {rank}: counter mismatch"
    print(f"Rank {rank}: sync with RMA OK")

    dist.barrier()
    ash.aclshmem_free_tensor(counter)
    _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_sync_ops_kernel(dist_test):
    dist_test(run_test_sync_ops, world_size=2)


@triton.jit
def udma_put_completed_kernel(dst, src, peer: tl.constexpr):
    libshmem_device.putmem_nbi_lowlevel(
        dst, src, 4, peer, transport=libshmem_device.ACLSHMEM_TRANSPORT_UDMA
    )
    libshmem_device.quiet_lowlevel(
        peer, transport=libshmem_device.ACLSHMEM_TRANSPORT_UDMA
    )


def run_test_sync_with_udma(rank, world_size, use_team):
    from test_atomic_transport_runtime import _init_aclshmem, _synchronize_ranks

    ash, _ = _init_aclshmem(rank, world_size, libshmem_device.ACLSHMEM_TRANSPORT_UDMA)
    src = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
    dst = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
    observed = []

    def synchronize():
        if use_team:
            sync_kernel[(1,)](libshmem_device.ACLSHMEM_TEAM_WORLD)
        else:
            sync_all_kernel[(1,)]()

    try:
        dst.copy_(torch.zeros(1, dtype=torch.int32))
        _synchronize_ranks()
        for epoch in range(1, 6):
            src.copy_(torch.tensor([epoch * 10 + rank], dtype=torch.int32))
            udma_put_completed_kernel[(1,)](dst, src, (rank + 1) % world_size)
            # No host PE barrier between the UDMA transfer and observation:
            # the collective under test must rendezvous with the remote writer.
            synchronize()
            observed.append(dst.cpu().item())
            synchronize()  # No writer may start the next epoch before all reads.
        _synchronize_ranks()
    finally:
        ash.aclshmem_free_tensor(src)
        ash.aclshmem_free_tensor(dst)
        ash.aclshmem_finalize()
    assert observed == [epoch * 10 + (rank - 1) % world_size for epoch in range(1, 6)]


@pytest.mark.dist
@pytest.mark.parametrize("use_team", [False, True], ids=["sync_all", "sync"])
def test_sync_with_udma(dist_test, use_team):
    from test_atomic_transport_runtime import _require_atomic_runtime

    _require_atomic_runtime(ascend950=True)
    dist_test(run_test_sync_with_udma, world_size=2, args=(use_team,))
