# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""RoCE runtime tests: no dist mark, pending official enablement."""

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


@triton.jit
def _collective_kernel(
    dst,
    src,
    status,
    peer,
    OP: tl.constexpr,
    HAS_TEAM: tl.constexpr,
    IS_SYNC: tl.constexpr,
):
    if tl.program_id(0) == 0:
        shmem.putmem_nbi_lowlevel(
            dst, src, 4, peer, transport=shmem.ACLSHMEM_TRANSPORT_ROCE
        )
        shmem.quiet_lowlevel(peer, transport=shmem.ACLSHMEM_TRANSPORT_ROCE)
        if IS_SYNC:
            if HAS_TEAM:
                result = OP(
                    shmem.ACLSHMEM_TEAM_WORLD, transport=shmem.ACLSHMEM_TRANSPORT_ROCE
                )
            else:
                OP(transport=shmem.ACLSHMEM_TRANSPORT_ROCE)
        elif HAS_TEAM:
            result = OP(shmem.ACLSHMEM_TEAM_WORLD)
        else:
            OP()
        if HAS_TEAM:
            tl.store(status, result)
        else:
            tl.store(status, 0)
        # Read after the device collective, before any host-side rank barrier.
        tl.store(status + 1, tl.load(dst))


def _run_collective(rank, world_size, name):
    ash, _ = _init_aclshmem(rank, world_size, shmem.ACLSHMEM_TRANSPORT_ROCE)
    tensors = []
    checks = []
    try:
        for _ in range(2):
            tensors.append(
                ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
            )
        src, dst = tensors
        status = torch.empty(2, dtype=torch.int32, device=f"npu:{rank}")
        for epoch in range(1, 4):
            src.fill_(epoch * 100 + rank)
            dst.fill_(-1)
            status.fill_(-1)
            _synchronize_ranks()
            _collective_kernel[(1,)](
                dst,
                src,
                status,
                (rank + 1) % world_size,
                getattr(shmem, name),
                "all" not in name,
                name.startswith("sync"),
            )
            _synchronize_ranks()
            checks.append(
                (status.cpu().tolist(), [0, epoch * 100 + (rank - 1) % world_size])
            )
    finally:
        torch.npu.synchronize()
        for tensor in reversed(tensors):
            ash.aclshmem_free_tensor(tensor)
        assert ash.aclshmem_finalize() == 0
    for actual, expected in checks:
        assert actual == expected, f"rank {rank}: {name}: {actual} != {expected}"


# Intentionally not marked dist until RoCE receives official enablement.
@pytest.mark.parametrize(
    "name",
    [
        "barrier_roce",
        "barrier_all_roce",
        "barrier_vec_roce",
        "barrier_all_vec_roce",
        "sync",
        "sync_all",
    ],
)
def test_roce_collective_runtime(dist_test, name):
    _require_atomic_runtime(ascend950=True)
    dist_test(_run_collective, world_size=2, args=(name,))
