# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import shmem as ash
import torch
import torch.distributed as dist

import triton.language as tl
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device

import triton_dist
from utils import init_aclshmem

G_ASH_SIZE = 1024 * 1024 * 1024


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, G_ASH_SIZE, ash.OpEngineType.MTE)


@triton_dist.jit(core_ratio=(0, 1))
def _increase_with_barrier_all_vec_kernel(
    counter_ptr, my_rank: tl.constexpr, world_size: tl.constexpr
):
    '''
    barrier_all_vec should work under mix (0, 1)
    '''
    value = tl.load(counter_ptr)

    libshmem_device.barrier_all_vec()

    if sub_vec_id() == 0:
        next_pe = (my_rank + 1) % world_size
        remote_counter = libshmem_device.remote_ptr(counter_ptr, next_pe)
        tl.store(remote_counter, value + 1)

    libshmem_device.barrier_all_vec()


@triton_dist.jit(core_ratio=(1, 1))
def _single_vector_core_kernel(marker_ptr):
    '''
    with core_ratio(1, 1) the conseq under sub_vec_id() == 1 should not be executed.
    '''
    if sub_vec_id() == 0:
        tl.store(marker_ptr, 1)
    if sub_vec_id() == 1:
        tl.store(marker_ptr + 1, 1)


def _test_core_ratio(rank, world_size):
    counter = ash.aclshmem_create_tensor([1], dtype=torch.int64, device_id=rank)
    marker = ash.aclshmem_create_tensor([2], dtype=torch.int64, device_id=rank)
    try:
        counter.zero_()
        marker.zero_()
        dist.barrier()

        _increase_with_barrier_all_vec_kernel[(1, 1, 1)](
            counter, rank, world_size
        )
        dist.barrier()
        actual_counter = counter.cpu().item()
        assert actual_counter == 1, (
            f"Rank {rank}: expected counter 1, got {actual_counter}"
        )

        _single_vector_core_kernel[(1, 1, 1)](marker)
        actual_marker = marker.cpu()
        assert torch.equal(actual_marker, torch.tensor([1, 0], dtype=torch.int64)), (
            f"Rank {rank}: sub_vec_id() == 1 unexpectedly executed: "
            f"marker={actual_marker.tolist()}"
        )
    finally:
        ash.aclshmem_free_tensor(marker)
        ash.aclshmem_free_tensor(counter)


def run_test_core_ratio(rank, world_size):
    _init_aclshmem(rank, world_size)
    _test_core_ratio(rank, world_size)
    _ = ash.aclshmem_finalize()


@pytest.mark.experimental
def test_core_ratio(dist_test):
    dist_test(run_test_core_ratio, world_size=2)
