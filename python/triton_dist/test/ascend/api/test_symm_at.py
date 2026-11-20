# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import shmem as ash
import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device
from utils import init_aclshmem

g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


@triton.jit
def _shmemx_symm_at(remote_ptr, output_ptr, rank, world_size):
    """Test symm_at by reading data from remote rank"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        # Get target rank (circular)
        target_rank = (rank + 1) % world_size

        # Use symm_at to get pointer to remote memory
        remote_mem_ptr = dl.symm_at(remote_ptr, target_rank)

        # Load data from remote memory
        offset = tl.arange(0, 32)
        data = tl.load(remote_mem_ptr + offset, None)

        # Store to output
        tl.store(output_ptr + offset, data, None)
    libshmem_device.barrier_all_vec()


def run_test_distributed(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    init_aclshmem(rank, world_size, g_ash_size, ash.OpEngineType.MTE)

    # Create local data
    local_data = torch.arange(32, dtype=torch.int64).npu() + rank * 100
    output = torch.zeros(32, dtype=torch.int64).npu()

    # Create shared memory for remote access
    remote_mem = ash.aclshmem_create_tensor([32], dtype=torch.int64, device_id=rank)
    remote_mem.copy_(local_data)

    # Gather all remote data for reference
    all_data = [torch.zeros(32, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather_object(all_data, local_data.cpu())

    # Run kernel
    _shmemx_symm_at[1, 1, 1](remote_mem, output, rank, world_size)

    # Verify result
    target_rank = (rank + 1) % world_size
    expected = all_data[target_rank]
    assert torch.equal(output.cpu(), expected), f"Rank {rank}: output mismatch"

    # Cleanup
    ash.aclshmem_free_tensor(remote_mem)
    _ = ash.aclshmem_finalize()


@pytest.mark.ci
@pytest.mark.dist
def test_symm_at(dist_test):
    dist_test(run_test_distributed, world_size=2)
