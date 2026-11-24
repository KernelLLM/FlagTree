# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import shmem as ash
import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cann.extension import sub_vec_id
from utils import init_aclshmem

g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024

# dl.notify supports int32/int64/uint64 signal pointers on Ascend for SET/ADD.
SIGNAL_DTYPE = torch.int32
# aclshmem_wait walks barriers 64 bytes apart, i.e. 64 / sizeof(int32) elements.
SIGNAL_STRIDE = 64 // 4
I64_SIGNAL_DTYPE = torch.int64
# aclshmem_wait walks barriers 64 bytes apart, i.e. 64 / sizeof(int64) elements.
I64_SIGNAL_STRIDE = 64 // 8
# "set" 1 then "add" 10 on the same signal word.
SET_ADD_WAIT_VALUE = 11


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, g_ash_size, ash.OpEngineType.MTE)


@triton.jit
def _producer_kernel(data_ptr, signal_ptr, rank, world_size):
    """Producer: write data and notify consumer"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        # Write data
        offset = tl.program_id(0)
        value = rank * 100 + offset
        data_ptr = dl.symm_at(data_ptr, 0)
        tl.store(data_ptr + offset, value, None)

        # Notify consumer (rank 0) that data is ready
        target_rank = 0
        dl.notify(
            signal_ptr, target_rank, signal=1, sig_op="set", comm_scope="intra_node"
        )


@triton.jit
def _consumer_kernel(data_ptr, signal_ptr, output_ptr, rank, world_size):
    """Consumer: wait for signal, then read data"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        # Wait for notification from all other ranks
        num_barriers = world_size - 1  # Wait for all producers
        barrier_ptrs = signal_ptr
        token = dl.wait(
            barrier_ptrs, num_barriers, scope="sys", semantic="acquire", waitValue=1
        )
        data_ptr = dl.consume_token(data_ptr, token)
        # Read data from all producers
        offset = tl.program_id(0)
        data = tl.load(data_ptr + offset, None)
        tl.store(output_ptr + offset, data, None)


@triton.jit
def _producer_set_add_kernel(data_ptr, signal_ptr, rank, world_size):
    """Producer: write data, then set the signal and add to it"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        offset = tl.program_id(0)
        value = rank * 100 + offset
        data_ptr = dl.symm_at(data_ptr, 0)
        tl.store(data_ptr + offset, value, None)

        target_rank = 0
        dl.notify(
            signal_ptr, target_rank, signal=1, sig_op="set", comm_scope="intra_node"
        )
        dl.notify(
            signal_ptr, target_rank, signal=10, sig_op="add", comm_scope="intra_node"
        )


@triton.jit
def _consumer_set_add_kernel(
    data_ptr, signal_ptr, output_ptr, rank, world_size, waitValue: tl.constexpr
):
    """Consumer: wait for the accumulated signal, then read data"""
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        num_barriers = world_size - 1
        token = dl.wait(
            signal_ptr,
            num_barriers,
            scope="sys",
            semantic="acquire",
            waitValue=waitValue,
        )
        data_ptr = dl.consume_token(data_ptr, token)
        offset = tl.program_id(0)
        data = tl.load(data_ptr + offset, None)
        tl.store(output_ptr + offset, data, None)


def run_test_distributed(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _init_aclshmem(rank, world_size)

    # Create shared memory for data and signals
    data_mem = ash.aclshmem_create_tensor([32], dtype=torch.int64, device_id=rank)
    signal_mem = ash.aclshmem_create_tensor(
        [(world_size - 1) * SIGNAL_STRIDE], dtype=SIGNAL_DTYPE, device_id=rank
    )

    # Initialize signal to 0
    signal_mem.zero_()

    if rank == 0:
        # Consumer (rank 0)
        output = torch.zeros(32, dtype=torch.int64).npu()

        # Run consumer kernel
        _consumer_kernel[1, 1, 1](data_mem, signal_mem, output, rank, world_size)

        # Verify: output should contain data from the last producer
        # (since all producers write to same location, last write wins)
        last_producer = world_size - 1
        expected = torch.zeros(32, dtype=torch.int64).npu()
        expected[0] = last_producer * 100
        assert torch.equal(output.cpu(), expected.cpu()), "Consumer: output mismatch"
    else:
        # Producers (rank 1, 2, ...)
        # Run producer kernel
        _producer_kernel[1, 1, 1](data_mem, signal_mem, rank, world_size)

    # Synchronize all ranks
    dist.barrier()

    # Cleanup
    ash.aclshmem_free_tensor(data_mem)
    ash.aclshmem_free_tensor(signal_mem)
    _ = ash.aclshmem_finalize()


def run_test_distributed_set_add(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _init_aclshmem(rank, world_size)

    data_mem = ash.aclshmem_create_tensor([32], dtype=torch.int64, device_id=rank)
    signal_mem = ash.aclshmem_create_tensor(
        [(world_size - 1) * SIGNAL_STRIDE], dtype=SIGNAL_DTYPE, device_id=rank
    )

    signal_mem.zero_()
    data_mem.zero_()
    torch.npu.synchronize()
    # Zero-fill must land before any producer notifies, or it erases the signal.
    dist.barrier()

    output = None
    try:
        if rank == 0:
            output = torch.zeros(32, dtype=torch.int64).npu()
            _consumer_set_add_kernel[1, 1, 1](
                data_mem, signal_mem, output, rank, world_size, SET_ADD_WAIT_VALUE
            )
        else:
            _producer_set_add_kernel[1, 1, 1](data_mem, signal_mem, rank, world_size)
        torch.npu.synchronize()
    finally:
        # Barrier before teardown so a failing rank cannot leave its peer blocked.
        dist.barrier()
        ash.aclshmem_free_tensor(data_mem)
        ash.aclshmem_free_tensor(signal_mem)
        _ = ash.aclshmem_finalize()

    if rank == 0:
        expected = torch.zeros(32, dtype=torch.int64)
        expected[0] = (world_size - 1) * 100
        assert torch.equal(output.cpu(), expected), "Consumer: output mismatch"


def run_test_distributed_set_i64(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _init_aclshmem(rank, world_size)

    data_mem = ash.aclshmem_create_tensor([32], dtype=torch.int64, device_id=rank)
    signal_mem = ash.aclshmem_create_tensor(
        [(world_size - 1) * I64_SIGNAL_STRIDE],
        dtype=I64_SIGNAL_DTYPE,
        device_id=rank,
    )

    signal_mem.zero_()
    data_mem.zero_()
    torch.npu.synchronize()
    # Zero-fill must land before the producer notifies, or it erases the signal.
    dist.barrier()

    output = None
    try:
        if rank == 0:
            output = torch.zeros(32, dtype=torch.int64).npu()
            _consumer_kernel[1, 1, 1](data_mem, signal_mem, output, rank, world_size)
        else:
            _producer_kernel[1, 1, 1](data_mem, signal_mem, rank, world_size)
        torch.npu.synchronize()
    finally:
        # Barrier before teardown so a failing rank cannot leave its peer blocked.
        dist.barrier()
        ash.aclshmem_free_tensor(data_mem)
        ash.aclshmem_free_tensor(signal_mem)
        _ = ash.aclshmem_finalize()

    if rank == 0:
        expected = torch.zeros(32, dtype=torch.int64)
        expected[0] = (world_size - 1) * 100
        assert torch.equal(output.cpu(), expected), "Consumer: output mismatch"


def run_test_distributed_set_add_i64(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _init_aclshmem(rank, world_size)

    data_mem = ash.aclshmem_create_tensor([32], dtype=torch.int64, device_id=rank)
    signal_mem = ash.aclshmem_create_tensor(
        [(world_size - 1) * I64_SIGNAL_STRIDE],
        dtype=I64_SIGNAL_DTYPE,
        device_id=rank,
    )

    signal_mem.zero_()
    data_mem.zero_()
    torch.npu.synchronize()
    dist.barrier()

    output = None
    try:
        if rank == 0:
            output = torch.zeros(32, dtype=torch.int64).npu()
            _consumer_set_add_kernel[1, 1, 1](
                data_mem, signal_mem, output, rank, world_size, SET_ADD_WAIT_VALUE
            )
        else:
            _producer_set_add_kernel[1, 1, 1](data_mem, signal_mem, rank, world_size)
        torch.npu.synchronize()
    finally:
        dist.barrier()
        ash.aclshmem_free_tensor(data_mem)
        ash.aclshmem_free_tensor(signal_mem)
        _ = ash.aclshmem_finalize()

    if rank == 0:
        expected = torch.zeros(32, dtype=torch.int64)
        expected[0] = (world_size - 1) * 100
        assert torch.equal(output.cpu(), expected), "Consumer: output mismatch"


@pytest.mark.ci
@pytest.mark.dist
def test_wait_notify(dist_test):
    dist_test(run_test_distributed, world_size=2)


# Single producer so the "add" accumulates onto its own "set", not a peer's.
@pytest.mark.ci
@pytest.mark.dist
def test_wait_notify_set_add(dist_test):
    dist_test(run_test_distributed_set_add, world_size=2)


@pytest.mark.ci
@pytest.mark.dist
def test_wait_notify_set_i64(dist_test):
    dist_test(run_test_distributed_set_i64, world_size=2)


# cannot be marked with CI as this will not run on A3
@pytest.mark.dist
def test_wait_notify_set_add_i64(dist_test):
    dist_test(run_test_distributed_set_add_i64, world_size=2)
