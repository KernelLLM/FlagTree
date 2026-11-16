# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ascend All-to-All-Single 2D kernel using UDMA.

The kernel uses a block-granular UDMA pipeline for remote messages. Odd Vector
Cores consume ready data and write it directly into the compact
``all_to_all_single`` output layout.
The local-rank path is copied directly to its compact output segment by the
local-copy cores.
"""

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # Register the torch.npu backend.
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.backends.ascend.driver import NPUUtils
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device

COMM_BLOCK_SIZE_H = 128
UB_SIZE_BYTES = 2031616 // 8  # 2031616 bits = 248 * 1024 bytes
COMM_BLOCK_SIZE_M_BYTES = UB_SIZE_BYTES // COMM_BLOCK_SIZE_H  # 1984 bytes, full use of ub
COMM_BLOCK_SIZE_M_BYTES = COMM_BLOCK_SIZE_M_BYTES // 2  # 992 bytes, reserved for double buffer
SIG_SLOT_STRIDE = 8  # 8 * sizeof(int64) = one 64-byte cache line.


@triton.jit
def _exclusive_split_offset(split_ptr, split_idx, world_size):
    """Return the row offset for one rank from all_to_all_single splits."""
    offset = tl.full((), 0, tl.int64)
    for rank_idx in range(0, world_size):
        split_rows = tl.load(split_ptr + rank_idx).to(tl.int64)
        offset += tl.where(rank_idx < split_idx, split_rows, 0)
    return offset


@triton.jit
def _copy_local_tile(
    a_ptr,
    c_ptr,
    self_input_row_start,
    self_output_row_start,
    self_rows,
    hidden_dim,
    global_id_m,
    block_id_h,
    stride_am,
    stride_ah,
    stride_cm,
    stride_ch,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_H: tl.constexpr,
):
    row_offsets = tl.arange(0, COMM_BLOCK_M)
    hidden_offsets = tl.arange(0, COMM_BLOCK_H)
    rows = global_id_m * COMM_BLOCK_M + row_offsets
    columns = block_id_h * COMM_BLOCK_H + hidden_offsets
    mask = (rows < self_rows)[:, None] & (columns < hidden_dim)[None, :]
    a_offsets = (self_input_row_start + rows[:, None]) * stride_am + columns[
        None, :
    ] * stride_ah
    c_offsets = (self_output_row_start + rows[:, None]) * stride_cm + columns[
        None, :
    ] * stride_ch
    data = tl.load(a_ptr + a_offsets, mask=mask, other=0)
    tl.store(c_ptr + c_offsets, data, mask=mask)


@triton.jit(do_not_specialize=["signal_version"])
def kernel_a2a_single_2d_pipelined(
    a_ptr,
    c_ptr,
    input_split_ptr,
    output_split_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    signal_version,
    rank,
    world_size,
    max_m,
    hidden_dim,
    elem_bytes,
    stride_am,
    stride_ah,
    stride_cm,
    stride_ch,
    COMM_BLOCK_M: tl.constexpr,
    COMM_BLOCK_H: tl.constexpr,
    SIG_STRIDE: tl.constexpr,
):
    """Forward All-to-All-Single 2D with block-granular remote transfers.

    Remote senders and receivers use designated Vector Core IDs. Local-copy
    cores copy the self segment directly into the final output.
    """
    n_vec = 1
    total_vec = tl.num_programs(axis=0) * n_vec
    global_vec_id = tl.program_id(axis=0) * n_vec + sub_vec_id()
    logical_core_id = global_vec_id // 2
    n_role_cores = total_vec // 2

    max_m = max_m.to(tl.int64)
    hidden_dim = hidden_dim.to(tl.int64)
    signal_version = signal_version.to(tl.int64)
    num_blocks_h = tl.cdiv(hidden_dim, COMM_BLOCK_H)
    remote_producer_cores = world_size - 1
    local_copy_ncores = n_role_cores - remote_producer_cores

    # Remote producer stage. A target rank has exactly one UDMA-issuing core,
    # while rank rotation spreads simultaneous traffic over different links.
    if global_vec_id % 2 == 0 and logical_core_id < remote_producer_cores:
        for peer_id in range(logical_core_id, remote_producer_cores, n_role_cores):
            target_rank = (rank + peer_id + 1) % world_size
            split_rows = tl.load(input_split_ptr + target_rank).to(tl.int64)
            input_row_start = _exclusive_split_offset(
                input_split_ptr, target_rank, world_size
            )
            # Notify each row block so consumers can overlap repacking with
            # the ongoing remote transfer.
            num_send_blocks = tl.cdiv(split_rows, COMM_BLOCK_M)
            for global_id_m in range(0, num_send_blocks):
                block_row_start = global_id_m * COMM_BLOCK_M
                cur_block_m = tl.minimum(COMM_BLOCK_M, split_rows - block_row_start)
                libshmem_device.putmem(
                    peer_mem_ptr + (rank * max_m + block_row_start) * hidden_dim,
                    a_ptr + (input_row_start + block_row_start) * stride_am,
                    cur_block_m * hidden_dim * elem_bytes,
                    target_rank,
                )
                dl.notify(
                    signal_mem_ptr + (global_id_m * world_size + rank) * SIG_STRIDE,
                    target_rank.to(tl.int32),
                    signal=signal_version,
                    sig_op="set",
                )

    # Even cores left over after the remote producers copy the local-rank
    # segment. Their logical IDs form a contiguous local-worker range.
    if global_vec_id % 2 == 0 and logical_core_id >= remote_producer_cores:
        local_copy_worker_id = logical_core_id - remote_producer_cores
        self_rows = tl.load(input_split_ptr + rank).to(tl.int64)
        self_input_row_start = _exclusive_split_offset(
            input_split_ptr, rank, world_size
        )
        self_output_row_start = _exclusive_split_offset(
            output_split_ptr, rank, world_size
        )
        num_self_blocks = tl.cdiv(self_rows, COMM_BLOCK_M)
        for global_id_m in range(0, num_self_blocks):
            first_block_id_h = (
                local_copy_worker_id
                - (global_id_m * num_blocks_h) % local_copy_ncores
            ) % local_copy_ncores
            for block_id_h in range(first_block_id_h, num_blocks_h, local_copy_ncores):
                _copy_local_tile(
                    a_ptr,
                    c_ptr,
                    self_input_row_start,
                    self_output_row_start,
                    self_rows,
                    hidden_dim,
                    global_id_m,
                    block_id_h,
                    stride_am,
                    stride_ah,
                    stride_cm,
                    stride_ch,
                    COMM_BLOCK_M,
                    COMM_BLOCK_H,
                )

    # Compactly number only remote (source rank, H tile) tasks on odd Vector
    # Cores. Skipping the local rank in the numbering avoids empty tasks.
    if global_vec_id % 2 == 1:
        row_offsets = tl.arange(0, COMM_BLOCK_M)
        hidden_offsets = tl.arange(0, COMM_BLOCK_H)
        remote_world_size = world_size - 1
        tasks_per_block = num_blocks_h * remote_world_size

        max_output_rows = tl.full((), 0, tl.int64)
        for r in range(0, world_size):
            source_rows = tl.load(output_split_ptr + r).to(tl.int64)
            if r != rank:
                max_output_rows = tl.maximum(max_output_rows, source_rows)
        num_recv_blocks = tl.cdiv(max_output_rows, COMM_BLOCK_M)

        for global_id_m in range(0, num_recv_blocks):
            for task_idx in range(logical_core_id, tasks_per_block, n_role_cores):
                remote_idx = task_idx % remote_world_size
                source_rank = remote_idx + (remote_idx >= rank)
                block_id_h = task_idx // remote_world_size
                split_rows = tl.load(output_split_ptr + source_rank).to(tl.int64)
                output_row_start = _exclusive_split_offset(
                    output_split_ptr, source_rank, world_size
                )
                rows = global_id_m * COMM_BLOCK_M + row_offsets
                columns = block_id_h * COMM_BLOCK_H + hidden_offsets
                mask = (rows < split_rows)[:, None] & (columns < hidden_dim)[None, :]
                if global_id_m * COMM_BLOCK_M < split_rows:
                    remote_sig_ptr = (
                        signal_mem_ptr
                        + (global_id_m * world_size + source_rank) * SIG_STRIDE
                    )
                    token = dl.wait(
                        remote_sig_ptr,
                        1,
                        "sys",
                        "acquire",
                        waitValue=signal_version,
                    )
                    load_ptr = dl.consume_token(peer_mem_ptr, token)
                    recv_offsets = (
                        source_rank * max_m + rows[:, None]
                    ) * hidden_dim + columns[None, :]
                    data = tl.load(load_ptr + recv_offsets, mask=mask, other=0)
                    c_offsets = (
                        output_row_start + rows[:, None]
                    ) * stride_cm + columns[None, :] * stride_ch
                    tl.store(c_ptr + c_offsets, data, mask=mask)


class AllToAllSingle2DContext:
    """
    initialize the shmem buffers & split role cores for all2all-single-2d kernel
    only run on local (intra-node) communication
    """

    def __init__(
        self,
        max_m: int,
        hidden_dim: int,
        rank: int,
        world_size: int,
        dtype=torch.bfloat16,
        device=None,
    ):
        self.max_m = max_m
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        if dtype not in (torch.bfloat16, torch.float16, torch.int8, torch.int32):
            raise ValueError(
                "dtype must be torch.bfloat16, torch.float16, torch.int8, or torch.int32"
            )
        self.device = (
            device
            if device is not None
            else torch.device("npu", torch.npu.current_device())
        )
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.comm_block_size_m = COMM_BLOCK_SIZE_M_BYTES // self.element_size
        self.comm_block_size_h = COMM_BLOCK_SIZE_H

        # role assignment across vector cores
        self.aivectorcore_num = NPUUtils().get_aivector_core_num()
        n_role_cores = self.aivectorcore_num // 2
        remote_producer_cores = world_size - 1
        if n_role_cores <= remote_producer_cores:
            raise ValueError(
                "the producer/consumer role split requires more logical role "
                "cores than remote ranks "
                f"(got {n_role_cores} and {remote_producer_cores})"
            )

        # initialize shared memory
        peer_mem_size = world_size * max_m * hidden_dim
        self.peer_mem = ash.aclshmem_create_tensor(
            [peer_mem_size], dtype=dtype, device_id=rank
        )
        signal_mem_size = (
            triton.cdiv(max_m, self.comm_block_size_m) * world_size * SIG_SLOT_STRIDE
        )
        self.signal_mem = ash.aclshmem_create_tensor(
            [signal_mem_size], dtype=torch.int64, device_id=rank
        )
        self._signal_version = 0
        self.reset()

    def reset(self):
        self.signal_mem.zero_()
        self._signal_version = 0
        torch.npu.synchronize()
        dist.barrier()

    def _next_signal_version(self):
        """Return a completion-signal value that is unique for this context."""
        self._signal_version += 1
        return self._signal_version

    def finalize(self):
        ash.aclshmem_free_tensor(self.peer_mem)
        ash.aclshmem_free_tensor(self.signal_mem)

def create_all_to_all_single_2d_context(
    max_m: int,
    hidden_dim: int,
    rank: int,
    world_size: int,
    dtype=torch.bfloat16,
):
    return AllToAllSingle2DContext(
        max_m,
        hidden_dim,
        rank,
        world_size,
        dtype,
    )


def all_to_all_single_2d(
    ctx: AllToAllSingle2DContext,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
):
    signal_version = ctx._next_signal_version()

    kernel_a2a_single_2d_pipelined[ctx.aivectorcore_num, 1, 1](
        input_tensor,
        output_tensor,
        input_splits,
        output_splits,
        ctx.peer_mem,
        ctx.signal_mem,
        signal_version,
        ctx.rank,
        ctx.world_size,
        ctx.max_m,
        ctx.hidden_dim,
        input_tensor.element_size(),
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        COMM_BLOCK_M=ctx.comm_block_size_m,
        COMM_BLOCK_H=ctx.comm_block_size_h,
        SIG_STRIDE=SIG_SLOT_STRIDE,
    )
    return output_tensor
