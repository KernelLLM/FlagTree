# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ascend mesh AllGather kernel using ACLSHMEM UDMA.

Each rank pushes its contiguous local tensor into every remote symmetric
workspace. Remote receivers copy ready tiles into the regular output tensor,
while the local tile is copied directly from the input.

shared data mem size: (world_size * m, hidden_dim)

Each UDMA transaction transfers min(256 MB, remaining data for one remote
rank). All receiver cores wait on the transaction's ready signal, copy their
share of that contiguous range, and publish one ack per core.

shared signal mem layout: readySignal             + ackSignal
                          [chunks * remote_ranks] + [chunks * remote_ranks * receiver_cores]
parity: even    wait ack 0 -> set ready 1 -> wait ready 1 -> set ack 1
parity: odd     wait ack 1 -> set ready 0 -> wait ready 0 -> set ack 0
"""

import shmem as ash
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.backends.ascend.driver import NPUUtils
from triton_dist.language.extra import libshmem_device


UDMA_MAX_SEND_BYTES = 256 * 1024 * 1024
UB_SIZE_BYTES = 2031616 // 8  # 2031616 bits = 248 * 1024 bytes
COPY_BLOCK_SIZE_BYTES = UB_SIZE_BYTES // 2  # 124 * 1024 bytes
SIGNAL_DTYPE = torch.int64
SIGNAL_SLOT_STRIDE = 8  # 8 * sizeof(int64) = one 64-byte cache line.


@triton.jit(do_not_specialize=["launch_parity"])
def _mesh_udma_allgather_kernel(
    input_ptr,
    output_ptr,
    data_buffer_ptr,
    signal_ptr,
    launch_parity,  # adjacent launches have inverted parity
    rank,
    world_size,
    numel,
    element_size,
    SEND_CHUNK_NUMEL: tl.constexpr,
    NUM_SEND_CHUNKS: tl.constexpr,
    NUM_ROLE_CORES: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """Push remote tiles and materialize remote and local output tiles."""
    global_vec_id = tl.program_id(axis=0)
    n_role_cores = NUM_ROLE_CORES
    logical_core_id = global_vec_id // 2
    role = global_vec_id % 2

    parity = launch_parity.to(tl.int64)
    numel = numel.to(tl.int64)
    num_remote_ranks = world_size - 1
    remote_producer_cores = num_remote_ranks
    ack_signal_offset = NUM_SEND_CHUNKS * num_remote_ranks * SIGNAL_STRIDE

    # One producer core owns each remote PE and sends at most 256 MB per put.
    if role == 0 and logical_core_id < remote_producer_cores:
        for producer_id in range(logical_core_id, remote_producer_cores, n_role_cores):
            target_rank = (rank + producer_id + 1) % world_size  # rank rotation

            for send_chunk_id in tl.static_range(0, NUM_SEND_CHUNKS):
                chunk_offset = send_chunk_id * SEND_CHUNK_NUMEL
                chunk_numel = tl.minimum(SEND_CHUNK_NUMEL, numel - chunk_offset)

                # Do not overwrite this target's workspace chunk until every
                # receiver core consumed it during the previous launch.
                target_slot = target_rank - tl.where(target_rank > rank, 1, 0)
                put_ptr = data_buffer_ptr
                # Keep the token-dependent pointer chain out of scf.for. A
                # loop-carried pointer can fail dominance verification during
                # the Ascend IR conversion.
                for receiver_core_id in tl.static_range(0, NUM_ROLE_CORES):
                    ack_slot = (
                        send_chunk_id * num_remote_ranks + target_slot
                    ) * n_role_cores + receiver_core_id
                    ack_ptr = signal_ptr + ack_signal_offset + ack_slot * SIGNAL_STRIDE
                    ack_token = dl.wait(
                        ack_ptr,
                        1,
                        "sys",
                        "acquire",
                        waitValue=parity,
                    )
                    put_ptr = dl.consume_token(put_ptr, ack_token)
                libshmem_device.putmem(
                    put_ptr + rank * numel + chunk_offset,
                    input_ptr + chunk_offset,
                    chunk_numel * element_size,
                    target_rank,
                )

                source_slot = rank - tl.where(rank > target_rank, 1, 0)
                ready_slot = send_chunk_id * num_remote_ranks + source_slot
                dl.notify(
                    signal_ptr + ready_slot * SIGNAL_STRIDE,
                    target_rank.to(tl.int32),
                    signal=parity ^ 1,
                    sig_op="set",
                )

    # Remaining even cores copy the local data to final output.
    if role == 0 and logical_core_id >= remote_producer_cores:
        local_copy_pid = logical_core_id - remote_producer_cores
        local_copy_ncores = n_role_cores - remote_producer_cores
        copy_offsets = tl.arange(0, COPY_BLOCK)
        num_copy_blocks = tl.cdiv(numel, COPY_BLOCK)

        for block_id in range(local_copy_pid, num_copy_blocks, local_copy_ncores):
            offsets = block_id * COPY_BLOCK + copy_offsets
            mask = offsets < numel
            values = tl.load(input_ptr + offsets, mask=mask)
            tl.store(output_ptr + rank * numel + offsets, values, mask=mask)

    # Odd cores receive remote ranks. Every core acknowledges every ready
    # chunk after copying its share.
    if role == 1:
        copy_offsets = tl.arange(0, COPY_BLOCK)
        blocks_per_chunk = tl.cdiv(SEND_CHUNK_NUMEL, COPY_BLOCK)

        for send_chunk_id in tl.static_range(0, NUM_SEND_CHUNKS):
            chunk_offset = send_chunk_id * SEND_CHUNK_NUMEL
            chunk_numel = tl.minimum(SEND_CHUNK_NUMEL, numel - chunk_offset)

            for remote_idx in range(0, num_remote_ranks):
                source_rank = remote_idx + (remote_idx >= rank)
                source_slot = source_rank - tl.where(source_rank > rank, 1, 0)
                ready_slot = send_chunk_id * num_remote_ranks + source_slot
                token = dl.wait(
                    signal_ptr + ready_slot * SIGNAL_STRIDE,
                    1,
                    "sys",
                    "acquire",
                    waitValue=parity ^ 1,
                )
                load_ptr = dl.consume_token(data_buffer_ptr, token)

                for block_id in range(logical_core_id, blocks_per_chunk, n_role_cores):
                    offsets = block_id * COPY_BLOCK + copy_offsets
                    mask = offsets < chunk_numel
                    source_offsets = source_rank * numel + chunk_offset + offsets
                    values = tl.load(load_ptr + source_offsets, mask=mask)
                    tl.store(output_ptr + source_offsets, values, mask=mask)

                libshmem_device.quiet()
                target_slot = rank - tl.where(rank > source_rank, 1, 0)
                ack_slot = (
                    send_chunk_id * num_remote_ranks + target_slot
                ) * n_role_cores + logical_core_id
                dl.notify(
                    signal_ptr + ack_signal_offset + ack_slot * SIGNAL_STRIDE,
                    source_rank.to(tl.int32),
                    signal=parity ^ 1,
                    sig_op="set",
                )


class AllGatherContext:
    """Own the symmetric buffers used by the intra-node AllGather kernel."""

    def __init__(
        self,
        m: int,
        hidden_dim: int,
        rank: int,
        world_size: int,
        dtype=torch.bfloat16,
        device=None,
    ):
        if m <= 0 or hidden_dim <= 0:
            raise ValueError("m and hidden_dim must be positive")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("dtype must be torch.bfloat16, torch.float16, or torch.float32")

        self.m = m
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.device = device if device is not None else torch.device("npu", torch.npu.current_device())
        self.element_size = torch.empty((), dtype=dtype).element_size()
        self.copy_block_size = COPY_BLOCK_SIZE_BYTES // self.element_size

        # core number requirement
        self.num_aivector_cores = NPUUtils().get_aivector_core_num()
        n_role_cores = self.num_aivector_cores // 2
        remote_producer_cores = world_size - 1
        if n_role_cores <= remote_producer_cores:
            raise ValueError(
                "AllGather requires at least one local-copy core beyond the "
                f"remote producer range (got {n_role_cores} role cores and "
                f"{remote_producer_cores} remote ranks)"
            )

        # create shared memory
        output_shape = (world_size * m, hidden_dim)
        self.data_mem = ash.aclshmem_create_tensor(
            output_shape, dtype=dtype, device_id=rank
        )
        send_data_per_remote_rank = m * hidden_dim * self.element_size
        self.send_chunk_numel = UDMA_MAX_SEND_BYTES // self.element_size
        self.num_send_chunks = triton.cdiv(
            send_data_per_remote_rank, UDMA_MAX_SEND_BYTES
        )
        num_ready_signal_slots = self.num_send_chunks * (world_size - 1)
        num_ack_signal_slots = num_ready_signal_slots * n_role_cores
        signal_mem_size = (
            num_ready_signal_slots + num_ack_signal_slots
        ) * SIGNAL_SLOT_STRIDE
        self.signal_mem = ash.aclshmem_create_tensor(
            [signal_mem_size], dtype=SIGNAL_DTYPE, device_id=rank
        )
        self.reset()

    def reset(self):
        self.signal_mem.zero_()
        dist.barrier()
        self.launch_parity = 1  # for later pre_launch() to set parity to 0

    def pre_launch(self):
        self.launch_parity ^= 1

    def finalize(self):
        ash.aclshmem_free_tensor(self.data_mem)
        ash.aclshmem_free_tensor(self.signal_mem)


def create_allgather_context(
    m: int,
    hidden_dim: int,
    rank: int,
    world_size: int,
    dtype=torch.bfloat16,
) -> AllGatherContext:
    return AllGatherContext(m, hidden_dim, rank, world_size, dtype)


def allgather(
    ctx: AllGatherContext,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
):
    if input_tensor.dim() != 2:
        raise ValueError("AllGather expects a 2D input tensor")
    if not input_tensor.is_contiguous():
        raise ValueError("AllGather requires a contiguous input tensor")
    if input_tensor.shape != (ctx.m, ctx.hidden_dim):
        raise ValueError(f"input shape must be {(ctx.m, ctx.hidden_dim)}, got {tuple(input_tensor.shape)}")
    if input_tensor.dtype != ctx.dtype:
        raise ValueError(f"input dtype must be {ctx.dtype}, got {input_tensor.dtype}")
    if output_tensor.shape != (ctx.world_size * ctx.m, ctx.hidden_dim):
        raise ValueError(
            f"output shape must be {(ctx.world_size * ctx.m, ctx.hidden_dim)}, got {tuple(output_tensor.shape)}"
        )
    if not output_tensor.is_contiguous():
        raise ValueError("AllGather requires a contiguous output tensor")
    if output_tensor.dtype != ctx.dtype:
        raise ValueError(f"output dtype must be {ctx.dtype}, got {output_tensor.dtype}")

    ctx.pre_launch()
    numel = input_tensor.numel()
    _mesh_udma_allgather_kernel[ctx.num_aivector_cores, 1, 1](
        input_tensor,
        output_tensor,
        ctx.data_mem,
        ctx.signal_mem,
        ctx.launch_parity,
        ctx.rank,
        ctx.world_size,
        numel,
        ctx.element_size,
        SEND_CHUNK_NUMEL=ctx.send_chunk_numel,
        NUM_SEND_CHUNKS=ctx.num_send_chunks,
        NUM_ROLE_CORES=ctx.num_aivector_cores // 2,
        COPY_BLOCK=ctx.copy_block_size,
        SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
    )
    return output_tensor
