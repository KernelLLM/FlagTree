# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Single-node SUM AllReduce for Ascend950.

The operator uses UDMA communication and Vector Cores in CV0:1 mode.
Both paths complete in one kernel launch.

All ranks accumulate in source-rank order ``0 .. world_size - 1`` using FP32,
with one final cast to the output dtype.

``allreduce(ctx, x)`` updates and returns a contiguous FP16/BF16/FP32 NPU
tensor in place. The input must not require gradients. All ranks must use
matching element counts, dtypes, configurations, and call order.

The caller initializes HCCL and ACLSHMEM with the UDMA engine. A reusable
``AllReduceContext`` owns the symmetric workspace; follow its stream and
synchronization requirements and finalize it collectively before shutting
down ACLSHMEM.
"""

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.backends.ascend.driver import NPUUtils
from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device as shmem

UDMA_MAX_BYTES = 256 * 1024 * 1024
UB_SIZE_BYTES = 2031616 // 8  # 248 KiB, converted from bits.
# Reserve UB for local preloads, computation, and double buffering.
COMM_BLOCK_SIZE = UB_SIZE_BYTES // 8
ONESHOT_TWOSHOT_BORDER_BYTE = 256 * 1024


@triton.jit(do_not_specialize=["padded_shard_size", "N"])
def allreduce_oneshot(
    input_ptr,
    ready_ptr,
    WORLD_SIZE: tl.constexpr,
    padded_shard_size,  # Padded per-rank buffer size in elements.
    N,
    ELEM_BYTES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Reduction block size in elements.
    BARRIER_AFTER_ALL: tl.constexpr,
):
    """Exchange rank inputs and reduce them in one Vector Core launch."""
    rank = dl.rank().to(tl.int32)
    n_vec = 1
    global_vec_id = tl.program_id(0) * n_vec + sub_vec_id()
    total_vec = tl.num_programs(0) * n_vec
    total_blocks = tl.cdiv(N, BLOCK_SIZE)
    producer_tasks = WORLD_SIZE - 1

    for task in range(global_vec_id, producer_tasks, total_vec):
        target_rank = task.to(tl.int32)
        target_rank += tl.where(target_rank >= rank, 1, 0)
        shmem.putmem(
            ready_ptr + rank * padded_shard_size,
            input_ptr,
            N * ELEM_BYTES,
            target_rank,
        )

    # The barrier orders the exchange before any core overwrites local input.
    shmem.barrier_all_vec()

    for block_id in range(global_vec_id, total_blocks, total_vec):
        indices = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = indices < N
        total = tl.zeros((BLOCK_SIZE,), tl.float32)
        for source in range(0, WORLD_SIZE):
            if source == rank:
                values = tl.load(input_ptr + indices, valid, other=0)
            else:
                values = tl.load(
                    ready_ptr + source * padded_shard_size + indices, valid, other=0
                )
            total = total + values
        tl.store(input_ptr + indices, total.to(input_ptr.dtype.element_ty), valid)

    if BARRIER_AFTER_ALL:
        shmem.barrier_all_vec()


@triton.jit(do_not_specialize=["padded_shard_size", "N"])
def allreduce_twoshot(
    input_ptr,
    ready_ptr,
    reduced_ptr,
    WORLD_SIZE: tl.constexpr,
    padded_shard_size,  # Padded shard size in elements.
    N,
    ELEM_BYTES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,  # UDMA transfer chunk size in elements.
    BLOCK_SIZE: tl.constexpr,  # Reduction block size in elements.
    BARRIER_AFTER_ALL: tl.constexpr,
):
    """Run reduce-scatter and all-gather in one Vector Core launch.

    Inbox slots are source-major; reduced shards are owner-major. A separate
    reduction buffer keeps the input stable while outgoing transfers run.
    """
    rank = dl.rank().to(tl.int32)
    n_vec = 1
    global_vec_id = tl.program_id(0) * n_vec + sub_vec_id()
    total_vec = tl.num_programs(0) * n_vec
    producer_count = WORLD_SIZE - 1
    blocks_per_shard = padded_shard_size // BLOCK_SIZE

    if global_vec_id < producer_count:
        target_rank = global_vec_id.to(tl.int32)
        target_rank += tl.where(target_rank >= rank, 1, 0)
        src_base = target_rank * padded_shard_size
        valid_count = tl.minimum(padded_shard_size, tl.maximum(0, N - src_base))
        dst_base = rank * padded_shard_size
        for offset in range(0, padded_shard_size, CHUNK_SIZE):
            if offset < valid_count:
                count = tl.minimum(CHUNK_SIZE, valid_count - offset)
                shmem.putmem(
                    ready_ptr + dst_base + offset,
                    input_ptr + src_base + offset,
                    count * ELEM_BYTES,
                    target_rank,
                )

    shmem.barrier_all_vec()

    if global_vec_id >= producer_count:
        consumer_id = global_vec_id - producer_count
        consumer_count = total_vec - producer_count
        input_base = rank * padded_shard_size
        for block_id in range(consumer_id, blocks_per_shard, consumer_count):
            indices = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            valid = input_base + indices < N
            local_values = tl.load(
                input_ptr + input_base + indices, valid, other=0
            )
            total = tl.zeros((BLOCK_SIZE,), tl.float32)
            for source in range(0, WORLD_SIZE):
                if source == rank:
                    values = local_values
                else:
                    values = tl.load(
                        ready_ptr + source * padded_shard_size + indices,
                        valid,
                        other=0,
                    )
                total = total + values
            tl.store(
                reduced_ptr + input_base + indices,
                total.to(reduced_ptr.dtype.element_ty),
                valid,
            )

    # The barrier publishes the reduced shards so all-gather sees them.
    shmem.barrier_all_vec()

    remote_count = WORLD_SIZE - 1
    if global_vec_id < remote_count:
        peer = global_vec_id
        peer += tl.where(peer >= rank, 1, 0)
        slice_start = peer * padded_shard_size
        valid_count = tl.minimum(padded_shard_size, tl.maximum(0, N - slice_start))
        for offset in range(0, padded_shard_size, CHUNK_SIZE):
            if offset < valid_count:
                count = tl.minimum(CHUNK_SIZE, valid_count - offset)
                shmem.getmem(
                    input_ptr + slice_start + offset,
                    reduced_ptr + slice_start + offset,
                    count * ELEM_BYTES,
                    peer,
                )
    else:
        consumer_id = global_vec_id - remote_count
        consumer_count = total_vec - remote_count
        input_base = rank * padded_shard_size
        local_count = tl.minimum(
            padded_shard_size, tl.maximum(0, N - input_base)
        )
        for block_id in range(
            consumer_id, tl.cdiv(local_count, BLOCK_SIZE), consumer_count
        ):
            indices = input_base + (block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
            valid = indices < input_base + local_count
            values = tl.load(reduced_ptr + indices, valid, other=0)
            tl.store(input_ptr + indices, values, valid)

    if BARRIER_AFTER_ALL:
        shmem.barrier_all_vec()


class AllReduceContext:
    """Collectively allocate and release reusable symmetric AllReduce buffers.

    Initialize HCCL and ACLSHMEM/UDMA before construction. The context owns its
    buffers, not the SHMEM runtime. Use it on its original device and stream;
    do not share it between concurrent streams or host threads. Call
    ``finalize()`` collectively before shutting down ACLSHMEM.

    Args:
        numel: Number of input elements on each rank; at most
            4GB // element_size + 1024.
        rank: Local rank in the SHMEM world.
        world_size: Number of participating ranks.
        dtype: Input and output dtype; FP16, BF16, or FP32.
        device: NPU device for the context; defaults to the current NPU device.
        cores: Number of Vector Cores to use; zero selects all available cores.
            Oneshot requires at least one core. Twoshot must leave at least one
            core for reduction and at least one for local copy in its respective
            phases. The requested count must not exceed the available cores.
        barrier_after_all: Whether to synchronize all ranks after the final
            phase. Defaults to True, allowing consecutive calls on the context's
            stream. If False, the caller must ensure completion on every rank
            before reusing the context or communication buffers, for example
            with torch.npu.synchronize(device) followed by dist.barrier()
            between calls.
        algorithm: "auto", "oneshot", or "twoshot". Auto selects oneshot below
            256 KiB and twoshot otherwise. Forced oneshot does not split a
            transfer into chunks: each put must be at most UDMA_MAX_BYTES
            (256 MB).
    """

    def __init__(
        self,
        numel: int,
        rank: int,
        world_size: int,
        dtype=torch.bfloat16,
        device=None,
        cores=0,
        barrier_after_all=True,
        algorithm="auto",
    ):
        device, available = self._validate_config(
            numel, rank, world_size, dtype, device, cores, algorithm
        )
        self.numel = numel
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.requested_algorithm = algorithm
        self.elem_bytes = torch.empty((), dtype=dtype).element_size()
        self.device = device
        self.cores = cores or available
        self.barrier_after_all = barrier_after_all

        self._init_launch_config()
        self.stream = torch.npu.current_stream(self.device)
        self.closed = False
        self._allocations = []
        self.n, self.world = self.numel, self.world_size

        self._create_data_buffers()
        torch.npu.synchronize(self.device)
        dist.barrier()

    @staticmethod
    def _validate_config(numel, rank, world_size, dtype, device, cores, algorithm):
        if not isinstance(numel, int) or numel <= 0:
            raise ValueError(f"numel must be a positive integer, got {numel!r}")
        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(
                f"world_size must be a positive integer, got {world_size!r}"
            )
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
        if algorithm not in ("auto", "oneshot", "twoshot"):
            raise ValueError(
                "algorithm must be 'auto', 'oneshot', or 'twoshot', "
                f"got {algorithm!r}"
            )
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if dtype not in supported:
            raise ValueError(f"dtype must be one of {supported}, got {dtype}")
        elem_size = torch.empty((), dtype=dtype).element_size()
        max_numel = 4 * 1024 * 1024 * 1024 // elem_size + 1024
        if numel > max_numel:
            raise ValueError(
                f"only support numel at most {max_numel} for {dtype}, got {numel}"
            )
        device = (
            device
            if device is not None
            else torch.device("npu", torch.npu.current_device())
        )
        if device.type != "npu":
            raise ValueError("device must be an NPU device")
        available = NPUUtils().get_aivector_core_num()
        if not isinstance(cores, int) or not 0 <= cores <= available:
            raise ValueError(
                f"cores must be an integer in [0, {available}], got {cores!r}"
            )

        return device, available

    def _init_launch_config(self):
        # UDMA transfers use chunks; local loads and stores use blocks.
        self.chunk = UDMA_MAX_BYTES // self.elem_bytes
        self.block = COMM_BLOCK_SIZE // self.elem_bytes
        self.oneshot_num_blocks = triton.cdiv(self.numel, self.block)
        self.twoshot_num_blocks = triton.cdiv(self.numel, self.world_size * self.block)
        self._init_algorithm()
        self._validate_core_count()
        if (
            self.algorithm == "oneshot"
            and self.numel * self.elem_bytes > UDMA_MAX_BYTES
        ):
            raise ValueError(
                f"allreduce by {self.algorithm} algorithm: "
                f"input size exceeds UDMA_MAX_BYTES ({UDMA_MAX_BYTES})"
            )

    def _init_algorithm(self):
        if self.requested_algorithm == "auto":
            input_size = self.numel * self.elem_bytes
            self.algorithm = (
                "oneshot" if input_size < ONESHOT_TWOSHOT_BORDER_BYTE else "twoshot"
            )
        else:
            self.algorithm = self.requested_algorithm

    def _validate_core_count(self):
        # Each remote rank uses one core; leave one core for local work.
        min_cores = 1 if self.algorithm == "oneshot" else self.world_size
        if self.cores < min_cores:
            raise ValueError(
                "AllReduceContext._validate_core_count() failed: "
                f"allreduce by {self.algorithm} algorithm requires at least "
                f"{min_cores} Vector Cores, got {self.cores}"
            )

    def _create_data_buffers(self):
        if self.algorithm == "oneshot":
            # pad up to a multiple of block
            self.padded = self.oneshot_num_blocks * self.block
            self.inbox = self._allocate(self.padded * self.world_size, self.dtype)
        else:
            # pad up to a multiple of block
            self.padded = self.twoshot_num_blocks * self.block
            # inbox holds raw remote shards; reduce_buf holds reduced shards.
            self.inbox = self._allocate(self.padded * self.world_size, self.dtype)
            self.reduce_buf = self._allocate(self.padded * self.world_size, self.dtype)

    def _allocate(self, count, dtype):
        tensor = ash.aclshmem_create_tensor(
            [count], dtype=dtype, device_id=self.device.index
        )
        self._allocations.append(tensor)
        return tensor

    def finalize(self):
        if not self.closed:
            torch.npu.synchronize(self.device)
            dist.barrier()
            for tensor in reversed(self._allocations):
                ash.aclshmem_free_tensor(tensor)
            self.closed = True


def allreduce(ctx: AllReduceContext, x: torch.Tensor):
    """Enqueue SUM AllReduce in place and return ``x`` with its shape preserved.

    All ranks must use matching context configurations and invoke this
    collective in the same order.

    Args:
        ctx: Context providing communication buffers and launch configuration.
        x: Contiguous NPU tensor that does not require gradients. Its element
            count, dtype, and device must match ``ctx``.
    """
    if ctx.closed:
        raise RuntimeError("context is closed")
    if (
        x.numel() != ctx.n
        or x.dtype != ctx.dtype
        or x.device != ctx.device
        or not x.is_contiguous()
        or x.requires_grad
    ):
        raise ValueError(
            "tensor must match the context's numel/dtype/device and contiguous no-grad contract"
        )
    if torch.npu.current_stream(ctx.device) != ctx.stream:
        raise ValueError("context must be used on its original NPU stream")
    if ctx.algorithm == "oneshot":
        allreduce_oneshot[(ctx.cores,)](
            x,
            ctx.inbox,
            ctx.world,
            ctx.padded,
            ctx.n,
            ctx.elem_bytes,
            ctx.block,
            BARRIER_AFTER_ALL=ctx.barrier_after_all,
        )
    else:
        allreduce_twoshot[(ctx.cores,)](
            x,
            ctx.inbox,
            ctx.reduce_buf,
            ctx.world,
            ctx.padded,
            ctx.n,
            ctx.elem_bytes,
            ctx.chunk,
            ctx.block,
            BARRIER_AFTER_ALL=ctx.barrier_after_all,
        )
    return x
