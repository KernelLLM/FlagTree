# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""QKV All2All UDMA variant with a torch.npu.Event timing comparison.

The input layout is ``(S, global_heads, D)`` and the output layout is
``(rank_size * S, heads_per_rank, D)``.  One mixed kernel packs only the
remote Q/K/V slices with MTE, then one physical AI Core issues exactly one
UDMA getmem from each remote PE directly into ordinary-GM output.  The
local-rank slice bypasses symmetric memory and is copied directly to output on
the second vector sub-core while UDMA is in flight.

The symmetric workspace contains two alternating send buffers, with one
contiguous payload per target PE in each buffer::

    [send_buffer=2, target_rank, S, QKV=3, heads_per_rank, D]

The ordinary-GM output uses the same combined ``(rank_size*S, 3*H, D)``
layout as Torch Golden.  Its Q/K/V results are zero-copy views of that tensor.

This standalone entry point runs the Golden reference and the UDMA kernel once
per swept shape and checks accuracy.  Event timing for this example lives in
``python/triton_dist/test/ascend/qkv-all2all-udma.py``.  This s32k variant also sweeps
32768/131072 sequence lengths in addition to the original 2k/4k/8k cases.  The
remote UDMA read is batched into ``MAX_GET_BYTES`` chunks instead of one
``getmem`` per S-block, which is the main fix for the large-S/small-H slowdown.
"""

import math
import os

import shmem as ash
import torch
import torch_npu  # noqa: F401  # Register the torch.npu backend.
import torch.distributed as dist
import triton
import triton.language as tl
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id, sub_vec_num
from triton_dist.language.extra import libshmem_device
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "20000-29999")
MAX_GET_BYTES = 128 * 1024 * 1024
COMM_BLOCK_S = 112
COMM_BLOCK_D = 128
COMM_HEAD_GROUP_OVERRIDE = int(os.environ.get("QKV_UDMA_HEAD_GROUP", "0"))
if COMM_HEAD_GROUP_OVERRIDE not in (0, 1, 2):
    raise ValueError("QKV_UDMA_HEAD_GROUP must be 0 (auto), 1, or 2")
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def qkv_fuse_a2a_golden(q, k, v, rank_size, group=None):
    """Return the standard Torch QKV All2All reference result."""
    q_transposed = torch.cat(torch.chunk(q, rank_size, dim=1), dim=0)
    k_transposed = torch.cat(torch.chunk(k, rank_size, dim=1), dim=0)
    v_transposed = torch.cat(torch.chunk(v, rank_size, dim=1), dim=0)
    heads_per_rank = q_transposed.shape[1]
    qkv = torch.cat([q_transposed, k_transposed, v_transposed], dim=1).contiguous()
    recv = torch.empty_like(qkv)
    dist.all_to_all_single(recv, qkv, group=group)
    return recv.split([heads_per_rank] * 3, dim=1)


def workspace_size(
    sequence_length,
    n_head,
    rank_size,
    head_dim=COMM_BLOCK_D,
    comm_block_s=COMM_BLOCK_S,
):
    """Return the number of elements required in the symmetric workspace."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if rank_size <= 0:
        raise ValueError("rank_size must be positive")
    if n_head <= 0 or n_head % rank_size != 0:
        raise ValueError("n_head must be positive and divisible by rank_size")
    if head_dim != COMM_BLOCK_D:
        raise ValueError(f"head_dim must be {COMM_BLOCK_D}")
    if comm_block_s <= 0:
        raise ValueError("comm_block_s must be positive")

    heads_per_rank = n_head // rank_size
    rank_payload_elements = 3 * heads_per_rank * sequence_length * head_dim
    return 2 * rank_size * rank_payload_elements


def symmetric_memory_size(rank_size, s_values, h_values) -> int:
    """Return the aligned ACLSHMEM heap capacity the sweep needs.

    The peer staging is a double-buffered mirror of every rank's bf16 q/k/v
    payload for one case, worst-cased over the sweep via workspace_size().
    """
    peer_bytes = max(
        workspace_size(s, h * rank_size, rank_size) * 2
        for s in s_values
        for h in h_values
    )
    required = peer_bytes
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(rank_size, s_values, h_values) -> int:
    """Symmetric heap: QKV_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("QKV_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(rank_size, s_values, h_values)
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the QKV_ASH_SIZE_GB override"
        )
    return pool_bytes


def _select_launch_ai_cores(
    sequence_length,
    heads_per_rank,
    rank_size,
    ai_cores,
    remote_pack_tasks,
    heads_per_group,
    comm_block_s,
):
    """Select the measured winner for each coarse S/head region."""
    launch_ai_cores = min(
        ai_cores,
        max(rank_size, math.ceil(remote_pack_tasks / 2)),
    )
    if comm_block_s != 112 or heads_per_group != 1:
        tuned_cores = None
    elif sequence_length < 3000:
        tuned_cores = {4: 24, 5: 32, 6: 32}.get(heads_per_rank)
    elif sequence_length < 6000:
        tuned_cores = {1: 32, 3: 32, 4: 24, 6: 32}.get(heads_per_rank)
    else:
        tuned_cores = {1: 32, 2: 32, 5: 32}.get(heads_per_rank)
    if tuned_cores is not None:
        launch_ai_cores = min(ai_cores, max(rank_size, tuned_cores))

    # Retain an explicit profiling override without changing normal dispatch.
    core_override = int(os.environ.get("QKV_UDMA_CORES", "0"))
    if core_override > 0:
        launch_ai_cores = min(ai_cores, max(rank_size, core_override))
    return launch_ai_cores


def _select_heads_per_group(sequence_length, heads_per_rank):
    """Select measured contiguous-head packing width for this shape."""
    if COMM_HEAD_GROUP_OVERRIDE > 0:
        return COMM_HEAD_GROUP_OVERRIDE
    if sequence_length < 3000:
        return 2 if heads_per_rank in (3, 6) else 1
    if sequence_length < 6000:
        return 2 if heads_per_rank in (2, 3, 4) else 1
    return 2 if heads_per_rank == 5 else 1


def _select_blocks_per_get(
    heads_per_rank,
    element_bytes,
    comm_block_s=COMM_BLOCK_S,
    comm_block_d=COMM_BLOCK_D,
):
    """Return how many S-blocks fit in one UDMA get under MAX_GET_BYTES."""
    bytes_per_block = comm_block_s * 3 * heads_per_rank * comm_block_d * element_bytes
    return max(1, MAX_GET_BYTES // bytes_per_block)


@triton.jit
def kernel_qkv_fuse_a2a_udma_get(
    q_ptr,
    k_ptr,
    v_ptr,
    qkv_out_ptr,
    peer_mem_ptr,
    rank,
    rank_size,
    buffer_id,
    sequence_length,
    stride_a_s,
    stride_a_h,
    stride_a_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    rank_payload_elements,
    ELEMENT_BYTES: tl.constexpr,
    BLOCKS_PER_GET: tl.constexpr,
):
    """Pack remote data and get once per PE directly into QKV output."""
    rank_payload_elements = rank_payload_elements.to(tl.int64)
    vec_num_per_aicore = 2
    actual_sub_vec_num = sub_vec_num()
    physical_ncores = tl.num_programs(axis=0) * actual_sub_vec_num // vec_num_per_aicore
    physical_pid = tl.program_id(axis=0) * actual_sub_vec_num // vec_num_per_aicore
    global_vec_id = physical_pid * actual_sub_vec_num + sub_vec_id()
    vector_ncores = tl.num_programs(axis=0) * actual_sub_vec_num
    num_blocks = tl.cdiv(sequence_length, BLOCK_S)
    head_groups = tl.cdiv(HEADS_PER_RANK, HEADS_PER_GROUP)
    qkv_row_stride = 3 * HEADS_PER_RANK * BLOCK_D
    qkv_group_stride = HEADS_PER_RANK * BLOCK_D
    send_buffer_base = buffer_id * rank_size * rank_payload_elements

    # Phase 1: flatten (remote rank, S block, head group).  Grouping adjacent
    # heads makes both the ordinary-GM load and symmetric-GM store contiguous
    # over a wider row while preserving the single contiguous PE payload.
    remote_pack_tasks = (rank_size - 1) * num_blocks * head_groups
    for task_idx in range(global_vec_id, remote_pack_tasks, vector_ncores):
        head_group = task_idx % head_groups
        rank_block_idx = task_idx // head_groups
        block_id = rank_block_idx % num_blocks
        remote_index = rank_block_idx // num_blocks
        target_rank = remote_index
        if remote_index >= rank:
            target_rank += 1

        # S-block offsets address the full [S, n_head, D] input, so they
        # need int64 once S * n_head * D crosses 2**31 elements.
        offs_s = (block_id * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
        if HEADS_PER_GROUP == 1:
            offs_d = tl.arange(0, BLOCK_D)
            local_head = head_group
            peer_head_offset = local_head * BLOCK_D
            mask = (offs_s < sequence_length)[:, None]
        else:
            offs_hd = tl.arange(0, HEADS_PER_GROUP * BLOCK_D)
            head_in_group = offs_hd // BLOCK_D
            offs_d = offs_hd % BLOCK_D
            local_head = head_group * HEADS_PER_GROUP + head_in_group
            peer_head_offset = local_head[None, :] * BLOCK_D
            mask = (offs_s < sequence_length)[:, None] & (local_head < HEADS_PER_RANK)[
                None, :
            ]

        global_head = target_rank * HEADS_PER_RANK + local_head
        input_offs = (
            offs_s[:, None] * stride_a_s
            + global_head * stride_a_h
            + offs_d[None, :] * stride_a_d
        )
        peer_base = (
            send_buffer_base
            + target_rank * rank_payload_elements
            + offs_s[:, None] * qkv_row_stride
            + peer_head_offset
            + offs_d[None, :]
        )
        q_data = tl.load(q_ptr + input_offs, mask=mask, other=0.0)
        k_data = tl.load(k_ptr + input_offs, mask=mask, other=0.0)
        v_data = tl.load(v_ptr + input_offs, mask=mask, other=0.0)
        tl.store(peer_mem_ptr + peer_base, q_data, mask=mask)
        tl.store(
            peer_mem_ptr + peer_base + qkv_group_stride,
            k_data,
            mask=mask,
        )
        tl.store(
            peer_mem_ptr + peer_base + 2 * qkv_group_stride,
            v_data,
            mask=mask,
        )

    if rank_size > 1:
        # The SHMEM barrier synchronizes all local vector cores and all PEs,
        # making every MTE store above visible before any UDMA source is read.
        libshmem_device.barrier_all()

    # Phase 2a: exactly one physical core owns each source PE.  getmem writes
    # directly to ordinary GM and is blocking (including per-PE quiet), so no
    # source can consume multiple quiet slots concurrently and no unpack pass
    # is required after communication.
    if global_vec_id % vec_num_per_aicore == 0:
        for source_rank in range(physical_pid, rank_size, physical_ncores):
            if source_rank != rank:
                # On source_rank, slot `rank` contains the payload packed for
                # this PE.  The destination slot is the final output rows for
                # source_rank, in the same QKV-interleaved layout as Golden.
                # UDMA imposes a maximum message length, so batch several
                # S-blocks per getmem while staying safely below that limit.
                # Large-S/small-H cases especially benefit from fewer, larger
                # getmem calls instead of one call per 112-row block.
                for block_start in range(0, num_blocks, BLOCKS_PER_GET):
                    block_end = tl.minimum(block_start + BLOCKS_PER_GET, num_blocks)
                    chunk_blocks = block_end - block_start
                    last_block_id = block_end - 1
                    last_block_s = tl.minimum(
                        BLOCK_S,
                        sequence_length - last_block_id * BLOCK_S,
                    )
                    chunk_s_elements = (chunk_blocks - 1) * BLOCK_S + last_block_s
                    source = (
                        peer_mem_ptr
                        + send_buffer_base
                        + rank * rank_payload_elements
                        + block_start * BLOCK_S * qkv_row_stride
                    )
                    destination = (
                        qkv_out_ptr
                        + source_rank * rank_payload_elements
                        + block_start * BLOCK_S * qkv_row_stride
                    )
                    libshmem_device.getmem(
                        destination,
                        source,
                        chunk_s_elements * qkv_row_stride * ELEMENT_BYTES,
                        source_rank,
                    )

    # Phase 2b: odd vector sub-cores copy the local-rank slice directly from
    # ordinary GM to ordinary GM via MTE while the even UDMA lane is busy.
    # This removes the former self pack+unpack round trip through SHMEM.
    if global_vec_id % vec_num_per_aicore == 1:
        local_copy_tasks = num_blocks * head_groups
        for local_task_idx in range(physical_pid, local_copy_tasks, physical_ncores):
            head_group = local_task_idx % head_groups
            block_id = local_task_idx // head_groups
            offs_s = (block_id * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
            if HEADS_PER_GROUP == 1:
                offs_d = tl.arange(0, BLOCK_D)
                local_head = head_group
                output_head_offset = local_head * BLOCK_D
                mask = (offs_s < sequence_length)[:, None]
            else:
                offs_hd = tl.arange(0, HEADS_PER_GROUP * BLOCK_D)
                head_in_group = offs_hd // BLOCK_D
                offs_d = offs_hd % BLOCK_D
                local_head = head_group * HEADS_PER_GROUP + head_in_group
                output_head_offset = local_head[None, :] * BLOCK_D
                mask = (offs_s < sequence_length)[:, None] & (
                    local_head < HEADS_PER_RANK
                )[None, :]
            output_base = (
                rank * rank_payload_elements
                + offs_s[:, None] * qkv_row_stride
                + offs_d[None, :]
            )

            global_head = rank * HEADS_PER_RANK + local_head
            input_offs = (
                offs_s[:, None] * stride_a_s
                + global_head * stride_a_h
                + offs_d[None, :] * stride_a_d
            )
            output_offs = output_base + output_head_offset
            q_data = tl.load(q_ptr + input_offs, mask=mask, other=0.0)
            k_data = tl.load(k_ptr + input_offs, mask=mask, other=0.0)
            v_data = tl.load(v_ptr + input_offs, mask=mask, other=0.0)
            tl.store(qkv_out_ptr + output_offs, q_data, mask=mask)
            tl.store(
                qkv_out_ptr + output_offs + qkv_group_stride,
                k_data,
                mask=mask,
            )
            tl.store(
                qkv_out_ptr + output_offs + 2 * qkv_group_stride,
                v_data,
                mask=mask,
            )

    # getmem is blocking, hence local output is complete at kernel return.  No
    # trailing world barrier is needed: the next invocation packs the other
    # send buffer, and its pre-get barrier proves that every PE has finished
    # the preceding get before that old buffer can be reused two calls later.


def _validate_tensors(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    peer_mem,
    rank,
    rank_size,
    comm_block_s,
    comm_block_d,
):
    if rank_size <= 0:
        raise ValueError("rank_size must be positive")
    if rank < 0 or rank >= rank_size:
        raise ValueError(f"rank={rank} is outside [0, {rank_size})")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.stride() != k.stride() or q.stride() != v.stride():
        raise ValueError("q, k, and v must have identical strides")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have identical dtypes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.ndim != 3:
        raise ValueError("q, k, and v must have shape (S, n_head, head_dim)")

    sequence_length, n_head, head_dim = q.shape
    if n_head % rank_size != 0:
        raise ValueError("n_head must be divisible by rank_size")
    if comm_block_s <= 0 or comm_block_d != COMM_BLOCK_D:
        raise ValueError(
            f"this implementation requires positive block_s and "
            f"block_d={COMM_BLOCK_D}"
        )
    required_elements = workspace_size(
        sequence_length,
        n_head,
        rank_size,
        head_dim,
        comm_block_s,
    )
    if peer_mem.numel() < required_elements:
        raise ValueError(
            f"peer_mem has {peer_mem.numel()} elements; "
            f"{required_elements} required"
        )
    if peer_mem.dtype != q.dtype or peer_mem.device != q.device:
        raise ValueError("peer_mem must match the input dtype and device")
    if not peer_mem.is_contiguous():
        raise ValueError("peer_mem must be contiguous")
    heads_per_rank = n_head // rank_size
    expected_output_shape = (
        sequence_length * rank_size,
        heads_per_rank,
        head_dim,
    )
    for name, output in (("q_out", q_out), ("k_out", k_out), ("v_out", v_out)):
        if tuple(output.shape) != expected_output_shape:
            raise ValueError(
                f"{name} has shape {tuple(output.shape)}; "
                f"expected {expected_output_shape}"
            )
        if output.dtype != q.dtype or output.device != q.device:
            raise ValueError(f"{name} must match the input dtype and device")
    if q_out.stride() != k_out.stride() or q_out.stride() != v_out.stride():
        raise ValueError("q_out, k_out, and v_out must have identical strides")
    expected_output_stride = (
        3 * heads_per_rank * head_dim,
        head_dim,
        1,
    )
    if q_out.stride() != expected_output_stride:
        raise ValueError(
            "outputs must be Q/K/V views of a contiguous "
            f"(rank_size*S, 3*H, D) tensor; got stride={q_out.stride()}, "
            f"expected {expected_output_stride}"
        )
    element_bytes = q.element_size()
    q_address = q_out.data_ptr()
    group_bytes = heads_per_rank * head_dim * element_bytes
    if (
        k_out.data_ptr() != q_address + group_bytes
        or v_out.data_ptr() != q_address + 2 * group_bytes
    ):
        raise ValueError(
            "q_out, k_out, and v_out must be adjacent views of one "
            "QKV-interleaved output tensor"
        )
    return sequence_length, heads_per_rank, head_dim, required_elements


def qkv_fuse_a2a(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    peer_mem,
    rank,
    rank_size,
    comm_block_s=COMM_BLOCK_S,
    comm_block_d=COMM_BLOCK_D,
):
    """Launch the fused remote-pack/UDMA-get QKV All2All kernel."""
    sequence_length, heads_per_rank, _, _ = _validate_tensors(
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        peer_mem,
        rank,
        rank_size,
        comm_block_s,
        comm_block_d,
    )
    rank_payload_elements = 3 * heads_per_rank * sequence_length * comm_block_d
    ai_cores = driver.active.utils.get_aicore_num()
    if ai_cores <= 0:
        raise RuntimeError(f"invalid AI Core count: {ai_cores}")
    num_blocks = math.ceil(sequence_length / comm_block_s)
    heads_per_group = _select_heads_per_group(sequence_length, heads_per_rank)
    head_groups = math.ceil(heads_per_rank / heads_per_group)
    remote_pack_tasks = (rank_size - 1) * num_blocks * head_groups
    launch_ai_cores = _select_launch_ai_cores(
        sequence_length,
        heads_per_rank,
        rank_size,
        ai_cores,
        remote_pack_tasks,
        heads_per_group,
        comm_block_s,
    )
    buffer_id = getattr(peer_mem, "_qkv_udma_next_buffer", 0)
    peer_mem._qkv_udma_next_buffer = 1 - buffer_id
    kernel_qkv_fuse_a2a_udma_get[launch_ai_cores, 1, 1](
        q,
        k,
        v,
        q_out,
        peer_mem,
        rank,
        rank_size,
        buffer_id,
        sequence_length,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        BLOCK_S=comm_block_s,
        BLOCK_D=comm_block_d,
        HEADS_PER_RANK=heads_per_rank,
        HEADS_PER_GROUP=heads_per_group,
        rank_payload_elements=rank_payload_elements,
        ELEMENT_BYTES=q.element_size(),
        BLOCKS_PER_GET=_select_blocks_per_get(
            heads_per_rank,
            q.element_size(),
            comm_block_s,
            comm_block_d,
        ),
    )


def _prepare_qkv_fuse_a2a(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    peer_mem,
    rank,
    rank_size,
    comm_block_s=COMM_BLOCK_S,
    comm_block_d=COMM_BLOCK_D,
):
    """Validate once and return the low-overhead timed kernel launcher."""
    sequence_length, heads_per_rank, _, _ = _validate_tensors(
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        peer_mem,
        rank,
        rank_size,
        comm_block_s,
        comm_block_d,
    )
    rank_payload_elements = 3 * heads_per_rank * sequence_length * comm_block_d
    ai_cores = driver.active.utils.get_aicore_num()
    if ai_cores <= 0:
        raise RuntimeError(f"invalid AI Core count: {ai_cores}")
    num_blocks = math.ceil(sequence_length / comm_block_s)
    heads_per_group = _select_heads_per_group(sequence_length, heads_per_rank)
    head_groups = math.ceil(heads_per_rank / heads_per_group)
    remote_pack_tasks = (rank_size - 1) * num_blocks * head_groups
    launch_ai_cores = _select_launch_ai_cores(
        sequence_length,
        heads_per_rank,
        rank_size,
        ai_cores,
        remote_pack_tasks,
        heads_per_group,
        comm_block_s,
    )
    input_strides = q.stride()
    element_bytes = q.element_size()
    next_buffer = 0

    def launch():
        nonlocal next_buffer
        buffer_id = next_buffer
        next_buffer = 1 - next_buffer
        kernel_qkv_fuse_a2a_udma_get[launch_ai_cores, 1, 1](
            q,
            k,
            v,
            q_out,
            peer_mem,
            rank,
            rank_size,
            buffer_id,
            sequence_length,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            BLOCK_S=comm_block_s,
            BLOCK_D=comm_block_d,
            HEADS_PER_RANK=heads_per_rank,
            HEADS_PER_GROUP=heads_per_group,
            rank_payload_elements=rank_payload_elements,
            ELEMENT_BYTES=element_bytes,
            BLOCKS_PER_GET=_select_blocks_per_get(
                heads_per_rank,
                element_bytes,
                comm_block_s,
                comm_block_d,
            ),
        )

    launch.heads_per_group = heads_per_group
    launch.launch_ai_cores = launch_ai_cores
    launch.block_s = comm_block_s
    return launch


def _parse_int_list(name, default_values):
    raw = os.environ.get(name)
    if raw is None:
        return list(default_values)
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def _check_accuracy(
    q,
    q_golden,
    k_golden,
    v_golden,
    q_out,
    k_out,
    v_out,
    rank,
    rank_size,
):
    """Compare the final benchmark outputs collectively on every rank."""
    passed = torch.tensor([1], dtype=torch.int32, device=q.device)
    error_message = ""
    try:
        torch.testing.assert_close(q_out, q_golden, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_out, k_golden, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v_out, v_golden, rtol=1e-3, atol=1e-3)
    except AssertionError as error:
        passed[0] = 0
        error_message = str(error)

    all_passed = [torch.zeros_like(passed) for _ in range(rank_size)]
    dist.all_gather(all_passed, passed)
    dist.barrier()
    for rank_id in range(rank_size):
        if rank == rank_id:
            if passed.item() == 1:
                print(
                    f"  {GREEN}[PASS]{RESET} Rank {rank}: "
                    "UDMA output matches Golden.",
                    flush=True,
                )
            else:
                print(
                    f"  {RED}[FAIL]{RESET} Rank {rank}: "
                    f"UDMA output failed. Details:\n{error_message}",
                    flush=True,
                )
        dist.barrier()
    if any(value.item() == 0 for value in all_passed):
        raise AssertionError("UDMA QKV output does not match Torch Golden")


def run_hccl_benchmark():
    """Run Golden and UDMA per shape and check accuracy."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")

    # Same knobs and defaults as qkv-all2all.py.
    default_sequence_lengths = (
        list(range(2040, 2049))
        + list(range(4088, 4097))
        + list(range(8184, 8193))
        + list(range(32760, 32769))
        + list(range(131064, 131073))
    )
    sequence_lengths = _parse_int_list("QKV_BENCH_S", default_sequence_lengths)
    local_heads = _parse_int_list("QKV_BENCH_H", range(1, 8))
    expected_cases = len(sequence_lengths) * len(local_heads)

    if rank == 0:
        print(
            "CONFIG "
            f"engine=UDMA-event-s32k-chunk ranks={rank_size} "
            f"aicore_num={driver.active.utils.get_aicore_num()} "
            f"vector_cores={driver.active.utils.get_aivector_core_num()} "
            f"block_s={COMM_BLOCK_S} "
            f"head_group_override={COMM_HEAD_GROUP_OVERRIDE} "
            f"cases={expected_cases}",
            flush=True,
        )

    dist.barrier()
    init_aclshmem(
        rank,
        rank_size,
        ash_pool_bytes(rank_size, sequence_lengths, local_heads),
        ash.OpEngineType.UDMA,
    )

    try:
        for sequence_length in sequence_lengths:
            for heads_per_rank in local_heads:
                n_head = heads_per_rank * rank_size
                case_block_s = COMM_BLOCK_S

                torch.manual_seed(42 + rank)
                q = torch.randn(
                    sequence_length,
                    n_head,
                    COMM_BLOCK_D,
                    dtype=torch.bfloat16,
                    device=device,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                # Keep the same contiguous QKV output contract as the UDMA
                # kernel requires, while exposing Golden-compatible views.
                qkv_out = torch.empty(
                    sequence_length * rank_size,
                    3 * heads_per_rank,
                    COMM_BLOCK_D,
                    dtype=q.dtype,
                    device=device,
                )
                q_out, k_out, v_out = qkv_out.split([heads_per_rank] * 3, dim=1)

                q_golden, k_golden, v_golden = qkv_fuse_a2a_golden(q, k, v, rank_size)

                peer_elements = workspace_size(
                    sequence_length,
                    n_head,
                    rank_size,
                    comm_block_s=case_block_s,
                )
                peer_mem = ash.aclshmem_create_tensor(
                    [peer_elements], dtype=q.dtype, device_id=local_rank
                )
                torch.npu.synchronize()
                dist.barrier()

                # Validate once and retain the low-overhead launcher.
                udma_call = _prepare_qkv_fuse_a2a(
                    q,
                    k,
                    v,
                    q_out,
                    k_out,
                    v_out,
                    peer_mem,
                    rank,
                    rank_size,
                    comm_block_s=case_block_s,
                )
                try:
                    udma_call()
                    torch.npu.synchronize()

                    _check_accuracy(
                        q,
                        q_golden,
                        k_golden,
                        v_golden,
                        q_out,
                        k_out,
                        v_out,
                        rank,
                        rank_size,
                    )
                finally:
                    ash.aclshmem_free_tensor(peer_mem)
                    dist.barrier()
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_hccl_benchmark()
