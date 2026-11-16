# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reverse All-to-All HCCL kernel correctness check (UDMA).

This is the UDMA sibling of ``reverse-all2all.py``.  The kernel is a
pipelined wait/notify double-buffered reverse all-to-all.  For every swept
shape the Torch Golden reference runs first, then the Triton kernel, then the
accuracy check.  Event timing for this example lives in
``python/triton_dist/test/ascend/reverse-all2all-udma.py``.
"""

import os

import shmem as ash
import torch
import torch_npu  # noqa: F401  # Register the torch.npu backend.
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id, sub_vec_num
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem

# Default sweeps match reverse-alltoall-wait-notify.py.
S_RANGES = (
    list(range(2040, 2049))  # 2040 to 2048 (9 values)
    + list(range(4088, 4097))  # 4088 to 4096 (9 values)
    + list(range(8184, 8193))  # 8184 to 8192 (9 values)
    + list(range(32760, 32769))  # 32760 to 32769 (9 values)
)
H_LIST = [3, 6, 8, 12, 16, 24, 28, 32, 40, 48, 56]
D = 128
COMM_BLOCK_S = 128
COMM_BLOCK_D = D
BUFFER_NUM = 2


def symmetric_memory_size(rank_size: int, s_values=None, h_values=None) -> int:
    """Return the aligned ACLSHMEM heap capacity a sweep needs.

    The peer staging buffer holds every rank's S x H x D bf16 tile, and the
    signal arena holds one int64 slot per (S block, rank) plus the
    double-buffered D blocks per head, both worst-cased over the sweep.
    """
    s_max = max(s_values if s_values is not None else S_RANGES)
    h_max = max(h_values if h_values is not None else H_LIST)
    peer_bytes = rank_size * s_max * h_max * D * 2
    num_blocks_s = -(-s_max // COMM_BLOCK_S)
    num_blocks_d = -(-D // COMM_BLOCK_D)
    signal_bytes = (
        (num_blocks_s * rank_size + BUFFER_NUM * num_blocks_d * h_max) * 8 * 8
    )
    required = peer_bytes + signal_bytes
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(rank_size: int, s_values=None, h_values=None) -> int:
    """Symmetric heap: A2A_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("A2A_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(rank_size, s_values, h_values)
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the A2A_ASH_SIZE_GB override"
        )
    return pool_bytes


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


@triton.jit
def kernel_hccl_reverse_a2a_pipelined(
    a_ptr,
    c_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    S,
    H,
    D,
    stride_as,
    stride_ah,
    stride_ad,
    stride_cs,
    stride_ch,
    stride_cd,
    COMM_BLOCK_S: tl.constexpr,
    COMM_BLOCK_D: tl.constexpr,
    UDMA_CORES: tl.constexpr,
):
    """
    Optimized HCCL Reverse All-to-All Triton kernel on Ascend AICore.
    - Uses Vector Core level launching (double buffering pipelining via program ID).
    - Cross-rank UDMA sends and the local self-copy are each synchronized with
      point-to-point wait/notify semaphores (signal_mem_ptr) instead of a
      global libshmem_device.barrier_all(), so a task only stalls on the
      exact (block, rank) or (block, task) it actually depends on.
    - Includes dynamic masks for non-aligned S and D dimensions.
    """
    n_vec = 1
    total_vec = tl.num_programs(axis=0) * n_vec
    global_vec_id = tl.program_id(axis=0) * n_vec + sub_vec_id()
    logical_core_id = global_vec_id // 2
    n_role_cores = total_vec // 2
    elem_bytes = 2
    S = S.to(tl.int64)
    H = H.to(tl.int64)
    D = D.to(tl.int64)

    num_blocks_s = tl.cdiv(S, COMM_BLOCK_S)
    num_blocks_d = tl.cdiv(D, COMM_BLOCK_D)
    tasks_per_block = num_blocks_d * H

    # Physical buffer stride configuration
    # Layout of symmetric memory: [buffer_num, S_block, H, D]
    stride_ps = H * D
    stride_ph = D

    # signal_mem_ptr is a flat int64 array of two regions, each flag padded to
    # SIG_STRIDE elements (one cache line) so concurrently-touched flags don't
    # false-share:
    #   Region A [num_blocks_s, rank_size]      cross-rank UDMA block-ready flags,
    #                                            written once, never reset (each
    #                                            (block, sender) pair uses a unique
    #                                            peer_mem_ptr address, so there is
    #                                            no slot to free/reuse).
    #   Region B [buffer_num, tasks_per_block]  local self-copy double-buffer
    #                                            flags, cycled every buffer_num
    #                                            iterations (ready -> consumed ->
    #                                            free -> ready ...).
    SIG_STRIDE = 8
    region_b_base = num_blocks_s * rank_size * SIG_STRIDE

    # Outer loop: iterate over blocks of dimension S
    for global_id_s in range(0, num_blocks_s):
        buffer_id = global_id_s % buffer_num
        cur_block_s = tl.minimum(COMM_BLOCK_S, S - global_id_s * COMM_BLOCK_S)

        # Sender Stage (Even Vector Cores, logical_core_id in [0, UDMA_CORES)): UDMA-put this
        # S-block to each assigned peer rank, then notify only that rank's own
        # (global_id_s, rank) flag on its local signal_mem via symm_at. Since
        # peer_mem_ptr has a unique address per (block, sender) pair, there is
        # nothing to wait for before sending -- block i+1's put can fire right
        # after block i's, and a consumer on another rank only ever blocks on
        # the specific sender it needs, not on a cluster-wide rendezvous.
        if global_vec_id % 2 == 0 and logical_core_id < UDMA_CORES:
            for udma_target_rank in range(logical_core_id, rank_size, UDMA_CORES):
                if udma_target_rank != rank:
                    libshmem_device.putmem(
                        peer_mem_ptr
                        + rank * S * H * D
                        + global_id_s * COMM_BLOCK_S * stride_ps,
                        a_ptr
                        + udma_target_rank * S * H * D
                        + global_id_s * COMM_BLOCK_S * stride_as,
                        cur_block_s * H * D * elem_bytes,
                        udma_target_rank,
                    )
            for udma_target_rank in range(logical_core_id, rank_size, UDMA_CORES):
                if udma_target_rank != rank:
                    dl.notify(
                        signal_mem_ptr + (global_id_s * rank_size + rank) * SIG_STRIDE,
                        udma_target_rank.to(tl.int32),
                    )

        # Producer Stage (self data, no cross-card comm needed): write to local
        # symmetric memory by mte, execute on logical_core_id[UDMA_CORES, n_role_cores). Each
        # (h, block_d) task owns its own double-buffer flag, so overwriting a
        # slot only waits on the specific consumer task that last drained it.
        if global_vec_id % 2 == 0 and logical_core_id >= UDMA_CORES:
            logical_mte_pid = logical_core_id - UDMA_CORES
            logical_mte_ncores = n_role_cores - UDMA_CORES
            for task_idx in range(logical_mte_pid, tasks_per_block, logical_mte_ncores):
                h_id = task_idx % H
                block_id_d = task_idx // H

                # Compute physical offsets and boundary masks
                offs_s = global_id_s * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
                offs_d = block_id_d * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (offs_s < S)[:, None] & (offs_d < D)[None, :]

                # Load data from local A tensor
                a_s = rank * S + offs_s
                a_offs = (
                    a_s[:, None] * stride_as
                    + h_id * stride_ah
                    + offs_d[None, :] * stride_ad
                )
                a_data = tl.load(a_ptr + a_offs, mask=mask, other=0.0)

                # Wait until the previous occupant of this double-buffer slot
                # has been fully consumed (freed) before overwriting it.
                self_sig_ptr = (
                    signal_mem_ptr
                    + region_b_base
                    + (buffer_id * tasks_per_block + task_idx) * SIG_STRIDE
                )
                token = dl.wait(self_sig_ptr, 1, "gpu", "acquire", waitValue=0)
                store_ptr = dl.consume_token(peer_mem_ptr, token)

                # Write to local symmetric memory
                peer_offs_write = (
                    S * rank * H * D
                    + buffer_id * (COMM_BLOCK_S * stride_ps)
                    + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                    + h_id * stride_ph
                    + offs_d[None, :]
                )
                tl.store(store_ptr + peer_offs_write, a_data, mask=mask)
            libshmem_device.quiet_lowlevel(
                transport=libshmem_device.ACLSHMEM_TRANSPORT_MTE
            )
            for task_idx in range(logical_mte_pid, tasks_per_block, logical_mte_ncores):
                self_sig_ptr = (
                    signal_mem_ptr
                    + region_b_base
                    + (buffer_id * tasks_per_block + task_idx) * SIG_STRIDE
                )
                dl.notify(self_sig_ptr, rank)

        # Consumer Stage (Odd Vector Cores): read from local Shmem and store to
        # C. Each task waits only on the exact (block, sender) or (buffer,
        # task) flag it needs, so one slow sender/task no longer stalls
        # unrelated ranks or blocks the way a global barrier would.
        if global_vec_id % 2 == 1:
            total_cons_tasks = tasks_per_block * rank_size
            for task_idx in range(logical_core_id, total_cons_tasks, n_role_cores):
                tmp = task_idx
                r = tmp % rank_size
                tmp //= rank_size
                h_id = tmp % H
                block_id_d = tmp // H

                # Compute offsets and boundary masks
                offs_s = global_id_s * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
                offs_d = block_id_d * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (offs_s < S)[:, None] & (offs_d < D)[None, :]
                if r == rank:
                    self_sig_ptr = (
                        signal_mem_ptr
                        + region_b_base
                        + (buffer_id * tasks_per_block + tmp) * SIG_STRIDE
                    )
                    token = dl.wait(self_sig_ptr, 1, "gpu", "acquire")
                    load_ptr = dl.consume_token(peer_mem_ptr, token)
                    peer_offs_read = (
                        S * rank * H * D
                        + buffer_id * (COMM_BLOCK_S * stride_ps)
                        + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                        + h_id * stride_ph
                        + offs_d[None, :]
                    )
                    peer_data = tl.load(load_ptr + peer_offs_read, mask=mask, other=0.0)
                else:
                    remote_sig_ptr = (
                        signal_mem_ptr + (global_id_s * rank_size + r) * SIG_STRIDE
                    )
                    token = dl.wait(remote_sig_ptr, 1, "gpu", "acquire")
                    load_ptr = dl.consume_token(peer_mem_ptr, token)
                    peer_offs_read = (
                        S * r * H * D
                        + global_id_s * COMM_BLOCK_S * stride_ps
                        + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                        + h_id * stride_ph
                        + offs_d[None, :]
                    )
                    peer_data = tl.load(load_ptr + peer_offs_read, mask=mask, other=0.0)

                # Transpose and write to local C tensor
                c_h_idx = r * H + h_id
                c_offs = (
                    offs_s[:, None] * stride_cs
                    + c_h_idx * stride_ch
                    + offs_d[None, :] * stride_cd
                )
                tl.store(c_ptr + c_offs, peer_data, mask=mask)
            libshmem_device.quiet_lowlevel(
                transport=libshmem_device.ACLSHMEM_TRANSPORT_MTE
            )
            # Free this iteration's self double-buffer slots so a later
            # producer reusing the same buffer_id is allowed to overwrite them.
            for task_idx in range(logical_core_id, total_cons_tasks, n_role_cores):
                tmp = task_idx
                r = tmp % rank_size
                tmp //= rank_size
                if r == rank:
                    self_sig_ptr = (
                        signal_mem_ptr
                        + region_b_base
                        + (buffer_id * tasks_per_block + tmp) * SIG_STRIDE
                    )
                    dl.notify(self_sig_ptr, rank, 0)


def torch_reverse_a2a(tensor, dsp_size, group=None):
    """Reference implementation using PyTorch native communication"""
    send = torch.cat(torch.chunk(tensor, dsp_size, dim=0), dim=1).contiguous()
    send = send.transpose(1, 0).contiguous()
    recv = torch.empty(send.shape, dtype=send.dtype, device=send.device)
    dist.all_to_all_single(recv, send, group=group)
    recv = recv.transpose(0, 1).contiguous()
    return recv


def _prepare_reverse_launch(
    A,
    C,
    peer_mem,
    signal_mem,
    rank,
    rank_size,
    buffer_num,
    S,
    H,
    D,
    COMM_BLOCK_S,
    COMM_BLOCK_D,
):
    """Validate once and return a low-overhead timed launcher.

    The caller zeroes ``signal_mem`` before each launch; this launcher only
    issues the kernel launch.
    """
    aivectorcore_num = driver.active.utils.get_aivector_core_num()
    # Logical vector cores reserved for cross-rank UDMA sends; the remaining
    # even lanes produce the self copy.  Decoupled from rank_size so large
    # worlds (rank_size > logical core count) cannot starve the producer role.
    # Must stay below n_role_cores so at least one lane produces.
    sender_override = os.environ.get("A2A_UDMA_SENDER_CORES")
    udma_sender_cores = (
        int(sender_override)
        if sender_override is not None
        else driver.active.utils.get_aicore_num() // 2
    )
    # UDMA sender lanes must leave at least one even lane for the self-copy
    # producer, otherwise the consumer deadlocks on its own double-buffer
    # flags.
    role_cores = aivectorcore_num // 2
    if not 0 < udma_sender_cores < role_cores:
        raise ValueError(
            f"A2A_UDMA_SENDER_CORES={udma_sender_cores} must be in "
            f"[1, {role_cores - 1}] for aivectorcore_num={aivectorcore_num}"
        )

    def launch():
        kernel_hccl_reverse_a2a_pipelined[aivectorcore_num, 1, 1](
            A,
            C,
            peer_mem,
            signal_mem,
            rank,
            rank_size,
            buffer_num,
            S,
            H,
            D,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            COMM_BLOCK_S=COMM_BLOCK_S,
            COMM_BLOCK_D=COMM_BLOCK_D,
            UDMA_CORES=udma_sender_cores,
        )

    return launch


def _check_accuracy(C_golden, C_local, rank, rank_size, S, H):
    """Compare the final benchmark outputs collectively on every rank."""
    passed = torch.tensor([1], dtype=torch.int32, device=C_local.device)
    error_message = ""
    try:
        torch.testing.assert_close(C_golden, C_local, rtol=1e-3, atol=1e-3)
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
                    f"  {GREEN}[PASS]{RESET} Rank {rank}: S={S}, H={H} "
                    "matches golden.",
                    flush=True,
                )
            else:
                print(
                    f"  {RED}[FAIL]{RESET} Rank {rank}: S={S}, H={H} failed. "
                    f"Details:\n{error_message}",
                    flush=True,
                )
        dist.barrier()
    if any(value.item() == 0 for value in all_passed):
        raise AssertionError(
            f"Reverse A2A kernel output does not match Torch Golden " f"at S={S}, H={H}"
        )


def run_reverse_benchmark():
    """Time Golden and kernel with torch.npu.Event and print speedups."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")

    pool_bytes = ash_pool_bytes(rank_size)
    if rank == 0:
        print(
            "CONFIG "
            f"engine=reverse-a2a-event ranks={rank_size} "
            f"pool={pool_bytes >> 20}MB "
            f"aicore_num={driver.active.utils.get_aicore_num()} "
            f"vector_cores={driver.active.utils.get_aivector_core_num()} "
            f"block_s={COMM_BLOCK_S} block_d={COMM_BLOCK_D} "
            f"buffer_num={BUFFER_NUM} "
            f"cases={len(S_RANGES) * len(H_LIST)}",
            flush=True,
        )

    dist.barrier()
    init_aclshmem(rank, rank_size, pool_bytes, ash.OpEngineType.UDMA)

    try:
        for S in S_RANGES:
            for H in H_LIST:
                S_total = S * rank_size
                peer_mem_size = rank_size * S * H * D
                peer_mem = ash.aclshmem_create_tensor(
                    [peer_mem_size], dtype=torch.bfloat16, device_id=local_rank
                )
                num_blocks_s = -(-S // COMM_BLOCK_S)
                num_blocks_d = -(-D // COMM_BLOCK_D)
                signal_mem_size = (
                    num_blocks_s * rank_size + BUFFER_NUM * num_blocks_d * H
                ) * 8
                signal_mem = ash.aclshmem_create_tensor(
                    [signal_mem_size], dtype=torch.int64, device_id=local_rank
                )
                torch.npu.synchronize()
                dist.barrier()

                torch.manual_seed(42 + rank)
                A_local = torch.randn(
                    [S_total, H, D], dtype=torch.bfloat16, device=device
                )
                C_local = torch.zeros(
                    [S, H * rank_size, D], dtype=torch.bfloat16, device=device
                )

                # Run the Torch Golden reference first.
                C_golden = torch_reverse_a2a(A_local, rank_size)

                launch = _prepare_reverse_launch(
                    A_local,
                    C_local,
                    peer_mem,
                    signal_mem,
                    rank,
                    rank_size,
                    BUFFER_NUM,
                    S,
                    H,
                    D,
                    COMM_BLOCK_S,
                    COMM_BLOCK_D,
                )
                try:
                    signal_mem.fill_(0)
                    dist.barrier()
                    launch()
                    torch.npu.synchronize()
                    _check_accuracy(C_golden, C_local, rank, rank_size, S, H)
                finally:
                    ash.aclshmem_free_tensor(peer_mem)
                    ash.aclshmem_free_tensor(signal_mem)
                    dist.barrier()
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_reverse_benchmark()
