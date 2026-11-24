# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import torch
import torch_npu
import shmem as ash
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.runtime import driver
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# dl.notify requires an int32/uint32 signal pointer on Ascend.
# aclshmem_wait walks barriers 64 bytes apart: 64 / sizeof(int32) = 16 elements.
SIGNAL_SLOT_STRIDE = 16

# Staging geometry, identical for every sweep case: the peer buffer holds one
# COMM_BLOCK_S chunk per (buffer, remote head row, D) and the int32 signal
# arena one slot-stride entry per (head, rank, buffer).
COMM_BLOCK_S = 64
COMM_BLOCK_D = 128
BUFFER_NUM = 2


def symmetric_memory_size(rank_size, d, h_values) -> int:
    """Return the aligned ACLSHMEM heap capacity the sweep needs.

    Neither arena scales with S, so only the head sweep matters.
    """
    h_max = max(h_values)
    peer_bytes = COMM_BLOCK_S * h_max * rank_size * d * BUFFER_NUM * 2
    signal_bytes = h_max * rank_size * BUFFER_NUM * SIGNAL_SLOT_STRIDE * 4
    required = peer_bytes + signal_bytes
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(rank_size, d, h_values) -> int:
    """Symmetric heap: A2A_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("A2A_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(rank_size, d, h_values)
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the A2A_ASH_SIZE_GB override"
        )
    return pool_bytes


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
    SIGNAL_STRIDE: tl.constexpr,
):
    """
    Optimized HCCL Reverse All-to-All Triton kernel on Ascend AICore.
    - Uses Vector Core level launching (double buffering pipelining via program ID).
    - Includes dynamic masks for non-aligned S and D dimensions.
    """
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    # Calculate role (Producer or Consumer) based on Vector Core program ID
    # Since we launch at Vector Core granularity, even pids are Producers, odd pids are Consumers
    role = pid % 2
    logical_core_id = pid // 2
    n_role_cores = ncore // 2

    num_blocks_s = tl.cdiv(S, COMM_BLOCK_S)
    num_blocks_d = tl.cdiv(D, COMM_BLOCK_D)

    # Physical buffer stride configuration
    # Layout of symmetric memory: [buffer_num, S_block, H * rank_size, D]
    buffer_chunk_size = (H * rank_size) * D
    stride_ps = H * rank_size * D
    stride_ph = D
    stride_pd = 1

    # Outer loop: iterate over blocks of dimension S
    for global_id_s in range(0, num_blocks_s):
        buffer_id = global_id_s % buffer_num

        # Producer Stage (Even Vector Cores): cross-card remote store
        if role == 0:
            total_prod_tasks = num_blocks_d * H * rank_size
            for task_idx in range(logical_core_id, total_prod_tasks, n_role_cores):
                tmp = task_idx
                rank_loop_id = tmp % rank_size
                tmp //= rank_size
                h_id = tmp % H
                block_id_d = tmp // H

                target_rank = (rank + rank_loop_id) % rank_size

                # Compute physical offsets and boundary masks
                offs_s = global_id_s * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
                offs_d = block_id_d * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (offs_s < S)[:, None] & (offs_d < D)[None, :]

                # Load data from local A tensor
                a_s = target_rank * S + offs_s
                a_offs = (
                    a_s[:, None] * stride_as
                    + h_id * stride_ah
                    + offs_d[None, :] * stride_ad
                )
                a_data = tl.load(a_ptr + a_offs, mask=mask, other=0.0)

                # Write to remote symmetric memory
                remote_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                peer_h_write = rank * H + h_id

                peer_offs_write = (
                    buffer_id * (COMM_BLOCK_S * buffer_chunk_size)
                    + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                    + peer_h_write * stride_ph
                    + offs_d[None, :] * stride_pd
                )
                remote_signal_ptr = dl.symm_at(signal_mem_ptr, target_rank)
                token = dl.wait(
                    remote_signal_ptr
                    + buffer_id * num_blocks_d * H * rank_size * SIGNAL_STRIDE
                    + num_blocks_d * H * rank * SIGNAL_STRIDE
                    + tmp * SIGNAL_STRIDE,
                    1,
                    "gpu",
                    "acquire",
                    waitValue=0,
                )
                remote_ptr_dummy = dl.consume_token(remote_ptr, token)
                tl.store(remote_ptr_dummy + peer_offs_write, a_data, mask=mask)
            # Sync barrier: ensure cross-card transfer of current buffer is complete
            libshmem_device.fence()
            for task_idx in range(logical_core_id, total_prod_tasks, n_role_cores):
                tmp = task_idx
                rank_loop_id = tmp % rank_size
                tmp //= rank_size
                target_rank = (rank + rank_loop_id) % rank_size
                dl.notify(
                    signal_mem_ptr
                    + buffer_id * num_blocks_d * H * rank_size * SIGNAL_STRIDE
                    + num_blocks_d * H * rank * SIGNAL_STRIDE
                    + tmp * SIGNAL_STRIDE,
                    target_rank,
                )

        # Consumer Stage (Odd Vector Cores): read from local Shmem and store to C
        if role == 1:
            local_peer_ptr = dl.symm_at(peer_mem_ptr, rank)
            total_cons_tasks = num_blocks_d * H * rank_size
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

                peer_h_read = r * H + h_id
                peer_offs_read = (
                    buffer_id * (COMM_BLOCK_S * buffer_chunk_size)
                    + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                    + peer_h_read * stride_ph
                    + offs_d[None, :] * stride_pd
                )
                token = dl.wait(
                    signal_mem_ptr
                    + buffer_id * num_blocks_d * H * rank_size * SIGNAL_STRIDE
                    + num_blocks_d * H * r * SIGNAL_STRIDE
                    + tmp * SIGNAL_STRIDE,
                    1,
                    "gpu",
                    "acquire",
                )
                local_peer_ptr = dl.consume_token(local_peer_ptr, token)
                peer_data = tl.load(
                    local_peer_ptr + peer_offs_read, mask=mask, other=0.0
                )

                # Transpose and write to local C tensor
                c_h_idx = r * H + h_id
                c_offs = (
                    offs_s[:, None] * stride_cs
                    + c_h_idx * stride_ch
                    + offs_d[None, :] * stride_cd
                )
                tl.store(c_ptr + c_offs, peer_data, mask=mask)
            libshmem_device.fence()
            for task_idx in range(logical_core_id, total_cons_tasks, n_role_cores):
                tmp = task_idx
                r = tmp % rank_size
                tmp //= rank_size
                h_id = tmp % H
                block_id_d = tmp // H
                dl.notify(
                    signal_mem_ptr
                    + buffer_id * num_blocks_d * H * rank_size * SIGNAL_STRIDE
                    + num_blocks_d * H * r * SIGNAL_STRIDE
                    + tmp * SIGNAL_STRIDE,
                    rank,
                    0,
                )


def hccl_reverse_a2a_kernel_launcher(
    A, C, peer_mem, signal_mem, rank, rank_size, buffer_num, COMM_BLOCK_S, COMM_BLOCK_D
):
    """Launcher wrapper function"""
    S_total, H, D = A.shape
    S = S_total // rank_size

    # Launch with total Vector Core count instead of AI Core count
    vec_num = driver.active.utils.get_aivector_core_num()

    kernel_hccl_reverse_a2a_pipelined[vec_num, 1, 1](
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
        SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
    )


def torch_reverse_a2a(tensor, dsp_size, group=None):
    """Reference implementation using PyTorch native communication"""
    send = torch.cat(torch.chunk(tensor, dsp_size, dim=0), dim=1).contiguous()
    send = send.transpose(1, 0).contiguous()
    recv = torch.empty(send.shape, dtype=send.dtype, device=send.device)
    dist.all_to_all_single(recv, send, group=group)
    recv = recv.transpose(0, 1).contiguous()
    return recv


def run_test_distributed():
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # Generate contiguous S and H dimension test suites
    S_ranges = (
        list(range(2040, 2049))  # 2040 to 2048 (9 values)
        + list(range(4088, 4097))  # 4088 to 4096 (9 values)
        + list(range(8184, 8193))  # 8184 to 8192 (9 values)
    )
    H_list = [8, 12, 24, 28, 32, 40, 48, 56]
    D = 128

    # Initialize ACL symmetric shared memory environment
    init_aclshmem(
        pe, world_size, ash_pool_bytes(world_size, D, H_list), ash.OpEngineType.MTE
    )

    # Generate 216 shape combinations for full Cartesian sweep testing
    test_cases = []
    for s_val in S_ranges:
        for h_val in H_list:
            test_cases.append((s_val, h_val))

    total_cases = len(test_cases)
    if pe == 0:
        print(
            f"{BOLD}[START]{RESET} Starting full parameter sweep tests ({total_cases} cases) on world_size={world_size}...",
            flush=True,
        )

    # Iterate over all shape combinations for verification
    try:
        for case_idx, (S, H) in enumerate(test_cases):
            dist.barrier()
            if pe == 0:
                print(
                    f"\n{BOLD}Test Case {case_idx + 1}/{total_cases}:{RESET} S={S}, H={H}, D={D} (Aligned and Non-aligned shapes)",
                    flush=True,
                )

            dtype = torch.bfloat16
            S_total = S * world_size

            peer_mem_size = COMM_BLOCK_S * (H * world_size) * D * BUFFER_NUM
            peer_mem = ash.aclshmem_create_tensor(
                [peer_mem_size],
                dtype=dtype,
                device_id=local_pe,
            )
            signal_mem = ash.aclshmem_create_tensor(
                [H * world_size * BUFFER_NUM * SIGNAL_SLOT_STRIDE],
                dtype=torch.int32,
                device_id=local_pe,
            )

            A_local = torch.ones([S_total, H, D], dtype=dtype).npu()
            C_local = torch.zeros([S, H * world_size, D], dtype=dtype).npu()

            C_golden = torch_reverse_a2a(A_local, world_size)
            signal_mem.fill_(0)
            dist.barrier()
            hccl_reverse_a2a_kernel_launcher(
                A_local,
                C_local,
                peer_mem,
                signal_mem,
                pe,
                world_size,
                BUFFER_NUM,
                COMM_BLOCK_S,
                COMM_BLOCK_D,
            )
            torch.npu.synchronize()

            # Numerical accuracy check (with rtol/atol tolerance)
            passed = torch.tensor([1], dtype=torch.int32).npu()
            error_msg = ""
            try:
                torch.testing.assert_close(C_golden, C_local, rtol=1e-3, atol=1e-3)
            except AssertionError as e:
                passed[0] = 0
                error_msg = str(e)

            all_passed = [
                torch.zeros(1, dtype=torch.int32).npu() for _ in range(world_size)
            ]
            dist.all_gather(all_passed, passed)

            dist.barrier()
            for rank_id in range(world_size):
                if pe == rank_id:
                    if all_passed[rank_id].item() == 1:
                        print(
                            f"  {GREEN}[PASS]{RESET} Rank {pe}: S={S}, H={H} matches golden.",
                            flush=True,
                        )
                    else:
                        print(
                            f"  {RED}[FAIL]{RESET} Rank {pe}: S={S}, H={H} failed. Details:\n{error_msg}",
                            flush=True,
                        )
                dist.barrier()

            # Release temporary symmetric memory resources
            ash.aclshmem_free_tensor(peer_mem)
            ash.aclshmem_free_tensor(signal_mem)

            if any(r.item() == 0 for r in all_passed):
                raise AssertionError(f"Parameter Sweep failed at case: S={S}, H={H}")
    finally:
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)
    world_size = dist.get_world_size()

    print(f"[INFO] Rank {local_pe} of {world_size} initialized")
    dist.barrier()
    run_test_distributed()
    if local_pe == 0:
        print(
            f"\n{GREEN}[SUCCESS]{RESET} All 216 swept shapes have been verified successfully!"
        )
