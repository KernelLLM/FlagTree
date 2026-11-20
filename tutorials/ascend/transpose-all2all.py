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

# Shape sweep (189 cases): S_RANGES x H_LIST, head_dim fixed to 128, dsp_size=8.
S_RANGES = list(range(2040, 2049)) + list(range(4088, 4097)) + list(range(8184, 8193))
H_LIST = [1, 2, 3, 4, 5, 6, 7]
HEAD_DIM = 128
DSP_SIZE = 8
COMM_BLOCK_S = 128
COMM_BLOCK_D = 128


def symmetric_memory_size(world_size) -> int:
    """Return the aligned ACLSHMEM heap capacity the sweep needs.

    The peer staging holds one COMM_BLOCK_S x D chunk per (remote head,
    S block) and the int32 signal arena one slot-stride entry per
    (S block, rank, D block, head); both scale with the swept S and H.
    """
    s_max, h_max = max(S_RANGES), max(H_LIST)
    buffer_num = -(-s_max // COMM_BLOCK_S)
    num_blocks_d = -(-HEAD_DIM // COMM_BLOCK_D)
    peer_bytes = COMM_BLOCK_S * h_max * world_size * HEAD_DIM * buffer_num * 2
    signal_bytes = (
        buffer_num * world_size * num_blocks_d * h_max * SIGNAL_SLOT_STRIDE * 4
    )
    required = peer_bytes + signal_bytes
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(world_size) -> int:
    """Symmetric heap: A2A_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("A2A_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(world_size)
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the A2A_ASH_SIZE_GB override"
        )
    return pool_bytes


def torch_transpose_a2a(tensor, dsp_size, group=None):
    """Forward All2All dispatch. tensor: (S, n_head, head_dim) with n_head = H * dsp_size.
    Returns recv: (S * dsp_size, H, head_dim)."""
    # [S, n_head, head_dim] -> [S * dsp_size, n_head / dsp_size, head_dim]
    transposed = torch.cat(torch.chunk(tensor, dsp_size, dim=1), dim=0).contiguous()
    recv = torch.empty_like(transposed)
    dist.all_to_all_single(recv, transposed, group=group)
    return recv


@triton.jit
def kernel_hccl_transpose_a2a(
    a_ptr,
    c_ptr,
    peer_mem_ptr,
    signal_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    S,
    D,
    stride_as,
    stride_ah,
    stride_ad,
    stride_cs,
    stride_ch,
    stride_cd,
    COMM_BLOCK_S: tl.constexpr,
    COMM_BLOCK_D: tl.constexpr,
    H: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
):
    """
    Fused Transpose-All2All kernel.

    Each S-block owns a dedicated symmetric-memory slot (buffer_num == num_blocks_s),
    eliminating circular-buffer backpressure.  Producer writes directly to remote
    peer_mem + fence + notify(value=1).  Consumer wait(waitValue=1) + acquire + read.
    No "buffer free" handshake needed.

    Input A: (S, H * rank_size, D) — token hidden states, n_head sharded.
    Output C: (S * rank_size, H, D) — tokens redistributed by expert.

    The producer for a target rank writes all full S-blocks, fences, and
    notifies (phase 1).  If there is a tail block, it then writes the tail,
    fences, and notifies again (phase 2).  The consumer waits for phase 1,
    reads all full blocks, waits for phase 2, and reads the tail.  This lets
    full-block consumption overlap tail-block production on non-exact shapes
    while keeping the single-signal fast path for exact shapes.
    """
    S_CHUNK: tl.constexpr = 1

    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    # Even 1:1 split: half producer (even pids), half consumer (odd pids).
    role = pid % 2
    logical_core_id = pid // 2
    n_role_cores = ncore // 2

    num_blocks_s = tl.cdiv(S, COMM_BLOCK_S)
    # Flatten the (target_rank, chunk) work space and distribute it across ALL
    # role cores.  The previous implementation iterated `for rank_loop_id in
    # range(logical_core_id, rank_size, n_role_cores)` which, for small
    # rank_size (e.g. 2), left the vast majority of vector cores idle.  Here
    # every role core owns a disjoint set of (rank, chunk) tiles, so the full
    # 72-core AIV array is utilised and per-chunk signalling pipelines
    # production and consumption.
    num_chunks = tl.cdiv(num_blocks_s, S_CHUNK)

    buffer_chunk_size = (H * rank_size) * D
    stride_ps = H * rank_size * D
    stride_ph = D
    stride_pd = 1

    # One signal per (source_rank, chunk).  The producer for (target, chunk)
    # notifies the target at offset rank*rank_stride + chunk*8; the target's
    # consumer waits at the symmetric offset source_rank*rank_stride + chunk*8.
    signal_slot_stride = SIGNAL_STRIDE
    signal_rank_stride = num_chunks * signal_slot_stride

    offs_d_full = tl.arange(0, COMM_BLOCK_D)

    # ---- Producer (even Vector Cores) --------------------------------
    if role == 0:
        total_work = rank_size * num_chunks
        for wid in range(logical_core_id, total_work, n_role_cores):
            target_rank = wid // num_chunks
            chunk_id = wid % num_chunks
            chunk_start = chunk_id * S_CHUNK

            remote_peer_ptr = dl.symm_at(peer_mem_ptr, target_rank)
            peer_h_write = rank * H

            for h_id in range(0, H):
                for local_s in range(0, S_CHUNK):
                    global_id_s = chunk_start + local_s
                    if global_id_s < num_blocks_s:
                        buffer_id = global_id_s % buffer_num
                        offs_s = global_id_s * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
                        mask = (offs_s < S)[:, None] & (offs_d_full < D)[None, :]

                        a_offs = (
                            offs_s[:, None] * stride_as
                            + (target_rank * H + h_id) * stride_ah
                            + offs_d_full[None, :] * stride_ad
                        )
                        a_data = tl.load(a_ptr + a_offs, mask=mask, other=0.0)

                        peer_offs_write = (
                            buffer_id * (COMM_BLOCK_S * buffer_chunk_size)
                            + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                            + (peer_h_write + h_id) * stride_ph
                            + offs_d_full[None, :] * stride_pd
                        )
                        tl.store(remote_peer_ptr + peer_offs_write, a_data, mask=mask)

            libshmem_device.fence()

            dl.notify(
                signal_mem_ptr
                + rank * signal_rank_stride
                + chunk_id * signal_slot_stride,
                target_rank,
                1,
            )

    # ---- Consumer (odd Vector Cores) ---------------------------------
    if role == 1:
        total_work = rank_size * num_chunks
        for wid in range(logical_core_id, total_work, n_role_cores):
            r = wid // num_chunks
            chunk_id = wid % num_chunks
            chunk_start = chunk_id * S_CHUNK

            token = dl.wait(
                signal_mem_ptr + r * signal_rank_stride + chunk_id * signal_slot_stride,
                1,
                "gpu",
                "acquire",
                waitValue=1,
            )
            local_peer_ptr = dl.consume_token(peer_mem_ptr, token)

            peer_h_read = r * H
            for h_id in range(0, H):
                for local_s in range(0, S_CHUNK):
                    global_id_s = chunk_start + local_s
                    if global_id_s < num_blocks_s:
                        buffer_id = global_id_s % buffer_num
                        offs_s = global_id_s * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
                        mask = (offs_s < S)[:, None] & (offs_d_full < D)[None, :]

                        peer_offs_read = (
                            buffer_id * (COMM_BLOCK_S * buffer_chunk_size)
                            + tl.arange(0, COMM_BLOCK_S)[:, None] * stride_ps
                            + (peer_h_read + h_id) * stride_ph
                            + offs_d_full[None, :] * stride_pd
                        )
                        peer_data = tl.load(
                            local_peer_ptr + peer_offs_read, mask=mask, other=0.0
                        )

                        c_offs = (
                            (r * S + offs_s)[:, None] * stride_cs
                            + h_id * stride_ch
                            + offs_d_full[None, :] * stride_cd
                        )
                        tl.store(c_ptr + c_offs, peer_data, mask=mask)


def hccl_transpose_a2a_kernel_launcher(
    A, C, peer_mem, signal_mem, rank, rank_size, buffer_num, COMM_BLOCK_S, COMM_BLOCK_D
):
    """Launcher for the fused Transpose-All2All kernel.

    A: (S, n_head, D) with n_head = H * rank_size
    C: (S * rank_size, H, D)
    """
    S, n_head, D = A.shape
    H = n_head // rank_size

    # The simplified kernel assumes one D tile covers the whole head dimension.
    assert D == COMM_BLOCK_D, f"D={D} must equal COMM_BLOCK_D={COMM_BLOCK_D}"

    vec_num = driver.active.utils.get_aivector_core_num()

    kernel_hccl_transpose_a2a[vec_num, 1, 1](
        A,
        C,
        peer_mem,
        signal_mem,
        rank,
        rank_size,
        buffer_num,
        S,
        D,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        COMM_BLOCK_S=COMM_BLOCK_S,
        COMM_BLOCK_D=COMM_BLOCK_D,
        H=H,
        SIGNAL_STRIDE=SIGNAL_SLOT_STRIDE,
    )


def run_test_distributed():
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # Initialize ACL symmetric shared memory environment
    init_aclshmem(pe, world_size, ash_pool_bytes(world_size), ash.OpEngineType.MTE)

    # Build test cases: S_RANGES x H_LIST (27 x 7 = 189 cases)
    test_cases = []
    for s_val in S_RANGES:
        for h_val in H_LIST:
            test_cases.append((s_val, h_val))

    total_cases = len(test_cases)
    if pe == 0:
        print(
            f"{BOLD}[START]{RESET} Starting transpose All2All sweep tests "
            f"({total_cases} cases) on world_size={world_size}...",
            flush=True,
        )

    try:
        for case_idx, (S, H) in enumerate(test_cases):
            dist.barrier()
            if pe == 0:
                print(
                    f"\n{BOLD}Test Case {case_idx + 1}/{total_cases}:{RESET} "
                    f"S={S}, H={H}, D={HEAD_DIM}",
                    flush=True,
                )

            dtype = torch.bfloat16
            n_head = H * world_size
            D = HEAD_DIM

            num_blocks_s = (S + COMM_BLOCK_S - 1) // COMM_BLOCK_S
            num_blocks_d = (D + COMM_BLOCK_D - 1) // COMM_BLOCK_D
            buffer_num = num_blocks_s

            # Allocate symmetric memory
            peer_mem_size = COMM_BLOCK_S * (H * world_size) * D * buffer_num
            peer_mem = ash.aclshmem_create_tensor(
                [peer_mem_size],
                dtype=dtype,
                device_id=local_pe,
            )
            signal_mem = ash.aclshmem_create_tensor(
                [buffer_num * world_size * num_blocks_d * H * SIGNAL_SLOT_STRIDE],
                dtype=torch.int32,
                device_id=local_pe,
            )

            # Create input/output tensors with seeded data
            torch.manual_seed(42 + case_idx * world_size + pe)
            A_local = torch.randn([S, n_head, D], dtype=dtype).npu()
            C_local = torch.zeros([S * world_size, H, D], dtype=dtype).npu()

            C_golden = torch_transpose_a2a(A_local, world_size)
            signal_mem.fill_(0)
            dist.barrier()
            hccl_transpose_a2a_kernel_launcher(
                A_local,
                C_local,
                peer_mem,
                signal_mem,
                pe,
                world_size,
                buffer_num,
                COMM_BLOCK_S,
                COMM_BLOCK_D,
            )
            torch.npu.synchronize()

            # Accuracy check
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
                            f"  {GREEN}[PASS]{RESET} Rank {pe}: "
                            f"S={S}, H={H} matches golden.",
                            flush=True,
                        )
                    else:
                        print(
                            f"  {RED}[FAIL]{RESET} Rank {pe}: "
                            f"S={S}, H={H} failed. Details:\n{error_msg}",
                            flush=True,
                        )
                dist.barrier()

            # Release symmetric memory
            ash.aclshmem_free_tensor(peer_mem)
            ash.aclshmem_free_tensor(signal_mem)

            if any(r.item() == 0 for r in all_passed):
                raise AssertionError(f"Parameter sweep failed at case: S={S}, H={H}")
    finally:
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)
    world_size = dist.get_world_size()

    dist.barrier()
    run_test_distributed()
    if local_pe == 0:
        print("have been verified successfully")
