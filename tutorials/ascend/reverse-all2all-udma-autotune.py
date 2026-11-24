# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bare autotune: PR #114 UDMA kernel + triton_dist autotune, nothing else.

No call-count, no reset kernel, no Region A chain, no per-launch host sync
inside the autotuned function. The only signal_mem.zero_() is done ONCE before
the autotune call so the first launch starts clean. Purpose: observe whether
the autotune decorator's internal back-to-back launches (warmup + timing per
config) cause a hang with the base value-0 toggle kernel.
"""

import os
import json

import shmem as ash
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
import triton_dist.tune
from triton_dist.language.extra import libshmem_device
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id, sub_vec_num
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem


def _ash_pool_bytes(required: int) -> int:
    """Symmetric heap: A2A_ASH_SIZE_GB override, else the requirement, page-aligned."""
    override = os.environ.get("A2A_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE
    )
    if pool_bytes > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {pool_bytes / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the sweep or lower "
            "the A2A_ASH_SIZE_GB override"
        )
    return pool_bytes


TUNE_CONFIG_SPACE = [
    {"COMM_BLOCK_S": bs, "COMM_BLOCK_D": bd, "buffer_num": bn}
    for bs in [32, 64, 128, 256, 512]
    for bd in [64, 128]
    for bn in [2, 3, 4, 8]
]
TUNE_SPACE_VERSION = "reverse-a2a-udma-autotune-v1"


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

    stride_ps = H * D
    stride_ph = D

    SIG_STRIDE = 8
    region_b_base = num_blocks_s * rank_size * SIG_STRIDE

    for global_id_s in range(0, num_blocks_s):
        buffer_id = global_id_s % buffer_num
        cur_block_s = tl.minimum(COMM_BLOCK_S, S - global_id_s * COMM_BLOCK_S)

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

        if global_vec_id % 2 == 0 and logical_core_id >= UDMA_CORES:
            logical_mte_pid = logical_core_id - UDMA_CORES
            logical_mte_ncores = n_role_cores - UDMA_CORES
            for task_idx in range(logical_mte_pid, tasks_per_block, logical_mte_ncores):
                h_id = task_idx % H
                block_id_d = task_idx // H

                offs_s = global_id_s * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
                offs_d = block_id_d * COMM_BLOCK_D + tl.arange(0, COMM_BLOCK_D)
                mask = (offs_s < S)[:, None] & (offs_d < D)[None, :]

                a_s = rank * S + offs_s
                a_offs = (
                    a_s[:, None] * stride_as
                    + h_id * stride_ah
                    + offs_d[None, :] * stride_ad
                )
                a_data = tl.load(a_ptr + a_offs, mask=mask, other=0.0)

                self_sig_ptr = (
                    signal_mem_ptr
                    + region_b_base
                    + (buffer_id * tasks_per_block + task_idx) * SIG_STRIDE
                )
                token = dl.wait(self_sig_ptr, 1, "gpu", "acquire", waitValue=0)
                store_ptr = dl.consume_token(peer_mem_ptr, token)

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

        if global_vec_id % 2 == 1:
            total_cons_tasks = tasks_per_block * rank_size
            for task_idx in range(logical_core_id, total_cons_tasks, n_role_cores):
                tmp = task_idx
                r = tmp % rank_size
                tmp //= rank_size
                h_id = tmp % H
                block_id_d = tmp // H

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


def _autotune_key(A, C, peer_mem, signal_mem, rank, rank_size):
    return (TUNE_SPACE_VERSION, tuple(A.shape), tuple(C.shape), rank_size)


def _prune(config_record, A, *args, **kwargs):
    import math

    rank_size = args[4]
    S = A.shape[0] // rank_size
    D = A.shape[2]
    bs = config_record["COMM_BLOCK_S"]
    bd = config_record["COMM_BLOCK_D"]
    bn = config_record["buffer_num"]
    num_blocks_s = math.ceil(S / bs)
    if bn > num_blocks_s:
        return False
    if bs == 128 and bd == 128:
        return True
    if bs > S or bd > D:
        return False
    if S / (num_blocks_s * bs) < 0.75:
        return False
    if D / (math.ceil(D / bd) * bd) < 0.75:
        return False
    if bs * bd * A.element_size() > 128 * 1024:
        return False
    return True


@triton_dist.tune.autotune(
    config_space=TUNE_CONFIG_SPACE,
    key_fn=_autotune_key,
    prune_fn=_prune,
)
def _function_autotuned_hccl_reverse_a2a_udma_bare(
    A,
    C,
    peer_mem,
    signal_mem,
    rank,
    rank_size,
    buffer_num=2,
    COMM_BLOCK_S=128,
    COMM_BLOCK_D=128,
):
    S_total, H, D = A.shape
    S = S_total // rank_size
    vec_num = driver.active.utils.get_aivector_core_num()
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
    role_cores = vec_num // 2
    if not 0 < udma_sender_cores < role_cores:
        raise ValueError(
            f"A2A_UDMA_SENDER_CORES={udma_sender_cores} must be in "
            f"[1, {role_cores - 1}] for aivectorcore_num={vec_num}"
        )
    signal_mem.fill_(0)
    dist.barrier()
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
        UDMA_CORES=udma_sender_cores,
    )


def torch_reverse_a2a(tensor, dsp_size, group=None):
    send = torch.cat(torch.chunk(tensor, dsp_size, dim=0), dim=1).contiguous()
    send = send.transpose(1, 0).contiguous()
    recv = torch.empty(send.shape, dtype=send.dtype, device=send.device)
    dist.all_to_all_single(recv, send, group=group)
    recv = recv.transpose(0, 1).contiguous()
    return recv


def _init_shmem(rank, rank_size, local_mem_size):
    init_aclshmem(rank, rank_size, local_mem_size, ash.OpEngineType.UDMA)


def _signal_mem_size(
    S, H, D, rank_size, max_buffer_num, max_num_blocks_s, max_num_blocks_d
):
    sig_stride = 8
    region_a = max_num_blocks_s * rank_size * sig_stride
    region_b = max_buffer_num * max_num_blocks_d * H * sig_stride
    return region_a + region_b


def run_test_distributed():
    pe = dist.get_rank()
    local_pe = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    process_group = dist.new_group(ranks=list(range(world_size)))
    S_ranges = (
        list(range(2040, 2049))
        + list(range(4088, 4097))
        + list(range(8184, 8193))
        + list(range(32760, 32769))
        + list(range(131064, 131073))
    )
    H_list = [3, 6, 8, 12, 16, 24, 28, 32, 40, 48, 56]
    D = 128
    test_cases = [(S, H_local) for S in S_ranges for H_local in H_list]
    warmup = int(os.environ.get("ASCEND_PROFILE_WARMUP", "0"))
    iters = int(os.environ.get("ASCEND_PROFILE_ITERS", "1"))
    total_loops = warmup + iters
    manifest_path = os.environ.get(
        "REVERSE_A2A_MANIFEST",
        "reverse_a2a_udma_autotune_manifest.json",
    )
    manifest = []

    dtype = torch.bfloat16
    min_bs = min(config["COMM_BLOCK_S"] for config in TUNE_CONFIG_SPACE)
    max_bn = max(config["buffer_num"] for config in TUNE_CONFIG_SPACE)
    max_num_blocks_d = max(
        triton.cdiv(D, config["COMM_BLOCK_D"]) for config in TUNE_CONFIG_SPACE
    )
    # The peer staging mirrors every rank's S x H x D bf16 tile and the int64
    # signal arena one slot row per (S block, rank) plus the buffered D blocks
    # per head; size the heap from both, page-aligned.
    s_max, h_max = max(S_ranges), max(H_list)
    peer_bytes = world_size * s_max * h_max * D * 2
    signal_bytes = (
        (triton.cdiv(s_max, min_bs) * world_size + max_bn * max_num_blocks_d * h_max)
        * 8
        * 8
    )
    pool_bytes = ash_pool_bytes(peer_bytes + signal_bytes)
    _init_shmem(pe, world_size, pool_bytes)

    try:
        for case_idx, (S, H_local) in enumerate(test_cases):
            H_global = H_local * world_size
            S_total = S * world_size
            peer_mem_size = world_size * S * H_local * D
            peer_mem = ash.aclshmem_create_tensor(
                [peer_mem_size], dtype=dtype, device_id=local_pe
            )
            max_num_blocks_s = triton.cdiv(S, min_bs)
            signal_mem_size = _signal_mem_size(
                S, H_local, D, world_size, max_bn, max_num_blocks_s, max_num_blocks_d
            )
            signal_mem = ash.aclshmem_create_tensor(
                [signal_mem_size], dtype=torch.int64, device_id=local_pe
            )
            signal_mem.zero_()
            torch.npu.synchronize()
            dist.barrier()

            A_local = torch.ones([S_total, H_local, D], dtype=dtype).npu()
            C_local = torch.zeros([S, H_global, D], dtype=dtype).npu()

            for _ in range(total_loops):
                C_local.zero_()
                dist.barrier()
                golden = torch_reverse_a2a(A_local, world_size)
                dist.barrier()

            for iter_idx in range(total_loops):
                C_local.zero_()
                dist.barrier()
                _function_autotuned_hccl_reverse_a2a_udma_bare(
                    A_local,
                    C_local,
                    peer_mem,
                    signal_mem,
                    pe,
                    world_size,
                    autotune=True,
                    autotune_pg=process_group,
                )
                dist.barrier()
                if iter_idx == total_loops - 1:
                    torch.testing.assert_close(
                        C_local,
                        golden,
                        rtol=1e-3,
                        atol=1e-3,
                    )

            key = _autotune_key(A_local, C_local, peer_mem, signal_mem, pe, world_size)
            best = _function_autotuned_hccl_reverse_a2a_udma_bare.best_configs.get(key)
            if pe == 0:
                record = {
                    "S": S,
                    "H_local": H_local,
                    "D": D,
                    "COMM_BLOCK_S": best["COMM_BLOCK_S"],
                    "COMM_BLOCK_D": best["COMM_BLOCK_D"],
                    "buffer_num": best["buffer_num"],
                }
                manifest.append(record)
                print(
                    f"[{case_idx + 1}/{len(test_cases)}] "
                    f"S={S} H={H_local} {record}",
                    flush=True,
                )

            ash.aclshmem_free_tensor(peer_mem)
            ash.aclshmem_free_tensor(signal_mem)

        if pe == 0 and manifest_path:
            with open(manifest_path, "w") as file:
                json.dump({"configs": manifest}, file, indent=2)
            print(f"[MANIFEST] wrote {manifest_path}")
    finally:
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl")
    from triton_dist import utils as _td_utils

    _td_utils._TRITON_DIST_LOCAL_WORLD_SIZE = dist.get_world_size()
    run_test_distributed()
    dist.destroy_process_group()
