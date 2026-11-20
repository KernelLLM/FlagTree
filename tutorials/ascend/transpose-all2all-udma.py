# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import torch
import torch_npu
import shmem as ash
import torch.distributed as dist
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton.runtime import driver
from triton.language.extra.cann.extension import sync_block_all
from utils import ACLSHMEM_MAX_LOCAL_SIZE, SHMEM_PAGE_SIZE, init_aclshmem

# Symmetric memory pool (per rank).  The only symmetric allocation is the
# output C, which scales with world_size * S * H * D, so the pool is
# computed from the largest swept case; A2A_ASH_SIZE_GB overrides it.


def symmetric_memory_size(world_size: int, s_values, h_values) -> int:
    """Return the aligned ACLSHMEM heap capacity the swept cases need."""
    required = world_size * max(s_values) * max(h_values) * HEAD_DIM * 2
    return -(-required // SHMEM_PAGE_SIZE) * SHMEM_PAGE_SIZE


def ash_pool_bytes(world_size: int, s_values, h_values) -> int:
    """Symmetric heap: A2A_ASH_SIZE_GB override, else the sweep requirement."""
    override = os.environ.get("A2A_ASH_SIZE_GB")
    pool_bytes = (
        int(override) * 1024 * 1024 * 1024
        if override is not None
        else symmetric_memory_size(world_size, s_values, h_values)
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

HEAD_DIM = 128
COMM_BLOCK_D = 128

# Case set: S/H are unioned and fully cross-producted (36 S values x 10 H
# values = 360 cases, so the matrix has no holes).
TEST_S = list(range(2040, 2049)) + list(range(4088, 4097)) + list(range(8184, 8193))
TEST_H = [1, 2, 3, 4, 5, 6, 7]
PERF_S = (
    list(range(8184, 8193)) + list(range(32760, 32769)) + list(range(131064, 131073))
)
PERF_H = [3, 6, 8, 12, 16]
SWEEP_S = sorted(set(TEST_S) | set(PERF_S))
SWEEP_H = sorted(set(TEST_H) | set(PERF_H))
SWEEP_CASES = [(s, h) for s in SWEEP_S for h in SWEEP_H]

# ---- Tiered scheduling knobs (cannbot #12/#14: parameters are picked per
# (S,H) region, can be overridden by env, and were frozen after measurement) ----
A2A_UDMA_BLOCK_S = int(os.environ.get("A2A_UDMA_BLOCK_S", "0"))  # 0 = auto(128)
A2A_UDMA_HGROUP = int(os.environ.get("A2A_UDMA_HGROUP", "0"))  # 0 = auto(1)
A2A_UDMA_PUT_CHUNK = int(
    os.environ.get("A2A_UDMA_PUT_CHUNK", "0")
)  # 0 = auto; elements/put


def torch_transpose_a2a(tensor, dsp_size, group=None):
    """Forward All2All dispatch. tensor: (S, n_head, head_dim) with n_head = H * dsp_size.
    Returns recv: (S * dsp_size, H, head_dim)."""
    transposed = torch.cat(torch.chunk(tensor, dsp_size, dim=1), dim=0).contiguous()
    recv = torch.empty_like(transposed)
    dist.all_to_all_single(recv, transposed, group=group)
    return recv


@triton.jit
def kernel_hccl_transpose_a2a_put(
    a_ptr,
    c_ptr,
    staging_ptr,
    rank,
    rank_size,
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
    HEADS_PER_GROUP: tl.constexpr,
    PUT_CHUNK_ELEMS: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
):
    """Put-based fused Transpose-All2All (v3).

    Proven-primitive design for this stack: putmem push + fence +
    intra-launch block sync + quiet.  (getmem-pull and putmem_signal
    variants both corrupt data on this machine — only the
    put+sync+local-consume skeleton of the original 0x kernel passed
    360/360.)

    The OUTPUT C lives in symmetric memory, so putmem writes the remote
    PE's C directly — the entire consumer pass of the original kernel is
    gone, along with its second barrier_all and the peer_mem receive
    buffer.

    Layouts:
        C (symmetric): (rank_size*S, H, D); rows [r*S, (r+1)*S) = data from
            source rank r.
        staging (regular GM): [rank_size, S, H, D]; slot t = this rank's
            data for target t (A[:, t*H:(t+1)*H, :] gathered contiguous —
            putmem needs one contiguous payload per target).

    Phases:
        1:  all lanes pack one unified task space: remote-target slices
            A -> staging, the self slice A -> C[self rows] direct copy,
            interleaved so no lane idles on self only.
        2:  fence + sync_block_all — staging writes of all local cores
            visible before any putmem reads them.  staging is regular
            GM, so no cross-PE barrier is needed.
        3:  even lanes (one physical core per target stripe) putmem
            staging[t] -> target t's C[my rows]; quiet() then blocks until
            every put has landed, so kernel completion implies C is fully
            written on every PE.  No trailing barrier, single staging
            buffer: the next launch runs on the same stream and cannot
            start before this kernel (and therefore quiet) completes.
    """
    vec_num_per_aicore = 2
    physical_ncores = tl.num_programs(axis=0) // vec_num_per_aicore
    physical_pid = tl.program_id(axis=0) // vec_num_per_aicore
    global_vec_id = tl.program_id(axis=0)
    vector_ncores = tl.num_programs(axis=0)
    sub_id = global_vec_id % vec_num_per_aicore

    num_blocks = tl.cdiv(S, COMM_BLOCK_S)
    head_groups = tl.cdiv(H, HEADS_PER_GROUP)
    S = S.to(tl.int64)
    D = D.to(tl.int64)
    row_stride = H * D
    slot_elements = S * H * D
    num_put_chunks = tl.cdiv(slot_elements, PUT_CHUNK_ELEMS)

    # ---- Phase 1: pack all targets, striped over ALL vector lanes ----
    # Remote-target slices go to staging slots (contiguous payload for
    # putmem); the self slice goes straight into C[rank rows] — a local
    # copy that needs no fence, no barrier, no UDMA.  Using the full
    # lane space balances pack and self in one task space instead of
    # idling half the lanes on self only.
    pack_tasks = rank_size * num_blocks * head_groups
    for task_idx in range(global_vec_id, pack_tasks, vector_ncores):
        head_group = task_idx % head_groups
        rank_block_idx = task_idx // head_groups
        block_id = rank_block_idx % num_blocks
        target_rank = rank_block_idx // num_blocks

        offs_s = block_id * COMM_BLOCK_S + tl.arange(0, COMM_BLOCK_S)
        if HEADS_PER_GROUP == 1:
            offs_d = tl.arange(0, COMM_BLOCK_D)
            local_head = head_group
            mask = (offs_s < S)[:, None]
        else:
            offs_hd = tl.arange(0, HEADS_PER_GROUP * COMM_BLOCK_D)
            head_in_group = offs_hd // COMM_BLOCK_D
            offs_d = offs_hd % COMM_BLOCK_D
            local_head = head_group * HEADS_PER_GROUP + head_in_group
            mask = (offs_s < S)[:, None] & (local_head < H)[None, :]

        global_head = target_rank * H + local_head
        input_offs = (
            offs_s[:, None] * stride_as
            + global_head * stride_ah
            + offs_d[None, :] * stride_ad
        )
        a_data = tl.load(a_ptr + input_offs, mask=mask, other=0.0)
        if target_rank == rank:
            # self slice: A -> C[rank rows], direct local store
            if HEADS_PER_GROUP == 1:
                c_head_offset = local_head * stride_ch
            else:
                c_head_offset = local_head * stride_ch
            c_offs = (
                (rank * S + offs_s)[:, None] * stride_cs
                + c_head_offset
                + offs_d[None, :] * stride_cd
            )
            tl.store(c_ptr + c_offs, a_data, mask=mask)
        else:
            if HEADS_PER_GROUP == 1:
                stage_head_offset = local_head * D
            else:
                stage_head_offset = local_head * D
            stage_base = (
                target_rank * slot_elements
                + offs_s[:, None] * row_stride
                + stage_head_offset
                + offs_d[None, :]
            )
            tl.store(staging_ptr + stage_base, a_data, mask=mask)

    if rank_size > 1:
        libshmem_device.fence()
        sync_block_all("all_vector", 10)

        # ---- Phase 3: push staging slots into each target's C ----------
        # Owner-per-target (single issuing core per target — concurrent
        # putmem to one PE from many cores corrupts on this stack), but
        # the slot goes out as PUT_CHUNK_ELEMS-sized contiguous chunks.
        if sub_id == 0 and physical_pid < rank_size:
            for target_rank in range(physical_pid, rank_size, physical_ncores):
                if target_rank != rank:
                    for chunk_id in range(num_put_chunks):
                        chunk_start = chunk_id * PUT_CHUNK_ELEMS
                        if chunk_start < slot_elements:
                            chunk_elems = tl.minimum(
                                slot_elements - chunk_start, PUT_CHUNK_ELEMS
                            )
                            libshmem_device.putmem(
                                c_ptr + rank * slot_elements + chunk_start,
                                staging_ptr + target_rank * slot_elements + chunk_start,
                                chunk_elems * ELEMENT_BYTES,
                                target_rank,
                            )


def _select_block_s(sequence_length, heads_per_rank):
    """Tiered COMM_BLOCK_S (cannbot #14), measured on the 4-card sweep:
    256 wins for bandwidth-bound large S, 128 keeps parallelism at 2K/4K."""
    if A2A_UDMA_BLOCK_S > 0:
        return A2A_UDMA_BLOCK_S
    # H=1 keeps 128: bigger blocks halve the pack task count and hurt.
    if sequence_length >= 8184 and heads_per_rank >= 2:
        return 256
    return 128


def _select_heads_per_group():
    """Contiguous-head packing width (cannbot #14): every measured tier
    picked 1, so the default is 1 and only A2A_UDMA_HGROUP overrides."""
    if A2A_UDMA_HGROUP > 0:
        return A2A_UDMA_HGROUP
    return 1


def _select_put_chunk(slot_elements, element_bytes):
    """Tiered putmem chunk size in elements (cannbot #3/#16).

    Monolithic put per target measured fastest up to 256MiB slots
    (S=131072 H=8 passes).  At ~512MiB (S=131072 H=16) a single putmem
    corrupts data and hits a ~20x slow path — rank-3's slot then ends
    exactly at 2GiB, pointing at an int32 offset/size limit in the UDMA
    engine.  Chunk such slots into 2MiB puts (proven correct + fast).
    """
    if A2A_UDMA_PUT_CHUNK > 0:
        return min(slot_elements, A2A_UDMA_PUT_CHUNK)
    slot_bytes = slot_elements * element_bytes
    if slot_bytes > 256 * 1024 * 1024:
        return min(slot_elements, (2 * 1024 * 1024) // element_bytes)
    return slot_elements  # monolithic put per target measured fastest


def hccl_transpose_a2a_kernel_launcher(A, C, staging, rank, rank_size):
    """Launcher for the put Transpose-All2All kernel (v3).

    A: (S, n_head, D) with n_head = H * rank_size
    C: SYMMETRIC output tensor (S * rank_size, H, D) — putmem writes the
       remote PEs' C directly, so C must live in the symmetric pool.
    staging: regular local gather buffer [rank_size, S, H, D]
    """
    S, n_head, D = A.shape
    H = n_head // rank_size
    assert D == COMM_BLOCK_D, f"D={D} must equal COMM_BLOCK_D={COMM_BLOCK_D}"

    comm_block_s = _select_block_s(S, H)
    heads_per_group = _select_heads_per_group()
    put_chunk_elems = _select_put_chunk(S * H * D, A.element_size())

    # Launch on the vector-core count the driver reports for this hardware.
    aivector_cores = driver.active.utils.get_aivector_core_num()

    kernel_hccl_transpose_a2a_put[aivector_cores, 1, 1](
        A,
        C,
        staging,
        rank,
        rank_size,
        S,
        D,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        COMM_BLOCK_S=comm_block_s,
        COMM_BLOCK_D=COMM_BLOCK_D,
        H=H,
        HEADS_PER_GROUP=heads_per_group,
        PUT_CHUNK_ELEMS=put_chunk_elems,
        ELEMENT_BYTES=A.element_size(),
    )
    return aivector_cores


def make_symmetric_output(world_size, S, H, D, dtype):
    """Allocate the symmetric output C: (S * rank_size, H, D)."""
    local_pe = int(os.environ["LOCAL_RANK"])
    try:
        C = ash.aclshmem_create_tensor(
            [world_size * S * H * D], dtype=dtype, device_id=local_pe
        ).view(world_size * S, H, D)
        return C
    except Exception:
        return None


def _parse_int_list(name, default_values):
    raw = os.environ.get(name)
    if raw is None:
        return list(default_values)
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def run_test_distributed():
    pe = dist.get_rank()
    world_size = dist.get_world_size()

    # Optional subset filter for quick validation / tuning sweeps.
    bench_s = _parse_int_list("A2A_BENCH_S", SWEEP_S)
    bench_h = _parse_int_list("A2A_BENCH_H", SWEEP_H)
    sweep_cases = [(s, h) for s in bench_s for h in bench_h]
    pool_bytes = ash_pool_bytes(world_size, bench_s, bench_h)

    # Initialize ACL symmetric shared memory environment with UDMA engine
    init_aclshmem(pe, world_size, pool_bytes, ash.OpEngineType.UDMA)

    total_cases = len(sweep_cases)
    if pe == 0:
        print(
            f"\n{BOLD}[START]{RESET} put transpose All2All sweep "
            f"({total_cases} cases) on world_size={world_size}...",
            flush=True,
        )
        print(
            f"CONFIG pool={pool_bytes >> 20}MB",
            flush=True,
        )

    try:
        for case_idx, (S, H) in enumerate(sweep_cases):
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

            # Symmetric output C (putmem dest) + local staging buffer.
            # Any rank failing to allocate -> all ranks skip this case.
            C_local = make_symmetric_output(world_size, S, H, D, dtype)
            can_alloc = torch.tensor([1], dtype=torch.int32).npu()
            if C_local is None:
                can_alloc[0] = 0
            dist.all_reduce(can_alloc, op=dist.ReduceOp.MIN)
            if can_alloc.item() == 0:
                if C_local is not None:
                    ash.aclshmem_free_tensor(C_local)
                if pe == 0:
                    print(f"  case S={S} H={H} SKIP (OOM)", flush=True)
                continue
            staging = torch.zeros([world_size * S * H * D], dtype=dtype).npu()

            # Create input tensor with seeded data
            torch.manual_seed(42 + case_idx * world_size + pe)
            A_local = torch.randn([S, n_head, D], dtype=dtype).npu()

            # Golden reference.
            C_golden = torch_transpose_a2a(A_local, world_size)

            # UDMA kernel.  quiet() inside the kernel means kernel completion
            # implies every put landed, so back-to-back launches are safe on
            # the same stream.
            hccl_transpose_a2a_kernel_launcher(
                A_local, C_local, staging, pe, world_size
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
                if os.environ.get("A2A_DEBUG"):
                    diff = (C_golden - C_local).abs() > 1e-2
                    for r in range(world_size):
                        blk = diff[r * S : (r + 1) * S]
                        n_bad = int(blk.sum().item())
                        if n_bad:
                            rows = blk.any(dim=2).any(dim=1).nonzero().flatten()
                            print(
                                f"  [dbg] rank {pe} source {r}: {n_bad} bad elems, "
                                f"bad rows {rows[:8].tolist()}"
                                f"{'...' if len(rows) > 8 else ''} of {S}",
                                flush=True,
                            )

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
            ash.aclshmem_free_tensor(C_local)
            del staging, A_local, C_golden

            if any(r.item() == 0 for r in all_passed):
                raise AssertionError(f"Parameter sweep failed at case: S={S}, H={H}")
    finally:
        # Skip explicit aclshmem_finalize — it SIGABRTs on this environment
        # after a clean run; the OS reclaims resources at process exit.
        dist.destroy_process_group()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)
    world_size = dist.get_world_size()

    dist.barrier()
    run_test_distributed()
    if local_pe == 0:
        print(f"\n{GREEN}[SUCCESS]{RESET} put transpose All2All sweep verified.")
