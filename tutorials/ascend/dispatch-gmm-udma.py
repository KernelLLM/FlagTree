# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ascend 950 UDMA Dispatch + grouped GMM, forward only.

Run one random-routing correctness example:
    torchrun --standalone --nproc_per_node=8 tutorials/ascend/dispatch-gmm-udma.py

The public inputs are hidden_states [M,K] BF16, scatter_index [M,top_k] INT32,
splits [R*E] INT32/INT64, weights [E,N,K] BF16, and a routed interval [start,end).
scatter_index is the stable global-expert-major permutation of token branches;
splits describes the expert buckets in that interval. All ranks use the same
shape, interval length and invocation order. Routing is supplied, not computed.

For source p and local expert e on receiver q, metadata derives count C[p,q,e].
Received expert e consists of source 0's rows, then source 1's rows, etc.
Output_q[e, row, n] = sum_k Received_q[e, row, k] * Weight_q[e, n, k].
The returned dictionary also preserves the packed, source-major and
expert-major inputs and host count lists of the Torch/HCCL reference.

There are TWO launches: inverse routing/counts/offsets metadata, then a mixed
Vector/Cube fused body. Four-rank wave sizes follow the same shape-only
boundaries as the direct multi-kernel method. Vector subcore 0 packs and
self-copies, and one sender per peer issues blocking UDMA transfers followed by
expert-ready epoch signals. One worker per rank on Vector subcore 1 joins
local pack completion and nonempty remote bucket signals. Their per-wave
completion counter independently releases Cube GMM and public-layout restore.
The fused path has no per-wave global barrier: neither next-wave pack nor
GMM waits for previous-wave restoration. Both consumers read the expert-major
receive buffer; restore also
writes the source-major and expert-major public layouts.
Counts are copied to the host on a separate stream after metadata, overlapping
the fused body. All input-derived work remains inside dispatch_gmm().

Initialize ACL SHMEM with the UDMA engine before constructing a workspace.
Only capacity buffers, streams, events and the JIT compilation cache are reusable.
Outputs borrow workspace storage and must be consumed before its next call.
Calls are collective over the torchrun world and must not overlap. The rank
count must be positive and smaller than the Cube core count; each Cube must
have two Vector subcores. Transport/topology and memory limits still apply.
"""

import math
import os

import shmem as ash
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton_dist.language as dl
import triton.language.extra.cann.extension as al
from triton.runtime import driver
from triton_dist.language.extra import libshmem_device as sh
from utils import ACLSHMEM_MAX_LOCAL_SIZE, init_aclshmem

# JIT globals must be explicit constexpr objects (including Triton 3.6).
# Host allocation uses .value; signal slots contain INT32 words, payloads BF16.
BF16_ELEMENT_BYTES = tl.constexpr(2)
COUNT_ELEMENT_BYTES = tl.constexpr(4)
SIGNAL_SLOT_BYTES = 64
SIGNAL_SLOT_ELEMENTS = tl.constexpr(SIGNAL_SLOT_BYTES // COUNT_ELEMENT_BYTES.value)
INVERSE_BLOCK_SIZE = tl.constexpr(1024)
# Pack gathers token rows; restore copies contiguous receive rows.
# The restore tile is 64*256*2 = 32 KiB before compiler buffering.
PACK_BLOCK_ROWS = tl.constexpr(16)
RESTORE_BLOCK_ROWS = tl.constexpr(64)
RESTORE_BLOCK_COLS = tl.constexpr(256)
MAX_PACK_COLS = 2048
UDMA_MAX_PUT_BYTES = tl.constexpr(256 * 1024 * 1024)
SIGNED_INT32_MAX = tl.constexpr(2**31 - 1)

# Cube tiles are separate from Vector pack/restore tiles.
GEMM_BLOCK_M = 128
GEMM_BLOCK_N = 256
GEMM_BLOCK_K = 256


def fused_wave_experts(ranks, experts, packed_bytes=None):
    """Choose experts per wave from shape alone; never inspect routing values.

    Measured four-rank rules balance per-wave synchronization against packing
    and GMM overlap. Smaller payloads tolerate wider waves; large payloads use
    narrower waves to start consumers earlier. These are tuning boundaries,
    not divisibility requirements or guarantees for unmeasured shapes.
    Other world sizes retain the conservative two-expert wave.
    """
    small_payload_bytes = 128 * 1024 * 1024
    medium_payload_bytes = 256 * 1024 * 1024
    if packed_bytes is None:
        packed_bytes = medium_payload_bytes + 1
    if ranks != 4:
        return min(2, experts)
    if packed_bytes <= small_payload_bytes:
        wave = 8 if experts <= 31 else (20 if experts <= 80 else 40)
    elif packed_bytes <= medium_payload_bytes and experts <= 8:
        wave = 4
    else:
        wave = 2 if experts <= 16 else (8 if experts <= 40 else 20)
    return min(wave, experts)


@triton.jit
def kernel_dispatch_gmm_metadata(
    routes,
    splits,
    inverse,
    packed,
    count_matrix,
    exchange,
    counts,
    expert_sizes,
    send_offsets,
    destinations,
    source_offsets,
    expert_offsets,
    tile_offsets,
    epoch_state,
    host_tables,
    public_counts,
    public_experts,
    R: tl.constexpr,
    E: tl.constexpr,
    ROUTES: tl.constexpr,
    TOPK: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CORES: tl.constexpr,
    WAVE: tl.constexpr,
):
    """Invert routing and exchange counts; derive final expert-major receive offsets."""
    pid = tl.program_id(0)
    TOTAL: tl.constexpr = R * E
    EB: tl.constexpr = triton.next_power_of_2(TOTAL)
    RB: tl.constexpr = triton.next_power_of_2(R)
    if pid < tl.cdiv(ROUTES, INVERSE_BLOCK_SIZE):
        # A route position owns exactly one inverse entry (token index).
        branch = pid * INVERSE_BLOCK_SIZE + tl.arange(0, INVERSE_BLOCK_SIZE)
        position = tl.load(routes + branch, branch < ROUTES, -1)
        valid = (branch < ROUTES) & (position >= START) & (position < END)
        tl.store(inverse + position - START, branch // TOPK, valid)
    else:
        for wave in tl.static_range((E + WAVE - 1) // WAVE):
            tl.store(
                packed + (tl.cdiv(E, WAVE) * (CORES - R) + wave) * SIGNAL_SLOT_ELEMENTS,
                0,
            )
        layout_epoch = tl.load(epoch_state) + 1
        tl.store(epoch_state, layout_epoch)
        layout_experts = tl.arange(0, EB)
        layout_valid = layout_experts < TOTAL
        layout_counts = tl.load(splits + layout_experts, mask=layout_valid, other=0)
        layout_starts = tl.cumsum(layout_counts, axis=0) - layout_counts
        tl.store(send_offsets + layout_experts, layout_starts, mask=layout_valid)
        tl.store(send_offsets + TOTAL, tl.sum(layout_counts, axis=0))
        layout_own_row = count_matrix + RANK * TOTAL
        tl.store(layout_own_row + layout_experts, layout_counts, mask=layout_valid)
        # Complete local count-row stores before UDMA reads that GM source.
        # Epoch signals announce completion of count_matrix rows, not payloads
        # at the signal address; subsequent loads consume those peer rows.
        sh.quiet_lowlevel(transport=sh.ACLSHMEM_TRANSPORT_MTE)
        for layout_peer in tl.static_range(R):
            if layout_peer != RANK:
                sh.putmem(
                    layout_own_row,
                    layout_own_row,
                    TOTAL * COUNT_ELEMENT_BYTES,
                    layout_peer,
                )
                # Blocking putmem completes the row before publishing its epoch.
                sh.signal_op(
                    exchange + RANK * SIGNAL_SLOT_ELEMENTS,
                    layout_epoch,
                    sh.ACLSHMEM_SIGNAL_SET,
                    layout_peer,
                )
        for layout_source in tl.static_range(R):
            if layout_source != RANK:
                sh.signal_wait_until(
                    exchange + layout_source * SIGNAL_SLOT_ELEMENTS,
                    sh.ACLSHMEM_CMP_GE,
                    layout_epoch,
                )
        layout_sources = tl.arange(0, RB)
        layout_source_valid = layout_sources < R
        layout_local_experts = tl.arange(0, EB)
        layout_local_valid = layout_local_experts < E
        layout_local_first = RANK * E
        layout_receive = tl.load(
            count_matrix
            + layout_sources[:, None] * TOTAL
            + (layout_local_first + layout_local_experts)[None, :],
            mask=layout_source_valid[:, None] & layout_local_valid[None, :],
            other=0,
        )
        tl.store(
            counts + layout_sources[:, None] * E + layout_local_experts[None, :],
            layout_receive,
            mask=layout_source_valid[:, None] & layout_local_valid[None, :],
        )
        layout_source_rows = tl.sum(layout_receive, axis=1)
        layout_source_prefix = (
            tl.cumsum(layout_source_rows, axis=0) - layout_source_rows
        )
        tl.store(
            source_offsets + layout_sources,
            layout_source_prefix,
            mask=layout_source_valid,
        )
        tl.store(source_offsets + R, tl.sum(layout_source_rows, axis=0))
        layout_expert_sizes = tl.sum(layout_receive, axis=0)
        layout_expert_prefix = (
            tl.cumsum(layout_expert_sizes, axis=0) - layout_expert_sizes
        )
        tl.store(
            expert_offsets + layout_local_experts,
            layout_expert_prefix,
            mask=layout_local_valid,
        )
        tl.store(expert_offsets + E, tl.sum(layout_expert_sizes, axis=0))
        tl.store(
            expert_sizes + layout_local_experts,
            layout_expert_sizes,
            mask=layout_local_valid,
        )
        layout_tiles = tl.cdiv(layout_expert_sizes, BLOCK_M)
        layout_tile_prefix = tl.cumsum(layout_tiles, axis=0) - layout_tiles
        tl.store(
            tile_offsets + layout_local_experts,
            layout_tile_prefix,
            mask=layout_local_valid,
        )
        tl.store(tile_offsets + E, tl.sum(layout_tiles, axis=0))
        sources = tl.arange(0, RB)
        experts = tl.arange(0, EB)
        for owner in tl.static_range(R):
            receive = tl.load(
                count_matrix + sources[:, None] * TOTAL + owner * E + experts[None, :],
                (sources[:, None] < R) & (experts[None, :] < E),
                0,
            )
            sizes = tl.sum(receive, axis=0)
            prefix = tl.cumsum(sizes, axis=0) - sizes
            preceding = tl.sum(tl.where(sources[:, None] < RANK, receive, 0), axis=0)
            tl.store(
                destinations + owner * E + experts, prefix + preceding, experts < E
            )
        local = tl.load(splits + experts, experts < TOTAL, 0)
        incoming = tl.load(counts + experts, experts < TOTAL, 0)
        tl.store(host_tables + experts, local, experts < TOTAL)
        tl.store(host_tables + TOTAL + experts, incoming, experts < TOTAL)
        tl.store(public_counts + experts, incoming, experts < TOTAL)
        sizes = tl.load(expert_sizes + experts, experts < E, 0)
        tl.store(public_experts + experts, sizes, experts < E)


@triton.jit
def kernel_dispatch_gmm_fused(
    hidden,
    weights,
    scattered,
    received,
    source_out,
    expert_out,
    output,
    inverse,
    packed,
    signals,
    counts,
    send_offsets,
    destinations,
    source_offsets,
    expert_offsets,
    tile_offsets,
    epoch_state,
    R: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    ROUTES: tl.constexpr,
    RANK: tl.constexpr,
    CORES: tl.constexpr,
    WAVE: tl.constexpr,
    PACK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pack/send expert waves; restore public layouts concurrently with Cube GMM."""
    pid = tl.program_id(0)
    PRODUCERS: tl.constexpr = CORES - R
    SLOT: tl.constexpr = SIGNAL_SLOT_ELEMENTS
    WIDE: tl.constexpr = (
        R * ROUTES * K > SIGNED_INT32_MAX or R * ROUTES * N > SIGNED_INT32_MAX
    )
    WEIGHTS_WIDE: tl.constexpr = E * N * K > SIGNED_INT32_MAX
    sub = al.sub_vec_id()
    with al.scope(core_mode="vector"):
        epoch = tl.load(epoch_state)
    for wave in tl.static_range((E + WAVE - 1) // WAVE):
        with al.scope(core_mode="vector"):
            if sub == 0:
                if pid < PRODUCERS:
                    size = tl.minimum(WAVE, E - wave * WAVE)
                    groups = R * size
                    # Divide the physical producer budget among this wave's
                    # buckets; keep at least one logical row shard per bucket.
                    pack_workers = tl.maximum(1, PRODUCERS // groups)
                    for group in range(pid, groups * pack_workers, PRODUCERS):
                        bucket = group // pack_workers
                        worker = group % pack_workers
                        peer = bucket // size
                        expert = wave * WAVE + bucket % size
                        index = peer * E + expert
                        start = tl.load(send_offsets + index)
                        count = tl.load(send_offsets + index + 1) - start
                        dst = tl.load(destinations + index)
                        if WIDE:
                            start = start.to(tl.int64)
                            dst = dst.to(tl.int64)
                        for begin in range(
                            worker * PACK_BLOCK_ROWS,
                            count,
                            pack_workers * PACK_BLOCK_ROWS,
                        ):
                            rows = begin + tl.arange(0, PACK_BLOCK_ROWS)
                            tokens = tl.load(inverse + start + rows, rows < count, 0)
                            if WIDE:
                                tokens = tokens.to(tl.int64)
                            for kt in range(tl.cdiv(K, PACK_K)):
                                cols = kt * PACK_K + tl.arange(0, PACK_K)
                                mask = (rows[:, None] < count) & (cols[None, :] < K)
                                values = tl.load(
                                    hidden + tokens[:, None] * K + cols[None, :],
                                    mask,
                                    0,
                                )
                                tl.store(
                                    scattered
                                    + (start + rows[:, None]) * K
                                    + cols[None, :],
                                    values,
                                    mask,
                                )
                                if peer == RANK:
                                    tl.store(
                                        received
                                        + (dst + rows[:, None]) * K
                                        + cols[None, :],
                                        values,
                                        mask,
                                    )
                    sh.quiet_lowlevel(transport=sh.ACLSHMEM_TRANSPORT_MTE)
                    tl.store(packed + (wave * PRODUCERS + pid) * SLOT, epoch)
                    sh.quiet_lowlevel(transport=sh.ACLSHMEM_TRANSPORT_MTE)
                else:
                    peer = pid - PRODUCERS
                    for worker in range(PRODUCERS):
                        sh.signal_wait_until(
                            packed + (wave * PRODUCERS + worker) * SLOT,
                            sh.ACLSHMEM_CMP_GE,
                            epoch,
                        )
                    if peer != RANK:
                        for expert in range(
                            wave * WAVE, tl.minimum((wave + 1) * WAVE, E)
                        ):
                            index = peer * E + expert
                            start = tl.load(send_offsets + index)
                            count = tl.load(send_offsets + index + 1) - start
                            dst = tl.load(destinations + index)
                            if WIDE:
                                start = start.to(tl.int64)
                                dst = dst.to(tl.int64)
                            for begin in range(
                                0, count, UDMA_MAX_PUT_BYTES // (K * BF16_ELEMENT_BYTES)
                            ):
                                rows = tl.minimum(
                                    count - begin,
                                    UDMA_MAX_PUT_BYTES // (K * BF16_ELEMENT_BYTES),
                                )
                                sh.putmem(
                                    received + (dst + begin) * K,
                                    scattered + (start + begin) * K,
                                    rows * K * 2,
                                    peer,
                                )
                            if count > 0:
                                sh.signal_op(
                                    signals + (RANK * E + expert) * SLOT,
                                    epoch,
                                    sh.ACLSHMEM_SIGNAL_SET,
                                    peer,
                                )
            if (sub == 1) & (pid < R):
                # Device equivalent of ready[wave]: all remote buckets plus
                # the self-copy completion event. No compute/restore dependency.
                if pid == RANK:
                    for worker in range(PRODUCERS):
                        sh.signal_wait_until(
                            packed + (wave * PRODUCERS + worker) * SLOT,
                            sh.ACLSHMEM_CMP_GE,
                            epoch,
                        )
                size = tl.minimum(WAVE, E - wave * WAVE)
                for bucket in range(pid, R * size, R):
                    source = bucket // size
                    expert = wave * WAVE + bucket % size
                    count = tl.load(counts + source * E + expert)
                    if source != RANK and count > 0:
                        sh.signal_wait_until(
                            signals + (source * E + expert) * SLOT,
                            sh.ACLSHMEM_CMP_GE,
                            epoch,
                        )
                tl.atomic_add(
                    packed + (tl.cdiv(E, WAVE) * PRODUCERS + wave) * SLOT,
                    1,
                    sem="release",
                    scope="sys",
                )
        # Consumers independently acquire this wave, without a team barrier.
        with al.scope(core_mode="vector"):
            restore_pid = pid - R
            if (restore_pid >= 0) & (restore_pid < PRODUCERS):
                if sub == 1:
                    restore_token = dl.wait(
                        packed + (tl.cdiv(E, WAVE) * PRODUCERS + wave) * SLOT,
                        1,
                        "sys",
                        "acquire",
                        waitValue=R,
                    )
                    restore_received = dl.consume_token(received, restore_token)
                    size = tl.minimum(WAVE, E - wave * WAVE)
                    for bucket in range(restore_pid, R * size, PRODUCERS):
                        source = bucket // size
                        expert = wave * WAVE + bucket % size
                        count = tl.load(counts + source * E + expert)
                        sb = tl.load(source_offsets + source)
                        for previous in range(expert):
                            sb += tl.load(counts + source * E + previous)
                        eb = tl.load(expert_offsets + expert)
                        for previous in range(source):
                            eb += tl.load(counts + previous * E + expert)
                        if WIDE:
                            sb = sb.to(tl.int64)
                            eb = eb.to(tl.int64)
                        for begin in range(0, count, RESTORE_BLOCK_ROWS):
                            rows = begin + tl.arange(0, RESTORE_BLOCK_ROWS)
                            for kt in range(tl.cdiv(K, RESTORE_BLOCK_COLS)):
                                cols = kt * RESTORE_BLOCK_COLS + tl.arange(
                                    0, RESTORE_BLOCK_COLS
                                )
                                mask = (rows[:, None] < count) & (cols[None, :] < K)
                                values = tl.load(
                                    restore_received
                                    + (eb + rows[:, None]) * K
                                    + cols[None, :],
                                    mask,
                                    0,
                                )
                                tl.store(
                                    source_out
                                    + (sb + rows[:, None]) * K
                                    + cols[None, :],
                                    values,
                                    mask,
                                )
                                tl.store(
                                    expert_out
                                    + (eb + rows[:, None]) * K
                                    + cols[None, :],
                                    values,
                                    mask,
                                )
        with al.scope(core_mode="cube"):
            compute_token = dl.wait(
                packed + (tl.cdiv(E, WAVE) * PRODUCERS + wave) * SLOT,
                1,
                "sys",
                "acquire",
                waitValue=R,
            )
            compute_received = dl.consume_token(received, compute_token)
            if pid < CORES:
                first = tl.load(tile_offsets + wave * WAVE) * tl.cdiv(N, BLOCK_N)
                last = tl.load(
                    tile_offsets + tl.minimum((wave + 1) * WAVE, E)
                ) * tl.cdiv(N, BLOCK_N)
                for tile in range(first + pid, last, CORES):
                    mt = tile // tl.cdiv(N, BLOCK_N)
                    nt = tile % tl.cdiv(N, BLOCK_N)
                    expert = wave * WAVE
                    for index in tl.static_range(
                        wave * WAVE + 1,
                        (wave + 1) * WAVE if (wave + 1) * WAVE < E else E,
                    ):
                        expert += tl.where(tl.load(tile_offsets + index) <= mt, 1, 0)
                    if WEIGHTS_WIDE:
                        expert = expert.to(tl.int64)
                    start = tl.load(expert_offsets + expert)
                    count = tl.load(expert_offsets + expert + 1) - start
                    if WIDE:
                        start = start.to(tl.int64)
                    rows = (mt - tl.load(tile_offsets + expert)) * BLOCK_M + tl.arange(
                        0, BLOCK_M
                    )
                    cols = nt * BLOCK_N + tl.arange(0, BLOCK_N)
                    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                    for kt in range(tl.cdiv(K, BLOCK_K)):
                        kk = kt * BLOCK_K + tl.arange(0, BLOCK_K)
                        a = tl.load(
                            compute_received
                            + (start + rows[:, None]) * K
                            + kk[None, :],
                            (rows[:, None] < count) & (kk[None, :] < K),
                            0,
                        )
                        b = tl.load(
                            weights + expert * N * K + cols[None, :] * K + kk[:, None],
                            (cols[None, :] < N) & (kk[:, None] < K),
                            0,
                        )
                        acc += tl.dot(a, b)
                    tl.store(
                        output + (start + rows[:, None]) * N + cols[None, :],
                        acc.to(tl.bfloat16),
                        (rows[:, None] < count) & (cols[None, :] < N),
                    )


class DispatchGmmWorkspace:
    """Reusable shape-only capacity and communication state; not thread-safe.

    All world ranks must construct and close workspaces in the same order.
    ACL SHMEM must already be initialized with enough symmetric heap capacity.
    start/end may vary between calls, but must describe equally long intervals
    on all ranks and match the sums of their corresponding splits.
    """

    def __init__(self, ep_group, m, k, n, top_k, experts_per_rank):
        self.rank = dist.get_rank(ep_group)
        self.ranks = dist.get_world_size(ep_group)
        if dist.get_process_group_ranks(ep_group) != list(range(dist.get_world_size())):
            raise ValueError("only the complete torchrun world is supported")
        if not 1 <= experts_per_rank <= 160:
            raise ValueError("requires 1..160 local experts")
        if min(m, k, n, top_k) <= 0 or top_k > self.ranks * experts_per_rank:
            raise ValueError("invalid shape or top-k")
        self.group = ep_group
        self.m, self.k, self.n = m, k, n
        self.top_k, self.experts = top_k, experts_per_rank
        self.device = torch.device("npu", torch.npu.current_device())
        self.cores = driver.active.utils.get_aicore_num()
        if not 1 <= self.ranks < self.cores:
            raise ValueError(
                "insufficient Cube cores for fused producer/sender ownership"
            )
        if driver.active.utils.get_aivector_core_num() != 2 * self.cores:
            raise ValueError("overlapped restore requires two Vector subcores per Cube")
        self.wave = fused_wave_experts(self.ranks, self.experts, m * top_k * k * 2)
        self.closed = False
        rows = m * top_k
        capacity = self.ranks * rows
        total = self.ranks * self.experts

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, device=self.device, dtype=dtype)

        self.received = ash.aclshmem_create_tensor(
            [capacity, k], dtype=torch.bfloat16, device_id=self.device.index
        )
        self.signals = ash.aclshmem_create_tensor(
            [(total + self.ranks) * SIGNAL_SLOT_ELEMENTS.value],
            dtype=torch.int32,
            device_id=self.device.index,
        )
        self.exchange = self.signals[total * SIGNAL_SLOT_ELEMENTS.value :]
        self.count_matrix = ash.aclshmem_create_tensor(
            [self.ranks * total], dtype=torch.int32, device_id=self.device.index
        )
        self.signals.zero_()
        self.scatter_output = empty((rows, k))
        self.source_output = empty((capacity, k))
        self.expert_output = empty((capacity, k))
        self.gmm_output = empty((capacity, n))
        self.inverse = empty(rows, torch.int32)
        self.packed = torch.zeros(
            triton.cdiv(self.experts, self.wave)
            * (self.cores - self.ranks + 1)
            * SIGNAL_SLOT_ELEMENTS.value,
            device=self.device,
            dtype=torch.int32,
        )
        self.counts = empty(total, torch.int32)
        self.expert_sizes = empty(self.experts, torch.int32)
        self.send_offsets = empty(total + 1, torch.int32)
        self.destinations = empty(total, torch.int32)
        self.source_offsets = empty(self.ranks + 1, torch.int32)
        self.expert_offsets = empty(self.experts + 1, torch.int32)
        self.tile_offsets = empty(self.experts + 1, torch.int32)
        self.epoch = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.host_tables = empty(2 * total, torch.int32)
        self.public_counts = empty(total, torch.int32)
        self.public_experts = empty(self.experts, torch.int32)
        self.counts_host = torch.empty(
            2 * total, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.counts_stream = torch.npu.Stream(device=self.device)
        self.metadata_done = torch.npu.Event()
        self.counts_done = torch.npu.Event()
        torch.npu.synchronize()
        dist.barrier(group=self.group)

    def _launch(self, kernel, grid, args, constants, **options):
        # Let each Triton release bind its own compiled launcher signature.
        grid = (grid[0], 1, 1)
        kernel[grid](*args, *constants, **options)

    def close(self):
        """Collectively finish queued work and release symmetric allocations."""
        if not self.closed:
            torch.npu.synchronize()
            dist.barrier(group=self.group)
            for tensor in (self.count_matrix, self.signals, self.received):
                ash.aclshmem_free_tensor(tensor)
            self.closed = True

    def forward(
        self, hidden_states, scatter_index, splits, weights, start_idx, end_idx
    ):
        if self.closed:
            raise RuntimeError("workspace is closed")
        tensors = (hidden_states, scatter_index, splits, weights)
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("all inputs must be on the workspace NPU")
        if hidden_states.shape != (self.m, self.k):
            raise ValueError("hidden_states shape differs from workspace")
        if scatter_index.shape != (self.m, self.top_k):
            raise ValueError("scatter_index shape differs from workspace")
        if splits.shape != (self.ranks * self.experts,):
            raise ValueError("splits shape differs from workspace")
        if weights.shape != (self.experts, self.n, self.k):
            raise ValueError("weights shape differs from workspace")
        if hidden_states.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
            raise ValueError("hidden_states and weights must be BF16")
        if scatter_index.dtype != torch.int32 or splits.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("scatter_index must be INT32; splits must be INT32/INT64")
        if hidden_states.requires_grad or weights.requires_grad:
            raise ValueError("dispatch_gmm is forward-only")
        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            raise ValueError("start/end must be host integers")
        if not 0 <= start_idx <= end_idx <= self.m * self.top_k:
            raise ValueError("invalid routed interval")
        # Value contract: scatter_index is a permutation; splits are nonnegative
        # counts of that interval. Avoid a host validation read in the hot path.
        hidden_states = hidden_states.contiguous()
        scatter_index = scatter_index.contiguous()
        weights = weights.contiguous()
        splits_i32 = splits.to(torch.int32).contiguous()
        metadata_args = (
            scatter_index,
            splits_i32,
            self.inverse,
            self.packed,
            self.count_matrix,
            self.exchange,
            self.counts,
            self.expert_sizes,
            self.send_offsets,
            self.destinations,
            self.source_offsets,
            self.expert_offsets,
            self.tile_offsets,
            self.epoch,
            self.host_tables,
            self.public_counts,
            self.public_experts,
        )
        metadata_constants = (
            self.ranks,
            self.experts,
            self.m * self.top_k,
            self.top_k,
            start_idx,
            end_idx,
            self.rank,
            GEMM_BLOCK_M,
            self.cores,
            self.wave,
        )
        self._launch(
            kernel_dispatch_gmm_metadata,
            (triton.cdiv(self.m * self.top_k, INVERSE_BLOCK_SIZE.value) + 1,),
            metadata_args,
            metadata_constants,
        )
        self.metadata_done.record()
        main_args = (
            hidden_states,
            weights,
            self.scatter_output,
            self.received,
            self.source_output,
            self.expert_output,
            self.gmm_output,
            self.inverse,
            self.packed,
            self.signals,
            self.counts,
            self.send_offsets,
            self.destinations,
            self.source_offsets,
            self.expert_offsets,
            self.tile_offsets,
            self.epoch,
        )
        main_constants = (
            self.ranks,
            self.experts,
            self.k,
            self.n,
            self.m * self.top_k,
            self.rank,
            self.cores,
            self.wave,
            min(MAX_PACK_COLS, triton.next_power_of_2(self.k)),
            GEMM_BLOCK_M,
            GEMM_BLOCK_N,
            GEMM_BLOCK_K,
        )
        self._launch(
            kernel_dispatch_gmm_fused,
            (self.cores,),
            main_args,
            main_constants,
            enable_dynamic_cv_pipeline=False,
            enable_mixed_cv=True,
            compile_mode="simd",
            enable_preload=True,
            multibuffer=True,
            disable_auto_inject_block_sync=True,
        )
        # Submit main compute before copying counts, so the copy submission
        # cannot delay this rank's pack/send arrival.
        with torch.npu.stream(self.counts_stream):
            self.counts_stream.wait_event(self.metadata_done)
            self.counts_host.copy_(self.host_tables, non_blocking=True)
            self.counts_done.record()
        self.counts_done.synchronize()
        tables = self.counts_host.tolist()
        total = self.ranks * self.experts
        local, incoming = tables[:total], tables[total:]
        send_rows = [
            sum(local[p * self.experts : (p + 1) * self.experts])
            for p in range(self.ranks)
        ]
        source_rows = [
            sum(incoming[p * self.experts : (p + 1) * self.experts])
            for p in range(self.ranks)
        ]
        receive_rows = sum(source_rows)
        return dict(
            scatter_output=self.scatter_output[: end_idx - start_idx],
            a2a_splits=self.public_counts,
            a2a_splits_cpu_list=source_rows,
            splits_cpu_list=send_rows,
            a2a_splits_cpu_l=incoming,
            a2a_expert_input=self.source_output[:receive_rows],
            permute_a2a_expert_input=self.expert_output[:receive_rows],
            experts_a2a_splits=self.public_experts,
            pre_silu_output_all=self.gmm_output[:receive_rows],
        )


def dispatch_gmm(
    ep_group,
    hidden_states,
    scatter_index,
    splits,
    fc1_1_2_weight,
    start_idx,
    end_idx,
    *,
    workspace,
):
    """Complete forward recipe; reuse a caller-owned shape-only workspace.

    Inputs must satisfy the routing/count contract in the module docstring.
    Returned NPU views are ready on the current stream and alias workspace.
    """
    if ep_group is not workspace.group:
        raise ValueError("workspace belongs to another process group")
    return workspace.forward(
        hidden_states, scatter_index, splits, fc1_1_2_weight, start_idx, end_idx
    )


def dispatch_gmm_golden(
    ep_group, hidden_states, scatter_index, splits, fc1_1_2_weight, start_idx, end_idx
):
    """Independent forward-only Torch/HCCL reference with the same nine outputs."""
    ranks = dist.get_world_size(ep_group)
    experts = fc1_1_2_weight.shape[0]
    top_k = scatter_index.shape[1]
    scattered = torch.empty(
        (scatter_index.numel(), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    scattered[scatter_index.flatten().long()] = hidden_states.repeat_interleave(
        top_k, dim=0
    )
    scattered = scattered[start_idx:end_idx]
    incoming = torch.empty_like(splits)
    dist.all_to_all_single(incoming, splits, group=ep_group)
    incoming_list = incoming.cpu().tolist()
    source_rows = incoming.reshape(ranks, experts).sum(dim=1).cpu().tolist()
    send_rows = splits.reshape(ranks, experts).sum(dim=1).cpu().tolist()
    received = torch.empty(
        (sum(source_rows), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    dist.all_to_all_single(
        received,
        scattered,
        output_split_sizes=source_rows,
        input_split_sizes=send_rows,
        group=ep_group,
    )
    buckets = torch.split(received, incoming_list)
    expert_input = torch.cat(
        [buckets[p * experts + e] for e in range(experts) for p in range(ranks)], dim=0
    )
    sizes = incoming.reshape(ranks, experts).sum(dim=0, dtype=torch.int32)
    outputs = []
    offset = 0
    for e, count in enumerate(sizes.cpu().tolist()):
        if count:
            outputs.append(expert_input[offset : offset + count] @ fc1_1_2_weight[e].T)
        offset += count
    output = (
        torch.cat(outputs, dim=0)
        if outputs
        else hidden_states.new_empty((0, fc1_1_2_weight.shape[1]))
    )
    return dict(
        scatter_output=scattered,
        a2a_splits=incoming,
        a2a_splits_cpu_list=source_rows,
        splits_cpu_list=send_rows,
        a2a_splits_cpu_l=incoming_list,
        a2a_expert_input=received,
        permute_a2a_expert_input=expert_input,
        experts_a2a_splits=sizes,
        pre_silu_output_all=output,
    )


def make_random_inputs(shape, rank, ranks, device, seed):
    """Random expert bases with consecutive distinct top-k branches, as in the reference."""
    m, k, n, top_k, experts = shape
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    expert_ids = (
        torch.randint(ranks * experts, (m, 1), generator=generator)
        + torch.arange(top_k)
    ) % (ranks * experts)
    flat = expert_ids.flatten()
    order = flat.argsort(stable=True)
    scatter = torch.empty_like(order, dtype=torch.int32)
    scatter[order] = torch.arange(order.numel(), dtype=torch.int32)
    splits = torch.bincount(flat, minlength=ranks * experts).to(torch.int32)
    hidden = torch.randn((m, k), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn((experts, n, k), generator=generator) / math.sqrt(k)).to(
        device=device, dtype=torch.bfloat16
    )
    return hidden, scatter.view(m, top_k).to(device), splits.to(device), weights


def verify_outputs(expected, actual, group):
    """Full output comparison; report failures collectively before cleanup."""
    error = None
    try:
        if expected.keys() != actual.keys():
            raise AssertionError("result keys differ")
        for name, reference in expected.items():
            value = actual[name]
            if isinstance(reference, torch.Tensor):
                tolerance = 1e-3 if name == "pre_silu_output_all" else 0
                torch.testing.assert_close(
                    value.cpu(), reference.cpu(), rtol=tolerance, atol=tolerance
                )
            elif value != reference:
                raise AssertionError(f"{name} differs")
    except AssertionError as exception:
        error = str(exception)
        print(f"[FAIL rank={dist.get_rank(group)}] {error}", flush=True)
    flag = torch.tensor([error is None], dtype=torch.int32, device="npu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if not flag.item():
        raise AssertionError("DispatchGMM output validation failed")


def run_test_distributed(cases, iterations=1, *, case_executor=None, verify=True):
    """Run random cases, optionally delegating execution to the performance entry.

    case_executor(golden_recipe, candidate_recipe, case_info) returns both final
    result dictionaries. Allocation and random public-input generation remain
    here; every input-derived operation stays inside the two complete recipes.
    Standalone execution uses one correctness invocation. Test callers may
    explicitly request additional changed-input rounds.
    """
    group = dist.group.WORLD
    rank, ranks = dist.get_rank(), dist.get_world_size()
    selected = list(cases)
    # One heap for the whole sweep; each workspace releases its own symmetric tensors.
    # Per rank: worst-case receive payload, peer count rows and signal slots.
    # m/k/top_k/experts are local generator variables from each selected shape.
    heap_alignment_bytes = 2 * 1024 * 1024
    required = max(
        ranks * m * top_k * k * BF16_ELEMENT_BYTES.value
        + ranks * ranks * experts * COUNT_ELEMENT_BYTES.value
        + (ranks * experts + ranks) * SIGNAL_SLOT_BYTES
        for _, (m, k, n, top_k, experts) in selected
    )
    heap_size = triton.cdiv(required, heap_alignment_bytes) * heap_alignment_bytes
    if heap_size > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {heap_size / 2**30:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the swept shapes"
        )
    init_aclshmem(rank, ranks, heap_size, ash.OpEngineType.UDMA)
    try:
        for index, shape in selected:
            m, k, n, top_k, experts = shape
            if rank == 0:
                print(f"[CASE] {index:02d} random shape={shape}", flush=True)
            workspace = DispatchGmmWorkspace(group, m, k, n, top_k, experts)
            try:
                for iteration in range(iterations):
                    inputs = make_random_inputs(
                        shape, rank, ranks, workspace.device, 20260903 + iteration * 997
                    )
                    arguments = (group, *inputs, 0, m * top_k)
                    if case_executor is None:
                        expected = dispatch_gmm_golden(*arguments)
                        actual = dispatch_gmm(*arguments, workspace=workspace)
                    else:
                        expected, actual = case_executor(
                            lambda: dispatch_gmm_golden(*arguments),
                            lambda: dispatch_gmm(*arguments, workspace=workspace),
                            dict(
                                case=index,
                                m=m,
                                k=k,
                                n=n,
                                topk=top_k,
                                local_experts=experts,
                                ranks=ranks,
                                route="random",
                            ),
                        )
                    torch.npu.synchronize()
                    if verify:
                        verify_outputs(expected, actual, group)
                    del expected, actual, inputs, arguments
                if rank == 0:
                    print(
                        f"[{'PASS' if verify else 'COMPLETE'}] case={index:02d} random "
                        f"M={m} K={k} N={n} topk={top_k} experts={experts} "
                        f"ranks={ranks} iterations={iterations}",
                        flush=True,
                    )
            finally:
                workspace.close()
            del workspace
            torch.npu.empty_cache()
    finally:
        ash.aclshmem_finalize()
    if rank == 0:
        print(
            f"[SUCCESS] {len(selected)} random cases; all {ranks} ranks "
            f"{'passed' if verify else 'completed'}.",
            flush=True,
        )


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        run_test_distributed([(1, (2048, 1536, 960, 8, 8))])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
