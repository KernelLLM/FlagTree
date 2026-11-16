# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.


import math

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.runtime import driver
from triton.language.extra.cann.extension import sub_vec_id, sync_block_all
from triton_dist.language.extra import libshmem_device


# Model dimension, GEMM tile dimensions, and Vector repack granularity.
# HEAD_DIM and BLOCK_M happen to share a default value but serve different roles.
HEAD_DIM = 128
BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 128
REPACK_ROWS = 8
# Spacing in int32 elements: each readiness signal occupies a 64-byte slot.
SIGNAL_STRIDE = 16


def workspace_sizes(ranks, seq_len, hidden_dim, head_dim, out_features, gqa=1, core_count=None):
    """S_total=seq_len, K=hidden_dim, D=head_dim, N=out_features (all QKV)."""
    if ranks not in (4, 8) or min(seq_len, hidden_dim, head_dim, out_features, gqa) <= 0:
        raise ValueError('requires 4/8 ranks and positive shape dimensions/gqa')
    if seq_len % ranks or hidden_dim % head_dim:
        raise ValueError('requires S_total % ranks == 0 and K % D == 0')
    if out_features % (head_dim * ranks * (gqa + 2)):
        raise ValueError('requires N % (D * ranks * (gqa + 2)) == 0')
    if core_count is None:
        core_count = driver.active.utils.get_aicore_num()
    if core_count < ranks or core_count % ranks:
        raise ValueError('AI Core count must be a positive multiple of ranks')
    if core_count > 64:
        raise ValueError('producer wait bitset supports up to 64 AI Cores')
    m = seq_len // ranks
    kv_heads = out_features // (head_dim * ranks * (gqa + 2))
    local_heads = (gqa * kv_heads, kv_heads, kv_heads)
    widths = tuple(h * head_dim for h in local_heads)
    blocks = core_count // ranks
    peer = blocks * BLOCK_M * sum(widths)
    waves = triton.cdiv(m, blocks * BLOCK_M)
    return dict(m=m, k=hidden_dim, n=out_features, local_heads=local_heads, widths=widths,
                blocks=blocks, waves=waves, peer_elements=peer,
                wave_elements=ranks*peer, staging_elements=waves*ranks*peer,
                signal_elements=2*ranks*waves*blocks*SIGNAL_STRIDE,
                output_elements=seq_len*sum(widths))


@triton.jit
def _gemm_signal(wave, producer, CORES: tl.constexpr, SIGNAL_STRIDE: tl.constexpr):
    return (wave * CORES + producer) * SIGNAL_STRIDE


@triton.jit
def _packed_signal(source, wave, member, BLOCKS: tl.constexpr, WAVES: tl.constexpr,
                   CORES: tl.constexpr, SIGNAL_STRIDE: tl.constexpr):
    # First bank is local GEMM readiness; second is remote/local packed readiness.
    return (CORES * WAVES + (source * WAVES + wave) * BLOCKS + member) * SIGNAL_STRIDE


@triton.jit(do_not_specialize=['rank'])
def kernel_sp_ulysses_qkv_gemm_all2all(x, w, projected, staging, signals, output, rank, M, K,
                           RANKS: tl.constexpr, QWIDTH: tl.constexpr, KVWIDTH: tl.constexpr, CORES: tl.constexpr,
                           WAVES: tl.constexpr, COPY_N: tl.constexpr,
                           HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                           REPACK_ROWS: tl.constexpr, SIGNAL_STRIDE: tl.constexpr):
    BLOCKS: tl.constexpr = CORES // RANKS
    WAIT_DTYPE: tl.constexpr = tl.uint32 if CORES <= 32 else tl.uint64
    WIDTH: tl.constexpr = QWIDTH + 2 * KVWIDTH
    N: tl.constexpr = RANKS * WIDTH
    # Keep compile-time tile counts as integers, not tl.cdiv Tensor results.
    TILES_N: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    GEMM_TASKS: tl.constexpr = BLOCKS * TILES_N
    WAVE_ROWS: tl.constexpr = BLOCKS * BLOCK_M
    PEER: tl.constexpr = WAVE_ROWS * WIDTH
    COPY_TASKS_PER_M: tl.constexpr = (BLOCK_M // REPACK_ROWS) * 3 * RANKS
    tl.static_assert(BLOCK_M % REPACK_ROWS == 0)
    tl.static_assert(COPY_N >= QWIDTH and COPY_N >= KVWIDTH)
    pid = tl.program_id(0)
    sub = sub_vec_id()
    for wave in range(0, WAVES):
        row_begin = wave.to(tl.int64) * WAVE_ROWS
        wave_base = wave.to(tl.int64) * RANKS * PEER
        payload = tl.cast(M, tl.int64) * WIDTH
        # Linear full-N tiles: no target rank or Q/K/V boundary in this loop.
        for task in range(pid, GEMM_TASKS, CORES):
            tile_m = task // TILES_N
            tile_n = task % TILES_N
            rows = row_begin + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            tl.extra.cann.extension.compile_hint(acc, 'matmul_at_least_once')
            for kt in range(0, tl.cdiv(K, BLOCK_K)):
                ks = kt * BLOCK_K + tl.arange(0, BLOCK_K)
                a = tl.load(x + rows[:, None] * K + ks[None, :],
                            (rows[:, None] < M) & (ks[None, :] < K), other=0)
                b = tl.load(w + columns[None, :].to(tl.int64) * K + ks[:, None],
                            (ks[:, None] < K) & (columns[None, :] < N), other=0)
                acc += tl.dot(a, b)
            tl.store(projected + rows[:, None] * N + columns[None, :],
                     acc.to(projected.dtype.element_ty), (rows[:, None] < M) & (columns[None, :] < N))

        # Publish before producer-signal waits. Full storage and per-wave flags
        # permit faster programs/ranks to advance without overwriting old data.
        epoch = wave + 1
        # Complete this wave's GEMM on all Cube and Vector cores before repacking.
        # TODO: switch to lowlevel MTE transport quiet after NPU-IR fix
        sync_block_all("all", 8)
        libshmem_device.signal_op(signals + _gemm_signal(wave, pid, CORES, SIGNAL_STRIDE), epoch,
                        libshmem_device.ACLSHMEM_SIGNAL_SET, rank)
        if sub == 0:
            # Producer readiness covers ALL its GEMM tiles in this wave. Once
            # observed, its bit is valid for every subsequent repack group.
            # Reset each wave: signals are distinct and epochs must not leak.
            waited = tl.full((), 0, WAIT_DTYPE)
            copy_target_group = pid % RANKS
            for copy_m in range(0, BLOCKS):
                # This program owns tasks for all three slices of this target
                # within BLOCK_M rows. Include BLOCK_N tiles crossing slice boundaries.
                for part in range(0, 3):
                    part_width = tl.where(part == 0, QWIDTH, KVWIDTH)
                    part_offset = tl.where(part == 0, 0, QWIDTH + (part - 1) * KVWIDTH)
                    first_column = part_offset * RANKS + copy_target_group * part_width
                    first_tile = first_column // BLOCK_N
                    end_tile = tl.cdiv(first_column + part_width, BLOCK_N)
                    for tile_n_ready in range(first_tile, end_tile):
                        producer = (copy_m * TILES_N + tile_n_ready) % CORES
                        producer_bit = tl.full((), 1, WAIT_DTYPE) << producer
                        if (waited & producer_bit) == 0:
                            libshmem_device.signal_wait_until(signals + _gemm_signal(wave, producer, CORES, SIGNAL_STRIDE),
                                                   libshmem_device.ACLSHMEM_CMP_EQ, epoch)
                            waited = waited | producer_bit
                # Restart the cyclic assignment within each M group, including
                # groups whose task count is not divisible by CORES.
                for copy_task in range(copy_m * COPY_TASKS_PER_M + pid,
                                       (copy_m + 1) * COPY_TASKS_PER_M, CORES):
                    copy_group = copy_task // (3 * RANKS)
                    copy_part = (copy_task // RANKS) % 3
                    copy_target = copy_task % RANKS
                    copy_rows = copy_group * REPACK_ROWS + tl.arange(0, REPACK_ROWS)
                    copy_cols = tl.arange(0, COPY_N)
                    source_base = (row_begin + copy_rows[:, None]) * N
                    copy_width = tl.where(copy_part == 0, QWIDTH, KVWIDTH)
                    copy_offset = tl.where(copy_part == 0, 0, QWIDTH + (copy_part - 1) * KVWIDTH)
                    source_cols = copy_offset * RANKS + copy_target * copy_width + copy_cols[None, :]
                    copy_mask = ((row_begin + copy_rows[:, None] < M)
                                 & (copy_cols[None, :] < copy_width))
                    data = tl.load(projected + source_base + source_cols,
                                   copy_mask, other=0)
                    # The task stride is CORES, divisible by RANKS: each program
                    # repacks one destination. Ordinary output is never symmetric.
                    if copy_target == rank:
                        dest_base = rank.to(tl.int64) * payload + (row_begin + copy_rows[:, None]) * WIDTH
                        tl.store(output + dest_base + copy_offset + copy_cols[None, :],
                                 data, copy_mask)
                    else:
                        dest_base = wave_base + copy_target.to(tl.int64) * PEER + copy_rows[:, None] * WIDTH
                        tl.store(staging + dest_base + copy_offset + copy_cols[None, :],
                                 data, copy_mask)
        # Every core must join before either core type publishes packed readiness.
        sync_block_all("all", 9)
        if sub == 0:
            packed_ready = _packed_signal(
                rank, wave, pid // RANKS, BLOCKS, WAVES, CORES, SIGNAL_STRIDE
            )
            if pid % RANKS != rank:
                libshmem_device.signal_op(signals + packed_ready, epoch,
                                  libshmem_device.ACLSHMEM_SIGNAL_SET, pid % RANKS)
        if sub == 1:
            if pid < RANKS:
                if pid != rank:
                    for member in range(0, BLOCKS):
                        ready = _packed_signal(pid, wave, member, BLOCKS, WAVES, CORES, SIGNAL_STRIDE)
                        libshmem_device.signal_wait_until(signals + ready, libshmem_device.ACLSHMEM_CMP_EQ, epoch)
                    libshmem_device.getmem(output + pid.to(tl.int64) * payload + row_begin * WIDTH,
                                 staging + wave_base + rank.to(tl.int64) * PEER,
                                 tl.minimum(M - row_begin, WAVE_ROWS) * WIDTH * 2, pid)


class SpUlyssesQKVGemmAll2AllKernel:
    """Fixed-shape operator with caller-owned, non-overlapping workspace.

    staging/signals must be symmetric aclshmem allocations; projected/output
    are ordinary NPU tensors. All ranks use the same AI Core count, shapes and
    allocation order. The caller supplies contiguous tensors on the same NPU: int32
    signals and matching BF16/FP16 data, with workspace capacity from workspace_sizes().
    Buffers must remain alive until collective completion.
    seq_len is global tokens, hidden_dim is independent input width K,
    out_features is total QKV width N; Q heads = gqa * K/V heads.
    Batch size is one. Weight storage is contiguous [N,K], consumed transposed.
    """

    def __init__(self, rank, ranks, seq_len, hidden_dim, head_dim, out_features,
                 staging, signals, output, projected, gqa=1):
        self.core_count = driver.active.utils.get_aicore_num()
        self.layout = workspace_sizes(ranks, seq_len, hidden_dim, head_dim, out_features,
                                      gqa, self.core_count)
        if not 0 <= rank < ranks:
            raise ValueError('requires 0 <= rank < ranks')
        self.rank, self.ranks = rank, ranks
        self.m, self.k, self.n = self.layout['m'], hidden_dim, out_features
        self.head_dim = head_dim
        self.staging, self.signals = staging, signals
        self.output = output.view(seq_len, sum(self.layout['local_heads']), head_dim)
        self.projected = projected.view(self.m, self.n)
        self.copy_n = triton.next_power_of_2(max(self.layout['widths']))

    def forward(self, x, w):
        """Return Q/K/V views; caller resets signals and handles completion."""
        if x.shape != (self.m, self.k) or w.shape != (self.n, self.k):
            raise ValueError('requires X[S_total/ranks,K], W[N,K]')
        kernel_sp_ulysses_qkv_gemm_all2all[(self.core_count,)](
            x, w, self.projected, self.staging, self.signals, self.output,
            self.rank, self.m, self.k, RANKS=self.ranks,
            QWIDTH=self.layout['widths'][0], KVWIDTH=self.layout['widths'][1],
            WAVES=self.layout['waves'], COPY_N=self.copy_n, CORES=self.core_count,
            HEAD_DIM=self.head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K, REPACK_ROWS=REPACK_ROWS, SIGNAL_STRIDE=SIGNAL_STRIDE,
        )
        return self.output.split(self.layout['local_heads'], dim=1)


def _cpu_reference(x, w, compute=True, gqa=1):
    """Stream source-rank FP64 QKV references; every rank joins broadcasts.

    Inputs and weights may differ between source ranks. Only the receiving
    rank's head slice is multiplied, in 64-row CPU chunks. Passing ranks join
    the same broadcasts without computing a reference.
    """
    rank, ranks = dist.get_rank(), dist.get_world_size()
    m, k = x.shape
    kv_width = w.shape[0] // ((gqa + 2) * ranks)
    widths = (gqa * kv_width, kv_width, kv_width)
    source_x, source_w = torch.empty_like(x), torch.empty_like(w)
    for source in range(ranks):
        if rank == source:
            source_x.copy_(x)
            source_w.copy_(w)
        dist.broadcast(source_x, src=source)
        dist.broadcast(source_w, src=source)
        if compute:
            parts = source_w.split([width * ranks for width in widths], dim=0)
            weight = torch.cat([part[rank * width:(rank + 1) * width]
                                for part, width in zip(parts, widths)], dim=0).detach().cpu().double().T
            for start in range(0, m, 64):
                rows = source_x[start:start + 64].detach().cpu().double()
                yield source * m + start, torch.matmul(rows, weight)


def _check_cpu_reference(actual, expected, chunks, small_threshold, error_threshold):
    """Compare full-output error metrics with golden against CPU FP64 truth."""
    statistics = [dict(max_relative=0.0, sum_relative=0.0, sum_squared=0.0, small_errors=0)
                  for _ in range(2)]
    normal_count = small_count = covered_rows = 0
    finite = True
    for start, truth in chunks:
        end = start + truth.shape[0]
        covered_rows += truth.shape[0]
        values = [torch.cat([part[start:end].detach().cpu().double().flatten(1) for part in result], dim=1)
                  .reshape(truth.shape) for result in (actual, expected)]
        if not torch.isfinite(truth).all() or any(not torch.isfinite(v).all() for v in values):
            # Keep draining the reference iterator so every rank completes the
            # source broadcasts, even after detecting invalid output.
            finite = False
            continue
        small = truth.abs() < small_threshold
        normal = ~small
        normal_count += int(normal.sum())
        small_count += int(small.sum())
        for value, stats in zip(values, statistics):
            difference = value - truth
            relative = difference[normal].abs() / (truth[normal].abs() + 1e-7)
            if relative.numel():
                stats['max_relative'] = max(stats['max_relative'], float(relative.max()))
                stats['sum_relative'] += float(relative.sum())
                stats['sum_squared'] += float(difference[normal].square().sum())
            stats['small_errors'] += int((difference[small].abs() > error_threshold).sum())
    if not finite or covered_rows != actual[0].shape[0]:
        return False, 'CPU FP64 check failed: non-finite values or incomplete reference'

    def ratio(custom, reference):
        return custom / reference if reference else (1.0 if custom == 0 else float('inf'))

    def within(value, limit):
        return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)

    metrics = [dict(mare=s['max_relative'], mere=s['sum_relative'] / max(normal_count, 1),
                    rmse=math.sqrt(s['sum_squared'] / max(normal_count, 1))) for s in statistics]
    ratios = {name: ratio(metrics[0][name], metrics[1][name]) for name in metrics[0]}
    small_ratio = statistics[0]['small_errors'] / max(statistics[1]['small_errors'], 1)
    passed = (all(within(ratios[name], limit)
                  for name, limit in (('mare', 2.0), ('mere', 1.2), ('rmse', 1.2)))
              and within(small_ratio, 2.0))
    return passed, (
        f'CPU FP64: normal_count={normal_count}, small_count={small_count}, '
        f'small_threshold={small_threshold}, error_threshold={error_threshold}, '
        f'kernel_metrics={metrics[0]}, torch_metrics={metrics[1]}, ratios={ratios} '
        f'(limits: mare=2.0, mere=1.2, rmse=1.2), '
        f"small_errors={statistics[0]['small_errors']}/{statistics[1]['small_errors']}, "
        f'small_count_ratio={small_ratio} (limit=2.0)'
    )


def check_accuracy(actual, expected, x, w, context="result", gqa=1):
    """Check QKV with dtype-specific tolerances and CPU FP64 fallback metrics."""
    if x.dtype == torch.bfloat16:
        tolerance, small_threshold = 5e-3, 2 ** (-8)
    else:
        tolerance, small_threshold = 1e-3, 2 ** (-11)
    error_threshold = 2 ** (-16)
    rank = dist.get_rank()
    passed = torch.ones(1, dtype=torch.int32, device=actual[0].device)
    error_msg = cpu_details = ""
    try:
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    except AssertionError as err:
        passed[0] = 0
        error_msg = str(err)
    all_passed = passed.clone()
    dist.all_reduce(all_passed, op=dist.ReduceOp.MIN)
    if not all_passed.item():
        chunks = _cpu_reference(x, w, compute=not bool(passed.item()), gqa=gqa)
        if passed.item():
            for _ in chunks:
                pass
        else:
            cpu_passed, cpu_details = _check_cpu_reference(
                actual, expected, chunks, small_threshold, error_threshold)
            passed[0] = int(cpu_passed)
        all_passed = passed.clone()
        dist.all_reduce(all_passed, op=dist.ReduceOp.MIN)
    for rank_id in range(dist.get_world_size()):
        if rank == rank_id:
            if passed.item():
                detail = (f'passed CPU fallback. {cpu_details}' if cpu_details else
                          f'matches golden (rtol={tolerance}, atol={tolerance}).')
                print(f'[PASS] Rank {rank}: {context} {detail}', flush=True)
            else:
                print(f'[FAIL] Rank {rank}: {context}\n{error_msg}\n{cpu_details}', flush=True)
        dist.barrier()
    if not all_passed.item():
        raise AssertionError(f'{context} correctness check failed')
