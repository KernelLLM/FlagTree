# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ulysses QKV GEMM + All2All correctness and event timing.

Run from the repository root (warmup=5, iters=50 by default)::

    ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4  \
        python/triton_dist/test/ascend/sp_ulysses_qkv_gemm_all2all.py --mode=both --case=random

--case=full covers 315 shapes. --mode=verify checks outputs without performance
events; perf skips accuracy; both checks the outputs returned by the timed calls.
Default: D=128, equal QKV H_local=1..7, four/eight ranks.
Use --seq-len S_total --hidden-dim K --out-features N --head-dim D --gqa G
for one custom shape (batch size one). K is independent of Q/K/V head counts.
One program per AI Core.
Core count must be divisible by ranks and at most 64 (producer wait bitset).
Partial final waves are masked; staging reserves a full region for each wave.
ULYSSES_SHMEM_BYTES sets the symmetric heap (default 8 GB).
--dtype selects bf16 (default) or fp16 for both golden and kernel.
Both case modes print mean/geomean of per-shape golden/kernel speedups at the end.

Times are means of per-iteration rank maxima. Full forward/QKV views are timed;
custom workspaces are preallocated, golden allocations occur inside events.
Signal reset and device/rank synchronization are outside both event windows.
No IR dumps, compilation logs or per-invocation progress are generated.
"""

import shmem as ash
import torch
import torch.distributed as dist
from prof import (
    distributed_max_times,
    load_kernel,
    perf_enabled,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
import argparse
import gc
import os
import statistics
import warnings

from utils import init_aclshmem

warnings.filterwarnings("ignore", message=r"No device id is provided via .*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"triton\.language\.extra\.cann\.extension\.(atan2|isfinited) is deprecated.*", category=FutureWarning)

kernel = load_kernel("sp_ulysses_qkv_gemm_all2all")


HEAD_DIM = kernel.HEAD_DIM
QKV_GROUPS = 3
SHMEM_BYTES = int(os.environ.get("ULYSSES_SHMEM_BYTES", 8 * 1024**3))
SEQUENCE_LENGTHS = tuple(s for end in (2048, 4096, 8192, 32768, 131072)
                         for s in range(end - 8, end + 1))


def make_cases(world_size):
    return [(s * world_size, world_size * h * HEAD_DIM, 3 * world_size * h * HEAD_DIM, HEAD_DIM, 1)
            for s in SEQUENCE_LENGTHS for h in range(1, 8)]


def input_factory(local_sequence_length, hidden_size, out_features, rank, device, dtype):
    attention_input = torch.rand(
        (local_sequence_length, hidden_size), dtype=dtype, device=device
    )
    attention_input.mul_(-2).add_(1).mul_(rank + 1)
    weight = torch.rand(
        (out_features, hidden_size), dtype=dtype, device=device
    )
    weight.mul_(-2).add_(1).mul_(rank + 1)
    return attention_input, weight


def golden(attention_input, weight, head_dim=HEAD_DIM, gqa=1):
    """Reference GEMM, unequal Q/K/V packing, and HCCL All2All."""
    world_size = dist.get_world_size()
    kv_heads = weight.shape[0] // ((gqa + 2) * head_dim)
    heads = (gqa * kv_heads, kv_heads, kv_heads)
    projected = torch.matmul(attention_input, weight.transpose(0, 1))
    parts = projected.split([h * head_dim for h in heads], dim=1)
    sends = [torch.cat(part.view(attention_input.shape[0], h, head_dim)
                       .chunk(world_size, dim=1), dim=0)
             for part, h in zip(parts, heads)]
    send = torch.cat(sends, dim=1).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return recv.split([h // world_size for h in heads], dim=1)


def synchronize():
    torch.npu.synchronize()
    dist.barrier()


def run_case(seq_len, hidden_dim, out_features, head_dim, gqa, rank, world_size, local_rank, dtype):
    layout = kernel.workspace_sizes(world_size, seq_len, hidden_dim, head_dim, out_features, gqa)
    m = layout['m']
    device = torch.device(f'npu:{local_rank}')
    x, w = input_factory(m, hidden_dim, out_features, rank, device, dtype)
    projected = torch.empty((m, out_features), dtype=dtype, device=device)
    output = torch.empty(layout['output_elements'], dtype=dtype, device=device)
    staging = ash.aclshmem_create_tensor([layout['staging_elements']], dtype=dtype, device_id=local_rank)
    try:
        signals = ash.aclshmem_create_tensor([layout['signal_elements']], dtype=torch.int32, device_id=local_rank)
        try:
            operator = kernel.SpUlyssesQKVGemmAll2AllKernel(
                rank, world_size, seq_len, hidden_dim, head_dim, out_features,
                staging, signals, output, projected, gqa=gqa)
            synchronize()

            def reset_signals():
                signals.zero_()
                # Complete reset BEFORE the profiler's prelaunch rank barrier.
                torch.npu.synchronize()

            @prof(label='torch_ulysses_qkv_gemm_all2all', prepare=reset_signals)
            def run_golden():
                return golden(x, w, head_dim, gqa)

            @prof(label='ulysses_qkv_gemm_all2all', prepare=reset_signals)
            def run_kernel():
                return operator.forward(x, w)

            if verify_enabled():
                projected.fill_(float('nan'))
                staging.fill_(float('nan'))
                output.fill_(float('nan'))
            expected = run_golden()
            actual = run_kernel()
            dtype_name = 'bf16' if dtype == torch.bfloat16 else 'fp16'
            context = (f'S_total={seq_len} S_local={m} K={hidden_dim} N={out_features} '
                       f'D={head_dim} gqa={gqa} ranks={world_size} dtype={dtype_name}')
            accuracy = 'SKIPPED'
            if verify_enabled():
                kernel.check_accuracy(actual, expected, x, w, context=context, gqa=gqa)
                accuracy = 'PASS'

            row = dict(dtype=dtype_name, ranks=world_size, S_local=m, S_total=seq_len,
                       D=head_dim, K=hidden_dim, N=out_features, gqa=gqa, accuracy=accuracy,
                       samples=0, golden_ms='', kernel_ms='', speedup='')
            if perf_enabled():
                # Reduce on EVERY rank before prof_report prints on rank zero.
                for fn in (run_golden, run_kernel):
                    fn.prof.times_ms = distributed_max_times(fn.prof.times_ms).tolist()
                prof_report(run_golden, run_kernel, context=context)
                row.update(samples=len(run_kernel.prof.times_ms), golden_ms=run_golden.prof.mean,
                           kernel_ms=run_kernel.prof.mean, speedup=run_golden.prof.mean / run_kernel.prof.mean)
            return row
        finally:
            synchronize()
            ash.aclshmem_free_tensor(signals)
    finally:
        synchronize()
        ash.aclshmem_free_tensor(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('verify', 'perf', 'both'), default='both')
    parser.add_argument('--case', choices=('random', 'full'), default='full')
    parser.add_argument('--dtype', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--seq-len', type=int, help='global sequence length S_total')
    parser.add_argument('--hidden-dim', type=int, help='independent input width K')
    parser.add_argument('--out-features', type=int, help='total QKV width N')
    parser.add_argument('--head-dim', type=int, default=HEAD_DIM)
    parser.add_argument('--gqa', type=int, default=1)
    args = parser.parse_args()
    custom = (args.seq_len, args.hidden_dim, args.out_features)
    if any(v is not None for v in custom) and not all(v is not None for v in custom):
        parser.error('--seq-len, --hidden-dim and --out-features must be supplied together')
    if args.seq_len is None and (args.head_dim != HEAD_DIM or args.gqa != 1):
        parser.error('--head-dim/--gqa require an explicit custom shape')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    run_options()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank)
    dist.init_process_group('hccl')
    initialized = False
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if rank != local_rank or int(os.environ['LOCAL_WORLD_SIZE']) != world_size:
            raise ValueError('this test requires a single node, rank == LOCAL_RANK')
        cases = ([(args.seq_len, args.hidden_dim, args.out_features, args.head_dim, args.gqa)]
                 if args.seq_len is not None else select_cases(make_cases(world_size)))
        for seq_len, hidden_dim, out_features, head_dim, gqa in cases:
            kernel.workspace_sizes(world_size, seq_len, hidden_dim, head_dim, out_features, gqa)
        init_aclshmem(rank, world_size, SHMEM_BYTES, ash.OpEngineType.UDMA)
        initialized = True
        dist.barrier()
        torch.manual_seed(42 + rank)
        if rank == 0:
            print(f'[CONFIG] ranks={world_size} mode={args.mode} dtype={args.dtype} case={args.case} selected={len(cases)}', flush=True)
        speedups = []
        for index, shape in enumerate(cases, 1):
            if rank == 0:
                print(f'[CASE {index}/{len(cases)}] S_total/K/N/D/gqa={shape}', flush=True)
            result = run_case(*shape, rank, world_size, local_rank, dtype)
            # After run_case returns, reclaim profiler closure cycles holding
            # this case's tensors. Device/rank completion is handled inside it.
            gc.collect()
            if rank == 0 and result['samples']:
                speedups.append(result['speedup'])
        if rank == 0:
            print(f'[DONE] Completed {len(cases)} cases in {args.mode} mode.', flush=True)
            if speedups:
                print(f'[AVERAGE] ranks={world_size} dtype={args.dtype} case={args.case} '
                      f'shapes={len(speedups)} '
                      f'mean_speedup={statistics.mean(speedups):.4f} '
                      f'geomean_speedup={statistics.geometric_mean(speedups):.4f} '
                      '(golden_ms/kernel_ms, equal weight per shape)', flush=True)
            else:
                print('[AVERAGE] mean_speedup=N/A geomean_speedup=N/A (no timed samples)', flush=True)
    finally:
        try:
            if initialized:
                synchronize()
                ash.aclshmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
