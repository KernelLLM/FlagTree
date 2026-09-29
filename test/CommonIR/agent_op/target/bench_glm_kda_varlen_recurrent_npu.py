"""Benchmark GLM5.3 KDA recurrent kernel on varlen sequences.

Compares optimized v0 kernel against the upstream sglang reference kernel.
Reports per-shape latency (p50, p90, mean) and speedup ratio.
"""

import os
import sys
import time
import torch
import torch_npu
import json

# Fused path enabled by default; single-seq cases will use _run_fused
# Set GLM53_KDA_ENABLE_FUSED=0 to force reference kernel path

sys.path.insert(0, '/public-flash/yuansheng/op/flagos/FlagTree/test/CommonIR/agent_op/target')
sys.path.insert(0, '/public-flash/yuansheng/models/sglang/python/sglang/srt/hardware_backend/npu/attention/glm53')

from glm_kda_varlen_recurrent_npu_v0 import glm_kda_varlen_recurrent_npu as optimized_fn
from kda_recurrent_npu import glm_kda_varlen_recurrent_npu as reference_fn


def offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return torch.tensor(result, dtype=torch.int32, device="npu:0")


def make_inputs(total, heads):
    result = {
        name: (torch.randn(1, total, heads, 128, device="npu:0") * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    result["b"] = torch.randn(1, total, heads, device="npu:0").bfloat16()
    return result


def make_constants(heads):
    return dict(
        A_log=torch.randn(heads, device="npu:0") * 0.1,
        dt_bias=torch.randn(heads * 128, device="npu:0") * 0.1,
        lower_bound=-5.0,
    )


def _bench_fn(fn, constants, inputs, initial, slots, starts, num_iters, warmup):
    """Run warmup + timed iterations for a single kernel function."""
    for _ in range(warmup):
        fn(
            **constants, **inputs,
            initial_state_source=initial.clone(),
            initial_state_indices=slots, cu_seqlens=starts, prefill=True,
        )
    torch.npu.synchronize()

    times = []
    for _ in range(num_iters):
        state = initial.clone()
        torch.npu.synchronize()
        start = time.perf_counter()
        fn(
            **constants, **inputs,
            initial_state_source=state,
            initial_state_indices=slots, cu_seqlens=starts, prefill=True,
        )
        torch.npu.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Trim top/bottom 10% as outliers
    trim = max(1, len(times) // 10)
    times = sorted(times)[trim:-trim]
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "p50_ms": times[len(times) // 2] * 1000,
        "p90_ms": times[int(len(times) * 0.9)] * 1000,
    }


def benchmark_shape(seq_len, heads, num_iters=50, warmup=10):
    """Benchmark both optimized and reference kernels for a given shape."""
    torch.manual_seed(42)
    initial = torch.randn(8, heads, 128, 128, device="npu:0", dtype=torch.float32) * 0.1
    initial = initial.transpose(-1, -2)
    slots = torch.tensor([3], dtype=torch.int64, device="npu:0")
    starts = offsets([seq_len])
    inputs = make_inputs(seq_len, heads)
    constants = make_constants(heads)

    opt_stats = _bench_fn(
        optimized_fn, constants, inputs, initial, slots, starts,
        num_iters, warmup,
    )
    ref_stats = _bench_fn(
        reference_fn, constants, inputs, initial, slots, starts,
        num_iters, warmup,
    )

    speedup = ref_stats["p50_ms"] / opt_stats["p50_ms"] if opt_stats["p50_ms"] > 0 else float("inf")

    return {
        "seq_len": seq_len,
        "heads": heads,
        "optimized": opt_stats,
        "reference": ref_stats,
        "speedup_p50": speedup,
    }


def benchmark_varlen(seq_lengths, heads, num_iters=50, warmup=10):
    """Benchmark varlen (multi-sequence) batch."""
    torch.manual_seed(42)
    total = sum(seq_lengths)
    num_seqs = len(seq_lengths)
    initial = torch.randn(num_seqs + 4, heads, 128, 128, device="npu:0", dtype=torch.float32) * 0.1
    initial = initial.transpose(-1, -2)
    slots = torch.arange(1, num_seqs + 1, dtype=torch.int64, device="npu:0")
    starts = offsets(seq_lengths)
    inputs = make_inputs(total, heads)
    constants = make_constants(heads)

    opt_stats = _bench_fn(
        optimized_fn, constants, inputs, initial, slots, starts,
        num_iters, warmup,
    )
    ref_stats = _bench_fn(
        reference_fn, constants, inputs, initial, slots, starts,
        num_iters, warmup,
    )

    speedup = ref_stats["p50_ms"] / opt_stats["p50_ms"] if opt_stats["p50_ms"] > 0 else float("inf")

    return {
        "seq_lengths": seq_lengths,
        "total_tokens": total,
        "heads": heads,
        "optimized": opt_stats,
        "reference": ref_stats,
        "speedup_p50": speedup,
    }


def _fmt_row(label, opt, ref, speedup):
    """Format one benchmark row."""
    return (
        f"  {label:<30s}  "
        f"opt p50={opt['p50_ms']:8.3f}ms  "
        f"ref p50={ref['p50_ms']:8.3f}ms  "
        f"speedup={speedup:.3f}x"
    )


if __name__ == "__main__":
    torch.npu.set_device(0)

    results = []

    # --- Single-sequence shapes ---
    print("=== Single-sequence benchmarks ===")
    single_shapes = [
        (128, 4), (256, 4), (512, 4), (1024, 4),
        (128, 8), (256, 8), (512, 8), (1024, 8),
        (3264, 32), (8192, 32), (16384, 32),
    ]

    for seq_len, heads in single_shapes:
        label = f"T={seq_len}, H={heads}"
        print(f"  Benchmarking {label}...", flush=True)
        result = benchmark_shape(seq_len, heads)
        results.append(result)
        print(_fmt_row(label, result["optimized"], result["reference"], result["speedup_p50"]))

    # --- Varlen (multi-sequence) shapes ---
    print("\n=== Varlen (multi-sequence) benchmarks ===")
    varlen_configs = [
        ([129, 257], 4, "2-seq varlen H=4"),
        ([64, 128, 256], 4, "3-seq varlen H=4"),
        ([129, 257], 8, "2-seq varlen H=8"),
        ([512, 1024, 2048], 32, "3-seq varlen H=32"),
    ]

    varlen_results = []
    for lengths, heads, desc in varlen_configs:
        print(f"  Benchmarking {desc}...", flush=True)
        result = benchmark_varlen(lengths, heads)
        varlen_results.append(result)
        print(_fmt_row(desc, result["optimized"], result["reference"], result["speedup_p50"]))

    # --- Summary ---
    print("\n=== Summary ===")
    opt_wins = sum(1 for r in results if r["speedup_p50"] > 1.0)
    print(f"  Single-seq: optimized faster in {opt_wins}/{len(results)} configs")
    if results:
        avg_speedup = sum(r["speedup_p50"] for r in results) / len(results)
        print(f"  Average speedup (p50): {avg_speedup:.3f}x")

    # --- Save results ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "bench_results.json")
    all_results = {
        "version": "v0",
        "single_seq": results,
        "varlen": varlen_results,
    }
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
