"""Benchmark: attention (06-fused-attention) vs torch_npu.npu_fusion_attention.

Two timing modes
----------------
wall   (default)
    sync -> start -> kernel -> sync -> stop.
    Measures end-to-end latency including Python dispatch and kernel-launch
    overhead.  Uses time.perf_counter + device synchronize.

kernel
    Uses do_bench_npu (embedded), which uses torch_npu.profiler.
    Brackets only kernel execution, eliminating host-side overhead.

Metrics
-------
- Latency    (ms)
- TFLOPS     (2 * 2 * B * H * S * S * D / latency)
- Bandwidth  (GB/s)  -- bytes read (Q+K+V) + bytes written (O)

Usage
-----
    python bench_06-fused-attention.py [--B 4] [--S 1024] [--H 16]
                                       [--D 128] [--causal]
                                       [--block-m 64] [--block-n 128]
                                       [--warmup 5] [--rep 20]
                                       [--mode wall|kernel]
                                       [--no-check]
                                       [--sweep]
"""

import argparse
import glob
import os
import shutil
import tempfile
import time

import pandas as pd
import torch
import torch_npu

# ---------------------------------------------------------------------------
# Import attention implementation from fused_attention_06.py
# ---------------------------------------------------------------------------
from fused_attention_06 import attention

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
BLOCK_M = 64
BLOCK_N = 128
DIM = 128

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_B = 4
_DEFAULT_S = 1024
_DEFAULT_H = 16
_DEFAULT_D = DIM
_DEFAULT_WARMUP = 2
_DEFAULT_REP = 3

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _device():
    return "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"


def _sync(device: str):
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _tflops(B, H, S, D, latency_ms):
    """Two matmuls (QK^T and PV), each B*H*S*S*D multiply-adds = 2 flops."""
    return 2 * 2.0 * B * H * S * S * D / (latency_ms * 1e-3) / 1e12


def _bandwidth_gbs(B, H, S, D, latency_ms):
    """Read Q+K+V and write O, all fp16 (2 bytes each)."""
    elem = 2
    bytes_io = elem * B * H * S * D * 4  # Q + K + V + O
    return bytes_io / (latency_ms * 1e-3) / 1e9


def _wall_stats(latencies):
    """Return (median, mean, min, max) from a list of ms samples."""
    s = sorted(latencies)
    median = s[len(s) // 2]
    mean = sum(s) / len(s)
    return median, mean, s[0], s[-1]


# ---------------------------------------------------------------------------
# Timing backends
# ---------------------------------------------------------------------------


def _bench_wall(fn, device, warmup, rep):
    """Wall-clock: sync -> perf_counter -> fn() -> sync -> perf_counter."""
    for _ in range(warmup):
        fn()
    _sync(device)

    latencies = []
    for _ in range(rep):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        latencies.append((time.perf_counter() - t0) * 1e3)

    return _wall_stats(latencies)


def do_bench_npu(funcs, warmup=2, active=3, clear_l2_cache=False, keep_res=False):
    """Simplified do_bench_npu using torch_npu.profiler.

    Args:
        funcs: list of callables
        warmup: number of warmup iterations
        active: number of profiling iterations
        clear_l2_cache: ignored (placeholder for compatibility)
        keep_res: ignored (placeholder for compatibility)

    Returns:
        list of average kernel times (ms) for each func
    """
    device = _device()
    results = []

    for fn in funcs:
        # Warmup
        for _ in range(warmup):
            fn()
        _sync(device)

        # Create temp dir for profiler output
        tmp_dir = tempfile.mkdtemp(prefix="npu_prof_")
        try:
            # Profile
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(tmp_dir),
            ) as prof:  # noqa: F841
                for _ in range(active):
                    fn()
                _sync(device)

            # Parse task_time CSV
            csv_files = glob.glob(os.path.join(tmp_dir, "**/task_time*.csv"), recursive=True)
            if not csv_files:
                # Fallback: wall time
                latencies = []
                for _ in range(active):
                    _sync(device)
                    t0 = time.perf_counter()
                    fn()
                    _sync(device)
                    latencies.append((time.perf_counter() - t0) * 1e3)
                results.append(sum(latencies) / len(latencies))
                continue

            df = pd.read_csv(csv_files[0])
            # Sum "Total Time(us)" column and convert to ms
            total_us = df["Total Time(us)"].sum()
            avg_ms = total_us / 1000.0 / active
            results.append(avg_ms)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return results if len(results) > 1 else results[0]


def _bench_kernel(fn, warmup, rep):
    """Kernel-only timing via do_bench_npu (torch_npu.profiler fallback).

    do_bench_npu returns the average time in ms for a single callable.
    warmup  -> do_bench_npu `warmup` parameter
    rep     -> do_bench_npu `active` parameter
    """
    avg_ms = do_bench_npu(
        [fn],
        warmup=warmup,
        active=rep,
        clear_l2_cache=True,
        keep_res=False,
    )
    # do_bench_npu returns a single float when given a one-element list
    if isinstance(avg_ms, list):
        avg_ms = avg_ms[0]
    # Return (median, mean, min, max) -- only avg is available from do_bench_npu
    return avg_ms, avg_ms, avg_ms, avg_ms


# ---------------------------------------------------------------------------
# Reference: torch_npu.npu_fusion_attention
# ---------------------------------------------------------------------------


def _ref_npu_fusion(q, k, v, H, sm_scale, is_causal):
    """Run torch_npu.npu_fusion_attention as the reference implementation."""
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        H,
        padding_mask=None,
        atten_mask=None,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=0,
    )[0]


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


def run_benchmark(B, H, S, D, bm, bn, sm_scale, is_causal, mode, warmup, rep, no_check, device):
    """Benchmark attention() for a single (B, H, S, D, BM, BN) config.

    Returns a dict with keys: attn_ms, ref_ms, speedup, tflops, bw.
    """
    torch.manual_seed(0)
    q = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    k = torch.randn((B, H, S, D), dtype=torch.float16, device=device)
    v = torch.randn((B, H, S, D), dtype=torch.float16, device=device)

    # ---- optional correctness check ----------------------------------------
    if not no_check:
        ref = _ref_npu_fusion(q, k, v, H, sm_scale, is_causal)
        out = attention(q, k, v, is_causal, sm_scale, bm, bn)
        torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
        print("  Correctness check passed.")

    attn_fn = lambda: attention(q, k, v, is_causal, sm_scale, bm, bn)
    ref_fn = lambda: _ref_npu_fusion(q, k, v, H, sm_scale, is_causal)

    # ---- benchmark ---------------------------------------------------------
    if mode == "kernel":
        attn_median, attn_mean, attn_min, attn_max = _bench_kernel(attn_fn, warmup, rep)
        ref_median, ref_mean, ref_min, ref_max = _bench_kernel(ref_fn, warmup, rep)
    else:
        attn_median, attn_mean, attn_min, attn_max = _bench_wall(attn_fn, device, warmup, rep)
        ref_median, ref_mean, ref_min, ref_max = _bench_wall(ref_fn, device, warmup, rep)

    tfl = _tflops(B, H, S, D, attn_median)
    bw = _bandwidth_gbs(B, H, S, D, attn_median)
    speedup = ref_median / attn_median if attn_median > 0 else 0.0

    return {
        "attn_ms": attn_median,
        "ref_ms": ref_median,
        "speedup": speedup,
        "tflops": tfl,
        "bw": bw,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--B", type=int, default=_DEFAULT_B, help="Batch size")
    p.add_argument("--S", type=int, default=_DEFAULT_S, help="Sequence length")
    p.add_argument("--H", type=int, default=_DEFAULT_H, help="Number of heads")
    p.add_argument("--D", type=int, default=_DEFAULT_D, help="Head dimension")
    p.add_argument("--causal", action="store_true", help="Use causal attention")
    p.add_argument("--block-m", type=int, default=BLOCK_M, help="Query block size")
    p.add_argument("--block-n", type=int, default=BLOCK_N, help="Key/Value block size")
    p.add_argument("--sm-scale", type=float, default=None, help="Softmax scale (default: 1/sqrt(D))")
    p.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP, help="Warmup iterations")
    p.add_argument("--rep", type=int, default=_DEFAULT_REP, help="Measurement iterations")
    p.add_argument("--mode", choices=["wall", "kernel"], default="kernel", help="Timing mode")
    p.add_argument("--no-check", action="store_true", help="Skip correctness check")
    p.add_argument("--sweep", action="store_true", help="Run sweep across multiple configs")
    return p.parse_args()


def _print_result(B, H, S, D, bm, bn, result):
    print(f"  B={B}, H={H}, S={S}, D={D}, BM={bm}, BN={bn}")
    print(f"    attention:  {result['attn_ms']:8.3f} ms  {result['tflops']:6.2f} TFLOPS  {result['bw']:7.2f} GB/s")
    print(f"    reference:  {result['ref_ms']:8.3f} ms")
    print(f"    speedup:    {result['speedup']:8.2f}x")


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------

_SWEEP_CONFIGS = [
    # (B, H, S)
    (1, 1, 128),
    (1, 2, 256),
    (2, 2, 128),
    (4, 16, 512),
    (4, 32, 1024),
    (4, 32, 2048),
    (8, 32, 1024),
]


def main():
    args = _parse_args()
    device = _device()

    B, H, S, D = args.B, args.H, args.S, args.D
    bm, bn = args.block_m, args.block_n
    sm_scale = args.sm_scale if args.sm_scale is not None else (1.0 / (D**0.5))
    is_causal = args.causal
    mode = args.mode
    warmup = args.warmup
    rep = args.rep
    no_check = args.no_check

    print("=" * 70)
    print("Benchmark: attention (06-fused-attention.py)")
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Warmup: {warmup}, Rep: {rep}")
    print("=" * 70)

    if args.sweep:
        print("\nRunning sweep...\n")
        for (b, h, s) in _SWEEP_CONFIGS:
            d = D  # Use the CLI-specified D
            print(f"\nConfig: B={b}, H={h}, S={s}, D={d}, BM={bm}, BN={bn}")
            try:
                r = run_benchmark(b, h, s, d, bm, bn, sm_scale, is_causal, mode, warmup, rep, no_check, device)
                _print_result(b, h, s, d, bm, bn, r)
            except Exception as e:
                print(f"  [FAILED] {e}")
    else:
        print(f"\nSingle config: B={B}, H={H}, S={S}, D={D}, BM={bm}, BN={bn}\n")
        r = run_benchmark(B, H, S, D, bm, bn, sm_scale, is_causal, mode, warmup, rep, no_check, device)
        _print_result(B, H, S, D, bm, bn, r)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
