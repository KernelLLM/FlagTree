"""Benchmark: flash_attention_fwd (fa_triton_arch) vs PyTorch SDPA.

Two timing modes
----------------
wall   (default)
    sync -> start -> kernel -> sync -> stop.
    Measures end-to-end latency including Python dispatch and kernel-launch
    overhead.  Uses time.perf_counter + device synchronize.

kernel
    Uses do_bench_npu from testing.py (same directory), which tries the
    lightweight mspti backend first and falls back to torch_npu.profiler.
    Brackets only kernel execution, eliminating host-side overhead.

Metrics
-------
- Latency    (ms)
- TFLOPS     (2 * 2 * B * Hq * Sq * Skv * Dq / latency)
- Bandwidth  (GB/s)  — bytes read (Q+K+V) + bytes written (O)

Sweep matrix (--sweep)
----------------------
  batches      : [1, 4, 8, 16, 32]
  seqlen_pairs : [[32, 8192], [2048, 2048], [4096, 4096], [8192, 8192], [16384, 16384]]
  head_dims    : [64, 96, 128]   (Dq == Dkv required; split Q/KV dim unsupported)
  head_pairs   : [[16, 16], [16, 8], [16, 4], [16, 1]]

  Total configurations: 5 × 5 × 4 × 4 = 400
  Note: kernel now supports variable head dimensions via constexpr DIM parameter.

Usage
-----
    python bench_fa_triton_arch.py [--B 4] [--S 1024] [--H 16]
                                   [--q-heads N] [--kv-heads N]
                                   [--D 128] [--kv-dim 128]
                                   [--causal]
                                   [--combine-batch 8]
                                   [--warmup 5] [--rep 20]
                                   [--mode wall|kernel]
                                   [--no-check]
                                   [--sweep]
"""

import argparse
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Import flash_attention_fwd from success_case/fa_triton_arch.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUCCESS_DIR = os.path.abspath(os.path.join(_HERE, "..", "success_case"))
sys.path.insert(0, _SUCCESS_DIR)

import fa_triton_arch as _fa  # noqa: E402

# ---------------------------------------------------------------------------
# Import do_bench_npu from testing.py in the same directory
# ---------------------------------------------------------------------------
sys.path.insert(0, _HERE)

from testing import do_bench_npu  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_B = 4
_DEFAULT_S = 1024
_DEFAULT_H = 16
_DEFAULT_D = 64  # default head dimension (kernel now supports variable DIM via constexpr)
_DEFAULT_COMBINE_BATCH = 8
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


def _tflops(B, Hq, Sq, Skv, Dq, latency_ms):
    """Two matmuls (QK^T and PV).
    QK^T: B * Hq * Sq * Skv * Dq  (2 flops per multiply-add)
    PV  : B * Hq * Sq * Dq * Skv  (same shape, same cost)
    """
    return 2 * 2.0 * B * Hq * Sq * Skv * Dq / (latency_ms * 1e-3) / 1e12


def _bandwidth_gbs(B, Hq, Hkv, Sq, Skv, Dq, Dkv, latency_ms):
    """Read Q+K+V and write O, all fp16 (2 bytes each).
    Q : B * Hq  * Sq  * Dq
    K : B * Hkv * Skv * Dkv
    V : B * Hkv * Skv * Dkv
    O : B * Hq  * Sq  * Dq  (same shape as Q)
    """
    elem = 2  # fp16 = 2 bytes
    bytes_io = elem * (
        B * Hq * Sq * Dq +  # Q
        B * Hkv * Skv * Dkv +  # K
        B * Hkv * Skv * Dkv +  # V
        B * Hq * Sq * Dq  # O
    )
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
    """Wall-clock: sync → perf_counter → fn() → sync → perf_counter."""
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


def _bench_kernel(fn, warmup, rep):
    """Kernel-only timing via do_bench_npu (mspti → torch_npu.profiler fallback).

    do_bench_npu returns the average time in ms for a single callable.
    warmup  → do_bench_npu `warmup` parameter
    rep     → do_bench_npu `active` parameter
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
    # Return (median, mean, min, max) — only avg is available from do_bench_npu
    return avg_ms, avg_ms, avg_ms, avg_ms


# ---------------------------------------------------------------------------
# Reference: PyTorch SDPA
# ---------------------------------------------------------------------------


def _ref_sdpa(q, k, v, is_causal):
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                            is_causal=is_causal).to(torch.float16)


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


def run_benchmark(B, Hq, Hkv, Sq, Skv, Dq, Dkv, combine_batch, is_causal, mode, warmup, rep, no_check, device):
    """Benchmark flash_attention_fwd for a single (B, Hq, Hkv, Sq, Skv, Dq, Dkv) config.

    Returns a dict with keys: fa_ms, sdpa_ms, speedup, tflops_fa, bw_fa.
    flash_attention_fwd enforces Dq == Dkv == DIM (compile-time constant).
    Configs that violate this constraint raise AssertionError and are surfaced
    as errors in sweep mode rather than crashing the entire run.
    """
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})"

    torch.manual_seed(0)
    q = torch.randn((B, Hq, Sq, Dq), dtype=torch.float16, device=device)
    k = torch.randn((B, Hkv, Skv, Dkv), dtype=torch.float16, device=device)
    v = torch.randn((B, Hkv, Skv, Dkv), dtype=torch.float16, device=device)

    # ---- optional correctness check ----------------------------------------
    if not no_check:
        ref = _ref_sdpa(q, k, v, is_causal)
        out = _fa.flash_attention_fwd(q, k, v, combine_batch, is_causal=is_causal)
        torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
        print("  Correctness check passed.")

    fa_fn = lambda: _fa.flash_attention_fwd(q, k, v, combine_batch, is_causal=is_causal)
    sdpa_fn = lambda: _ref_sdpa(q, k, v, is_causal)

    # ---- benchmark ---------------------------------------------------------
    if mode == "kernel":
        fa_median, fa_mean, fa_min, fa_max = _bench_kernel(fa_fn, warmup, rep)
        sdpa_median, sdpa_mean, sdpa_min, sdpa_max = _bench_kernel(sdpa_fn, warmup, rep)
    else:
        fa_median, fa_mean, fa_min, fa_max = _bench_wall(fa_fn, device, warmup, rep)
        sdpa_median, sdpa_mean, sdpa_min, sdpa_max = _bench_wall(sdpa_fn, device, warmup, rep)

    tfl_fa = _tflops(B, Hq, Sq, Skv, Dq, fa_median)
    bw_fa = _bandwidth_gbs(B, Hq, Hkv, Sq, Skv, Dq, Dkv, fa_median)
    speedup = sdpa_median / fa_median if fa_median > 0 else float("inf")

    return dict(
        fa_ms=fa_median,
        fa_mean=fa_mean,
        fa_min=fa_min,
        fa_max=fa_max,
        sdpa_ms=sdpa_median,
        speedup=speedup,
        tflops_fa=tfl_fa,
        bw_fa=bw_fa,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def _print_result(cfg_label, r, mode):
    tag = "avg" if mode == "kernel" else "median"
    print(f"  flash_attention_fwd  {tag}={r['fa_ms']:8.3f} ms  "
          f"| {r['tflops_fa']:.3f} TFLOPS  "
          f"| {r['bw_fa']:.1f} GB/s")
    if mode == "wall":
        print(f"    (mean={r['fa_mean']:.3f} ms  min={r['fa_min']:.3f} ms  max={r['fa_max']:.3f} ms)")
    print(f"  pytorch SDPA         {tag}={r['sdpa_ms']:8.3f} ms")
    direction = "faster" if r["speedup"] > 1 else "slower"
    print(f"  speedup vs SDPA:  {r['speedup']:.3f}x {direction}")


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

# Full benchmark matrix.  Each entry is (B, Hq, Hkv, Sq, Skv, Dq, Dkv).
# When Dq == Dkv, head_dim is uniform; [192, 128] means Q-dim=192, KV-dim=128.
# seqlen_pairs: [Sq, Skv] — currently Sq == Skv for all cases (square attention).
_BATCHES = [1, 4, 8, 16, 32]
_SEQLEN_PAIRS = [
    (32, 8192),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
_HEAD_DIMS = [(64, 64),  # uniform 64
              (96, 96),  # uniform 96
              (128, 128),  # uniform 128
              # (192, 128),  # unsupported: Dq != Dkv (kernel requires Dq == Dkv)
              ]
_HEAD_PAIRS = [(16, 16),  # MHA
               (16, 8),  # GQA 2:1
               (16, 4),  # GQA 4:1
               (16, 1),  # MQA
               ]


def _build_sweep_configs():
    """Return list of (B, Hq, Hkv, Sq, Skv, Dq, Dkv) tuples."""
    configs = []
    for B in _BATCHES:
        for (Sq, Skv) in _SEQLEN_PAIRS:
            for (Dq, Dkv) in _HEAD_DIMS:
                for (Hq, Hkv) in _HEAD_PAIRS:
                    configs.append((B, Hq, Hkv, Sq, Skv, Dq, Dkv))
    return configs


_SWEEP_CONFIGS = _build_sweep_configs()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark flash_attention_fwd (fa_triton_arch)")
    p.add_argument("--B", type=int, default=_DEFAULT_B, help="batch size")
    p.add_argument("--S", type=int, default=_DEFAULT_S, help="sequence length (used for both Sq and Skv)")
    p.add_argument("--H", type=int, default=_DEFAULT_H, help="number of heads (shorthand for --q-heads)")
    p.add_argument("--q-heads", type=int, default=None, help="query heads (overrides --H)")
    p.add_argument("--kv-heads", type=int, default=None, help="key/value heads (default: same as q-heads)")
    p.add_argument("--D", type=int, default=_DEFAULT_D,
                   help="Q head dim (kernel now supports variable DIM via constexpr parameter)")
    p.add_argument("--kv-dim", dest="Dkv", type=int, default=None,
                   help="KV head dim (default: same as --D).  Use e.g. --D 192 --kv-dim 128 for split Q/KV dims.")
    p.add_argument("--causal", action="store_true", help="causal masking")
    p.add_argument("--combine-batch", type=int, default=_DEFAULT_COMBINE_BATCH,
                   help="combine_batch passed to flash_attention_fwd")
    p.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP, help="warmup iterations")
    p.add_argument("--rep", type=int, default=_DEFAULT_REP, help="measurement iterations (active runs for kernel mode)")
    p.add_argument("--mode", choices=["wall", "kernel"], default="kernel",
                   help="wall: perf_counter+sync; kernel: do_bench_npu (mspti/profiler)")
    p.add_argument("--no-check", action="store_true", help="skip correctness verification")
    p.add_argument("--sweep", action="store_true",
                   help="run full sweep over batches × seqlen_pairs × head_dims × head_pairs")
    p.add_argument("--sweep-verify-only", action="store_true",
                   help="sweep mode: run correctness check only (no benchmarking)")
    return p.parse_args()


def main():
    args = _parse_args()
    device = _device()

    Hq = args.q_heads or args.H
    Hkv = args.kv_heads or Hq

    print(f"Device : {device}")
    print(f"Mode   : {args.mode}" +
          (" (do_bench_npu, mspti→profiler)" if args.mode == "kernel" else " (perf_counter+sync)"))
    print(f"Warmup : {args.warmup}   Rep/Active: {args.rep}")
    print(f"D      : {args.D}   causal={args.causal}   combine_batch={args.combine_batch}")
    print()

    if args.sweep or args.sweep_verify_only:
        # ---- sweep mode: table over all (B, Hq, Hkv, Sq, Skv, Dq, Dkv) ----
        if args.sweep_verify_only:
            print("=" * 80)
            print("CORRECTNESS VERIFICATION MODE (no benchmarking)")
            print("=" * 80)
            print()

        hdr = (f"{'B':>4} {'Hq':>4} {'Hkv':>4} {'Sq':>6} {'Skv':>6} {'Dq':>4} {'Dkv':>4} "
               f"{'FA(ms)':>9} {'SDPA(ms)':>10} {'speedup':>8} "
               f"{'TFLOPS':>8} {'BW(GB/s)':>10}")
        sep = "-" * len(hdr)

        if not args.sweep_verify_only:
            print(hdr)
            print(sep)

        skipped = 0
        passed = 0
        failed = 0

        for (sB, sHq, sHkv, sSq, sSkv, sDq, sDkv) in _SWEEP_CONFIGS:
            cfg_label = f"B={sB} Hq={sHq} Hkv={sHkv} Sq={sSq} Skv={sSkv} Dq={sDq} Dkv={sDkv}"
            try:
                r = run_benchmark(
                    B=sB,
                    Hq=sHq,
                    Hkv=sHkv,
                    Sq=sSq,
                    Skv=sSkv,
                    Dq=sDq,
                    Dkv=sDkv,
                    combine_batch=args.combine_batch,
                    is_causal=args.causal,
                    mode=args.mode,
                    warmup=args.warmup,
                    rep=args.rep,
                    no_check=args.sweep_verify_only is False,  # verify in verify-only mode
                    device=device,
                )
                if args.sweep_verify_only:
                    passed += 1
                    print(f"✓ {cfg_label}")
                else:
                    direction = "↑" if r["speedup"] > 1 else "↓"
                    print(f"{sB:>4} {sHq:>4} {sHkv:>4} {sSq:>6} {sSkv:>6} {sDq:>4} {sDkv:>4} "
                          f"{r['fa_ms']:>9.3f} {r['sdpa_ms']:>10.3f} "
                          f"{r['speedup']:>7.3f}{direction} "
                          f"{r['tflops_fa']:>8.3f} {r['bw_fa']:>10.1f}")
            except AssertionError as exc:
                skipped += 1
                if args.sweep_verify_only:
                    print(f"⊘ {cfg_label}  SKIP: {exc}")
                else:
                    print(f"{sB:>4} {sHq:>4} {sHkv:>4} {sSq:>6} {sSkv:>6} {sDq:>4} {sDkv:>4} "
                          f"  SKIP (unsupported config: {exc})")
            except Exception as exc:
                failed += 1
                if args.sweep_verify_only:
                    print(f"✗ {cfg_label}  ERROR: {exc}")
                else:
                    print(f"{sB:>4} {sHq:>4} {sHkv:>4} {sSq:>6} {sSkv:>6} {sDq:>4} {sDkv:>4} "
                          f"  ERROR: {exc}")

        if not args.sweep_verify_only:
            print(sep)
        else:
            print()
            print("=" * 80)

        total = len(_SWEEP_CONFIGS)
        print(f"Total configs: {total}  |  Skipped (unsupported): {skipped}  |  "
              f"Ran: {total - skipped}")
        if args.sweep_verify_only:
            print(f"Passed: {passed}  |  Failed: {failed}")
        return

    # ---- single-config mode -------------------------------------------------
    cfg_label = f"B={args.B} Hq={Hq} Hkv={Hkv} Sq={args.S} Skv={args.S} Dq={args.D} Dkv={args.Dkv or args.D}"
    print(f"Config : {cfg_label}")
    print()

    r = run_benchmark(
        B=args.B,
        Hq=Hq,
        Hkv=Hkv,
        Sq=args.S,
        Skv=args.S,
        Dq=args.D,
        Dkv=args.Dkv if args.Dkv is not None else args.D,
        combine_batch=args.combine_batch,
        is_causal=args.causal,
        mode=args.mode,
        warmup=args.warmup,
        rep=args.rep,
        no_check=args.no_check,
        device=device,
    )
    _print_result(cfg_label, r, args.mode)
    print()


if __name__ == "__main__":
    main()
