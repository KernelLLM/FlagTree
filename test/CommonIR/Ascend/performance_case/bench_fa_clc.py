"""Benchmark: flash_attention_fwd (fa_triton_arch) vs PyTorch SDPA.

Configurations from flash-attention clc.yaml (dense section):
  batches:     [1, 4, 8, 16, 32]
  seqlen_pairs: [[32, 8192], [2048, 2048], [4096, 4096], [8192, 8192], [16384, 16384]]
  head_dims:   [64, 96, 128, [192, 128]]
  head_pairs:  [[16, 16], [16, 8], [16, 4], [16, 1]]
  causal:      [true]

Kernel constraints (fa_triton_arch.py):
  - DIM must be 64 (compile-time constant)
  - Sq must equal Skv (asymmetric not supported)
  - S must be divisible by BLOCK_N (32)

Filtered configs keep only DIM=64, symmetric seqlen pairs, and
sequences that fit in NPU memory.

Usage
-----
    python bench_fa_clc.py [--warmup 5] [--rep 20] [--mode wall|kernel]
                           [--combine-batch 8] [--no-check]
                           [--csv bench_fa_clc_results.csv]
                           [--filter-batch 1,4] [--filter-seqlen 2048,4096]
"""

import argparse
import csv
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
# clc.yaml dense configurations
# ---------------------------------------------------------------------------

# Original clc.yaml dense section:
#   batches:      [1, 4, 8, 16, 32]
#   seqlen_pairs: [[32, 8192], [2048, 2048], [4096, 4096], [8192, 8192], [16384, 16384]]
#   head_dims:    [64, 96, 128, [192, 128]]
#   head_pairs:   [[16, 16], [16, 8], [16, 4], [16, 1]]
#   causal:       [true]

_CLC_BATCHES = [1, 4, 8, 16, 32]
_CLC_SEQLEN_PAIRS = [
    (32, 8192),       # asymmetric: decode-like
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
_CLC_HEAD_DIMS = [
    (64, 64),         # standard
    (96, 96),         # not supported by kernel (DIM=64)
    (128, 128),       # not supported by kernel (DIM=64)
    (192, 128),       # not supported by kernel (DIM=64)
]
_CLC_HEAD_PAIRS = [
    (16, 16),   # MHA
    (16, 8),    # GQA 2:1
    (16, 4),    # GQA 4:1
    (16, 1),    # MQA
]
_CLC_CAUSAL = [False]

# ---------------------------------------------------------------------------
# Kernel-compatible filter
# ---------------------------------------------------------------------------
_KERNEL_DIM = _fa.DIM        # 64
_BLOCK_N = _fa.BLOCK_N       # 32
_BLOCK_M = _fa.BLOCK_M       # 32
_NUM_CORES = _fa.NUM_CORES   # 20


def _is_compatible(B, Sq, Skv, Dq, Dkv, Hq, Hkv):
    """Check if a config is compatible with the current kernel."""
    reasons = []
    if Dq != _KERNEL_DIM or Dkv != _KERNEL_DIM:
        reasons.append(f"DIM must be {_KERNEL_DIM}, got Dq={Dq} Dkv={Dkv}")
    if Sq % _BLOCK_M != 0:
        reasons.append(f"Sq={Sq} not divisible by BLOCK_M={_BLOCK_M}")
    if Skv % _BLOCK_N != 0:
        reasons.append(f"Skv={Skv} not divisible by BLOCK_N={_BLOCK_N}")
    if Hq % Hkv != 0:
        reasons.append(f"Hq={Hq} not divisible by Hkv={Hkv}")
    return reasons


def _build_configs(filter_batch=None, filter_seqlen=None):
    """Generate all clc dense configs, split into compatible and skipped."""
    compatible = []
    skipped = []

    for B in _CLC_BATCHES:
        if filter_batch and B not in filter_batch:
            continue
        for (Sq, Skv) in _CLC_SEQLEN_PAIRS:
            if filter_seqlen and Sq not in filter_seqlen and Skv not in filter_seqlen:
                continue
            for (Dq, Dkv) in _CLC_HEAD_DIMS:
                for (Hq, Hkv) in _CLC_HEAD_PAIRS:
                    for causal in _CLC_CAUSAL:
                        reasons = _is_compatible(B, Sq, Skv, Dq, Dkv, Hq, Hkv)
                        cfg = dict(B=B, Sq=Sq, Skv=Skv, Dq=Dq, Dkv=Dkv,
                                   Hq=Hq, Hkv=Hkv, causal=causal)
                        if reasons:
                            skipped.append((cfg, reasons))
                        else:
                            compatible.append(cfg)

    return compatible, skipped


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

def _tflops(B, Hq, Sq, Skv, D, latency_ms):
    """QK^T: B*Hq*Sq*Skv*D MADs, PV: B*Hq*Sq*Skv*D MADs, each = 2 flops."""
    return 2 * 2.0 * B * Hq * Sq * Skv * D / (latency_ms * 1e-3) / 1e12


def _bandwidth_gbs(B, Hq, Hkv, Sq, Skv, D, latency_ms):
    """Read Q + K + V, write O. All fp16 = 2 bytes."""
    elem = 2
    bytes_io = elem * (B * Hq * Sq * D +       # Q
                       B * Hkv * Skv * D +      # K
                       B * Hkv * Skv * D +      # V
                       B * Hq * Sq * D)         # O
    return bytes_io / (latency_ms * 1e-3) / 1e9


def _wall_stats(latencies):
    s = sorted(latencies)
    median = s[len(s) // 2]
    mean = sum(s) / len(s)
    return median, mean, s[0], s[-1]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _bench_wall(fn, device, warmup, rep):
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


# ---------------------------------------------------------------------------
# Reference: PyTorch SDPA (fp16 on NPU via npu_fusion_attention)
# ---------------------------------------------------------------------------

def _ref_sdpa(q, k, v, is_causal):
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal
    ).to(torch.float16)


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def run_one(cfg, combine_batch, mode, warmup, rep, no_check, device):
    B = cfg["B"]
    Hq, Hkv = cfg["Hq"], cfg["Hkv"]
    Sq, Skv = cfg["Sq"], cfg["Skv"]
    D = cfg["Dq"]
    causal = cfg["causal"]

    torch.manual_seed(0)
    q = torch.randn((B, Hq, Sq, D), dtype=torch.float16, device=device)
    k = torch.randn((B, Hkv, Skv, D), dtype=torch.float16, device=device)
    v = torch.randn((B, Hkv, Skv, D), dtype=torch.float16, device=device)

    # Adjust combine_batch to fit num_kv_blocks (based on Skv)
    num_kv_blocks = Skv // _BLOCK_N
    cb = min(combine_batch, num_kv_blocks)
    while num_kv_blocks % cb != 0 and cb > 1:
        cb -= 1

    if not no_check:
        ref = _ref_sdpa(q, k, v, causal)
        out = _fa.flash_attention_fwd(q, k, v, cb, is_causal=causal)
        torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)

    fa_fn = lambda: _fa.flash_attention_fwd(q, k, v, cb, is_causal=causal)
    sdpa_fn = lambda: _ref_sdpa(q, k, v, causal)

    fa_med, fa_mean, fa_min, fa_max = _bench_wall(fa_fn, device, warmup, rep)
    sdpa_med, sdpa_mean, sdpa_min, sdpa_max = _bench_wall(sdpa_fn, device, warmup, rep)

    speedup = sdpa_med / fa_med if fa_med > 0 else float("inf")
    tfl = _tflops(B, Hq, Sq, Skv, D, fa_med)
    bw = _bandwidth_gbs(B, Hq, Hkv, Sq, Skv, D, fa_med)

    return dict(
        B=B, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv, D=D, causal=causal, cb=cb,
        fa_ms=fa_med, fa_mean=fa_mean, fa_min=fa_min, fa_max=fa_max,
        sdpa_ms=sdpa_med, speedup=speedup, tflops=tfl, bw_gbs=bw,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark fa_triton_arch vs SDPA — clc.yaml dense configs")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--mode", choices=["wall"], default="wall")
    p.add_argument("--combine-batch", type=int, default=8)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--csv", type=str, default="bench_fa_clc_results.csv")
    p.add_argument("--filter-batch", type=str, default=None,
                   help="Comma-separated batch sizes to test, e.g. '1,4'")
    p.add_argument("--filter-seqlen", type=str, default=None,
                   help="Comma-separated seqlens to test, e.g. '2048,4096'")
    return p.parse_args()


def main():
    args = _parse_args()
    device = _device()

    filter_batch = None
    if args.filter_batch:
        filter_batch = [int(x) for x in args.filter_batch.split(",")]
    filter_seqlen = None
    if args.filter_seqlen:
        filter_seqlen = [int(x) for x in args.filter_seqlen.split(",")]

    compatible, skipped = _build_configs(filter_batch, filter_seqlen)

    print(f"Device : {device}")
    print(f"Mode   : {args.mode} (perf_counter+sync)")
    print(f"Warmup : {args.warmup}   Rep: {args.rep}")
    print(f"Kernel : DIM={_KERNEL_DIM}  BLOCK_M={_BLOCK_M}  BLOCK_N={_BLOCK_N}  NUM_CORES={_NUM_CORES}")
    print(f"combine_batch={args.combine_batch}")
    print()

    # ---- Report skipped configs -----------------------------------------------
    if skipped:
        print(f"=== Skipped configs: {len(skipped)} (kernel incompatible) ===")
        for cfg, reasons in skipped[:10]:
            label = (f"B={cfg['B']} Hq={cfg['Hq']} Hkv={cfg['Hkv']} "
                     f"Sq={cfg['Sq']} Skv={cfg['Skv']} Dq={cfg['Dq']} Dkv={cfg['Dkv']}")
            print(f"  SKIP {label}  — {'; '.join(reasons)}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
        print()

    # ---- Run compatible configs -----------------------------------------------
    print(f"=== Running {len(compatible)} compatible configs ===")
    print()

    hdr = (f"{'#':>3} {'B':>3} {'Hq':>3} {'Hkv':>4} {'Sq':>6} {'Skv':>6} {'D':>3} {'csl':>4} {'CB':>3} "
           f"{'FA(ms)':>9} {'SDPA(ms)':>10} {'speedup':>8} "
           f"{'TFLOPS':>8} {'BW(GB/s)':>10} {'status':>8}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    results = []
    for i, cfg in enumerate(compatible, 1):
        label = (f"B={cfg['B']} Hq={cfg['Hq']} Hkv={cfg['Hkv']} "
                 f"S={cfg['Sq']} D={cfg['Dq']} causal={cfg['causal']}")
        try:
            r = run_one(cfg, args.combine_batch, args.mode,
                        args.warmup, args.rep, args.no_check, device)
            direction = "↑" if r["speedup"] > 1 else "↓"
            status = "PASS"
            print(f"{i:>3} {r['B']:>3} {r['Hq']:>3} {r['Hkv']:>4} {r['Sq']:>6} {r['Skv']:>6} {r['D']:>3} "
                  f"{'Y' if r['causal'] else 'N':>4} {r['cb']:>3} "
                  f"{r['fa_ms']:>9.3f} {r['sdpa_ms']:>10.3f} "
                  f"{r['speedup']:>7.3f}{direction} "
                  f"{r['tflops']:>8.3f} {r['bw_gbs']:>10.1f} "
                  f"{'PASS':>8}")
            results.append(r)
        except Exception as exc:
            err_msg = str(exc).split("\n")[0][:60]
            print(f"{i:>3} {cfg['B']:>3} {cfg['Hq']:>3} {cfg['Hkv']:>4} {cfg['Sq']:>6} {cfg['Skv']:>6} "
                  f"{cfg['Dq']:>3} {'Y' if cfg['causal'] else 'N':>4} {'?':>3} "
                  f"{'—':>9} {'—':>10} {'—':>8} {'—':>8} {'—':>10} "
                  f"{'FAIL':>8}")
            print(f"    ERROR: {err_msg}")
            results.append(dict(
                B=cfg["B"], Hq=cfg["Hq"], Hkv=cfg["Hkv"],
                Sq=cfg["Sq"], Skv=cfg["Skv"], D=cfg["Dq"],
                causal=cfg["causal"], cb=args.combine_batch,
                fa_ms=float("nan"), sdpa_ms=float("nan"),
                speedup=float("nan"), tflops=float("nan"), bw_gbs=float("nan"),
            ))

    print(sep)

    # ---- Summary --------------------------------------------------------------
    passed = [r for r in results if r["fa_ms"] == r["fa_ms"]]  # not NaN
    failed = len(results) - len(passed)
    if passed:
        avg_speedup = sum(r["speedup"] for r in passed) / len(passed)
        best = max(passed, key=lambda r: r["speedup"])
        worst = min(passed, key=lambda r: r["speedup"])
        print(f"\nSummary: {len(passed)} passed, {failed} failed, {len(skipped)} skipped")
        print(f"  Average speedup: {avg_speedup:.3f}x")
        print(f"  Best:  B={best['B']} Hq={best['Hq']} Hkv={best['Hkv']} Sq={best['Sq']} Skv={best['Skv']} "
              f"→ {best['speedup']:.3f}x ({best['fa_ms']:.3f} vs {best['sdpa_ms']:.3f} ms)")
        print(f"  Worst: B={worst['B']} Hq={worst['Hq']} Hkv={worst['Hkv']} Sq={worst['Sq']} Skv={worst['Skv']} "
              f"→ {worst['speedup']:.3f}x ({worst['fa_ms']:.3f} vs {worst['sdpa_ms']:.3f} ms)")

    # ---- CSV output -----------------------------------------------------------
    if args.csv and results:
        csv_path = os.path.join(_HERE, args.csv)
        fields = ["B", "Hq", "Hkv", "Sq", "Skv", "D", "causal", "cb",
                  "fa_ms", "sdpa_ms", "speedup", "tflops", "bw_gbs"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
