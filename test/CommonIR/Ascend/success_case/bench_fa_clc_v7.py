"""Benchmark: fa_triton_arch_v7.flash_attention_fwd vs PyTorch SDPA — clc.yaml dense.

Same structure and metrics as bench_fa_clc.py, retargeted at v7. Runs the dense
(non-causal) half of the sweep by default; `--causal` runs the causal half
instead, through v7's balanced sparse schedule unless `--no-balance` is given.

Configurations from flash-attention clc.yaml (dense section):
  batches:      [1, 4, 8, 16, 32]
  seqlen_pairs: [[32, 8192], [2048, 2048], [4096, 4096], [8192, 8192], [16384, 16384]]
  head_dims:    [64, 96, 128, [192, 128]]
  head_pairs:   [[16, 16], [16, 8], [16, 4], [16, 1]]
  causal:       [false] by default, [true] under --causal

Kernel constraints (fa_triton_arch_v7.py):
  - DIM must be 64 (compile-time constant)
  - Sq must equal Skv (asymmetric not supported)
  - S divisible by BLOCK_M (128) and by BLOCK_N (256)
  - num_kv_blocks divisible by CB, and CB <= MAX_CB (2)

Two references, matching bench_fa_clc.py's split:
  - fp16 SDPA  — the *timed* baseline, so like is compared with like
  - fp32 SDPA  — the accuracy reference only, never timed

Usage
-----
    python bench_fa_clc_v7.py [--causal] [--no-balance]
                              [--warmup 5] [--rep 20] [--no-check]
                              [--autotune] [--config M,N,CB,SV,R,ns]
                              [--outdir result] [--filter-batch 1,4]
                              [--filter-seqlen 2048,4096]
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback

import torch

# ---------------------------------------------------------------------------
# Import flash_attention_fwd from success_case/fa_triton_arch_v7.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import fa_triton_arch_v7 as _fa  # noqa: E402

# ---------------------------------------------------------------------------
# clc.yaml dense configurations
# ---------------------------------------------------------------------------
_CLC_BATCHES = [1, 4, 8, 16, 32]
_CLC_SEQLEN_PAIRS = [
    (32, 8192),  # asymmetric: decode-like
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
_CLC_HEAD_DIMS = [
    (64, 64),  # standard
    (96, 96),  # not supported by kernel (DIM=64)
    (128, 128),  # not supported by kernel (DIM=64)
    (192, 128),  # not supported by kernel (DIM=64)
]
_CLC_HEAD_PAIRS = [
    (16, 16),  # MHA
    (16, 8),  # GQA 2:1
    (16, 4),  # GQA 4:1
    (16, 1),  # MQA
]
_CLC_CAUSAL = [False]  # default; --causal switches this run to [True]

# ---------------------------------------------------------------------------
# Kernel-compatible filter
# ---------------------------------------------------------------------------
_KERNEL_DIM = _fa.DIM
_NUM_CORES = _fa.NUM_CORES


def _is_compatible(B, Sq, Skv, Dq, Dkv, Hq, Hkv, cfg_shape):
    """Reasons this config cannot run, or [] if it can.

    Tile-shape divisibility is asked of the actual launch config, since v7's
    BLOCK_M/BLOCK_N are tunable rather than module constants.
    """
    reasons = []
    if Dq != _KERNEL_DIM or Dkv != _KERNEL_DIM:
        reasons.append(f"DIM must be {_KERNEL_DIM}, got Dq={Dq} Dkv={Dkv}")
    if Sq != Skv:
        reasons.append(f"asymmetric seqlen Sq={Sq} != Skv={Skv} not supported")
    if Hq % Hkv:
        reasons.append(f"Hq={Hq} not divisible by Hkv={Hkv}")
    if Sq == Skv:
        ok, why = _fa.config_is_valid(cfg_shape, Sq)
        if not ok:
            reasons.append(f"config {cfg_shape.label()}: {why}")
    return reasons


def _build_configs(cfg_shape, filter_batch=None, filter_seqlen=None, causal_list=None):
    """Generate all clc dense configs, split into compatible and skipped."""
    causal_list = _CLC_CAUSAL if causal_list is None else causal_list
    compatible, skipped = [], []
    for B in _CLC_BATCHES:
        if filter_batch and B not in filter_batch:
            continue
        for (Sq, Skv) in _CLC_SEQLEN_PAIRS:
            if filter_seqlen and Sq not in filter_seqlen and Skv not in filter_seqlen:
                continue
            for (Dq, Dkv) in _CLC_HEAD_DIMS:
                for (Hq, Hkv) in _CLC_HEAD_PAIRS:
                    for causal in causal_list:
                        reasons = _is_compatible(B, Sq, Skv, Dq, Dkv, Hq, Hkv, cfg_shape)
                        cfg = dict(B=B, Sq=Sq, Skv=Skv, Dq=Dq, Dkv=Dkv, Hq=Hq, Hkv=Hkv, causal=causal)
                        (skipped if reasons else compatible).append((cfg, reasons) if reasons else cfg)
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


def _free(device: str):
    gc.collect()
    if device == "npu":
        torch.npu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _tflops(B, Hq, Sq, Skv, D, latency_ms):
    """QK^T: B*Hq*Sq*Skv*D MADs, PV: B*Hq*Sq*Skv*D MADs, each = 2 flops."""
    return 2 * 2.0 * B * Hq * Sq * Skv * D / (latency_ms * 1e-3) / 1e12


def _bandwidth_gbs(B, Hq, Hkv, Sq, Skv, D, latency_ms):
    """Read Q + K + V, write O. All fp16 = 2 bytes."""
    elem = 2
    bytes_io = elem * (B * Hq * Sq * D +  # Q
                       B * Hkv * Skv * D +  # K
                       B * Hkv * Skv * D +  # V
                       B * Hq * Sq * D)  # O
    return bytes_io / (latency_ms * 1e-3) / 1e9


def _wall_stats(latencies):
    s = sorted(latencies)
    return s[len(s) // 2], sum(s) / len(s), s[0], s[-1]


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
# References
# ---------------------------------------------------------------------------
def _expand_kv(q, k, v):
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    return k, v


def _ref_sdpa(q, k, v, is_causal):
    """fp16 SDPA — the timed baseline."""
    k, v = _expand_kv(q, k, v)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _ref_sdpa_fp32(q, k, v, is_causal):
    """fp32 SDPA — accuracy reference, never timed."""
    k, v = _expand_kv(q, k, v)
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                            is_causal=is_causal).to(torch.float16)


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------
def run_one(cfg, mode, warmup, rep, no_check, device, launch_cfg, autotune, balance=True):
    B = cfg["B"]
    Hq, Hkv = cfg["Hq"], cfg["Hkv"]
    Sq, Skv = cfg["Sq"], cfg["Skv"]
    D = cfg["Dq"]
    causal = cfg["causal"]

    torch.manual_seed(0)
    q = torch.randn((B, Hq, Sq, D), dtype=torch.float16, device=device)
    k = torch.randn((B, Hkv, Skv, D), dtype=torch.float16, device=device)
    v = torch.randn((B, Hkv, Skv, D), dtype=torch.float16, device=device)

    used = launch_cfg
    if autotune:
        used = _fa.autotune_config(q, k, v, is_causal=causal, balance_causal=balance, verbose=False)

    max_err = float("nan")
    check = "skip"
    if not no_check:
        out = _fa.flash_attention_fwd(q, k, v, is_causal=causal, balance_causal=balance, config=used)
        try:
            ref = _ref_sdpa_fp32(q, k, v, causal)
            ref_kind = "fp32"
        except RuntimeError:
            # fp32 SDPA needs 2x the activation memory; fall back at the top shapes.
            _free(device)
            ref = _ref_sdpa(q, k, v, causal)
            ref_kind = "fp16"
        max_err = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
        check = ref_kind
        del ref, out
        _free(device)

    fa_med, fa_mean, fa_min, fa_max = _bench_wall(
        lambda: _fa.flash_attention_fwd(q, k, v, is_causal=causal, balance_causal=balance, config=used), device,
        warmup, rep)
    sdpa_med, sdpa_mean, sdpa_min, sdpa_max = _bench_wall(lambda: _ref_sdpa(q, k, v, causal), device, warmup, rep)

    speedup = sdpa_med / fa_med if fa_med > 0 else float("inf")
    return dict(
        B=B, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv, D=D, causal=causal, cb=used.combine_batch, config=used.label(),
        fa_ms=fa_med, fa_mean=fa_mean, fa_min=fa_min, fa_max=fa_max, sdpa_ms=sdpa_med, sdpa_mean=sdpa_mean,
        speedup=speedup, tflops=_tflops(B, Hq, Sq, Skv, D, fa_med),
        sdpa_tflops=_tflops(B, Hq, Sq, Skv, D, sdpa_med), bw_gbs=_bandwidth_gbs(B, Hq, Hkv, Sq, Skv, D, fa_med),
        max_err=max_err, check=check)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark fa_triton_arch_v7 vs SDPA — clc.yaml dense, non-causal")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--mode", choices=["wall"], default="wall")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--autotune", action="store_true", help="tune the launch config per shape (costs many compiles)")
    p.add_argument("--config", type=str, default=None, help="pin a config: BLOCK_M,BLOCK_N,CB,SUB_VEC_NUM,RING,ns")
    p.add_argument("--outdir", type=str, default="result", help="directory for the CSV / JSON / report")
    p.add_argument("--causal", action="store_true", help="run the causal half of the sweep instead of the dense one")
    p.add_argument("--no-balance", action="store_true",
                   help="causal: contiguous dense split instead of v7's balanced sparse schedule")
    p.add_argument("--tag", type=str, default=None, help="filename tag; defaults to causal/noncausal")
    p.add_argument("--filter-batch", type=str, default=None, help="Comma-separated batch sizes, e.g. '1,4'")
    p.add_argument("--filter-seqlen", type=str, default=None, help="Comma-separated seqlens, e.g. '2048,4096'")
    return p.parse_args()


def main():
    args = _parse_args()
    device = _device()

    causal_list = [True] if args.causal else _CLC_CAUSAL
    balance = not args.no_balance
    tag = args.tag or ("causal" if args.causal else "noncausal")

    launch_cfg = _fa.DEFAULT_CONFIG
    if args.config:
        launch_cfg = _fa.FAConfig(*[int(x) for x in args.config.replace(" ", "").split(",")])

    filter_batch = [int(x) for x in args.filter_batch.split(",")] if args.filter_batch else None
    filter_seqlen = [int(x) for x in args.filter_seqlen.split(",")] if args.filter_seqlen else None

    compatible, skipped = _build_configs(launch_cfg, filter_batch, filter_seqlen, causal_list)

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"bench_fa_clc_v7_{tag}_{stamp}"

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"Device : {device}")
    emit(f"Mode   : {args.mode} (perf_counter+sync)")
    emit(f"Warmup : {args.warmup}   Rep: {args.rep}")
    emit(f"Kernel : fa_triton_arch_v7  DIM={_KERNEL_DIM}  NUM_CORES={_NUM_CORES}")
    emit(f"Config : {'autotune per shape' if args.autotune else launch_cfg.label()}")
    emit(f"Causal : {causal_list}" + (f"  (schedule: {'balanced sparse' if balance else 'dense split'})"
                                      if args.causal else ""))
    emit(f"Ref    : timed baseline = fp16 SDPA;  accuracy = fp32 SDPA")
    emit()

    if skipped:
        emit(f"=== Skipped configs: {len(skipped)} (kernel incompatible) ===")
        shown = {}
        for cfg, reasons in skipped:
            key = "; ".join(reasons)
            shown.setdefault(key, 0)
            shown[key] += 1
        for key, n in shown.items():
            emit(f"  SKIP x{n:<4} — {key}")
        emit()

    emit(f"=== Running {len(compatible)} compatible configs ===")
    emit()

    hdr = (f"{'#':>3} {'B':>3} {'Hq':>3} {'Hkv':>4} {'Sq':>6} {'Skv':>6} {'D':>3} {'csl':>4} {'CB':>3} "
           f"{'FA(ms)':>9} {'SDPA(ms)':>10} {'speedup':>8} "
           f"{'TFLOPS':>8} {'BW(GB/s)':>10} {'maxerr':>8} {'status':>7}")
    sep = "-" * len(hdr)
    emit(hdr)
    emit(sep)

    results = []
    for i, cfg in enumerate(compatible, 1):
        try:
            r = run_one(cfg, args.mode, args.warmup, args.rep, args.no_check, device, launch_cfg, args.autotune,
                        balance)
            direction = "^" if r["speedup"] > 1 else "v"
            emit(f"{i:>3} {r['B']:>3} {r['Hq']:>3} {r['Hkv']:>4} {r['Sq']:>6} {r['Skv']:>6} {r['D']:>3} "
                 f"{'Y' if r['causal'] else 'N':>4} {r['cb']:>3} "
                 f"{r['fa_ms']:>9.3f} {r['sdpa_ms']:>10.3f} {r['speedup']:>7.3f}{direction} "
                 f"{r['tflops']:>8.3f} {r['bw_gbs']:>10.1f} {r['max_err']:>8.2e} {'PASS':>7}")
            results.append(r)
        except Exception as exc:
            err_msg = str(exc).split("\n")[0][:70]
            emit(f"{i:>3} {cfg['B']:>3} {cfg['Hq']:>3} {cfg['Hkv']:>4} {cfg['Sq']:>6} {cfg['Skv']:>6} "
                 f"{cfg['Dq']:>3} {'Y' if cfg['causal'] else 'N':>4} {'?':>3} "
                 f"{'-':>9} {'-':>10} {'-':>8} {'-':>8} {'-':>10} {'-':>8} {'FAIL':>7}")
            emit(f"    ERROR: {err_msg}")
            fail_log = os.path.join(outdir, f"{base}_fail_B{cfg['B']}_S{cfg['Sq']}_H{cfg['Hq']}x{cfg['Hkv']}.log")
            with open(fail_log, "w") as f:
                f.write(f"config: {cfg}\nlaunch: {launch_cfg}\n\n")
                f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            emit(f"    full error -> {fail_log}")
            results.append(
                dict(B=cfg["B"], Hq=cfg["Hq"], Hkv=cfg["Hkv"], Sq=cfg["Sq"], Skv=cfg["Skv"], D=cfg["Dq"],
                     causal=cfg["causal"], cb=launch_cfg.combine_batch, config=launch_cfg.label(),
                     fa_ms=float("nan"), sdpa_ms=float("nan"), speedup=float("nan"), tflops=float("nan"),
                     sdpa_tflops=float("nan"), bw_gbs=float("nan"), max_err=float("nan"), check="fail"))
        _free(device)

    emit(sep)

    # ---- Summary --------------------------------------------------------------
    passed = [r for r in results if r["fa_ms"] == r["fa_ms"]]
    failed = len(results) - len(passed)
    if passed:
        avg = sum(r["speedup"] for r in passed) / len(passed)
        geo = 1.0
        for r in passed:
            geo *= r["speedup"]
        geo **= 1.0 / len(passed)
        best = max(passed, key=lambda r: r["speedup"])
        worst = min(passed, key=lambda r: r["speedup"])
        peak = max(passed, key=lambda r: r["tflops"])
        emit()
        emit(f"Summary: {len(passed)} passed, {failed} failed, {len(skipped)} skipped")
        emit(f"  Average speedup: {avg:.3f}x   geomean: {geo:.3f}x")
        emit(f"  Peak v7:  {peak['tflops']:.2f} TFLOP/s at B={peak['B']} Hkv={peak['Hkv']} S={peak['Sq']}")
        emit(f"  Best:  B={best['B']} Hq={best['Hq']} Hkv={best['Hkv']} S={best['Sq']} "
             f"-> {best['speedup']:.3f}x ({best['fa_ms']:.3f} vs {best['sdpa_ms']:.3f} ms)")
        emit(f"  Worst: B={worst['B']} Hq={worst['Hq']} Hkv={worst['Hkv']} S={worst['Sq']} "
             f"-> {worst['speedup']:.3f}x ({worst['fa_ms']:.3f} vs {worst['sdpa_ms']:.3f} ms)")
        errs = [r["max_err"] for r in passed if r["max_err"] == r["max_err"]]
        if errs:
            emit(f"  Max abs error across configs: {max(errs):.2e}")

    # ---- Outputs --------------------------------------------------------------
    csv_path = os.path.join(outdir, base + ".csv")
    fields = [
        "B", "Hq", "Hkv", "Sq", "Skv", "D", "causal", "cb", "config", "fa_ms", "fa_mean", "fa_min", "fa_max",
        "sdpa_ms", "speedup", "tflops", "sdpa_tflops", "bw_gbs", "max_err", "check"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    json_path = os.path.join(outdir, base + ".json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "kernel": "fa_triton_arch_v7",
                "device": device,
                "warmup": args.warmup,
                "rep": args.rep,
                "causal": causal_list,
                "balance_causal": balance,
                "launch_config": (None if args.autotune else launch_cfg._asdict()),
                "autotune": bool(args.autotune),
                "timed_baseline": "fp16 SDPA",
                "accuracy_reference": "fp32 SDPA",
                "results": results,
            }, f, indent=1)

    txt_path = os.path.join(outdir, base + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nCSV    -> {csv_path}")
    print(f"JSON   -> {json_path}")
    print(f"Report -> {txt_path}")


if __name__ == "__main__":
    main()
