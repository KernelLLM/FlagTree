#!/usr/bin/env python3
"""Performance comparison: TLA vs cannbot-dsl for QK, softmax, and PV.

Separates compilation from execution timing:
  Phase 1: Warmup / compile — first N calls compile the kernel binary
  Phase 2: Measure — subsequent calls dispatch the cached binary

Usage:
    python3 test_perf_stages.py
    python3 test_perf_stages.py --warmup 20 --repeat 200
"""
import sys, os, argparse, statistics

TLA_UT = "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut"
sys.path.insert(0, TLA_UT)
MEGA_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla"
sys.path.insert(0, MEGA_DIR)

import torch
import triton.language as tl
import triton.language.extra.tla as tla

from test_precision_stages import (
    tla_qk_tiled, tla_softmax, tla_pv_tiled,
    BASE_N, BASE_K,
)


def bench_npu(fn, warmup=20, repeat=100, label=""):
    """NPU event timing. Returns (median_us, min_us, p5_us, p95_us, mean_trimmed_us)."""
    # Phase 1: warmup (includes compilation on first call)
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()

    # Phase 2: measure
    times_us = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)  # ms → us

    times_us.sort()
    n = len(times_us)
    trim = max(1, n // 10)
    trimmed = times_us[trim:-trim] if n > 2 * trim else times_us
    p5 = times_us[max(0, int(n * 0.05))]
    p95 = times_us[min(n - 1, int(n * 0.95))]
    return statistics.median(times_us), min(times_us), p5, p95, statistics.mean(trimmed)


def fmt_row(label, med, mn, p5, p95, mean_t):
    return f"  {label:35s}  median={med:8.1f}us  min={mn:8.1f}us  p5={p5:8.1f}us  p95={p95:8.1f}us  mean={mean_t:8.1f}us"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--k", type=int, default=640)
    parser.add_argument("--d", type=int, default=512)
    args = parser.parse_args()

    import torch_npu
    if not torch.npu.is_available():
        print("SKIP: NPU unavailable"); return
    torch.npu.set_device(0)

    H, K, D = args.heads, args.k, args.d
    W, R = args.warmup, args.repeat

    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16).npu()
    kv = (torch.randn(K, D) * 0.1).to(torch.bfloat16).npu()
    sink = (torch.randn(H) * 0.5).to(torch.float32).npu()
    s = torch.empty(H, K, dtype=torch.float32, device="npu")
    p = torch.empty(H, K, dtype=torch.bfloat16, device="npu")
    o = torch.empty(H, D, dtype=torch.bfloat16, device="npu")
    s2 = torch.empty(H, K, dtype=torch.float32, device="npu")
    o2 = torch.empty(H, D, dtype=torch.bfloat16, device="npu")

    from attention_matmul import QueryKeyMatmulOp, ProbabilityValueMatmulOp
    qk_dsl_op = QueryKeyMatmulOp(H, K, D)
    pv_dsl_op = ProbabilityValueMatmulOp(H, D, K, base_n=64)

    print(f"=== Perf: H={H}, K={K}, D={D}, warmup={W}, repeat={R} ===")
    print(f"    (all times in microseconds, lower is better)\n")

    # ----- QK matmul -----
    print("Stage 1: QK matmul  S[H,K] = Q[H,D] @ KV[K,D]^T")

    def tla_qk():
        tla_qk_tiled[(1,)](q, kv, s, H=H, KTOTAL=K, D=D,
                           BASE_N_=BASE_N, BASE_K_=BASE_K)
    def dsl_qk():
        qk_dsl_op.run(s2, q, kv)

    r = bench_npu(tla_qk, W, R)
    print(fmt_row("TLA  tla_qk_tiled", *r))
    r = bench_npu(dsl_qk, W, R)
    print(fmt_row("DSL  QueryKeyMatmulOp", *r))

    # ----- Softmax -----
    print(f"\nStage 2: Softmax  P[H,K] = softmax(S + sink)")

    # Prepare input
    dsl_qk(); torch.npu.synchronize()

    def tla_sm():
        tla_softmax[(1,)](s2, sink, p, H=H, K=K)

    r = bench_npu(tla_sm, W, R)
    print(fmt_row("TLA  tla_softmax", *r))
    print(f"  {'DSL  (fused, no standalone)':35s}  —")

    # ----- PV matmul -----
    print(f"\nStage 3: PV matmul  O[H,D] = P[H,K] @ KV[K,D]")

    tla_sm(); torch.npu.synchronize()  # prepare P

    def tla_pv():
        tla_pv_tiled[(1,)](p, kv, o, H=H, K=K, D=D,
                           PV_BN=BASE_N, PV_BK=BASE_K)
    def dsl_pv():
        pv_dsl_op.run(o2, p, kv)

    r = bench_npu(tla_pv, W, R)
    print(fmt_row("TLA  tla_pv_tiled", *r))
    r = bench_npu(dsl_pv, W, R)
    print(fmt_row("DSL  ProbabilityValueMatmulOp", *r))

    # ----- Full pipeline -----
    print(f"\n=== Full pipeline (3 separate kernel launches) ===")

    def tla_full():
        tla_qk_tiled[(1,)](q, kv, s, H=H, KTOTAL=K, D=D,
                           BASE_N_=BASE_N, BASE_K_=BASE_K)
        tla_softmax[(1,)](s, sink, p, H=H, K=K)
        tla_pv_tiled[(1,)](p, kv, o, H=H, K=K, D=D,
                           PV_BN=BASE_N, PV_BK=BASE_K)

    def dsl_qkpv():
        qk_dsl_op.run(s2, q, kv)
        pv_dsl_op.run(o2, p, kv)

    r = bench_npu(tla_full, W, R)
    print(fmt_row("TLA  QK + softmax + PV", *r))
    r = bench_npu(dsl_qkpv, W, R)
    print(fmt_row("DSL  QK + PV (no softmax)", *r))

    print()
    print("Note: DSL operators recompile each call via @host launcher.")
    print("The fused mega_mla kernel amortizes compilation and pipelines")
    print("all stages in one NPU dispatch.")


if __name__ == "__main__":
    main()
