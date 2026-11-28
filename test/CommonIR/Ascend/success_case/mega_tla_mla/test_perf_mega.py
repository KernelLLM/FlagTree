#!/usr/bin/env python3
"""Full mega_mla performance comparison: TLA fused attention vs cannbot-dsl.

Tests multiple shapes covering production decode scenarios.
Each shape runs warmup + timed iterations with NPU event timing.
"""
import sys, os, statistics, argparse

TLA_UT = "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut"
sys.path.insert(0, TLA_UT)
MEGA_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla"
sys.path.insert(0, MEGA_DIR)
TEST_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/test/attention/mega_mla"
sys.path.insert(0, TEST_DIR)

import torch
import torch_npu
import triton.language as tl
import triton.language.extra.tla as tla

torch.npu.set_device(0)
GRID = int(torch.npu.get_device_properties(0).cube_core_num)

from test_fused_lag import fused_attn_lag
from test_precision_stages import tla_qk_tiled, tla_softmax, tla_pv_tiled, BASE_N, BASE_K
from mega_mla import MegaMlaOp
from test_mega_mla import quantize_fp8, quantize_fp4


def bench(fn, warmup=20, repeat=200):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    trim = max(1, repeat // 10)
    trimmed = times[trim:-trim]
    return statistics.median(times), min(times), statistics.mean(trimmed)


def make_dsl_op(H, K, T):
    """Build and warm the cannbot-dsl fused mega_mla operator."""
    ps, wl = 128, 128
    op = MegaMlaOp(k=K, page_size=ps, win_len=wl, grid=GRID, mode="attn",
                   tokens=T, heads=H, cmp_decode_strategy="indexed")
    torch.manual_seed(31)
    x = (torch.randn(T, 5120) * 0.1).to(torch.bfloat16).npu()
    gfq = (torch.rand(1, 1280) + 0.5).npu()
    gfk = (torch.rand(1, 512) + 0.5).npu()
    nz = lambda t: torch_npu.npu_format_cast(t.npu(), torch_npu.Format.FRACTAL_NZ)
    wa = nz((torch.randn(1280, 5120) * 0.05).to(torch.float8_e4m3fn))
    wkv = nz((torch.randn(512, 5120) * 0.05).to(torch.float8_e4m3fn))
    wb = nz((torch.randn(H * 512, 1280) * 0.05).to(torch.float8_e4m3fn))
    sa = torch.full((1280, 80, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    skv = torch.full((512, 80, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    sb = torch.full((H * 512, 20, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    D = 512
    wkc = quantize_fp8((torch.randn(2, ps, 1, D) * 0.1).to(torch.bfloat16)).npu()
    ckc = quantize_fp4((torch.randn(4, ps, 1, D) * 0.1).to(torch.bfloat16)).npu()
    wi = torch.randint(ps, 2 * ps, (T, wl), dtype=torch.int32).npu()
    ci = torch.randint(0, 4 * ps, (T, K - wl), dtype=torch.int32).npu()
    snk = torch.randn(1, H, dtype=torch.float32).npu()
    cos = torch.randn(T, 32).npu()
    sin = torch.randn(T, 32).npu()
    slot = torch.arange(ps, ps + T, dtype=torch.int32).npu()
    # First call compiles
    op.run(x, gfq, gfk, wa, wkv, wb, sa, skv, sb, wkc, ckc,
           wi, ci, snk, cos, sin, slot)
    torch.npu.synchronize()
    args = (x, gfq, gfk, wa, wkv, wb, sa, skv, sb, wkc, ckc,
            wi, ci, snk, cos, sin, slot)
    return op, args


def run_shape(H, K, D, T, W, R):
    """Benchmark all three paths for one shape."""
    BN, BK = 64, 128
    torch.manual_seed(42)
    q = (torch.randn(T * H, D) * 0.1).to(torch.bfloat16).npu()
    kv = (torch.randn(T * K, D) * 0.1).to(torch.bfloat16).npu()
    sink = torch.zeros(T * H, dtype=torch.float32, device="npu")

    # Pre-alloc outputs
    s = torch.empty(T * H, K, dtype=torch.float32, device="npu")
    p = torch.empty(T * H, K, dtype=torch.bfloat16, device="npu")
    o_sep = torch.empty(T * H, D, dtype=torch.bfloat16, device="npu")
    o_fused = torch.empty(T * H, D, dtype=torch.bfloat16, device="npu")

    results = {}

    # --- TLA 3-launch (per token, sequential) ---
    def run_3launch():
        for tok in range(T):
            q_t = q[tok * H:(tok + 1) * H]
            kv_t = kv[tok * K:(tok + 1) * K]
            s_t = s[tok * H:(tok + 1) * H]
            p_t = p[tok * H:(tok + 1) * H]
            o_t = o_sep[tok * H:(tok + 1) * H]
            tla_qk_tiled[(1,)](q_t, kv_t, s_t, H=H, KTOTAL=K, D=D,
                               BASE_N_=BN, BASE_K_=BK)
            tla_softmax[(1,)](s_t, sink[tok * H:(tok + 1) * H], p_t, H=H, K=K)
            tla_pv_tiled[(1,)](p_t, kv_t, o_t, H=H, K=K, D=D,
                               PV_BN=BN, PV_BK=BK)

    med, mn, mean = bench(run_3launch, W, R)
    results["tla_3launch"] = (med, mn, mean)

    # --- TLA fused (per token, 1 launch each) ---
    def run_fused():
        for tok in range(T):
            q_t = q[tok * H:(tok + 1) * H]
            kv_t = kv[tok * K:(tok + 1) * K]
            o_t = o_fused[tok * H:(tok + 1) * H]
            fused_attn_lag[(1,)](q_t, kv_t, o_t, H=H, K=K, D=D, BN=BN, BK=BK)

    try:
        med, mn, mean = bench(run_fused, W, R)
        results["tla_fused"] = (med, mn, mean)
    except Exception as e:
        results["tla_fused"] = None

    # --- DSL fused mega_mla (full pipeline) ---
    try:
        op, dsl_args = make_dsl_op(H, K, T)
        def run_dsl():
            op.run(*dsl_args)
        med, mn, mean = bench(run_dsl, W, R)
        results["dsl_fused"] = (med, mn, mean)
    except Exception as e:
        results["dsl_fused"] = None

    return results


def fmt(label, r):
    if r is None:
        return f"  {label:36s}  FAILED"
    med, mn, mean = r
    return f"  {label:36s}  median={med:8.1f}us  min={mn:8.1f}us  mean={mean:8.1f}us"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    W, R = args.warmup, args.repeat

    shapes = [
        # (H, K, D, T, description)
        (16, 640, 512, 1, "H16 K640 T1 (decode, main config)"),
        (16, 640, 512, 8, "H16 K640 T8 (small batch)"),
        (16, 2176, 512, 1, "H16 K2176 T1 (decode, full model K)"),
        (64, 640, 512, 1, "H64 K640 T1 (decode, 64-head)"),
    ]

    print(f"=" * 80)
    print(f"  mega_mla Performance Comparison — Ascend 950, grid={GRID}")
    print(f"  warmup={W}, repeat={R}")
    print(f"=" * 80)

    for H, K, D, T, desc in shapes:
        print(f"\n--- {desc} ---")
        try:
            results = run_shape(H, K, D, T, W, R)
        except Exception as e:
            print(f"  SHAPE FAILED: {str(e).split(chr(10))[0]}")
            continue
        print(fmt("TLA 3-launch (attn only)", results["tla_3launch"]))
        print(fmt("TLA fused attn (1-launch/tok)", results["tla_fused"]))
        print(fmt("DSL fused mega_mla (full pipe)", results["dsl_fused"]))
        if results["tla_fused"] and results["tla_3launch"]:
            speedup = results["tla_3launch"][0] / results["tla_fused"][0]
            print(f"  {'Fusion speedup':36s}  {speedup:.2f}x")
        if results["dsl_fused"] and results["tla_fused"]:
            print(f"  {'Note':36s}  DSL includes projection+cache+gather+RoPE")

    print(f"\n{'=' * 80}")
    print("  TLA attn = QK matmul + softmax + PV matmul (attention only)")
    print("  DSL full = projection P0-P3 + cache decode + gather + RoPE + attention")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
