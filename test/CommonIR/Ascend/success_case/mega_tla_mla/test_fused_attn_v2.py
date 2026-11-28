#!/usr/bin/env python3
"""Fused TLA attention kernel with correct cross-core PV pipeline.

Pattern adapted from dsa_kernel.py:
  - Pre-allocated L1 slots for Q, KV, P
  - Balanced flag pairs: every set has a matching wait
  - Sequential N-tile loop (not yet pipelined across tiles)
  - Cube fixpipe dual-dst row-split for score/PV delivery to both AIVs

Per N-tile:
  Cube : load KV[BN,D] → QK(K-tiled) → fixpipe score to UB → PV → fixpipe partial O to UB
  Vector: softmax(score) → P → copy P to L1 (between QK fixpipe and PV)
  Vector: accumulate partial O into FP32 running sum (after PV fixpipe)

Usage:
    python3 test_fused_attn_v2.py
    python3 test_fused_attn_v2.py --warmup 20 --repeat 200
"""
import sys, os, argparse, statistics

TLA_UT = "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut"
sys.path.insert(0, TLA_UT)
MEGA_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla"
sys.path.insert(0, MEGA_DIR)

import torch
import triton.language as tl
import triton.language.extra.tla as tla

# Cross-core event IDs (balanced set/wait pairs)
FLAG_SCORE = tl.constexpr(0)       # cube(F) → vector(S): score tile in UB
FLAG_SCORE_FREE = tl.constexpr(1)  # vector(V) → cube(S): vector done reading score UB
FLAG_P_READY = tl.constexpr(2)     # vector(MTE3) → cube(S): P tile in L1
FLAG_PV_DONE = tl.constexpr(3)     # cube(F) → vector(S): PV partial in UB
FLAG_PV_FREE = tl.constexpr(4)     # vector(V) → cube(S): vector done folding PV


@tla.jit
def fused_attn_v2(
    Q_ptr, KV_ptr, Sink_ptr, O_ptr,
    H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr,   # N-tile (K rows per pass, 64)
    BK: tl.constexpr,   # K-tile for QK inner loop (128)
):
    """Fused QK + online-softmax + PV in one kernel dispatch.

    Sequential per-tile: cube QK → vector softmax → cube PV → vector accum.
    All buffers pre-allocated. No allocs inside loops.
    Pattern follows dsa_kernel.py for cross-core handshake.
    """
    gm_q = tla.view(Q_ptr, (H, D))
    gm_kv = tla.view(KV_ptr, (K, D))
    gm_o = tla.view(O_ptr, (H, D))

    N_TILES: tl.constexpr = K // BN
    QK_K_TILES: tl.constexpr = D // BK
    VM: tl.constexpr = H // 2

    # All L1/L0C/UB allocations up front (dsa_kernel pattern)
    q_l1 = tla.alloc((BK // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
    kv_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
    p_l1 = tla.alloc((BN // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
    pflat = tla.reshape(p_l1, (BN // 16, H * 16))
    qk_l0c = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")
    pv_l0c = tla.alloc((D // 16, H // 16, 16, 16), tl.float32, space="l0c")
    score = tla.alloc((VM, BN), tl.float32, space="ub")
    pv_ub = tla.alloc((VM, D), tl.float32, space="ub")
    acc = tla.alloc((VM, D), tl.float32, space="ub")
    prob_ub = tla.alloc((BN // 16, VM * 16 + 16), tl.bfloat16, space="ub")
    maximum = tla.alloc((64,), tl.float32, space="ub")
    denom_buf = tla.alloc((64,), tl.float32, space="ub")
    output = tla.alloc((VM, D), tl.bfloat16, space="ub")

    # Init softmax state and accumulator
    with tla.scope.vector():
        with tla.scope.simd():
            zero = tla.simd.vfull(0.0, tl.float32, 64)
            neg_inf = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
            tla.simd.vsts(maximum[0], neg_inf)
            tla.simd.vsts(denom_buf[0], zero)
            for row in tl.range(VM):
                for col in tl.static_range(D // 64):
                    tla.simd.vsts(acc[row, col * 64], zero)

    for t in tl.range(N_TILES):
        # ---- CUBE: load KV tile, K-tiled QK, fixpipe score ----
        with tla.scope.cube():
            tla.load_gm_to_l1(
                tla.subview(gm_kv, offsets=[t * BN, 0], sizes=[BN, D]),
                kv_l1)
            # First K-tile of QK
            tla.load_gm_to_l1(
                tla.subview(gm_q, offsets=[0, 0], sizes=[H, BK]),
                q_l1)
            tla.cube.mmad(
                q_l1,
                tla.view(kv_l1, (BK // 16, BN // 16, 16, 16)),
                qk_l0c, transpose_b=True, clear_accum=True)
            # Remaining K-tiles of QK
            for kt in tl.static_range(1, QK_K_TILES):
                tla.load_gm_to_l1(
                    tla.subview(gm_q, offsets=[0, kt * BK], sizes=[H, BK]),
                    q_l1)
                tla.cube.mmad(
                    q_l1,
                    tla.view(kv_l1, (BK // 16, BN // 16, 16, 16),
                             offsets=(kt * (BK // 16), 0, 0, 0)),
                    qk_l0c, transpose_b=True, clear_accum=False)
            # Wait for previous score read to finish
            if t >= 1:
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_SCORE_FREE)
            tla.copy_l0c_to_ub(
                qk_l0c, score,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F,
                                    FLAG_SCORE)

        # ---- VECTOR: online softmax, write P to L1 ----
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                     FLAG_SCORE)
            with tla.scope.simd():
                old_max = tla.simd.vlds(maximum[0], dist="BRC_B32")
                new_max = old_max
                for row in tl.range(VM):
                    for c in tl.static_range(BN // 64):
                        tile_max = tla.simd.vreduce_max(tla.simd.vlds(score[row, c * 64]))
                        new_max = tla.simd.vmax(new_max, tile_max)
                new_max = tla.simd.vdupv(new_max)
                alpha = tla.simd.vexp(tla.simd.vsub(old_max, new_max))
                old_den = tla.simd.vlds(denom_buf[0], dist="BRC_B32")
                new_den = tla.simd.vmul(old_den, alpha)
                for row in tl.range(VM):
                    rsum = tla.simd.vdups(0.0, tl.float32)
                    for c in tl.static_range(BN // 64):
                        e = tla.simd.vexp(tla.simd.vsub(
                            tla.simd.vlds(score[row, c * 64]), new_max))
                        rsum = tla.simd.vadd(rsum, tla.simd.vreduce_sum(e))
                        tla.simd.vsts(prob_ub[c, row * 16],
                                      tla.simd.vtruncf(e, tl.bfloat16))
                    new_den = tla.simd.vadd(new_den, rsum)
                    for col in tl.static_range(D // 64):
                        tla.simd.vsts(acc[row, col * 64],
                                      tla.simd.vmul(tla.simd.vlds(acc[row, col * 64]), alpha))
                tla.simd.vsts(maximum[0], new_max, dist="ONEPT_B32")
                tla.simd.vsts(denom_buf[0], new_den, dist="ONEPT_B32")
                tla.simd.membar("VST_VLD")
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V,
                                    FLAG_SCORE_FREE)
            # P UB → L1 for cube PV
            if t >= 1:
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_PV_FREE)
            tla.copy_ub_to_l1(
                tla.view(prob_ub, (BN // 16, VM * 16)),
                tla.view(pflat, (BN // 16, VM * 16),
                         offsets=(0, sub * VM * 16)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3,
                                    FLAG_P_READY)

        # ---- CUBE: PV(t) = P @ KV, fixpipe partial O ----
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                     FLAG_P_READY)
            tla.cube.mmad(p_l1, kv_l1, pv_l0c, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv_l0c, pv_ub,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F,
                                    FLAG_PV_DONE)

        # ---- VECTOR: fold PV partial into accumulator ----
        with tla.scope.vector():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                     FLAG_PV_DONE)
            with tla.scope.simd():
                for row in tl.range(VM):
                    for col in tl.static_range(D // 64):
                        tla.simd.vsts(
                            acc[row, col * 64],
                            tla.simd.vadd(tla.simd.vlds(acc[row, col * 64]),
                                          tla.simd.vlds(pv_ub[row, col * 64])))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V,
                                    FLAG_PV_FREE)

    # ---- Normalize and store ----
    with tla.scope.vector():
        sub = tla.sub_block_id()
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                 FLAG_PV_FREE)
        with tla.scope.simd():
            den = tla.simd.vlds(denom_buf[0], dist="BRC_B32")
            for row in tl.range(VM):
                for col in tl.static_range(D // 64):
                    tla.simd.vsts(
                        output[row, col * 64],
                        tla.simd.vtruncf(
                            tla.simd.vdiv(tla.simd.vlds(acc[row, col * 64]), den),
                            tl.bfloat16))
            tla.simd.membar("VST_VLD")
        tla.store_ub_to_gm(
            output,
            tla.view(gm_o, (VM, D), offsets=(sub * VM, 0)))


# ===========================================================================
# Benchmark
# ===========================================================================
def bench_npu(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn(); torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return statistics.median(times), min(times), statistics.mean(trimmed)


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
    BN, BK = 64, 128
    W, R = args.warmup, args.repeat

    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16).npu()
    kv = (torch.randn(K, D) * 0.1).to(torch.bfloat16).npu()
    sink = (torch.randn(H) * 0.5).to(torch.float32).npu()
    o_fused = torch.empty(H, D, dtype=torch.bfloat16, device="npu")

    # 3-launch baseline
    from test_precision_stages import tla_qk_tiled, tla_softmax, tla_pv_tiled, BASE_N, BASE_K
    s = torch.empty(H, K, dtype=torch.float32, device="npu")
    p = torch.empty(H, K, dtype=torch.bfloat16, device="npu")
    o_sep = torch.empty(H, D, dtype=torch.bfloat16, device="npu")

    def run_3launch():
        tla_qk_tiled[(1,)](q, kv, s, H=H, KTOTAL=K, D=D,
                           BASE_N_=BASE_N, BASE_K_=BASE_K)
        tla_softmax[(1,)](s, sink, p, H=H, K=K)
        tla_pv_tiled[(1,)](p, kv, o_sep, H=H, K=K, D=D,
                           PV_BN=BASE_N, PV_BK=BASE_K)

    def run_fused():
        fused_attn_v2[(1,)](q, kv, sink, o_fused, H=H, K=K, D=D, BN=BN, BK=BK)

    # DSL fused mega_mla
    TEST_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/test/attention/mega_mla"
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)
    from mega_mla import MegaMlaOp
    from test_mega_mla import quantize_fp8, quantize_fp4
    grid = int(torch.npu.get_device_properties(0).cube_core_num)
    op = MegaMlaOp(k=K, page_size=128, win_len=128, grid=grid, mode='attn',
                   tokens=1, heads=H, cmp_decode_strategy='indexed')
    torch.manual_seed(31)
    x = (torch.randn(1, 5120) * 0.1).to(torch.bfloat16).npu()
    gfq, gfk = (torch.rand(1, 1280) + 0.5).npu(), (torch.rand(1, 512) + 0.5).npu()
    def nz(t): return torch_npu.npu_format_cast(t.npu(), torch_npu.Format.FRACTAL_NZ)
    wa = nz((torch.randn(1280, 5120)*0.05).to(torch.float8_e4m3fn))
    wkv = nz((torch.randn(512, 5120)*0.05).to(torch.float8_e4m3fn))
    wb = nz((torch.randn(H*D, 1280)*0.05).to(torch.float8_e4m3fn))
    sa = torch.full((1280,80,2),125,dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    skv = torch.full((512,80,2),125,dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    sb = torch.full((H*D,20,2),125,dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    wkc = quantize_fp8((torch.randn(2,128,1,D)*0.1).to(torch.bfloat16)).npu()
    ckc = quantize_fp4((torch.randn(4,128,1,D)*0.1).to(torch.bfloat16)).npu()
    wi = torch.randint(128, 256, (1,128), dtype=torch.int32).npu()
    ci = torch.randint(0, 512, (1,K-128), dtype=torch.int32).npu()
    snk = torch.randn(1, H, dtype=torch.float32).npu()
    cos, sin = torch.randn(1,32).npu(), torch.randn(1,32).npu()
    slot = torch.arange(128, 129, dtype=torch.int32).npu()
    def run_dsl():
        op.run(x, gfq, gfk, wa, wkv, wb, sa, skv, sb, wkc, ckc,
               wi, ci, snk, cos, sin, slot)

    print(f"=== Fused attention v2: H={H}, K={K}, D={D}, grid={grid} ===\n")

    med, mn, mean = bench_npu(run_3launch, W, R)
    print(f"TLA 3-launch (QK+sm+PV):   median={med:7.1f}us  min={mn:7.1f}us  mean={mean:7.1f}us")

    try:
        med, mn, mean = bench_npu(run_fused, W, R)
        print(f"TLA fused v2 (1 launch):    median={med:7.1f}us  min={mn:7.1f}us  mean={mean:7.1f}us")
    except Exception as e:
        err = str(e).split('\n')[0]
        print(f"TLA fused v2 FAILED: {err}")

    med, mn, mean = bench_npu(run_dsl, W, R)
    print(f"DSL fused mega_mla (full):  median={med:7.1f}us  min={mn:7.1f}us  mean={mean:7.1f}us")


if __name__ == "__main__":
    main()
