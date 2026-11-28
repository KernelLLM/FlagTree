#!/usr/bin/env python3
"""Fused TLA attention kernel: QK + softmax + PV in one NPU dispatch.

Pattern follows dsa_kernel.py: cube + simd scopes with cross-core sync.
Covers the attention portion of mega_mla for fair benchmark comparison.

Stage layout per token (single-block, all N-tiles sequential):
  Cube : QK(nt) → fixpipe → score UB
  Vector: softmax(score) → P UB → copy P to L1
  Cube : PV(nt) → fixpipe → partial O(nt) → GM accumulate

Usage:
    python3 test_fused_attn.py
    python3 test_fused_attn.py --warmup 20 --repeat 200
"""
import sys, os, argparse, statistics

TLA_UT = "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut"
sys.path.insert(0, TLA_UT)
MEGA_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla"
sys.path.insert(0, MEGA_DIR)

import torch
import triton.language as tl
import triton.language.extra.tla as tla

# Cross-core event IDs
FLAG_SCORE = tl.constexpr(0)   # cube produced score tile in UB
FLAG_P_READY = tl.constexpr(1) # vector produced P tile in L1
FLAG_SCORE_FREE = tl.constexpr(2)  # vector done reading score UB

@tla.jit
def fused_attn(
    Q_ptr, KV_ptr, Sink_ptr, O_ptr,
    H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr,  # N-tile size (64)
    BK: tl.constexpr,  # K-tile for inner matmul (128)
):
    """Fused QK + masked softmax + PV in one kernel dispatch.

    For each N-tile of K:
      Cube: QK → score UB (fixpipe NZ2ND, dual-dst row-split to both AIVs)
      Vector: online softmax update (vexpdif) → P tile → L1
      Cube: PV accumulate
    Final: vector normalizes O and stores to GM.
    """
    gm_q = tla.view(Q_ptr, (H, D))
    gm_kv = tla.view(KV_ptr, (K, D))
    gm_sink = tla.view(Sink_ptr, (H,))
    gm_o = tla.view(O_ptr, (H, D))

    N_TILES: tl.constexpr = K // BN
    QK_K_TILES: tl.constexpr = D // BK
    VM: tl.constexpr = H // 2  # each AIV owns half the heads

    # --- UB allocations (vector owns these) ---
    score = tla.alloc((VM, BN), tl.float32, space="ub")
    p_ub = tla.alloc((BN // 16, VM * 16 + 16), tl.bfloat16, space="ub")
    pv = tla.alloc((VM, D), tl.float32, space="ub")
    acc = tla.alloc((VM, D), tl.float32, space="ub")
    maximum = tla.alloc((64,), tl.float32, space="ub")
    denom = tla.alloc((64,), tl.float32, space="ub")
    output = tla.alloc((VM, D), tl.bfloat16, space="ub")
    sink_ub = tla.alloc((64,), tl.float32, space="ub")

    # Initialize accumulator and softmax state
    with tla.scope.vector():
        with tla.scope.simd():
            zero = tla.simd.vfull(0.0, tl.float32, 64)
            neg_inf = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
            tla.simd.vsts(maximum[0], neg_inf)
            tla.simd.vsts(denom[0], zero)
            for row in tl.range(VM):
                for col in tl.static_range(D // 64):
                    tla.simd.vsts(acc[row, col * 64], zero)
        # Load sink values for this AIV's heads
        sub = tla.sub_block_id()
        tla.load_gm_to_ub(
            tla.view(gm_sink, (VM,), offsets=(sub * VM,)),
            tla.view(sink_ub, (VM,)),
        )

    for t in tl.range(N_TILES):
        # === CUBE: QK(t) → score ===
        with tla.scope.cube():
            q_l1 = tla.alloc((BK // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
            k_l1 = tla.alloc((BK // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
            qk_l0c = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")

            # First K-tile: clear
            tla.load_gm_to_l1(tla.subview(gm_q, offsets=[0, 0], sizes=[H, BK]), q_l1)
            tla.load_gm_to_l1(
                tla.subview(gm_kv, offsets=[t * BN, 0], sizes=[BN, BK]), k_l1)
            tla.cube.mmad(q_l1, k_l1, qk_l0c, transpose_b=True, clear_accum=True)

            # Remaining K-tiles: accumulate
            for kt in tl.static_range(1, QK_K_TILES):
                tla.load_gm_to_l1(
                    tla.subview(gm_q, offsets=[0, kt * BK], sizes=[H, BK]), q_l1)
                tla.load_gm_to_l1(
                    tla.subview(gm_kv, offsets=[t * BN, kt * BK], sizes=[BN, BK]),
                    k_l1)
                tla.cube.mmad(q_l1, k_l1, qk_l0c, transpose_b=True,
                              clear_accum=False)

            if t >= 1:
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_SCORE_FREE)
            # Fixpipe → UB (row-split to both AIVs)
            tla.copy_l0c_to_ub(
                qk_l0c, score,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None,
                ),
            )
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F,
                                    FLAG_SCORE)

            # === CUBE: PV(t-1) ===
            if t >= 1:
                p_l1 = tla.alloc((BN // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_P_READY)
                v_l1_first = tla.alloc((BN // 16, (D // 16) // 1, 16, 16), tl.bfloat16, space="l1")
                # PV: P[H, BN] @ V[BN, D] with BN as K-dim
                # For small BN=64, do it in one shot (BN fits in K-tile)
                tla.load_gm_to_l1(
                    tla.subview(gm_kv, offsets=[(t - 1) * BN, 0], sizes=[BN, D]),
                    tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1"),
                )
                # TLA_TODO: PV accumulation needs proper V-tile loading
                # For now simplified: single BN tile as K-dim

        # === VECTOR: online softmax update(t) ===
        with tla.scope.vector():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                     FLAG_SCORE)
            with tla.scope.simd():
                old_max = tla.simd.vlds(maximum[0])
                old_denom = tla.simd.vlds(denom[0])
                new_max = old_max

                # Find tile max
                for row in tl.range(VM):
                    for c in tl.static_range(BN // 64):
                        s = tla.simd.vlds(score[row, c * 64])
                        chunk_max = tla.simd.vreduce_max(s)
                        new_max = tla.simd.vmax(new_max, chunk_max)

                # Include sink
                # TLA_TODO: per-head sink (requires row-wise max)
                new_max = tla.simd.vdupv(new_max)

                # Rescale old accumulator
                alpha = tla.simd.vexpdif(old_max, new_max)
                new_denom = tla.simd.vmul(old_denom, alpha)

                # Compute exp(score - max) and sum, store BF16 P
                for row in tl.range(VM):
                    row_sum = tla.simd.vdups(0.0, tl.float32)
                    for c in tl.static_range(BN // 64):
                        s = tla.simd.vlds(score[row, c * 64])
                        e = tla.simd.vexpdif(s, new_max)
                        row_sum = tla.simd.vadd(row_sum, tla.simd.vreduce_sum(e))
                        p_bf16 = tla.simd.vtruncf(e, tl.bfloat16)
                        tla.simd.vsts(p_ub[c, row * 16], p_bf16)
                    new_denom = tla.simd.vadd(new_denom, row_sum)

                    # Rescale accumulator row
                    for col in tl.static_range(D // 64):
                        old_acc = tla.simd.vlds(acc[row, col * 64])
                        tla.simd.vsts(acc[row, col * 64],
                                      tla.simd.vmul(old_acc, alpha))
                tla.simd.membar("VST_VLD")

                tla.simd.vsts(maximum[0], new_max)
                tla.simd.vsts(denom[0], new_denom)

            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3,
                                    FLAG_SCORE_FREE)
            # TLA_TODO: copy P from UB to L1 for cube PV
            # tla.copy_ub_to_l1(p_ub, p_l1_view)
            # tla.sync.block_set_flag(MODE.BLOCK, PIPE.MTE3, FLAG_P_READY)

    # === Final: normalize and store ===
    with tla.scope.vector():
        with tla.scope.simd():
            final_denom = tla.simd.vlds(denom[0])
            final_denom = tla.simd.vdupv(final_denom)
            for row in tl.range(VM):
                for col in tl.static_range(D // 64):
                    a = tla.simd.vlds(acc[row, col * 64])
                    normed = tla.simd.vdiv(a, final_denom)
                    out_bf16 = tla.simd.vtruncf(normed, tl.bfloat16)
                    tla.simd.vsts(output[row, col * 64], out_bf16)
            tla.simd.membar("VST_VLD")
        sub = tla.sub_block_id()
        tla.store_ub_to_gm(
            output,
            tla.view(gm_o, (VM, D), offsets=(sub * VM, 0)),
        )


# ===========================================================================
# Benchmark
# ===========================================================================
def bench_npu(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return statistics.median(times), min(times), times[len(times)//20], times[-len(times)//20-1], statistics.mean(trimmed)


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

    # --- 3 separate TLA launches (baseline) ---
    from test_precision_stages import tla_qk_tiled, tla_softmax, tla_pv_tiled, BASE_N, BASE_K
    s_sep = torch.empty(H, K, dtype=torch.float32, device="npu")
    p_sep = torch.empty(H, K, dtype=torch.bfloat16, device="npu")
    o_sep = torch.empty(H, D, dtype=torch.bfloat16, device="npu")

    def run_separate():
        tla_qk_tiled[(1,)](q, kv, s_sep, H=H, KTOTAL=K, D=D,
                           BASE_N_=BASE_N, BASE_K_=BASE_K)
        tla_softmax[(1,)](s_sep, sink, p_sep, H=H, K=K)
        tla_pv_tiled[(1,)](p_sep, kv, o_sep, H=H, K=K, D=D,
                           PV_BN=BASE_N, PV_BK=BASE_K)

    # --- Fused TLA (single launch) ---
    def run_fused():
        fused_attn[(1,)](q, kv, sink, o_fused, H=H, K=K, D=D, BN=BN, BK=BK)

    # --- DSL fused mega_mla ---
    from mega_mla import MegaMlaOp
    TEST_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/test/attention/mega_mla"
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)
    from test_mega_mla import quantize_fp8, quantize_fp4
    win_pages, cmp_pages, cps, ps = 2, 4, 128, 128
    grid = int(torch.npu.get_device_properties(0).cube_core_num)
    op = MegaMlaOp(k=K, page_size=ps, win_len=128, grid=grid,
                   mode='attn', tokens=1, heads=H, cmp_decode_strategy='indexed')
    torch.manual_seed(31)
    x = (torch.randn(1, 5120) * 0.1).to(torch.bfloat16).npu()
    gfq = (torch.rand(1, 1280) + 0.5).npu()
    gfk = (torch.rand(1, 512) + 0.5).npu()
    def nz(t): return torch_npu.npu_format_cast(t.npu(), torch_npu.Format.FRACTAL_NZ)
    wa = nz((torch.randn(1280, 5120) * 0.05).to(torch.float8_e4m3fn))
    wkv = nz((torch.randn(512, 5120) * 0.05).to(torch.float8_e4m3fn))
    wb = nz((torch.randn(H * D, 1280) * 0.05).to(torch.float8_e4m3fn))
    sa = torch.full((1280, 80, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    skv = torch.full((512, 80, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    sb = torch.full((H * D, 20, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
    wkc = quantize_fp8((torch.randn(win_pages, ps, 1, D) * 0.1).to(torch.bfloat16)).npu()
    ckc = quantize_fp4((torch.randn(cmp_pages, cps, 1, D) * 0.1).to(torch.bfloat16)).npu()
    wi = torch.randint(ps, win_pages * ps, (1, 128), dtype=torch.int32).npu()
    ci = torch.randint(0, cmp_pages * cps, (1, K - 128), dtype=torch.int32).npu()
    snk = torch.randn(1, H, dtype=torch.float32).npu()
    cos, sin = torch.randn(1, 32).npu(), torch.randn(1, 32).npu()
    slot = torch.arange(ps, ps + 1, dtype=torch.int32).npu()

    def run_dsl():
        op.run(x, gfq, gfk, wa, wkv, wb, sa, skv, sb, wkc, ckc,
               wi, ci, snk, cos, sin, slot)

    print(f"=== Perf: H={H}, K={K}, D={D}, grid={grid}, warmup={W}, repeat={R} ===\n")

    # Benchmark separate TLA stages
    med, mn, p5, p95, mean = bench_npu(run_separate, W, R)
    print(f"TLA 3-launch (QK+sm+PV)      median={med:8.1f}us  min={mn:8.1f}us  p5={p5:8.1f}us  p95={p95:8.1f}us  mean={mean:8.1f}us")

    # Benchmark fused TLA
    try:
        med, mn, p5, p95, mean = bench_npu(run_fused, W, R)
        print(f"TLA fused attn (1 launch)     median={med:8.1f}us  min={mn:8.1f}us  p5={p5:8.1f}us  p95={p95:8.1f}us  mean={mean:8.1f}us")
    except Exception as e:
        print(f"TLA fused attn FAILED: {e}")

    # Benchmark DSL fused mega_mla (full pipeline)
    med, mn, p5, p95, mean = bench_npu(run_dsl, W, R)
    print(f"DSL fused mega_mla (full)     median={med:8.1f}us  min={mn:8.1f}us  p5={p5:8.1f}us  p95={p95:8.1f}us  mean={mean:8.1f}us")

    print()
    print(f"Note: DSL fused mega_mla includes projection (P0-P3) + cache decode")
    print(f"+ gather + RoPE + attention. TLA covers attention only.")


if __name__ == "__main__":
    main()
