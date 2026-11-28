"""Fused QK+softmax+PV following dsa_kernel.py's lagged pipeline exactly.

Key insight: ONE cube scope per tile doing QK(t) + PV(t-1).
ONE vector scope per tile doing softmax(t) + accumulate(t-1).
Epilogue drains the last PV.

Flag protocol (identical to dsa_kernel, adapted for pre-gathered KV):
  FLAG_SCORE:  cube(F) → vector(S)      score tile in UB
  FLAG_SFREE:  vector(V) → cube(S)      vector done reading score
  FLAG_P:      vector(MTE3) → cube(S)   P tile in L1
  FLAG_PFREE:  cube(M) → vector(S)      cube done reading P from L1
  FLAG_PV:     cube(F) → vector(S)      PV partial in UB
  FLAG_PVFREE: vector(V) → cube(S)      vector done folding PV
"""
import sys
sys.path.insert(0, "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut")
sys.path.insert(0, "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla")
import torch, torch_npu, statistics
import triton.language as tl
import triton.language.extra.tla as tla
torch.npu.set_device(0)

FLAG_SCORE = tl.constexpr(0)
FLAG_SFREE = tl.constexpr(1)
FLAG_P     = tl.constexpr(2)
FLAG_PFREE = tl.constexpr(3)
FLAG_PV    = tl.constexpr(4)
FLAG_PVFREE= tl.constexpr(5)

@tla.jit
def fused_attn_lag(
    Q_ptr, KV_ptr, O_ptr,
    H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    gm_q = tla.view(Q_ptr, (H, D))
    gm_kv = tla.view(KV_ptr, (K, D))
    gm_o = tla.view(O_ptr, (H, D))

    TILES: tl.constexpr = K // BN
    K_TILES: tl.constexpr = D // BK
    VM: tl.constexpr = H // 2

    # All allocations up front (dsa_kernel pattern)
    q_l1 = tla.alloc((BK // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
    # 2 KV L1 slots for the lag: QK(t) uses slot t%2, PV(t-1) uses (t-1)%2
    kv_l1 = tla.alloc((2 * (D // 16), BN // 16, 16, 16), tl.bfloat16, space="l1")
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
    corrections = tla.alloc((128,), tl.float32, space="ub")
    output = tla.alloc((VM, D), tl.bfloat16, space="ub")

    # Initialize
    with tla.scope.vector():
        with tla.scope.simd():
            zero = tla.simd.vfull(0.0, tl.float32, 64)
            neg_inf = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
            tla.simd.vsts(maximum[0], neg_inf)
            tla.simd.vsts(denom_buf[0], zero)
            for row in tl.range(VM):
                for col in tl.static_range(D // 64):
                    tla.simd.vsts(acc[row, col * 64], zero)

    # === Main tile loop ===
    for t in tl.range(TILES):
        kv_cur = tla.view(kv_l1, (D // 16, BN // 16, 16, 16),
                          offsets=((t % 2) * (D // 16), 0, 0, 0))

        # --- CUBE: load KV(t), QK(t), [PV(t-1)] ---
        with tla.scope.cube():
            # Load KV tile t into slot t%2
            tla.load_gm_to_l1(
                tla.subview(gm_kv, offsets=[t * BN, 0], sizes=[BN, D]),
                kv_cur)
            # QK: first K-tile
            tla.load_gm_to_l1(
                tla.subview(gm_q, offsets=[0, 0], sizes=[H, BK]),
                q_l1)
            tla.cube.mmad(
                q_l1,
                tla.view(kv_cur, (BK // 16, BN // 16, 16, 16)),
                qk_l0c, transpose_b=True, clear_accum=True)
            # QK: remaining K-tiles
            for kt in tl.static_range(1, K_TILES):
                tla.load_gm_to_l1(
                    tla.subview(gm_q, offsets=[0, kt * BK], sizes=[H, BK]),
                    q_l1)
                tla.cube.mmad(
                    q_l1,
                    tla.view(kv_cur, (BK // 16, BN // 16, 16, 16),
                             offsets=(kt * (BK // 16), 0, 0, 0)),
                    qk_l0c, transpose_b=True, clear_accum=False)
            # Fixpipe score → UB (wait for vector to finish reading previous)
            if t >= 1:
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_SFREE)
            tla.copy_l0c_to_ub(
                qk_l0c, score,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F,
                                    FLAG_SCORE)
            # PV(t-1): P(t-1) × KV((t-1)%2)
            if t >= 1:
                kv_prev = tla.view(kv_l1, (D // 16, BN // 16, 16, 16),
                                   offsets=(((t - 1) % 2) * (D // 16), 0, 0, 0))
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_P)
                tla.cube.mmad(p_l1, kv_prev, pv_l0c, clear_accum=True)
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.M,
                                        FLAG_PFREE)
                if t >= 2:
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                             FLAG_PVFREE)
                tla.copy_l0c_to_ub(
                    pv_l0c, pv_ub,
                    tla.FixpipeL0C2UBParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                        sub_block_idx=None))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F,
                                        FLAG_PV)

        # --- VECTOR: softmax(t), [copy P to L1], [accumulate PV(t-1)] ---
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                     FLAG_SCORE)
            corr_view = tla.view(corrections, (64,), offsets=((t % 2) * 64,))
            with tla.scope.simd():
                old_max = tla.simd.vlds(maximum[0], dist="BRC_B32")
                new_max = old_max
                for row in tl.range(VM):
                    for c in tl.static_range(BN // 64):
                        tile_max = tla.simd.vreduce_max(
                            tla.simd.vlds(score[row, c * 64]))
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
                tla.simd.vsts(maximum[0], new_max, dist="ONEPT_B32")
                tla.simd.vsts(denom_buf[0], new_den, dist="ONEPT_B32")
                tla.simd.vsts(corr_view[0], alpha, dist="ONEPT_B32")
                tla.simd.membar("VST_VLD")
            # Signal score consumed (not on last tile per dsa protocol)
            if t < TILES - 1:
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V,
                                        FLAG_SFREE)
            # Wait for cube done reading previous P before overwriting
            if t >= 1:
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_PFREE)
            # Copy P to L1
            tla.copy_ub_to_l1(
                tla.view(prob_ub, (BN // 16, VM * 16)),
                tla.view(pflat, (BN // 16, VM * 16),
                         offsets=(0, sub * VM * 16)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3,
                                    FLAG_P)
            # Accumulate PV(t-1)
            if t >= 1:
                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S,
                                         FLAG_PV)
                corr_prev = tla.view(corrections, (64,),
                                     offsets=(((t - 1) % 2) * 64,))
                with tla.scope.simd():
                    for row in tl.range(VM):
                        a = tla.simd.vlds(corr_prev[row], dist="BRC_B32")
                        for col in tl.static_range(D // 64):
                            tla.simd.vsts(
                                acc[row, col * 64],
                                tla.simd.vfma(
                                    tla.simd.vlds(pv_ub[row, col * 64]),
                                    tla.simd.vlds(acc[row, col * 64]),
                                    a))
                tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V,
                                        FLAG_PVFREE)

    # === Epilogue: drain last PV ===
    with tla.scope.cube():
        kv_last = tla.view(kv_l1, (D // 16, BN // 16, 16, 16),
                           offsets=(((TILES - 1) % 2) * (D // 16), 0, 0, 0))
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_P)
        tla.cube.mmad(p_l1, kv_last, pv_l0c, clear_accum=True)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PVFREE)
        tla.copy_l0c_to_ub(
            pv_l0c, pv_ub,
            tla.FixpipeL0C2UBParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                sub_block_idx=None))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_PV)

    with tla.scope.vector():
        sub = tla.sub_block_id()
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV)
        with tla.scope.simd():
            corr_last = tla.view(corrections, (64,),
                                 offsets=(((TILES - 1) % 2) * 64,))
            for row in tl.range(VM):
                a = tla.simd.vlds(corr_last[row], dist="BRC_B32")
                for col in tl.static_range(D // 64):
                    tla.simd.vsts(
                        acc[row, col * 64],
                        tla.simd.vfma(
                            tla.simd.vlds(pv_ub[row, col * 64]),
                            tla.simd.vlds(acc[row, col * 64]),
                            a))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, FLAG_PVFREE)
        # Normalize and store
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
# Test + benchmark
# ===========================================================================
def bench(fn, w=20, r=200):
    for _ in range(w): fn(); torch.npu.synchronize()
    times = []
    for _ in range(r):
        s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    trim = max(1, r // 10)
    return statistics.median(times), min(times), statistics.mean(times[trim:-trim])


if __name__ == "__main__":
    H, K, D, BN, BK = 16, 640, 512, 64, 128
    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16).npu()
    kv = (torch.randn(K, D) * 0.1).to(torch.bfloat16).npu()
    o_fused = torch.empty(H, D, dtype=torch.bfloat16, device="npu")

    print(f"=== Fused attention (lagged pipeline): H={H}, K={K}, D={D} ===\n")

    # Correctness: compare with separate-launch reference
    from test_precision_stages import tla_qk_tiled, tla_softmax, tla_pv_tiled, BASE_N, BASE_K
    s_ref = torch.empty(H, K, dtype=torch.float32, device="npu")
    p_ref = torch.empty(H, K, dtype=torch.bfloat16, device="npu")
    o_ref = torch.empty(H, D, dtype=torch.bfloat16, device="npu")
    sink = torch.zeros(H, dtype=torch.float32, device="npu")  # zero sink for fair comparison
    tla_qk_tiled[(1,)](q, kv, s_ref, H=H, KTOTAL=K, D=D, BASE_N_=64, BASE_K_=128)
    tla_softmax[(1,)](s_ref, sink, p_ref, H=H, K=K)
    tla_pv_tiled[(1,)](p_ref, kv, o_ref, H=H, K=K, D=D, PV_BN=64, PV_BK=128)
    torch.npu.synchronize()

    try:
        fused_attn_lag[(1,)](q, kv, o_fused, H=H, K=K, D=D, BN=BN, BK=BK)
        torch.npu.synchronize()
        err = (o_fused.cpu().float() - o_ref.cpu().float()).abs().max()
        print(f"Correctness: fused vs 3-launch max_err={err:.6e}")

        # Benchmark
        def run_fused():
            fused_attn_lag[(1,)](q, kv, o_fused, H=H, K=K, D=D, BN=BN, BK=BK)
        def run_3launch():
            tla_qk_tiled[(1,)](q, kv, s_ref, H=H, KTOTAL=K, D=D, BASE_N_=64, BASE_K_=128)
            tla_softmax[(1,)](s_ref, sink, p_ref, H=H, K=K)
            tla_pv_tiled[(1,)](p_ref, kv, o_ref, H=H, K=K, D=D, PV_BN=64, PV_BK=128)

        med, mn, mean = bench(run_fused)
        print(f"Fused (1 launch):   median={med:7.1f}us  min={mn:7.1f}us  mean={mean:7.1f}us")
        med, mn, mean = bench(run_3launch)
        print(f"3-launch baseline:  median={med:7.1f}us  min={mn:7.1f}us  mean={mean:7.1f}us")

    except Exception as e:
        err_msg = str(e).split('\n')[0]
        print(f"FAILED: {err_msg}")
