"""Minimal cube→vector→GM handshake with dual_dst ROW_SPLIT."""
import sys
sys.path.insert(0, "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut")
import torch, torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
torch.npu.set_device(0)

FLAG_S = tl.constexpr(0)

@tla.jit
def mini_mix(Q_ptr, K_ptr, O_ptr,
             H: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    VM: tl.constexpr = H // 2
    gm_q = tla.view(Q_ptr, (H, D))
    gm_k = tla.view(K_ptr, (BN, D))
    gm_o = tla.view(O_ptr, (H, BN))
    q_l1 = tla.alloc((D // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
    k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
    l0c = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")
    # Each AIV receives VM rows via dual_dst ROW_SPLIT
    score = tla.alloc((VM, BN), tl.float32, space="ub")
    out_ub = tla.alloc((VM, BN), tl.bfloat16, space="ub")

    with tla.scope.cube():
        tla.load_gm_to_l1(gm_q, q_l1)
        tla.load_gm_to_l1(gm_k, k_l1)
        tla.cube.mmad(q_l1, k_l1, l0c, transpose_b=True, clear_accum=True)
        tla.copy_l0c_to_ub(
            l0c, score,
            tla.FixpipeL0C2UBParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                sub_block_idx=None))
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_S)

    with tla.scope.vector():
        sub = tla.sub_block_id()
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_S)
        # Each AIV reads its VM rows and converts to BF16
        with tla.scope.simd():
            for row in tl.range(VM):
                for c in tl.static_range(BN // 64):
                    v = tla.simd.vlds(score[row, c * 64])
                    tla.simd.vsts(out_ub[row, c * 64], tla.simd.vtruncf(v, tl.bfloat16))
            tla.simd.membar("VST_VLD")
        # Each AIV writes its half to the correct GM offset
        tla.store_ub_to_gm(
            out_ub,
            tla.view(gm_o, (VM, BN), offsets=(sub * VM, 0)))


if __name__ == "__main__":
    H, BN, D = 16, 64, 128
    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16).npu()
    k = (torch.randn(BN, D) * 0.1).to(torch.bfloat16).npu()
    o = torch.empty(H, BN, dtype=torch.bfloat16, device="npu")

    mini_mix[(1,)](q, k, o, H=H, BN=BN, D=D)
    torch.npu.synchronize()

    ref = (q.cpu().float() @ k.cpu().float().T).to(torch.bfloat16)
    err = (o.cpu().float() - ref.float()).abs().max()
    result = "PASS" if err < 0.01 else "FAIL"
    print(f"cube->vector dual_dst: max_err={err:.6e} {result}")

    # Also test: cube→vector→cube→vector (the PV pipeline pattern)
    FLAG_P = tl.constexpr(1)
    FLAG_PV = tl.constexpr(2)
    FLAG_SF = tl.constexpr(3)

    @tla.jit
    def mini_pipeline(Q_ptr, K_ptr, O_ptr,
                      H: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
        """QK(cube) → truncate(vector) → PV(cube) → store(vector)."""
        VM: tl.constexpr = H // 2
        gm_q = tla.view(Q_ptr, (H, D))
        gm_k = tla.view(K_ptr, (BN, D))
        gm_o = tla.view(O_ptr, (H, D))
        q_l1 = tla.alloc((D // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
        k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        p_l1 = tla.alloc((BN // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
        qk_l0c = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")
        pv_l0c = tla.alloc((D // 16, H // 16, 16, 16), tl.float32, space="l0c")
        score = tla.alloc((VM, BN), tl.float32, space="ub")
        prob_ub = tla.alloc((BN // 16, VM * 16 + 16), tl.bfloat16, space="ub")
        pflat = tla.reshape(p_l1, (BN // 16, H * 16))
        pv_ub = tla.alloc((VM, D), tl.float32, space="ub")
        out_ub = tla.alloc((VM, D), tl.bfloat16, space="ub")

        # Stage 1: Cube QK
        with tla.scope.cube():
            tla.load_gm_to_l1(gm_q, q_l1)
            tla.load_gm_to_l1(gm_k, k_l1)
            tla.cube.mmad(q_l1, k_l1, qk_l0c, transpose_b=True, clear_accum=True)
            tla.copy_l0c_to_ub(
                qk_l0c, score,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_S)

        # Stage 2: Vector softmax → P → L1
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_S)
            with tla.scope.simd():
                for row in tl.range(VM):
                    for c in tl.static_range(BN // 64):
                        v = tla.simd.vlds(score[row, c * 64])
                        # Simple softmax: just truncate to BF16 for now
                        tla.simd.vsts(prob_ub[c, row * 16],
                                      tla.simd.vtruncf(v, tl.bfloat16))
                tla.simd.membar("VST_VLD")
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, FLAG_SF)
            tla.copy_ub_to_l1(
                tla.view(prob_ub, (BN // 16, VM * 16)),
                tla.view(pflat, (BN // 16, VM * 16),
                         offsets=(0, sub * VM * 16)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, FLAG_P)

        # Stage 3: Cube PV
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_P)
            tla.cube.mmad(p_l1, k_l1, pv_l0c, clear_accum=True)
            tla.copy_l0c_to_ub(
                pv_l0c, pv_ub,
                tla.FixpipeL0C2UBParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                    sub_block_idx=None))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, FLAG_PV)

        # Stage 4: Vector store
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, FLAG_PV)
            with tla.scope.simd():
                for row in tl.range(VM):
                    for col in tl.static_range(D // 64):
                        v = tla.simd.vlds(pv_ub[row, col * 64])
                        tla.simd.vsts(out_ub[row, col * 64],
                                      tla.simd.vtruncf(v, tl.bfloat16))
                tla.simd.membar("VST_VLD")
            tla.store_ub_to_gm(
                out_ub,
                tla.view(gm_o, (VM, D), offsets=(sub * VM, 0)))

    o2 = torch.empty(H, D, dtype=torch.bfloat16, device="npu")
    mini_pipeline[(1,)](q, k, o2, H=H, BN=BN, D=D)
    torch.npu.synchronize()
    # Reference: QK → truncate → PV (no real softmax, just truncation)
    s_ref = q.cpu().float() @ k.cpu().float().T
    p_ref = s_ref.to(torch.bfloat16)
    o_ref = (p_ref.float() @ k.cpu().float()).to(torch.bfloat16)
    err2 = (o2.cpu().float() - o_ref.float()).abs().max()
    result2 = "PASS" if err2 < 0.1 else "FAIL"
    print(f"cube->vec->cube->vec pipeline: max_err={err2:.6e} {result2}")
