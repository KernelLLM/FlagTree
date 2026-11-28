#!/usr/bin/env python3
"""Precision comparison: TLA vs cannbot-dsl for QK, softmax, and PV.

Validates the three core attention stages independently:
  1. QK matmul:  S[H, K] = Q[H, D] @ KV[K, D]^T          (Cube, BF16→FP32)
  2. Softmax:    P[H, K] = softmax(S + sink)                (SIMD, FP32→BF16)
  3. PV matmul:  O[H, D] = P[H, K] @ KV[K, D]             (Cube, BF16→BF16)

Usage:
    python3 test_precision_stages.py
"""
import sys, os

TLA_UT = "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut"
sys.path.insert(0, TLA_UT)
MEGA_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla"
sys.path.insert(0, MEGA_DIR)

import torch
import triton.language as tl
import triton.language.extra.tla as tla


# ===========================================================================
# Stage 1: QK matmul — tiled N64 K128  S[H, K] = Q[H, D] @ KV[K, D]^T
# ===========================================================================
BASE_N = 64   # N-tile (key rows per cube pass)
BASE_K = 128  # K-tile (feature columns per L1 load)


@tla.jit
def tla_qk_tiled(Q_ptr, K_ptr, S_ptr,
                  H: tl.constexpr, KTOTAL: tl.constexpr, D: tl.constexpr,
                  BASE_N_: tl.constexpr, BASE_K_: tl.constexpr):
    """S[H, KTOTAL] = Q[H, D] @ K[KTOTAL, D]^T, N-tiled and K-tiled."""
    gm_q = tla.view(Q_ptr, (H, D))
    gm_k = tla.view(K_ptr, (KTOTAL, D))
    gm_s = tla.view(S_ptr, (H, KTOTAL))
    N_TILES: tl.constexpr = KTOTAL // BASE_N_
    K_TILES: tl.constexpr = D // BASE_K_

    for nt in range(N_TILES):
        with tla.scope.cube():
            q_l1 = tla.alloc((BASE_K_ // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
            k_l1 = tla.alloc((BASE_K_ // 16, BASE_N_ // 16, 16, 16), tl.bfloat16, space="l1")
            acc = tla.alloc((BASE_N_ // 16, H // 16, 16, 16), tl.float32, space="l0c")

            # First K-tile: clear accumulator
            tla.load_gm_to_l1(
                tla.subview(gm_q, offsets=[0, 0], sizes=[H, BASE_K_]),
                q_l1)
            tla.load_gm_to_l1(
                tla.subview(gm_k, offsets=[nt * BASE_N_, 0],
                            sizes=[BASE_N_, BASE_K_]),
                k_l1)
            tla.cube.mmad(q_l1, k_l1, acc, transpose_b=True,
                          clear_accum=True)

            # Remaining K-tiles: accumulate
            for kt in range(1, K_TILES):
                tla.load_gm_to_l1(
                    tla.subview(gm_q, offsets=[0, kt * BASE_K_], sizes=[H, BASE_K_]),
                    q_l1)
                tla.load_gm_to_l1(
                    tla.subview(gm_k, offsets=[nt * BASE_N_, kt * BASE_K_],
                                sizes=[BASE_N_, BASE_K_]),
                    k_l1)
                tla.cube.mmad(q_l1, k_l1, acc, transpose_b=True,
                              clear_accum=False)

            tla.store_l0c_to_gm(
                acc,
                tla.subview(gm_s, offsets=[0, nt * BASE_N_], sizes=[H, BASE_N_]),
                tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


# ===========================================================================
# Stage 2: masked softmax with attn_sink (pure SIMD)
#
# For each head row:
#   m = max(max(S[row, :]), sink[row])
#   exp_i = exp(S[row, i] - m)
#   denom = sum(exp_i) + exp(sink[row] - m)
#   P[row, i] = bf16(exp_i / denom)
# ===========================================================================
@tla.jit
def tla_softmax(S_ptr, Sink_ptr, P_ptr,
                H: tl.constexpr, K: tl.constexpr):
    s_gm = tla.view(S_ptr, (H, K))
    sink_gm = tla.view(Sink_ptr, (H,))
    p_gm = tla.view(P_ptr, (H, K))

    s_ub = tla.alloc((1, K), tl.float32, space="ub")
    exp_ub = tla.alloc((1, K), tl.float32, space="ub")
    p_ub = tla.alloc((1, K), tl.bfloat16, space="ub")
    sink_ub = tla.alloc((1,), tl.float32, space="ub")

    NC: tl.constexpr = K // 64  # 64 fp32 lanes per SIMD register

    for row in range(H):
        # Load row of scores
        tla.copy(tla.subview(s_gm, offsets=[row, 0], sizes=[1, K]), s_ub)
        tla.copy(tla.subview(sink_gm, offsets=[row], sizes=[1]), sink_ub)
        tla.sync.barrier(tla.sync.PIPE.ALL)

        with tla.scope.simd():
            # Pass 1: find row max (across all K chunks and sink)
            sink_v = tla.simd.vlds(sink_ub[0])
            row_max = tla.simd.vdupv(sink_v)  # broadcast lane 0 (POS_LOWEST default)
            for c in range(NC):
                s = tla.simd.vlds(s_ub[0, c * 64])
                chunk_max = tla.simd.vreduce_max(s)
                row_max = tla.simd.vmax(row_max, chunk_max)
            # Broadcast the max to all lanes
            row_max = tla.simd.vdupv(row_max)

            # Pass 2: exp(s - max) and sum
            row_sum = tla.simd.vdups(0.0, tl.float32)
            for c in range(NC):
                s = tla.simd.vlds(s_ub[0, c * 64])
                e = tla.simd.vexpdif(s, row_max)
                tla.simd.vsts(exp_ub[0, c * 64], e)
                row_sum = tla.simd.vadd(row_sum, tla.simd.vreduce_sum(e))
            tla.simd.membar("VST_VLD")

            # Add exp(sink - max) to denominator
            denom = tla.simd.vadd(row_sum, tla.simd.vexpdif(sink_v, row_max))
            denom = tla.simd.vdupv(denom)

            # Pass 3: normalize to BF16
            for c in range(NC):
                e = tla.simd.vlds(exp_ub[0, c * 64])
                prob = tla.simd.vdiv(e, denom)
                prob_bf16 = tla.simd.vtruncf(prob, tl.bfloat16)
                tla.simd.vsts(p_ub[0, c * 64], prob_bf16)
            tla.simd.membar("VST_VLD")

        tla.copy(p_ub, tla.subview(p_gm, offsets=[row, 0], sizes=[1, K]))
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# Stage 3: PV matmul  O[H, D] = P[H, K] @ V[K, D]
# ===========================================================================
@tla.jit
def tla_pv_tiled(P_ptr, V_ptr, O_ptr,
                  H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                  PV_BN: tl.constexpr, PV_BK: tl.constexpr):
    """O[H, D] = P[H, K] @ V[K, D], N-tiled over D, K-tiled over K."""
    gm_p = tla.view(P_ptr, (H, K))
    gm_v = tla.view(V_ptr, (K, D))
    gm_o = tla.view(O_ptr, (H, D))
    N_TILES: tl.constexpr = D // PV_BN
    K_TILES: tl.constexpr = K // PV_BK

    for nt in range(N_TILES):
        with tla.scope.cube():
            p_l1 = tla.alloc((PV_BK // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
            v_l1 = tla.alloc((PV_BN // 16, PV_BK // 16, 16, 16), tl.bfloat16, space="l1")
            acc = tla.alloc((PV_BN // 16, H // 16, 16, 16), tl.float32, space="l0c")

            # First K-tile: clear
            tla.load_gm_to_l1(
                tla.subview(gm_p, offsets=[0, 0], sizes=[H, PV_BK]),
                p_l1)
            tla.load_gm_to_l1(
                tla.subview(gm_v, offsets=[0, nt * PV_BN],
                            sizes=[PV_BK, PV_BN]),
                v_l1)
            tla.cube.mmad(p_l1, v_l1, acc, transpose_b=False,
                          clear_accum=True)

            # Remaining K-tiles: accumulate
            for kt in range(1, K_TILES):
                tla.load_gm_to_l1(
                    tla.subview(gm_p, offsets=[0, kt * PV_BK], sizes=[H, PV_BK]),
                    p_l1)
                tla.load_gm_to_l1(
                    tla.subview(gm_v, offsets=[kt * PV_BK, nt * PV_BN],
                                sizes=[PV_BK, PV_BN]),
                    v_l1)
                tla.cube.mmad(p_l1, v_l1, acc, transpose_b=False,
                              clear_accum=False)

            tla.store_l0c_to_gm(
                acc,
                tla.subview(gm_o, offsets=[0, nt * PV_BN], sizes=[H, PV_BN]),
                tla.FixpipeL0C2GMParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    pre_quant=tla.FixpipePreQuantMode.F322BF16))


# ===========================================================================
# CPU reference
# ===========================================================================
def cpu_softmax_with_sink(scores, sink):
    """FP32 softmax with per-head attn_sink scalar."""
    maximum = torch.maximum(scores.amax(-1, keepdim=True), sink[:, None])
    exp = torch.exp(scores - maximum)
    denom = exp.sum(-1, keepdim=True) + torch.exp(sink[:, None] - maximum)
    return (exp / denom).to(torch.bfloat16)


# ===========================================================================
# Main
# ===========================================================================
def main():
    import torch_npu
    if not torch.npu.is_available():
        print("SKIP: NPU unavailable")
        return
    torch.npu.set_device(0)

    H, K, D = 16, 640, 512
    torch.manual_seed(42)

    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16)
    kv = (torch.randn(K, D) * 0.1).to(torch.bfloat16)
    sink = torch.randn(H, dtype=torch.float32) * 0.5

    print(f"=== Precision test: H={H}, K={K}, D={D} ===\n")

    # --- Stage 1: QK ---
    print("Stage 1: QK matmul (Cube BF16→FP32)")
    s_tla = torch.full((H, K), float("nan"), dtype=torch.float32, device="npu")
    tla_qk_tiled[(1,)](q.npu(), kv.npu(), s_tla,
                        H=H, KTOTAL=K, D=D, BASE_N_=BASE_N, BASE_K_=BASE_K)
    torch.npu.synchronize()

    from attention_matmul import QueryKeyMatmulOp
    s_dsl = torch.empty(H, K, dtype=torch.float32, device="npu")
    QueryKeyMatmulOp(H, K, D).run(s_dsl, q.npu(), kv.npu())
    torch.npu.synchronize()

    s_ref = q.float() @ kv.float().T
    diff_qk = (s_tla.cpu() - s_dsl.cpu()).abs()
    print(f"  TLA vs DSL:  max_err={diff_qk.max():.6e}  {'BITWISE' if diff_qk.max()==0 else 'CLOSE'}")
    print(f"  vs FP32 ref: max_err={(s_tla.cpu()-s_ref).abs().max():.6e}")

    # --- Stage 2: Softmax ---
    print("\nStage 2: Softmax with attn_sink (SIMD FP32→BF16)")
    # Use DSL scores as input (proven correct)
    p_tla = torch.full((H, K), float("nan"), dtype=torch.bfloat16, device="npu")
    tla_softmax[(1,)](s_dsl, sink.npu(), p_tla, H=H, K=K)
    torch.npu.synchronize()

    p_ref = cpu_softmax_with_sink(s_dsl.cpu(), sink)
    diff_sm = (p_tla.cpu().float() - p_ref.float()).abs()
    print(f"  TLA vs CPU:  max_err={diff_sm.max():.6e}  mean={diff_sm.mean():.6e}")

    # --- Stage 3: PV ---
    print("\nStage 3: PV matmul (Cube BF16→BF16)")
    o_tla = torch.full((H, D), float("nan"), dtype=torch.bfloat16, device="npu")
    tla_pv_tiled[(1,)](p_ref.npu(), kv.npu(), o_tla,
                        H=H, K=K, D=D, PV_BN=BASE_N, PV_BK=BASE_K)
    torch.npu.synchronize()

    from attention_matmul import ProbabilityValueMatmulOp
    o_dsl = torch.empty(H, D, dtype=torch.bfloat16, device="npu")
    ProbabilityValueMatmulOp(H, D, K, base_n=64).run(o_dsl, p_ref.npu(), kv.npu())
    torch.npu.synchronize()

    o_ref = (p_ref.float() @ kv.float()).to(torch.bfloat16)
    diff_pv = (o_tla.cpu().float() - o_dsl.cpu().float()).abs()
    print(f"  TLA vs DSL:  max_err={diff_pv.max():.6e}  {'BITWISE' if diff_pv.max()==0 else 'CLOSE'}")
    print(f"  vs FP32 ref: max_err={(o_tla.cpu().float()-o_ref.float()).abs().max():.6e}")

    # --- Full pipeline: QK → softmax → PV ---
    print("\n=== Full pipeline: QK → softmax → PV ===")
    full_tla = torch.full((H, D), float("nan"), dtype=torch.bfloat16, device="npu")
    s_full = torch.full((H, K), float("nan"), dtype=torch.float32, device="npu")
    p_full = torch.full((H, K), float("nan"), dtype=torch.bfloat16, device="npu")
    tla_qk_tiled[(1,)](q.npu(), kv.npu(), s_full,
                        H=H, KTOTAL=K, D=D, BASE_N_=BASE_N, BASE_K_=BASE_K)
    torch.npu.synchronize()
    tla_softmax[(1,)](s_full, sink.npu(), p_full, H=H, K=K)
    torch.npu.synchronize()
    tla_pv_tiled[(1,)](p_full, kv.npu(), full_tla,
                        H=H, K=K, D=D, PV_BN=BASE_N, PV_BK=BASE_K)
    torch.npu.synchronize()

    full_ref = cpu_softmax_with_sink(s_ref, sink)
    full_ref_out = (full_ref.float() @ kv.float()).to(torch.bfloat16)
    diff_full = (full_tla.cpu().float() - full_ref_out.float()).abs()
    print(f"  TLA pipeline vs CPU ref: max_err={diff_full.max():.6e}  mean={diff_full.mean():.6e}")
    torch.testing.assert_close(full_tla.cpu(), full_ref_out, rtol=0.02, atol=0.015)
    print("  PASS: full pipeline within (rtol=0.02, atol=0.015)")


if __name__ == "__main__":
    main()
