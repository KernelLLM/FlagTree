#!/usr/bin/env python3
"""Minimal precision comparison: TLA vs cannbot-dsl QK matmul.

Stage 1: Validate a single BF16 matmul tile produces bitwise-identical
results between TLA cube.mmad and cannbot-dsl matmul.

Usage:
    python3 test_precision_qk.py          # run on NPU
    python3 test_precision_qk.py --ir     # IR-only (no NPU needed)
"""
import sys
import os

# TLA test utilities
TLA_UT = "/home/ricky/git/Triton-Ascend-tla/third_party/ascend/unittest/tla_ut"
sys.path.insert(0, TLA_UT)

# cannbot-dsl mega_mla samples
MEGA_DIR = "/dev/shm/yuansheng/CANN/cannbot-dsl/samples/attention/mega_mla"
sys.path.insert(0, MEGA_DIR)

import torch
import triton.language as tl
import triton.language.extra.tla as tla

# ---------------------------------------------------------------------------
# TLA kernel: single-tile QK matmul   S[H, BN] = Q[H, D] @ K[BN, D]^T
#
# Following the proven gemm_basic.py pattern exactly:
#   L1 uses 4D NZ layout (k//16, m//16, 16, 16)
#   L0C uses 4D NZ layout (n//16, m//16, 16, 16)
# ---------------------------------------------------------------------------
@tla.jit
def tla_qk_tile(
    Q_ptr, K_ptr, S_ptr,
    H: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
    BASE_K: tl.constexpr,
):
    """S[H, BN] = Q[H, D] @ K[BN, D]^T, BF16 inputs, FP32 output."""
    with tla.scope.cube():
        q_l1 = tla.alloc((D // 16, H // 16, 16, 16), tl.bfloat16, space="l1")
        k_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        acc = tla.alloc((BN // 16, H // 16, 16, 16), tl.float32, space="l0c")

        gm_q = tla.view(Q_ptr, (H, D))
        gm_k = tla.view(K_ptr, (BN, D))
        gm_s = tla.view(S_ptr, (H, BN))

        tla.load_gm_to_l1(gm_q, q_l1)
        tla.load_gm_to_l1(gm_k, k_l1)

        tla.cube.mmad(q_l1, k_l1, acc,
                      transpose_a=False, transpose_b=True,
                      clear_accum=True)

        tla.store_l0c_to_gm(
            acc, gm_s,
            tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))


# ---------------------------------------------------------------------------
# Host test
# ---------------------------------------------------------------------------
def run_tla_qk(H, BN, D):
    """Launch the TLA QK kernel on NPU and return FP32 scores on CPU."""
    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16)
    k = (torch.randn(BN, D) * 0.1).to(torch.bfloat16)
    s_tla = torch.full((H, BN), float("nan"), dtype=torch.float32, device="npu")

    tla_qk_tile[(1,)](
        q.npu(), k.npu(), s_tla,
        H=H, BN=BN, D=D, BASE_K=D,
    )
    torch.npu.synchronize()
    return q, k, s_tla.cpu()


def run_dsl_qk(H, BN, D):
    """Launch the cannbot-dsl QK kernel on NPU and return FP32 scores on CPU."""
    from attention_matmul import QueryKeyMatmulOp

    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16)
    k = (torch.randn(BN, D) * 0.1).to(torch.bfloat16)
    s_dsl = torch.empty(H, BN, dtype=torch.float32, device="npu")

    QueryKeyMatmulOp(H, BN, D).run(s_dsl, q.npu(), k.npu())
    torch.npu.synchronize()
    return q, k, s_dsl.cpu()


def main():
    ir_only = "--ir" in sys.argv
    H, BN, D = 16, 64, 512  # 16 heads, 64 K-tiles, 512 dim

    if ir_only:
        from utils.tla_test_utils import compile_offline, lower_offline
        SIGNATURE = {"Q_ptr": "*bf16", "K_ptr": "*bf16", "S_ptr": "*fp32"}
        consts = {"H": H, "BN": BN, "D": D, "BASE_K": D}
        module, ctx = compile_offline(tla_qk_tile, SIGNATURE, consts)
        text = str(lower_offline(module, ctx))
        assert "hivm.hir.mmadL1" in text, "mmad not found in lowered IR"
        assert "hivm.hir.fixpipe" in text, "fixpipe not found"
        print(f"IR PASS: tla_qk_tile H={H} BN={BN} D={D}")
        return

    import torch_npu
    if not torch.npu.is_available():
        print("SKIP: NPU unavailable")
        return
    torch.npu.set_device(0)

    print(f"Running QK matmul: H={H}, BN={BN}, D={D}")

    # Reference: FP32 math
    torch.manual_seed(42)
    q = (torch.randn(H, D) * 0.1).to(torch.bfloat16)
    k = (torch.randn(BN, D) * 0.1).to(torch.bfloat16)
    ref = q.float() @ k.float().T
    print(f"  FP32 reference: min={ref.min():.6f} max={ref.max():.6f}")

    # TLA path
    q_tla, k_tla, s_tla = run_tla_qk(H, BN, D)
    print(f"  TLA output:     min={s_tla.min():.6f} max={s_tla.max():.6f}")

    # cannbot-dsl path
    q_dsl, k_dsl, s_dsl = run_dsl_qk(H, BN, D)
    print(f"  DSL output:     min={s_dsl.min():.6f} max={s_dsl.max():.6f}")

    # Compare TLA vs reference
    tla_err = (s_tla - ref).abs()
    print(f"  TLA vs ref:     max_err={tla_err.max():.6e} mean_err={tla_err.mean():.6e}")

    # Compare DSL vs reference
    dsl_err = (s_dsl - ref).abs()
    print(f"  DSL vs ref:     max_err={dsl_err.max():.6e} mean_err={dsl_err.mean():.6e}")

    # Compare TLA vs DSL (should be bitwise identical)
    diff = (s_tla - s_dsl).abs()
    print(f"  TLA vs DSL:     max_err={diff.max():.6e} mean_err={diff.mean():.6e}")

    if diff.max() == 0:
        print("PASS: TLA and DSL produce bitwise-identical results")
    else:
        torch.testing.assert_close(s_tla, s_dsl, rtol=1e-5, atol=1e-5)
        print("PASS: TLA and DSL match within tolerance (rtol=1e-5, atol=1e-5)")


if __name__ == "__main__":
    main()
