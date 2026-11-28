# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TLA port of cannbot-dsl qkv_projection.py.

Variable-head Q/KV projection with split-K=2 FP32 atomic accumulation.

Projection phases:
  P0 AIV : device MX-quant of x                     -> x8, xs
  barrier
  P1 AIC : wq_a on blocks 0..19, wkv on blocks 20..27; M<=64,K=2560/core
  barrier
  P2 AIV : q_norm + MX-quant on qr -> qn8, qns;  kv_norm on kv -> kvn
  barrier / rewind
  P3 AIC : wq_b with baseN=256 and block-stride N scheduling -> qfull

Original: cannbot-dsl/samples/attention/mega_mla/qkv_projection.py
"""

import triton
import triton.language as tl
import triton.language.extra.tla as tla

# ---------------------------------------------------------------------------
# Constants (match cannbot-dsl qkv_projection.py)
# ---------------------------------------------------------------------------
HEAD_DIM = 512
HID = 5120
QLORA = 1280
KVD = 512
CUBES = 32
BN1, BK1, KL1_1, SKL1_1 = 128, 256, 512, 5120
BN2, BK2, KL1_2, SKL1_2 = 256, 128, 256, 1280
GROUP = 32
VECTOR_CHUNK_ELEMENTS = 64  # Logical feature elements per SIMD chunk (= 64 fp32 lanes)
E4M3_ELE_EMAX = 8
E8M0_BIAS = 127
E4M3_MAX = 448.0


# ===========================================================================
# P0: per-row MX-FP8 quantization of hidden states
# ===========================================================================
@tla.jit
def quantize_mxfp8_rows(
    x_gm,     # GM: (rows, n) bfloat16 input
    y_gm,     # GM: (rows, n) fp8e4m3 output
    s_gm,     # GM: (rows, n//64, 2) uint8 E8M0 scales
    block,    # AIC block index
    nblk,     # total blocks
    rows: tl.constexpr,
    n: tl.constexpr,
):
    """P0: per-row MX quantization. Each AIV quantizes disjoint rows."""
    nc: tl.constexpr = n // VECTOR_CHUNK_ELEMENTS

    x_ub = tla.alloc((1, n), tl.bfloat16, space="ub")
    y_ub = tla.alloc((1, n), tl.uint8, space="ub")
    s_ub = tla.alloc((1, nc * 2), tl.uint8, space="ub")

    for r in range(block, rows, nblk):
        # GM -> UB：disable L2 cache for streaming input
        tla.load_gm_to_ub(
            tla.subview(x_gm, offsets=[r, 0], sizes=[1, n]),
            x_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst),
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)

        with tla.scope.simd():
            lane32 = tla.simd.pge(32, 64)
            # TLA_TODO: mask_not — approximate with XOR or vcmp
            # high mask = ~lane32 (upper 32 of 64 lanes)
            high = tla.simd.pnot(lane32)
            zero = tla.simd.vdups(0.0, tl.float32)
            c254 = tla.simd.vdups(2 * E8M0_BIAS, tl.int32)

            for c in range(nc):
                off = c * VECTOR_CHUNK_ELEMENTS
                # Load bf16, unpack to fp32
                raw = tla.simd.vlds(x_ub[0, off], dist="UNPK_B16")
                x = tla.simd.vextf(raw, tl.float32, part="part_even")

                # Split halves for per-32-element scale computation
                x_lo = tla.simd.vsel(high, x, zero)   # keep low 32 lanes
                x_hi = tla.simd.vsel(lane32, x, zero)  # keep high 32 lanes

                amax_lo = tla.simd.vreduce_max(tla.simd.vabs(x_lo))
                amax_hi = tla.simd.vreduce_max(tla.simd.vabs(x_hi))

                # Compute E8M0 biased exponent: b = max(0, floor(log2(amax)) - 8)
                b_lo = tla.simd.vmaxs(
                    tla.simd.vadds(
                        tla.simd.vbitcast(
                            tla.simd.vshr(tla.simd.vbitcast(amax_lo, tl.uint32), 23),
                            tl.int32,
                        ),
                        -E4M3_ELE_EMAX,
                    ),
                    0,
                )
                b_hi = tla.simd.vmaxs(
                    tla.simd.vadds(
                        tla.simd.vbitcast(
                            tla.simd.vshr(tla.simd.vbitcast(amax_hi, tl.uint32), 23),
                            tl.int32,
                        ),
                        -E4M3_ELE_EMAX,
                    ),
                    0,
                )

                # Inverse scale: 2^(254 - b) reinterpreted as fp32
                inv_lo = tla.simd.vdupv(
                    tla.simd.vbitcast(
                        tla.simd.vshl(tla.simd.vsub(c254, b_lo), 23),
                        tl.float32,
                    ),
                    tla.simd.POS.LOWEST,
                )
                inv_hi = tla.simd.vdupv(
                    tla.simd.vbitcast(
                        tla.simd.vshl(tla.simd.vsub(c254, b_hi), 23),
                        tl.float32,
                    ),
                    tla.simd.POS.LOWEST,
                )
                inv = tla.simd.vsel(high, inv_lo, inv_hi)

                # Scale and clamp to E4M3 range [-448, 448]
                y = tla.simd.vmins(
                    tla.simd.vmaxs(tla.simd.vmul(x, inv), -E4M3_MAX),
                    E4M3_MAX,
                )
                # Cast to fp8 E4M3
                y_fp8 = tla.simd.vtruncf(y, tl.float8e4nv)
                # Pack and store
                tla.simd.vsts(y_ub[0, off], y_fp8)

                # Store E8M0 scales (lowest lane of b_lo, b_hi → one byte each)
                # TLA_TODO: vstore_first equivalent — store lowest lane to UB
                tla.simd.vsts(s_ub[0, 2 * c], tla.simd.vbitcast(b_lo, tl.uint8), dist="ONEPT_B32")
                tla.simd.vsts(s_ub[0, 2 * c + 1], tla.simd.vbitcast(b_hi, tl.uint8), dist="ONEPT_B32")
            tla.simd.membar("VST_VLD")

        tla.sync.barrier(tla.sync.PIPE.ALL)
        # UB -> GM
        tla.store_ub_to_gm(y_ub, tla.subview(y_gm, offsets=[r, 0], sizes=[1, n]))
        tla.store_ub_to_gm(s_ub, tla.subview(s_gm, offsets=[r, 0, 0], sizes=[1, nc, 2]))
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# P2a: RMSNorm + MX-FP8 quantization for query
# ===========================================================================
@tla.jit
def rms_norm_quantize_query(
    h_gm,     # GM: (rows, n) float32 — accumulated QR
    g_gm,     # GM: (1, n) float32 — gamma weights
    y_gm,     # GM: (rows, n) fp8e4m3 output
    s_gm,     # GM: (rows, n//64, 2) uint8 E8M0 scales
    block,
    nblk,
    rows: tl.constexpr,
    n: tl.constexpr,
):
    """P2a: RMSNorm + MX quant on the FP32 QR accumulator."""
    nc: tl.constexpr = n // VECTOR_CHUNK_ELEMENTS

    h_ub = tla.alloc((1, n), tl.float32, space="ub")
    h_bf16_ub = tla.alloc((1, n), tl.bfloat16, space="ub")
    g_ub = tla.alloc((1, n), tl.float32, space="ub")
    xf_ub = tla.alloc((1, n), tl.float32, space="ub")
    y_ub = tla.alloc((1, n), tl.uint8, space="ub")
    s_ub = tla.alloc((1, nc * 2), tl.uint8, space="ub")

    tla.load_gm_to_ub(tla.subview(g_gm, offsets=[0, 0], sizes=[1, n]), g_ub)

    for r in range(block, rows, nblk):
        tla.load_gm_to_ub(
            tla.subview(h_gm, offsets=[r, 0], sizes=[1, n]),
            h_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst),
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)

        with tla.scope.simd():
            lane32 = tla.simd.pge(32, 64)
            high = tla.simd.pnot(lane32)

            # r=FP32(BF16_RN(h))；平方和与归一化分子均使用 r。
            # pack/unpack 保持 64 个值连续，禁止把窄化寄存器当 FP32 重读。
            for c in range(nc):
                val = tla.simd.vlds(h_ub[0, c * VECTOR_CHUNK_ELEMENTS])
                rounded = tla.simd.vtruncf(val, tl.bfloat16)
                tla.simd.vsts(h_bf16_ub[0, c * VECTOR_CHUNK_ELEMENTS], rounded)
            tla.simd.membar("VST_VLD")

            # Compute sum of squares
            acc = tla.simd.vdups(0.0, tl.float32)
            for c in range(nc):
                raw = tla.simd.vlds(h_bf16_ub[0, c * VECTOR_CHUNK_ELEMENTS], dist="UNPK_B16")
                xf = tla.simd.vextf(raw, tl.float32, part="part_even")
                tla.simd.vsts(xf_ub[0, c * VECTOR_CHUNK_ELEMENTS], xf)
                acc = tla.simd.vadd(acc, tla.simd.vmul(xf, xf))
            tla.simd.membar("VST_VLD")

            # RMS: rstd = 1 / sqrt(mean + eps)
            mean = tla.simd.vmuls(tla.simd.vreduce_sum(acc), 1.0 / n)
            rstd = tla.simd.vdupv(
                tla.simd.vdiv(
                    tla.simd.vdups(1.0, tl.float32),
                    tla.simd.vsqrt(tla.simd.vadds(mean, 1.0e-20)),
                ),
                tla.simd.POS.LOWEST,
            )

            zero = tla.simd.vdups(0.0, tl.float32)
            c254 = tla.simd.vdups(2 * E8M0_BIAS, tl.int32)

            for c in range(nc):
                off = c * VECTOR_CHUNK_ELEMENTS
                xf = tla.simd.vlds(xf_ub[0, off])
                g = tla.simd.vlds(g_ub[0, off])
                y = tla.simd.vmul(tla.simd.vmul(xf, rstd), g)

                # Compute per-half-vector E8M0 scales (same as quantize_mxfp8_rows)
                y_lo = tla.simd.vsel(high, y, zero)
                y_hi = tla.simd.vsel(lane32, y, zero)
                amax_lo = tla.simd.vreduce_max(tla.simd.vabs(y_lo))
                amax_hi = tla.simd.vreduce_max(tla.simd.vabs(y_hi))

                b_lo = tla.simd.vmaxs(
                    tla.simd.vadds(
                        tla.simd.vbitcast(
                            tla.simd.vshr(tla.simd.vbitcast(amax_lo, tl.uint32), 23),
                            tl.int32,
                        ),
                        -E4M3_ELE_EMAX,
                    ),
                    0,
                )
                b_hi = tla.simd.vmaxs(
                    tla.simd.vadds(
                        tla.simd.vbitcast(
                            tla.simd.vshr(tla.simd.vbitcast(amax_hi, tl.uint32), 23),
                            tl.int32,
                        ),
                        -E4M3_ELE_EMAX,
                    ),
                    0,
                )

                inv_lo = tla.simd.vdupv(
                    tla.simd.vbitcast(
                        tla.simd.vshl(tla.simd.vsub(c254, b_lo), 23),
                        tl.float32,
                    ),
                    tla.simd.POS.LOWEST,
                )
                inv_hi = tla.simd.vdupv(
                    tla.simd.vbitcast(
                        tla.simd.vshl(tla.simd.vsub(c254, b_hi), 23),
                        tl.float32,
                    ),
                    tla.simd.POS.LOWEST,
                )
                inv = tla.simd.vsel(high, inv_lo, inv_hi)
                yq = tla.simd.vmins(
                    tla.simd.vmaxs(tla.simd.vmul(y, inv), -E4M3_MAX),
                    E4M3_MAX,
                )
                yq_fp8 = tla.simd.vtruncf(yq, tl.float8e4nv)
                tla.simd.vsts(y_ub[0, off], yq_fp8)
                tla.simd.vsts(s_ub[0, 2 * c], tla.simd.vbitcast(b_lo, tl.uint8), dist="ONEPT_B32")
                tla.simd.vsts(s_ub[0, 2 * c + 1], tla.simd.vbitcast(b_hi, tl.uint8), dist="ONEPT_B32")
            tla.simd.membar("VST_VLD")

        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.store_ub_to_gm(y_ub, tla.subview(y_gm, offsets=[r, 0], sizes=[1, n]))
        tla.store_ub_to_gm(s_ub, tla.subview(s_gm, offsets=[r, 0, 0], sizes=[1, nc, 2]))
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# P2b: RMSNorm for KV (BF16 output)
# ===========================================================================
@tla.jit
def rms_norm_key_value_vf(
    h_ub,       # UB: (1, n) float32
    g_ub,       # UB: (1, n) float32
    o_ub,       # UB: (1, n) bfloat16 output
    h_bf16_ub,  # UB: (1, n) bfloat16 scratch
    xf_ub,      # UB: (1, n) float32 scratch
    n: tl.constexpr,
):
    """RMS arithmetic fragment; caller owns scratch and an active simd scope.

    r=FP32(BF16_RN(h))；先舍入完整 atomic 累加结果，再计算 RMS。
    """
    nc: tl.constexpr = n // VECTOR_CHUNK_ELEMENTS

    # Round FP32 accumulator to BF16 first (match baseline precision)
    for c in range(nc):
        val = tla.simd.vlds(h_ub[0, c * VECTOR_CHUNK_ELEMENTS])
        rounded = tla.simd.vtruncf(val, tl.bfloat16)
        tla.simd.vsts(h_bf16_ub[0, c * VECTOR_CHUNK_ELEMENTS], rounded)
    tla.simd.membar("VST_VLD")

    # Widen back and accumulate squared sum
    acc = tla.simd.vdups(0.0, tl.float32)
    for c in range(nc):
        raw = tla.simd.vlds(h_bf16_ub[0, c * VECTOR_CHUNK_ELEMENTS], dist="UNPK_B16")
        xf = tla.simd.vextf(raw, tl.float32, part="part_even")
        tla.simd.vsts(xf_ub[0, c * VECTOR_CHUNK_ELEMENTS], xf)
        acc = tla.simd.vadd(acc, tla.simd.vmul(xf, xf))
    tla.simd.membar("VST_VLD")

    mean = tla.simd.vmuls(tla.simd.vreduce_sum(acc), 1.0 / n)
    rstd = tla.simd.vdupv(
        tla.simd.vdiv(
            tla.simd.vdups(1.0, tl.float32),
            tla.simd.vsqrt(tla.simd.vadds(mean, 1.0e-20)),
        ),
        tla.simd.POS.LOWEST,
    )

    for c in range(nc):
        xf = tla.simd.vlds(xf_ub[0, c * VECTOR_CHUNK_ELEMENTS])
        g = tla.simd.vlds(g_ub[0, c * VECTOR_CHUNK_ELEMENTS])
        y = tla.simd.vmul(tla.simd.vmul(xf, rstd), g)
        y_bf16 = tla.simd.vtruncf(y, tl.bfloat16)
        tla.simd.vsts(o_ub[0, c * VECTOR_CHUNK_ELEMENTS], y_bf16)
    tla.simd.membar("VST_VLD")


@tla.jit
def rms_norm_key_value(
    h_gm,       # GM: (rows, n) float32
    g_gm,       # GM: (1, n) float32
    out_gm,     # GM: (rows, n) bfloat16
    block,
    nblk,
    rows: tl.constexpr,
    n: tl.constexpr,
):
    """P2b: plain RMSNorm (BF16 output in GM)."""
    h_ub = tla.alloc((1, n), tl.float32, space="ub")
    g_ub = tla.alloc((1, n), tl.float32, space="ub")
    o_ub = tla.alloc((1, n), tl.bfloat16, space="ub")
    h_bf16_ub = tla.alloc((1, n), tl.bfloat16, space="ub")
    xf_ub = tla.alloc((1, n), tl.float32, space="ub")

    tla.load_gm_to_ub(tla.subview(g_gm, offsets=[0, 0], sizes=[1, n]), g_ub)

    for r in range(block, rows, nblk):
        tla.load_gm_to_ub(
            tla.subview(h_gm, offsets=[r, 0], sizes=[1, n]),
            h_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst),
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)
        with tla.scope.simd():
            rms_norm_key_value_vf(h_ub, g_ub, o_ub, h_bf16_ub, xf_ub, n)
        tla.sync.barrier(tla.sync.PIPE.ALL)
        tla.store_ub_to_gm(o_ub, tla.subview(out_gm, offsets=[r, 0], sizes=[1, n]))
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# P3: WqB output tile (MXfp8 matmul)
# ===========================================================================
@tla.jit
def compute_wqb_output_tile(
    out_gm,       # GM: (m, q_width) bfloat16  — full Q output
    a_gm,         # GM: (m, QLORA) fp8e4m3 — quantized query norm
    b_gm,         # GM: (q_width, QLORA) fp8e4m3 — WqB weight (FRACTAL_NZ)
    sa_gm,        # GM: (m, QLORA//32, 2) E8M0 — query scale
    sb_gm,        # GM: (q_width, QLORA//32, 2) E8M0 — WqB scale
    m_idx,        # M tile index
    n_idx,        # N tile index
    BASE_M: tl.constexpr,
    BASE_N: tl.constexpr = BN2,
    BASE_K: tl.constexpr = BK2,
    KL1: tl.constexpr = KL1_2,
    SKL1: tl.constexpr = SKL1_2,
):
    """One (M, N) output tile of the phase-3 WqB matmul.

    This is a simplified TLA representation; the full pipeline uses L1 ring
    buffers and K-tiled MMAD with scale loading.
    """
    with tla.scope.cube():
        # NZ layout for WqB MXfp8 matmul
        # A[BASE_M, KL1] → (KL1//16, BASE_M//16, 16, 16)
        # B[BASE_N, KL1] with transpose_b=True → (KL1//16, BASE_N//16, 16, 16)
        # acc[BASE_M, BASE_N] → (BASE_N//16, BASE_M//16, 16, 16)
        a_l1 = tla.alloc((KL1 // 16, BASE_M // 16, 16, 16), tl.float8e4nv, space="l1")
        b_l1 = tla.alloc((KL1 // 16, BASE_N // 16, 16, 16), tl.float8e4nv, space="l1")
        # TLA_TODO: MX scale layout may differ; keep 2D for load_gm_to_l1_mx_scale
        sa_l1 = tla.alloc((BASE_M, KL1 // 32), tl.uint8, space="l1")
        sb_l1 = tla.alloc((BASE_N, KL1 // 32), tl.uint8, space="l1")
        acc = tla.alloc((BASE_N // 16, BASE_M // 16, 16, 16), tl.float32, space="l0c")

        k_tiles: tl.constexpr = QLORA // KL1
        k_l0_tiles: tl.constexpr = KL1 // BASE_K

        for kt in range(k_tiles):
            # Load A tile and scale
            tla.load_gm_to_l1(
                tla.subview(a_gm, offsets=[m_idx * BASE_M, kt * KL1], sizes=[BASE_M, KL1]),
                a_l1,
            )
            tla.load_gm_to_l1(
                tla.subview(b_gm, offsets=[n_idx * BASE_N, kt * KL1], sizes=[BASE_N, KL1]),
                b_l1,
            )
            # Load MX scales
            tla.load_gm_to_l1_mx_scale(
                tla.subview(sa_gm, offsets=[m_idx * BASE_M, kt * (KL1 // 32)], sizes=[BASE_M, KL1 // 32]),
                sa_l1,
            )
            tla.load_gm_to_l1_mx_scale(
                tla.subview(sb_gm, offsets=[n_idx * BASE_N, kt * (KL1 // 32)], sizes=[BASE_N, KL1 // 32]),
                sb_l1,
            )

            # MXfp8 matmul accumulation
            tla.cube.mmad_mx(
                a_l1, b_l1, sa_l1, sb_l1, acc,
                format_a=tla.MxFormat.E4M3,
                format_b=tla.MxFormat.E4M3,
                clear_accum=(kt == 0),
            )

        # Fixpipe write back to GM (NZ2ND layout conversion, FP32->BF16 quant)
        tla.store_l0c_to_gm(
            acc,
            tla.subview(out_gm, offsets=[m_idx * BASE_M, n_idx * BASE_N], sizes=[BASE_M, BASE_N]),
            params=tla.FixpipeL0C2GMParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                pre_quant=tla.FixpipePreQuantMode.F322BF16,
            ),
        )
