# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TLA port of rotary position embedding and query scaling helpers.

Ported from cannbot-dsl CompactRotaryEmbeddingKernel and mega_mla_kernel
inline helpers. Preserves FP32 arithmetic and BF16 rounding.
"""

import triton
import triton.language as tl
import triton.language.extra.tla as tla

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_CHUNK_ELEMENTS: tl.constexpr = 64  # 64 FP32 lanes per SIMD register


# ---------------------------------------------------------------------------
# upcast_bf16_input: BF16 UB → FP32 UB for a single row (m=1, d features)
# ---------------------------------------------------------------------------
@tla.jit
def upcast_bf16_input(
    xb_ub,       # UB buffer (1, d) bf16
    x_f_ub,      # UB buffer (1, d) f32
    D: tl.constexpr,
):
    """BF16 → FP32 upcast for one row of D features."""
    with tla.scope.simd():
        for c in range(D // VECTOR_CHUNK_ELEMENTS):
            off = c * VECTOR_CHUNK_ELEMENTS
            # vlds UNPK_B16 → vextf to fp32
            raw = tla.simd.vlds(xb_ub[0, off], dist="UNPK_B16")
            fp32 = tla.simd.vextf(raw, tl.float32, part="part_even")
            tla.simd.vsts(x_f_ub[0, off], fp32)
        tla.simd.membar("VST_VLD")


# ---------------------------------------------------------------------------
# apply_rotary_embedding_vf: compact RoPE (m=1, rope_dim=64) inside VF scope
# ---------------------------------------------------------------------------
@tla.jit
def apply_rotary_embedding_vf(
    x_f_ub,       # UB (1, D) f32  — input features (already FP32)
    cos_ub,       # UB (1, 64) f32 — 32 cosine coefficients (compact layout)
    sin_ub,       # UB (1, 64) f32 — 32 sine coefficients
    outb_ub,      # UB (1, D) bf16 — output
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,       # 64
    FORWARD: tl.constexpr,        # True for forward, False for inverse
    output_offset: tl.constexpr = 0,  # BF16 element offset in outb_ub
):
    """Rotate the last ROPE_DIM features, pass-through the prefix. No wrapping VF scope."""
    mask64_16 = tla.simd.pge(64, 128)  # 64 bf16 lanes
    one = tla.simd.vdups(1, tl.uint32)
    rnd = tla.simd.vdups(0x7FFF, tl.uint32)
    hi = tla.simd.vdups(0xFFFF0000, tl.uint32)
    head_chunks = (D - ROPE_DIM) // VECTOR_CHUNK_ELEMENTS

    # --- helper: round FP32 bit pattern to nearest BF16 ---
    # t = (b + 0x7FFF + ((b>>16)&1)); return t & 0xFFFF0000
    # This preserves the intermediate BF16 quantization before scale.

    for c in range(D // VECTOR_CHUNK_ELEMENTS):
        off = c * VECTOR_CHUNK_ELEMENTS
        if c < head_chunks:
            # Non-RoPE prefix: narrow back to bf16 unchanged
            x = tla.simd.vlds(x_f_ub[0, off])
            bf = tla.simd.vtruncf(x, tl.bfloat16)
            tla.simd.vsts(outb_ub[0, output_offset + off], bf)
        else:
            # Deinterleave even/odd pairs
            vec = tla.simd.vlds(x_f_ub[0, off])
            even = tla.simd.vdintlv(vec, tla.simd.POS.LOWEST)
            odd = tla.simd.vdintlv(vec, tla.simd.POS.HIGHEST)

            # Load compact cos/sin (32 unique values → broadcast to 64 lanes)
            co = tla.simd.vlds(cos_ub[0, 0])
            si = tla.simd.vlds(sin_ub[0, 0])

            sgn = -1.0 if FORWARD else 1.0
            # oe = even*cos + sgn*(odd*sin)
            oe = tla.simd.vadd(
                tla.simd.vmul(even, co),
                tla.simd.vmuls(tla.simd.vmul(odd, si), sgn),
            )
            # oo = odd*cos - sgn*(even*sin)
            oo = tla.simd.vsub(
                tla.simd.vmul(odd, co),
                tla.simd.vmuls(tla.simd.vmul(even, si), sgn),
            )

            # Round-to-BF16 bit manipulation (preserves exact baseline rounding)
            oe_u = tla.simd.vbitcast(oe, tl.uint32)
            oo_u = tla.simd.vbitcast(oo, tl.uint32)
            # round_to_bf16_bits(b) = (b + 0x7FFF + ((b>>16)&1)) & 0xFFFF0000
            oe_t = tla.simd.vadd(
                tla.simd.vadd(oe_u, rnd),
                tla.simd.vand(tla.simd.vshr(oe_u, 16), one),
            )
            oe_r = tla.simd.vand(oe_t, hi)
            oo_t = tla.simd.vadd(
                tla.simd.vadd(oo_u, rnd),
                tla.simd.vand(tla.simd.vshr(oo_u, 16), one),
            )
            oo_r = tla.simd.vand(oo_t, hi)

            # Pack: interleave two bf16 halves into one 32-bit word
            packed = tla.simd.vor(
                tla.simd.vshr(oe_r, 16),
                tla.simd.vand(oo_r, hi),
            )
            bf_packed = tla.simd.vbitcast(packed, tl.bfloat16)
            tla.simd.vsts(outb_ub[0, output_offset + off], bf_packed)
    tla.simd.membar("VST_VLD")


# ---------------------------------------------------------------------------
# apply_rotary_embedding: wrapper that opens the SIMD scope
# ---------------------------------------------------------------------------
@tla.jit
def apply_rotary_embedding(
    x_f_ub, cos_ub, sin_ub, outb_ub,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    FORWARD: tl.constexpr,
):
    with tla.scope.simd():
        apply_rotary_embedding_vf(x_f_ub, cos_ub, sin_ub, outb_ub,
                                  D, ROPE_DIM, FORWARD)


# ---------------------------------------------------------------------------
# scale_bf16_query_row: multiply one BF16 row by a FP32 scalar and write GM
# ---------------------------------------------------------------------------
@tla.jit
def scale_bf16_query_row(
    x_ub,         # UB (1, N) bf16 — source
    y_gm,         # GM output (rows, N) bf16
    r,            # row index in y_gm
    N: tl.constexpr,
    mul_const,    # FP32 scalar multiplier (1/sqrt(d))
):
    nc = N // VECTOR_CHUNK_ELEMENTS
    y_ub = tla.alloc((1, N), tl.bfloat16, space="ub")
    with tla.scope.simd():
        cmul = tla.simd.vdups(mul_const, tl.float32)
        for c in range(nc):
            off = c * VECTOR_CHUNK_ELEMENTS
            # BF16 → FP32
            raw = tla.simd.vlds(x_ub[0, off], dist="UNPK_B16")
            x = tla.simd.vextf(raw, tl.float32, part="part_even")
            x = tla.simd.vmul(x, cmul)
            bf = tla.simd.vtruncf(x, tl.bfloat16)
            tla.simd.vsts(y_ub[0, off], bf)
        tla.simd.membar("VST_VLD")
    # UB → GM
    dst = tla.subview(y_gm, offsets=[r, 0], sizes=[1, N])
    tla.copy(y_ub, dst)


# ---------------------------------------------------------------------------
# apply_query_rope_scale_paired: fused D=512, RoPE=64 specialization
# Preserves BF16 rounding after RoPE and again after scaling.
# ---------------------------------------------------------------------------
@tla.jit
def apply_query_rope_scale_paired(
    src,          # UB (1, D) bf16  — raw query row
    scratch,      # UB (1, ROPE_DIM) f32 — scratch for RoPE range
    cos_ub,       # UB (1, 64) f32
    sin_ub,       # UB (1, 64) f32
    dst,          # UB (1, D) bf16  — output (scaled + RoPE'd)
    D: tl.constexpr,              # 512
    ROPE_DIM: tl.constexpr,       # 64
    scale,                         # 1/sqrt(d) FP32
):
    """D=512, RoPE=64 specialization. Preserve BF16 rounding after RoPE
    and again after scaling, matching the baseline exactly."""
    prefix = D - ROPE_DIM  # 448

    with tla.scope.simd():

        # --- prefix (non-RoPE): BF16 upcast → scale → BF16 → store ---
        for c in range(prefix // (2 * VECTOR_CHUNK_ELEMENTS)):
            off0 = 2 * c * VECTOR_CHUNK_ELEMENTS
            off1 = (2 * c + 1) * VECTOR_CHUNK_ELEMENTS
            raw0 = tla.simd.vlds(src[0, off0], dist="UNPK_B16")
            raw1 = tla.simd.vlds(src[0, off1], dist="UNPK_B16")
            f0 = tla.simd.vextf(raw0, tl.float32, part="part_even")
            f1 = tla.simd.vextf(raw1, tl.float32, part="part_even")
            f0 = tla.simd.vmuls(f0, scale)
            f1 = tla.simd.vmuls(f1, scale)
            b0 = tla.simd.vtruncf(f0, tl.bfloat16)
            b1 = tla.simd.vtruncf(f1, tl.bfloat16)
            tla.simd.vsts(dst[0, off0], b0)
            tla.simd.vsts(dst[0, off1], b1)

        # Handle odd trailing chunk if (prefix // VCE) is odd
        if (prefix // VECTOR_CHUNK_ELEMENTS) % 2:
            offset = prefix - VECTOR_CHUNK_ELEMENTS
            raw = tla.simd.vlds(src[0, offset], dist="UNPK_B16")
            x = tla.simd.vextf(raw, tl.float32, part="part_even")
            b = tla.simd.vtruncf(tla.simd.vmuls(x, scale),
                                 tl.bfloat16)
            tla.simd.vsts(dst[0, offset], b)

        # --- RoPE tail: upcast into scratch FP32 buffer ---
        for c in range(ROPE_DIM // VECTOR_CHUNK_ELEMENTS):
            raw = tla.simd.vlds(src[0, prefix + c * VECTOR_CHUNK_ELEMENTS], dist="UNPK_B16")
            x = tla.simd.vextf(raw, tl.float32, part="part_even")
            tla.simd.vsts(scratch[0, c * VECTOR_CHUNK_ELEMENTS], x)
        tla.simd.membar("VST_VLD")

    # --- second VF region: RoPE + scale with explicit BF16 rounding ---
    with tla.scope.simd():
        mask64 = tla.simd.pge(64, 128)  # 64 bf16 lanes
        one = tla.simd.vdups(1, tl.uint32)
        rnd = tla.simd.vdups(0x7FFF, tl.uint32)
        hi = tla.simd.vdups(0xFFFF0000, tl.uint32)

        for c in range(ROPE_DIM // VECTOR_CHUNK_ELEMENTS):
            vec = tla.simd.vlds(scratch[0, c * VECTOR_CHUNK_ELEMENTS])
            even = tla.simd.vdintlv(vec, tla.simd.POS.LOWEST)
            odd = tla.simd.vdintlv(vec, tla.simd.POS.HIGHEST)
            co = tla.simd.vlds(cos_ub[0, c * (VECTOR_CHUNK_ELEMENTS // 2)])
            si = tla.simd.vlds(sin_ub[0, c * (VECTOR_CHUNK_ELEMENTS // 2)])
            ec = tla.simd.vmul(even, co)
            oc = tla.simd.vmul(odd, co)
            es = tla.simd.vmul(even, si)
            os_ = tla.simd.vmul(odd, si)
            oe = tla.simd.vadd(ec, tla.simd.vmuls(os_, -1.0))
            oo = tla.simd.vsub(oc, tla.simd.vmuls(es, -1.0))

            # Preserve the intermediate BF16 quantization before scale
            # round_to_bf16_bits(v) helper inlined
            oe_u = tla.simd.vbitcast(oe, tl.uint32)
            oe_t = tla.simd.vadd(
                tla.simd.vadd(oe_u, rnd),
                tla.simd.vand(tla.simd.vshr(oe_u, 16), one))
            oe_r = tla.simd.vand(oe_t, hi)
            oe_scaled = tla.simd.vmuls(
                tla.simd.vbitcast(oe_r, tl.float32), scale)

            oo_u = tla.simd.vbitcast(oo, tl.uint32)
            oo_t = tla.simd.vadd(
                tla.simd.vadd(oo_u, rnd),
                tla.simd.vand(tla.simd.vshr(oo_u, 16), one))
            oo_r = tla.simd.vand(oo_t, hi)
            oo_scaled = tla.simd.vmuls(
                tla.simd.vbitcast(oo_r, tl.float32), scale)

            # Final BF16 round + interleaved pack
            oe_su = tla.simd.vbitcast(oe_scaled, tl.uint32)
            oe_st = tla.simd.vadd(
                tla.simd.vadd(oe_su, rnd),
                tla.simd.vand(tla.simd.vshr(oe_su, 16), one))
            oe_sr = tla.simd.vand(oe_st, hi)

            oo_su = tla.simd.vbitcast(oo_scaled, tl.uint32)
            oo_st = tla.simd.vadd(
                tla.simd.vadd(oo_su, rnd),
                tla.simd.vand(tla.simd.vshr(oo_su, 16), one))
            oo_sr = tla.simd.vand(oo_st, hi)

            packed = tla.simd.vor(
                tla.simd.vshr(oe_sr, 16),
                oo_sr,
            )
            bf_packed = tla.simd.vbitcast(packed, tl.bfloat16)
            tla.simd.vsts(dst[0, prefix + c * VECTOR_CHUNK_ELEMENTS],
                          bf_packed, mask=mask64)
        tla.simd.membar("VST_VLD")
