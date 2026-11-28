# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TLA port of cache decode, quantize, dequantize, gather and paged addressing.

Ported from cannbot-dsl cache_decode.py, compressed_cache_gather.py,
paged_cache.py, and mega_mla_kernel.py inline cache helpers.
"""

import triton
import triton.language as tl
import triton.language.extra.tla as tla

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D: tl.constexpr = 512          # HEAD_DIM
VECTOR_CHUNK_ELEMENTS: tl.constexpr = 64
COMPRESSED_DECODE_DEPTH: tl.constexpr = 2
COMPRESSED_DECODE_UNROLL: tl.constexpr = 1


# ===========================================================================
# Paged cache addressing
# ===========================================================================

@tla.jit
def paged_cache_page(
    index,             # int64 logical index
    block_table_gm,    # GM page table (may be unused)
    tok,               # token index
    PAGE_SIZE: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
):
    """Resolve physical page number.
    负索引和有效前缀外的索引由调用方先屏蔽，不访问页表或cache。
    """
    page = tl.cast(index // PAGE_SIZE, tl.int64)
    if BATCH_STRIDE > 0:
        # 按token所属batch将逻辑页映射到物理页
        page = tl.cast(block_table_gm[tok // BATCH_STRIDE, page], tl.int64)
    return page


# ===========================================================================
# FP8 E4M3 cache decode (from Fp8CacheDecoder)
# ===========================================================================

@tla.jit
def fp8_cache_decode(
    data,       # UB buffer with raw FP8 data + BF16 scales
    dst,        # UB output buffer (bf16)
    row,        # row in data
    base,       # byte offset within row for data start
    DATA_W: tl.constexpr,  # data.shape[1] (e.g., 640 or 544)
):
    """Decode FP8 E4M3FN values with BF16 stored scales → BF16 output.

    Scale layout: 16 BF16 values at byte offset (base+512) in the record.
    Data layout: 512 E4M3FN bytes at byte offset (base) in the record.
    Each group of 32 elements shares one BF16 scale.
    """
    # Reinterpret data as bf16 for scale loading
    scale_half_w = DATA_W // 2  # scale buffer width in bf16 elements

    with tla.scope.simd():
        # Full mask removed: ops default to all-true
        # TLA_TODO: varange not in TLA API reference — using manual group index computation
        # groups = rr.vshr(rr.varange(0, dtypes.uint32), 5)
        # In TLA: build group indices via vdup + sequential pattern
        # For now, we approximate with a precomputed pattern

        # Load 16 BF16 scales → fp32 (unpacked to 64 lanes, only 16 meaningful)
        scale_offset = row * scale_half_w + (base + 512) // 2
        # TLA_TODO: need buffer reinterpret for scale_data view
        # scale_data = tla.view(data, shape=(data_rows, scale_half_w), ...)
        raw_scales = tla.simd.vlds(data[row, (base + 512) // 2], dist="UNPK_B16")
        sb = tla.simd.vextf(raw_scales, tl.float32, part="part_even")

        # Process 8 chunks of 64 E4M3 values, 2 at a time
        for c in range(0, 8, 2):
            # Load E4M3 bytes → unpack to fp32
            b0 = tla.simd.vlds(data[row, base + c * 64], dist="UNPK4_B8")
            b1 = tla.simd.vlds(data[row, base + (c + 1) * 64], dist="UNPK4_B8")

            # Extend FP8 → FP32
            f0 = tla.simd.vextf(b0, tl.float32, part="part_even")
            f1 = tla.simd.vextf(b1, tl.float32, part="part_even")

            # Gather per-group scales (each group of 32 elements shares one scale)
            # TLA_TODO: vgather_reg equivalent — using vgather with index vector
            # s0 corresponds to scale indices c*2 and c*2+1 for low/high halves
            s0_idx = tla.simd.vdups(c * 2, tl.uint32)
            s1_idx = tla.simd.vdups((c + 1) * 2, tl.uint32)
            s0 = tla.simd.vbitcast(
                tla.simd.vgather(sb, s0_idx, 64, "u32"),
                tl.float32)
            s1 = tla.simd.vbitcast(
                tla.simd.vgather(sb, s1_idx, 64, "u32"),
                tl.float32)

            # Scale and convert to BF16
            f0 = tla.simd.vmul(f0, s0)
            f1 = tla.simd.vmul(f1, s1)
            o0 = tla.simd.vtruncf(f0, tl.bfloat16)
            o1 = tla.simd.vtruncf(f1, tl.bfloat16)

            tla.simd.vsts(dst[row, c * 64], o0, dist="NORM_B16")
            tla.simd.vsts(dst[row, (c + 1) * 64], o1, dist="NORM_B16")
        tla.simd.membar("VST_VLD")


# ===========================================================================
# FP4 E2M1 cache decode (from Fp4CacheDecoder, unroll=1)
# ===========================================================================

@tla.jit
def fp4_cache_decode(
    data_ub,    # UB buffer with packed FP4 data + scales
    dst_ub,     # UB output buffer (bf16)
    row,        # row index
    DATA_W: tl.constexpr,     # data_ub.shape[1] (e.g., 384 for compressed)
    NATIVE_BF16_SCALES: tl.constexpr = True,
):
    """Decode packed E2M1 FP4 values with group scales → BF16.

    FP4 data: 256 bytes (512 E2M1 values packed 2 per byte) at row start.
    Scales: 64 bytes of BF16 (32 group scales) at byte offset 256.
    """
    scale_half_w = DATA_W // 2

    with tla.scope.simd():
        # Full mask removed: ops default to all-true
        # TLA_TODO: fp4x2_e2m1 reinterpret — TLA uses vunpack for 4-bit data

        # Load scales
        if NATIVE_BF16_SCALES:
            raw_s = tla.simd.vlds(data_ub[row, 128],  # scale offset = 256/2 = 128 in bf16 view
                                   dist="UNPK_B16")
            sb = tla.simd.vextf(raw_s, tl.float32, part="part_even")
        else:
            # Non-native: load raw bytes and reconstruct BF16 scales
            sb = tla.simd.vlds(data_ub[row, 256], dist="UNPK4_B8")
            # TLA_TODO: complex scale reconstruction for non-native path

        # TLA_TODO: varange for group index computation
        # groups = rr.vshr(rr.varange(0, dtypes.uint32), 3)

        # Process 4 chunks (each 64 packed FP4 bytes = 128 BF16 output elements)
        for c in range(4):
            # Load 64 bytes of packed FP4
            packed = tla.simd.vlds(data_ub[row, c * 64], dist="UNPK4_B8")

            # Unpack FP4 → BF16 (using vunpack for 4-bit data)
            # TLA_TODO: vunpack(src, elem, signed=True) for E2M1 conversion
            h0 = tla.simd.vunpack(packed, "i4", signed=True, part="part_even")

            # Gather scale for this group
            s_idx = tla.simd.vdups(c * 8, tl.uint32)
            s0 = tla.simd.vbitcast(
                tla.simd.vgather(sb, s_idx, 64, "u32"),
                tl.float32)

            # Convert to FP32, scale, then BF16
            e0 = tla.simd.vextf(h0, tl.float32, part="part_even")
            o0 = tla.simd.vextf(h0, tl.float32, part="part_odd")
            e0 = tla.simd.vmul(e0, s0)
            o0 = tla.simd.vmul(o0, s0)

            # Interleave even/odd and convert to BF16
            a0 = tla.simd.vintlv(e0, o0, dist="NORM_B32")
            # TLA_TODO: vintlv returns single vector; need proper even/odd interleave
            a0_bf = tla.simd.vtruncf(a0, tl.bfloat16)

            tla.simd.vsts(dst_ub[row, c * 128], a0_bf, dist="NORM_B16")
            # Second half
            # TLA_TODO: handle the z0 (high interleave half) separately
        tla.simd.membar("VST_VLD")


# ===========================================================================
# Window cache E4M3 quantization (from mega_mla_kernel._quantize_window_cache_row_vf)
# ===========================================================================

@tla.jit
def quantize_window_cache_row_vf(
    x_ub,       # UB (1, D) bf16  — input normalized KV row
    data_ub,    # UB (1, 544) uint8  — output record (512 E4M3 + 32 scale bytes)
    scale_ub,   # UB view into data_ub for scales (int16)
    scale_offset: tl.constexpr = 0,
):
    """Full-row E4M3/BF16-scale arithmetic inside an active VF region.

    For each 64-element chunk: find per-half-chunk max, compute E8M0 scale,
    quantize to E4M3, store data and scale bytes.
    """
    low = tla.simd.pge(32, 64)  # lower 32 lanes
    # TLA_TODO: mask_not for 'high' mask — using vnot or XOR
    # high = rr.mask_not(low, exec_mask=full)
    # In TLA, we can XOR with ALL mask or use complement pattern
    zero = tla.simd.vdups(0.0, tl.float32)
    one = tla.simd.vdups(1, tl.int32)
    no = tla.simd.vdups(0, tl.int32)
    bias_twice = tla.simd.vdups(254, tl.int32)

    for c in range(8):
        # Load BF16 → FP32
        raw = tla.simd.vlds(x_ub[0, c * 64], dist="UNPK_B16")
        x = tla.simd.vextf(raw, tl.float32, part="part_even")

        # Split into low/high 32-element halves using predicate masks
        x0 = tla.simd.vsel(low, x, zero)   # low half active
        # TLA_TODO: high mask = NOT low; for now assume symmetric split
        x1 = tla.simd.vsel(low, zero, x)   # high half active (inverted)

        # Per-half absolute max
        a0 = tla.simd.vreduce_max(tla.simd.vabs(x0))
        a1 = tla.simd.vreduce_max(tla.simd.vabs(x1))

        # Extract exponent: e = (reinterpret_uint32(amax) >> 23) as int32
        e0 = tla.simd.vbitcast(
            tla.simd.vshr(tla.simd.vbitcast(a0, tl.uint32), 23),
            tl.int32)
        e1 = tla.simd.vbitcast(
            tla.simd.vshr(tla.simd.vbitcast(a1, tl.uint32), 23),
            tl.int32)

        # Biased scale: b = max(e - 8, 114)
        b0 = tla.simd.vmaxs(tla.simd.vadds(e0, -8), 114)
        b1 = tla.simd.vmaxs(tla.simd.vadds(e1, -8), 114)

        # Threshold test: t = float(2^b) * 448.0
        t0 = tla.simd.vmuls(
            tla.simd.vbitcast(tla.simd.vshl(b0, 23), tl.float32),
            448.0)
        t1 = tla.simd.vmuls(
            tla.simd.vbitcast(tla.simd.vshl(b1, 23), tl.float32),
            448.0)

        # Adjust scale if amax > threshold
        gt0 = tla.simd.vcmp(a0, t0, op=tla.simd.CMP.GT)
        gt1 = tla.simd.vcmp(a1, t1, op=tla.simd.CMP.GT)
        b0 = tla.simd.vadd(b0, tla.simd.vsel(gt0, one, no))
        b1 = tla.simd.vadd(b1, tla.simd.vsel(gt1, one, no))

        # Inverse scale: inv = 2^(254-b) → reinterpret as float
        inv0 = tla.simd.vdupv(
            tla.simd.vbitcast(
                tla.simd.vshl(tla.simd.vsub(bias_twice, b0), 23),
                tl.float32),
            tla.simd.POS.LOWEST)
        inv1 = tla.simd.vdupv(
            tla.simd.vbitcast(
                tla.simd.vshl(tla.simd.vsub(bias_twice, b1), 23),
                tl.float32),
            tla.simd.POS.LOWEST)

        # Quantize: y = clamp(x * inv, -448, 448)
        inv_merged = tla.simd.vsel(low, inv0, inv1)  # low=inv0, high=inv1
        y = tla.simd.vmul(x, inv_merged)
        y = tla.simd.vmins(tla.simd.vmaxs(y, -448.0), 448.0)

        # Cast to E4M3 and store
        y_e4m3 = tla.simd.vtruncf(y, tl.float8_e4m3fn)
        tla.simd.vsts(data_ub[0, c * 64], y_e4m3)

        # Store E8M0 scale bytes: b << 7 as int16
        # E8M0 byte b encodes 2**(b-127). Its BF16 bits are b << 7.
        s0_bits = tla.simd.vbitcast(tla.simd.vshl(b0, 7), tl.int16)
        s1_bits = tla.simd.vbitcast(tla.simd.vshl(b1, 7), tl.int16)
        # TLA_TODO: vstore_first — store lowest lane to scalar buffer position
        # Using vsts with single-element store
        tla.simd.vsts(scale_ub[0, scale_offset + c * 2], s0_bits,
                       dist="ONEPT_B32")
        tla.simd.vsts(scale_ub[0, scale_offset + c * 2 + 1], s1_bits,
                       dist="ONEPT_B32")
    tla.simd.membar("VST_VLD")


# ---------------------------------------------------------------------------
# quantize_window_cache_chunk: one 64-value cache chunk
# ---------------------------------------------------------------------------
@tla.jit
def quantize_window_cache_chunk(x_ub, data_ub, scale_ub):
    """One 64-value cache chunk: 64 E4M3 bytes and two BF16 scales."""
    with tla.scope.simd():
        # Same logic as quantize_window_cache_row_vf but for a single chunk (c=0 only)
        low = tla.simd.pge(32, 64)
        zero = tla.simd.vdups(0.0, tl.float32)
        one = tla.simd.vdups(1, tl.int32)
        no = tla.simd.vdups(0, tl.int32)
        bias_twice = tla.simd.vdups(254, tl.int32)

        raw = tla.simd.vlds(x_ub[0, 0], dist="UNPK_B16")
        x = tla.simd.vextf(raw, tl.float32, part="part_even")
        x0 = tla.simd.vsel(low, x, zero)
        x1 = tla.simd.vsel(low, zero, x)
        a0 = tla.simd.vreduce_max(tla.simd.vabs(x0))
        a1 = tla.simd.vreduce_max(tla.simd.vabs(x1))
        e0 = tla.simd.vbitcast(
            tla.simd.vshr(tla.simd.vbitcast(a0, tl.uint32), 23), tl.int32)
        e1 = tla.simd.vbitcast(
            tla.simd.vshr(tla.simd.vbitcast(a1, tl.uint32), 23), tl.int32)
        b0 = tla.simd.vmaxs(tla.simd.vadds(e0, -8), 114)
        b1 = tla.simd.vmaxs(tla.simd.vadds(e1, -8), 114)
        t0 = tla.simd.vmuls(
            tla.simd.vbitcast(tla.simd.vshl(b0, 23), tl.float32),
            448.0)
        t1 = tla.simd.vmuls(
            tla.simd.vbitcast(tla.simd.vshl(b1, 23), tl.float32),
            448.0)
        gt0 = tla.simd.vcmp(a0, t0, op=tla.simd.CMP.GT)
        gt1 = tla.simd.vcmp(a1, t1, op=tla.simd.CMP.GT)
        b0 = tla.simd.vadd(b0, tla.simd.vsel(gt0, one, no))
        b1 = tla.simd.vadd(b1, tla.simd.vsel(gt1, one, no))
        inv0 = tla.simd.vdupv(
            tla.simd.vbitcast(
                tla.simd.vshl(tla.simd.vsub(bias_twice, b0), 23),
                tl.float32),
            tla.simd.POS.LOWEST)
        inv1 = tla.simd.vdupv(
            tla.simd.vbitcast(
                tla.simd.vshl(tla.simd.vsub(bias_twice, b1), 23),
                tl.float32),
            tla.simd.POS.LOWEST)
        inv_merged = tla.simd.vsel(low, inv0, inv1)
        y = tla.simd.vmul(x, inv_merged)
        y = tla.simd.vmins(tla.simd.vmaxs(y, -448.0), 448.0)
        y_e4m3 = tla.simd.vtruncf(y, tl.float8_e4m3fn)
        tla.simd.vsts(data_ub[0, 0], y_e4m3)

        # E8M0 byte b encodes 2**(b-127).  Its BF16 bits are b << 7.
        s0 = tla.simd.vbitcast(tla.simd.vshl(b0, 7), tl.int16)
        s1 = tla.simd.vbitcast(tla.simd.vshl(b1, 7), tl.int16)
        tla.simd.vsts(scale_ub[0, 0], s0, dist="ONEPT_B32")
        tla.simd.vsts(scale_ub[0, 1], s1, dist="ONEPT_B32")
        tla.simd.membar("VST_VLD")


# ===========================================================================
# Window cache E4M3 dequantization
# ===========================================================================

@tla.jit
def dequantize_window_cache_row(
    data_ub,     # UB raw FP8 record
    dst_ub,      # UB output (bf16)
    row,         # row index
    base,        # byte offset
    DATA_ROW_W: tl.constexpr,   # data_ub row width
):
    """E4M3 times stored BF16 scale → BF16, one selected cache row.

    Manual bit-level decode of E4M3FN:
      sign(1) | exponent(4) | mantissa(3)
      normal: 2^(exp-7) * (1 + mantissa/8)
      subnormal (exp=0): mantissa / 512
    Then multiply by per-32-element BF16 scale.
    """
    scale_half_w = DATA_ROW_W // 2

    with tla.scope.simd():
        zero_u = tla.simd.vdups(0, tl.uint32)
        one_u = tla.simd.vdups(1, tl.uint32)

        # Load all 16 BF16 scales for this row
        scale_offset = row * scale_half_w + (base + 512) // 2
        raw_s = tla.simd.vlds(data_ub[row, (base + 512) // 2], dist="UNPK_B16")
        scale_values = tla.simd.vextf(raw_s, tl.float32, part="part_even")

        for c in range(8):
            # Load E4M3 bytes → uint32
            bits = tla.simd.vlds(data_ub[row, base + c * 64], dist="UNPK4_B8")

            # Extract fields
            sign = tla.simd.vand(tla.simd.vshr(bits, 7), one_u)
            exponent = tla.simd.vand(
                tla.simd.vshr(bits, 3),
                tla.simd.vdups(15, tl.uint32))
            mantissa = tla.simd.vand(
                bits, tla.simd.vdups(7, tl.uint32))

            # Normal path: 2^(exp+120) * (1 + mantissa*0.125)
            exp_biased = tla.simd.vadd(
                exponent, tla.simd.vdups(120, tl.uint32))
            exp_float = tla.simd.vbitcast(
                tla.simd.vshl(exp_biased, 23), tl.float32)
            mant_int = tla.simd.vbitcast(mantissa, tl.int32)
            mant_float = tla.simd.vmuls(
                tla.simd.vextf(mant_int, tl.float32),
                0.125)
            one_f = tla.simd.vdups(1.0, tl.float32)
            normal = tla.simd.vmul(
                exp_float,
                tla.simd.vadd(one_f, mant_float))

            # Subnormal path: mantissa / 512.0
            sub = tla.simd.vmuls(
                tla.simd.vextf(tla.simd.vbitcast(mantissa, tl.int32),
                               tl.float32),
                1.0 / 512.0)

            # Select: exponent==0 → subnormal, else normal
            is_zero = tla.simd.vcmp(exponent, zero_u, op=tla.simd.CMP.EQ)
            val = tla.simd.vsel(is_zero, sub, normal)

            # Apply sign
            is_neg = tla.simd.vcmp(sign, one_u, op=tla.simd.CMP.EQ)
            neg_val = tla.simd.vmuls(val, -1.0)
            val = tla.simd.vsel(is_neg, neg_val, val)

            # Gather per-half scale (each chunk c uses scales c*2 and c*2+1)
            s0_idx = tla.simd.vdups(c * 2, tl.uint32)
            s1_idx = tla.simd.vdups(c * 2 + 1, tl.uint32)
            s0 = tla.simd.vbitcast(
                tla.simd.vgather(
                    tla.simd.vbitcast(scale_values, tl.uint32),
                    s0_idx, 64, "u32"),
                tl.float32)
            s1 = tla.simd.vbitcast(
                tla.simd.vgather(
                    tla.simd.vbitcast(scale_values, tl.uint32),
                    s1_idx, 64, "u32"),
                tl.float32)

            half = tla.simd.pge(32, 64)
            # low half uses s0, high half uses s1
            scale_merged = tla.simd.vsel(half, s0, s1)
            value = tla.simd.vmul(val, scale_merged)

            bf_out = tla.simd.vtruncf(value, tl.bfloat16)
            tla.simd.vsts(dst_ub[row, c * 64], bf_out)
        tla.simd.membar("VST_VLD")


# ===========================================================================
# align_window_cache_record: 544B alignment copy
# ===========================================================================

@tla.jit
def align_window_cache_record(raw_ub, aligned_ub, row, base,
                               RAW_W: tl.constexpr, ALIGNED_W: tl.constexpr):
    """A 544-byte record is 32-byte aligned: copy 512 data bytes and
    32 BF16-scale bytes into the padded UB row."""
    with tla.scope.simd():
        # Full mask removed: ops default to all-true
        mask_scale = tla.simd.pge(32, 64)  # 32 bytes for scale

        src = row * RAW_W + base
        dst_off = row * ALIGNED_W

        # TLA_TODO: vload_unalign_init / vload_unalign — no direct TLA equivalent
        # Using standard vlds with byte addressing (may require alignment handling)
        r0 = tla.simd.vlds(raw_ub[0, src])
        tla.simd.vsts(aligned_ub[0, dst_off], r0, dist="NORM_B8")

        r1 = tla.simd.vlds(raw_ub[0, src + 256])
        tla.simd.vsts(aligned_ub[0, dst_off + 256], r1, dist="NORM_B8")

        rs = tla.simd.vlds(raw_ub[0, src + 512])
        tla.simd.vsts(aligned_ub[0, dst_off + 512], rs, dist="NORM_B8", mask=mask_scale)
        tla.simd.membar("VST_VLD")


# ===========================================================================
# rms_norm_rope_quantize_window_cache
# AIV1 owns a token's KV norm → RoPE → 544-byte cache record in UB
# ===========================================================================

@tla.jit
def rms_norm_rope_quantize_window_cache(
    kv_gm,                    # GM (tokens, D) fp32 — raw KV projection
    gamma_gm,                 # GM (1, D) fp32 — RMS norm weights
    rope_cos_gm,              # GM (tokens, 32) fp32 — RoPE cosine
    rope_sin_gm,              # GM (tokens, 32) fp32 — RoPE sine
    window_cache_slots_gm,    # GM (tokens,) int64 — cache slot indices
    cache_gm,                 # GM (pages, ps, 544) int8 — output cache
    TOKENS: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    block, nblk,
):
    """AIV1 owns a token's KV norm → RoPE → 544-byte cache record in UB."""
    input_ub = tla.alloc((1, D), tl.float32, space="ub")
    gamma_ub = tla.alloc((1, D), tl.float32, space="ub")
    norm_ub = tla.alloc((1, D), tl.bfloat16, space="ub")
    norm_bf16_round_ub = tla.alloc((1, D), tl.bfloat16, space="ub")
    norm_fp32_ub = tla.alloc((1, D), tl.float32, space="ub")
    rope_input_ub = tla.alloc((1, 64), tl.float32, space="ub")
    cos_ub = tla.alloc((1, 64), tl.float32, space="ub")
    sin_ub = tla.alloc((1, 64), tl.float32, space="ub")
    record_ub = tla.alloc((1, 544), tl.uint8, space="ub")
    # scale_ub is a view into record_ub as int16
    scale_ub = tla.alloc((1, 272), tl.int16, space="ub")  # TLA_TODO: should alias record_ub

    tla.copy(tla.subview(gamma_gm, offsets=[0, 0], sizes=[1, D]), gamma_ub)

    for tok in range(block, TOKENS, nblk):
        tla.load_gm_to_ub(
            tla.subview(kv_gm, offsets=[tok, 0], sizes=[1, D]),
            input_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst))
        tla.copy(
            tla.subview(rope_cos_gm, offsets=[tok, 0], sizes=[1, 32]),
            tla.subview(cos_ub, offsets=[0, 0], sizes=[1, 32]))
        tla.copy(
            tla.subview(rope_sin_gm, offsets=[tok, 0], sizes=[1, 32]),
            tla.subview(sin_ub, offsets=[0, 0], sizes=[1, 32]))
        tla.sync.barrier(tla.sync.PIPE.ALL)

        # One VF owns the complete UB chain
        with tla.scope.simd():
            # TLA_TODO: inline rms_norm_key_value_row_vf here
            # (RMS norm arithmetic: sum-of-squares → rsqrt → scale → bf16)
            pass

        with tla.scope.simd():
            # Extract last 64 features for RoPE
            tail = tla.simd.vlds(norm_ub[0, D - 64], dist="UNPK_B16")
            tail_f32 = tla.simd.vextf(tail, tl.float32, part="part_even")
            tla.simd.vsts(rope_input_ub[0, 0], tail_f32)
            tla.simd.membar("VST_VLD")

            # Apply RoPE to last 64 features
            apply_rotary_embedding_vf(
                rope_input_ub, cos_ub, sin_ub, norm_ub,
                64, 64, True,  # D=64, ROPE_DIM=64, FORWARD=True
                output_offset=D - 64)

            # Quantize full row to E4M3 + BF16 scales
            quantize_window_cache_row_vf(norm_ub, record_ub, scale_ub,
                                          scale_offset=D // 2)
        tla.sync.barrier(tla.sync.PIPE.ALL)

        # Write record to cache page
        # TLA_TODO: read window_cache_slots_gm[tok] as int64 scalar
        # slot = tl.cast(window_cache_slots_gm[tok], tl.int64)
        # if slot >= PAGE_SIZE:
        #     page = slot // PAGE_SIZE
        #     offset = slot % PAGE_SIZE
        #     tla.copy(record_ub, tla.subview(cache_gm, [page, offset, 0], [1, 1, 544]))
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# update_window_cache: distributed (token, 64-feature chunk) tasks
# ===========================================================================

@tla.jit
def update_window_cache(
    kvn_gm, rope_cos_gm, rope_sin_gm, window_cache_slots_gm,
    wkc_gm, wvc_gm, wsc_gm,
    TOKENS: tl.constexpr,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    block, nblk,
):
    """Distribute (token, 64-feature chunk) tasks across the actual AIV count."""
    cos_ub = tla.alloc((1, ROPE_DIM), tl.float32, space="ub")
    sin_ub = tla.alloc((1, ROPE_DIM), tl.float32, space="ub")
    kvb_ub = tla.alloc((1, 64), tl.bfloat16, space="ub")
    kv_f_ub = tla.alloc((1, 64), tl.float32, space="ub")
    kv_rot_ub = tla.alloc((1, 64), tl.bfloat16, space="ub")
    data_ub = tla.alloc((1, 64), tl.int8, space="ub")
    scale_ub = tla.alloc((1, 16), tl.int16, space="ub")

    vec_id = block * 2 + tla.sync.sub_block_id()
    vec_num = nblk * 2

    for item in range(vec_id, TOKENS * (D // 64), vec_num):
        tok = item // (D // 64)
        chunk = item % (D // 64)

        tla.load_gm_to_ub(
            tla.subview(kvn_gm, offsets=[tok, chunk * 64], sizes=[1, 64]),
            kvb_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst))

        if chunk == (D // 64) - 1:
            tla.copy(
                tla.subview(rope_cos_gm, offsets=[tok, 0], sizes=[1, 32]),
                tla.subview(cos_ub, offsets=[0, 0], sizes=[1, 32]))
            tla.copy(
                tla.subview(rope_sin_gm, offsets=[tok, 0], sizes=[1, 32]),
                tla.subview(sin_ub, offsets=[0, 0], sizes=[1, 32]))
        tla.sync.barrier(tla.sync.PIPE.ALL)

        if chunk == (D // 64) - 1:
            # Last chunk gets RoPE
            upcast_bf16_input(kvb_ub, kv_f_ub, 64)
            apply_rotary_embedding(kv_f_ub, cos_ub, sin_ub, kv_rot_ub,
                                   64, 64, True)
            tla.sync.barrier(tla.sync.PIPE.ALL)
            quantize_window_cache_chunk(kv_rot_ub, data_ub, scale_ub)
        else:
            quantize_window_cache_chunk(kvb_ub, data_ub, scale_ub)
        tla.sync.barrier(tla.sync.PIPE.ALL)

        # Write to paged cache
        # TLA_TODO: scalar read of window_cache_slots_gm[tok]
        # slot = tl.cast(window_cache_slots_gm[tok], tl.int64)
        # if slot >= PAGE_SIZE:
        #     blk = slot // PAGE_SIZE
        #     off = slot % PAGE_SIZE
        #     ... write data_ub and scale_ub to cache page ...
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# decode_full_compressed_cache: FP4 bulk decode
# ===========================================================================

@tla.jit
def decode_full_compressed_cache(
    data_gm,     # GM compressed cache (pages, page_size, 320)
    dst_gm,      # GM output (pages, page_size, D) bf16
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    block, nblk,
):
    """Native FP4 conversion with stored BF16 scales, 2*nblk AIV owners.

    RingBuffer edges replace per-row sync_all; the existing B1 barrier
    drains the ring.
    """
    # TLA multi-buffer allocation
    fp4_ub = tla.alloc((1, 384), tl.uint8, space="ub", versions=COMPRESSED_DECODE_DEPTH)
    dst_ub = tla.alloc((1, D), tl.bfloat16, space="ub", versions=COMPRESSED_DECODE_DEPTH)

    vec_id = block * 2 + tla.sync.sub_block_id()
    vec_num = nblk * 2

    for flat in range(vec_id, NUM_PAGES * COMPRESSED_PAGE_SIZE, vec_num):
        page = flat // COMPRESSED_PAGE_SIZE
        off = flat % COMPRESSED_PAGE_SIZE

        # Load 320 bytes of compressed record
        # TLA_TODO: multi-buffer version management with mutex_lock/unlock
        tla.copy(
            tla.subview(data_gm, offsets=[page, off, 0], sizes=[1, 1, 320]),
            tla.subview(fp4_ub, offsets=[0, 0], sizes=[1, 320]))

        # Decode FP4 → BF16
        fp4_cache_decode(fp4_ub, dst_ub, 0, 384, True)

        # Store decoded BF16 row
        tla.copy(
            dst_ub,
            tla.subview(dst_gm, offsets=[page, off, 0], sizes=[1, 1, D]))


# ===========================================================================
# compressed_cache_gather_run: indexed compressed cache gathering
# ===========================================================================

@tla.jit
def compressed_cache_gather_run(
    cache_gm,               # GM compressed cache pages
    index_gm,               # GM (tokens, topk) int32 indices
    lengths_gm,             # GM lengths (may be unused)
    zero_gm,                # GM (1, D) bf16 zero row
    kv_gm,                  # GM output workspace
    vec_id, vec_num,
    block_table_gm,         # GM page table
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIN_LEN: tl.constexpr,
    D: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    BEGIN_ITEM: tl.constexpr,
    END_ITEM: tl.constexpr,
):
    """AIV-owned raw-record load, BF16-scale decode and KV assembly.

    按每个AIV的循环轮次划分任务，三个区间互斥且覆盖全部索引槽位。
    """
    fp4_ub = tla.alloc((1, 384), tl.uint8, space="ub")
    dst_ub = tla.alloc((1, D), tl.bfloat16, space="ub")

    for item in range(vec_id + BEGIN_ITEM, END_ITEM, vec_num):
        tok = item // TOPK
        pos = item % TOPK

        # Read index
        # TLA_TODO: scalar read of index_gm[tok, pos]
        # index = tl.cast(index_gm[tok, pos], tl.int32)

        # if index >= 0:
        #     page = paged_cache_page(index, block_table_gm, tok,
        #                             PAGE_SIZE, BATCH_STRIDE)
        #     off = index % PAGE_SIZE
        #     # Load compressed record
        #     tla.copy(
        #         tla.subview(cache_gm, [page, off, 0], [1, 1, 320]),
        #         tla.subview(fp4_ub, [0, 0], [1, 320]))
        #     # Decode
        #     fp4_cache_decode(fp4_ub, dst_ub, 0, 384, True)
        # else:
        #     tla.copy(zero_gm, dst_ub)
        # tla.sync.barrier(tla.sync.PIPE.ALL)

        # Write to output workspace
        # kg_offset = tok * (WIN_LEN + TOPK) + WIN_LEN + pos
        # tla.copy(dst_ub,
        #          tla.subview(kv_gm, [kg_offset, 0], [1, D]))
        # tla.sync.barrier(tla.sync.PIPE.ALL)
        pass  # TLA_TODO: complete with scalar index reads
