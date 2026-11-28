# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TLA port of QK/softmax/PV matmul, gather, and attention pipeline operations.

Ported from cannbot-dsl mega_mla_kernel.py attention methods,
attention_matmul.py, and attention_pipeline.py.

Pipeline phases covered:
  - Q RoPE + scaled gather (prepare_query_and_gather_kv)
  - Masked attention score computation
  - Online softmax with attn_sink
  - BF16 probability normalization
  - QK and PV matmul dispatch
  - Small-K CV attention pipeline
"""

import triton
import triton.language as tl
import triton.language.extra.tla as tla

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 512  # HEAD_DIM
VECTOR_CHUNK_ELEMENTS = 64  # FP32 lanes per SIMD register
NEG_INF = -1.7e38


# ===========================================================================
# Sync helper: publish QK scores to paired AIVs
# ===========================================================================
@tla.jit
def publish_qk_to_paired_aivs():
    """All QK AIC complete, then each AIC releases its paired 1:2 AIV.

    Port of _publish_qk_to_paired_aivs(flag_ids).
    """
    with tla.scope.cube():
        tla.sync.barrier(tla.sync.PIPE.ALL)
    with tla.scope.vector():
        tla.sync.barrier(tla.sync.PIPE.ALL)
    with tla.scope.cube():
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, flagid=1)
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, flagid=1)
        tla.sync.block_set_flag(tla.sync.MODE.SUBBLOCK, tla.sync.PIPE.MTE3, flagid=2)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.SUBBLOCK, tla.sync.PIPE.S, flagid=2)


# ===========================================================================
# Index masking: set positions beyond valid_length to -1
# ===========================================================================
@tla.jit
def mask_index_prefix(
    idx_ub,                # UB (1, width) int32
    width: tl.constexpr,   # total index count
    valid_length,          # runtime valid count
):
    """Mask UB indices: positions >= valid_length set to -1.

    Port of attention_pipeline._mask_index_prefix.
    """
    with tla.scope.simd():
        neg_one = tla.simd.vdups(-1, tl.int32)
        for c in range(width // VECTOR_CHUNK_ELEMENTS):
            off = c * VECTOR_CHUNK_ELEMENTS
            idx = tla.simd.vlds(idx_ub[0, off])
            # Build threshold: position indices from off to off+63
            pos_base = tla.simd.vdups(off, tl.int32)
            # TLA_TODO: ideally use vci() for lane index + vadds
            threshold = tla.simd.vdups(valid_length, tl.int32)
            # Compare: lane offset >= valid_length → invalid
            lane_idx = tla.simd.vadds(pos_base, 0)  # broadcast off
            invalid = tla.simd.vcmp(lane_idx, threshold, op=tla.simd.CMP.GE)
            masked = tla.simd.vsel(invalid, neg_one, idx)
            tla.simd.vsts(idx_ub[0, off], masked)
        tla.simd.membar("VST_VLD")


# ===========================================================================
# Mask attention scores: apply index validity mask
# ===========================================================================
@tla.jit
def mask_attention_scores(
    s_ub,            # UB (1, K) f32 raw scores
    win_idx_ub,      # UB (1, win_len) i32 window indices
    cmp_idx_ub,      # UB (1, K-win_len) i32 compressed indices
    score_ub,        # UB (K,) f32 output masked scores
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
):
    """Apply index mask to attention scores. Negative index → NEG_INF.

    Port of MegaMlaKernel._mask_attention_scores.
    """
    with tla.scope.simd():
        zero = tla.simd.vdups(0, tl.int32)
        neg = tla.simd.vdups(NEG_INF, tl.float32)
        for c in range(K // VECTOR_CHUNK_ELEMENTS):
            off = c * VECTOR_CHUNK_ELEMENTS
            if off < WIN_LEN:
                idx = tla.simd.vlds(win_idx_ub[0, off])
            else:
                idx = tla.simd.vlds(cmp_idx_ub[0, off - WIN_LEN])
            invalid = tla.simd.vcmp(idx, zero, op=tla.simd.CMP.LT)
            s = tla.simd.vlds(s_ub[0, off])
            # invalid → NEG_INF, valid → original score
            # vselect: vsel(mask, if_true, if_false)
            masked = tla.simd.vsel(invalid, neg, s)
            tla.simd.vsts(score_ub[off], masked)
        tla.simd.membar("VST_VLD")


# ===========================================================================
# Online softmax with attn_sink
# ===========================================================================
@tla.jit
def compute_sink_softmax_statistics_online(
    score_ub,    # UB (K,) f32 masked scores
    sink_ub,     # UB (1, 1) f32 attn_sink value
    p_ub,        # UB (K,) f32 output exponentials
    denom_ub,    # UB (1, 1) f32 output denominator
    K: tl.constexpr,
):
    """Online (m,l) reduction over K chunks, including the attn_sink logit.

    Port of MegaMlaKernel._compute_sink_softmax_statistics_online.
    Uses vexpdif = exp(a - b) fusion for numerical stability.

    UB footprint: score/p = 2*K*4 B, attn_sink/denom = 8 B;
    at K=640 this is 5,128 B, far below the 256 KiB UB limit.
    """
    with tla.scope.simd():
        # Initialize running max from sink value
        sink_raw = tla.simd.vlds(sink_ub[0, 0])
        m = tla.simd.vdupv(sink_raw)
        sum_exp = tla.simd.vdups(1.0, tl.float32)

        # Pass 1: online max + running sum
        for c in range(K // VECTOR_CHUNK_ELEMENTS):
            s = tla.simd.vlds(score_ub[c * VECTOR_CHUNK_ELEMENTS])
            tile_max = tla.simd.vreduce_max(s)
            new_m = tla.simd.vdupv(
                tla.simd.vmax(m, tile_max),
                
            )
            # alpha = exp(old_m - new_m) — rescale factor
            alpha = tla.simd.vexpdif(m, new_m)
            # tile_sum = sum(exp(s - new_m))
            tile_sum = tla.simd.vreduce_sum(
                tla.simd.vexpdif(s, new_m)
            )
            sum_exp = tla.simd.vadd(
                tla.simd.vmul(sum_exp, alpha),
                tile_sum
            )
            m = new_m

        # Pass 2: compute final exp(s - m) for all chunks
        for c in range(K // VECTOR_CHUNK_ELEMENTS):
            s = tla.simd.vlds(score_ub[c * VECTOR_CHUNK_ELEMENTS])
            p = tla.simd.vexpdif(s, m)
            tla.simd.vsts(p_ub[c * VECTOR_CHUNK_ELEMENTS], p)

        # Store denominator
        # TLA_TODO: vstore_first → store lowest lane to scalar UB location
        tla.simd.vsts(denom_ub[0, 0], sum_exp)
        tla.simd.membar("VST_VLD")


# ===========================================================================
# Fused mask + sink softmax + BF16 normalization (multi-row)
# ===========================================================================
@tla.jit
def compute_masked_sink_softmax_rows(
    scores_ub,                # UB (rows_per_owner, K) f32
    window_indices_ub,        # UB (1, win_len) i32
    compressed_indices_ub,    # UB (1, K-win_len) i32
    attention_sink_ub,        # UB (1, heads) f32
    masked_scores_ub,         # UB (K,) f32 scratch
    exponentials_ub,          # UB (K,) f32 scratch
    probabilities_ub,         # UB (rows_per_owner, K) bf16 output
    head_start,               # first head index for this batch
    row_count,                # number of rows to process
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
):
    """Keep mask, FP32 statistics and BF16 normalization in one VF.

    Port of MegaMlaKernel._compute_masked_sink_softmax_rows.
    """
    with tla.scope.simd():
        for row in range(row_count):
            zero_idx = tla.simd.vdups(0, tl.int32)
            neg_logit = tla.simd.vdups(NEG_INF, tl.float32)
            row_max = tla.simd.vdups(NEG_INF, tl.float32)

            # Pass 1: mask scores and find row max
            for chunk in range(K // VECTOR_CHUNK_ELEMENTS):
                off = chunk * VECTOR_CHUNK_ELEMENTS
                if off < WIN_LEN:
                    cache_idx = tla.simd.vlds(window_indices_ub[0, off])
                else:
                    cache_idx = tla.simd.vlds(
                        compressed_indices_ub[0, off - WIN_LEN])

                score = tla.simd.vlds(scores_ub[row, off])
                invalid = tla.simd.vcmp(cache_idx, zero_idx, op=tla.simd.CMP.LT)
                masked_score = tla.simd.vsel(invalid, neg_logit, score)
                tla.simd.vsts(masked_scores_ub[off], masked_score)
                row_max = tla.simd.vmax(
                    row_max,
                    tla.simd.vreduce_max(masked_score)
                )

            tla.simd.membar("VST_VLD")

            # Include attn_sink in max
            sink_raw = tla.simd.vlds(attention_sink_ub[0, head_start + row])
            sink = tla.simd.vdupv(sink_raw)
            row_max = tla.simd.vdupv(
                tla.simd.vmax(row_max, sink),
                
            )

            # Pass 2: compute exp(score - max) and sum
            row_sum = tla.simd.vdups(0.0, tl.float32)
            for chunk in range(K // VECTOR_CHUNK_ELEMENTS):
                off = chunk * VECTOR_CHUNK_ELEMENTS
                val = tla.simd.vexpdif(
                    tla.simd.vlds(masked_scores_ub[off]),
                    row_max
                )
                tla.simd.vsts(exponentials_ub[off], val)
                row_sum = tla.simd.vadd(
                    row_sum, tla.simd.vreduce_sum(val)
                )

            # Add exp(sink - max) to denominator
            denominator = tla.simd.vdupv(
                tla.simd.vadd(
                    row_sum,
                    tla.simd.vexpdif(sink, row_max)
                ),
                
            )
            tla.simd.membar("VST_VLD")

            # Pass 3: normalize to BF16 probabilities
            for chunk in range(K // VECTOR_CHUNK_ELEMENTS):
                off = chunk * VECTOR_CHUNK_ELEMENTS
                exp_val = tla.simd.vlds(exponentials_ub[off])
                prob = tla.simd.vdiv(exp_val, denominator)
                # FP32 → BF16 truncation then store
                prob_bf16 = tla.simd.vtruncf(prob, tl.bfloat16)
                tla.simd.vsts(probabilities_ub[row, off], prob_bf16)
            tla.simd.membar("VST_VLD")


# ===========================================================================
# Softmax dispatch: batched heads (fastest path for dual-AIV)
# ===========================================================================
@tla.jit
def run_softmax_batched_heads(
    scores_gm,                 # GM (T, H, K) f32
    window_indices_gm,         # GM (T, win_len) i32
    compressed_indices_gm,     # GM (T, K-win_len) i32
    attention_sink_gm,         # GM (1, H) f32
    probabilities_gm,          # GM (T, H, K) bf16
    window_topk_length_gm,    # GM (T,) i32
    compressed_topk_length_gm, # GM (T,) i32
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
    GRID: tl.constexpr,
    SOFTMAX_BATCH_ROWS: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """One contiguous Score/P transaction per token/head owner.

    Port of MegaMlaKernel._run_softmax_batched_heads.
    """
    rows_per_owner = SOFTMAX_BATCH_ROWS
    owners_per_token = 2 * GRID // TOKENS
    vec_id = block * 2 + tla.sync.sub_block_id()
    token = vec_id // owners_per_token
    owner_index = vec_id % owners_per_token
    head_start = owner_index * rows_per_owner
    row_count = tla.minimum(rows_per_owner, HEADS - head_start)

    scores_ub = tla.alloc((rows_per_owner, K), tl.float32, space="ub")
    probabilities_ub = tla.alloc((rows_per_owner, K), tl.bfloat16, space="ub")
    win_idx_ub = tla.alloc((1, WIN_LEN), tl.int32, space="ub")
    cmp_idx_ub = tla.alloc((1, K - WIN_LEN), tl.int32, space="ub")
    sink_ub = tla.alloc((1, HEADS), tl.float32, space="ub")
    masked_ub = tla.alloc((K,), tl.float32, space="ub")
    exp_ub = tla.alloc((K,), tl.float32, space="ub")

    if head_start < HEADS:
        # Load scores, indices, and sink
        token_scores = tla.subview(
            scores_gm, offsets=[token * HEADS + head_start, 0],
            sizes=[rows_per_owner, K]
        )
        tla.load_gm_to_ub(
            token_scores, scores_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst),
        )
        tla.copy(
            tla.subview(window_indices_gm, offsets=[token, 0], sizes=[1, WIN_LEN]),
            win_idx_ub,
        )
        tla.copy(
            tla.subview(compressed_indices_gm, offsets=[token, 0],
                        sizes=[1, K - WIN_LEN]),
            cmp_idx_ub,
        )
        tla.copy(attention_sink_gm, sink_ub)
        tla.sync.barrier(tla.sync.PIPE.ALL)

        # Apply length masking if needed
        if WIN_LENGTH_STRIDE > 0:
            win_valid = tl.cast(window_topk_length_gm[token // WIN_LENGTH_STRIDE],
                                tl.int32)
            mask_index_prefix(win_idx_ub, WIN_LEN, win_valid)
        if CMP_LENGTH_STRIDE > 0:
            cmp_valid = tl.cast(
                compressed_topk_length_gm[token // CMP_LENGTH_STRIDE], tl.int32)
            mask_index_prefix(cmp_idx_ub, K - WIN_LEN, cmp_valid)

        compute_masked_sink_softmax_rows(
            scores_ub, win_idx_ub, cmp_idx_ub, sink_ub,
            masked_ub, exp_ub, probabilities_ub,
            head_start, row_count, K, WIN_LEN,
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)

        # Write back BF16 probabilities
        token_probs = tla.subview(
            probabilities_gm, offsets=[token * HEADS + head_start, 0],
            sizes=[rows_per_owner, K]
        )
        tla.store_ub_to_gm(probabilities_ub, token_probs)
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# Softmax dispatch: all AIVs (dual-AIV, no batching)
# ===========================================================================
@tla.jit
def run_softmax_all_aivs(
    s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
    attn_sink_gm, p_gm,
    topk_length_gm, extra_topk_length_gm,
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """Distribute T*heads softmax rows over both AIVs of every block.

    Port of MegaMlaKernel._run_softmax_all_aivs.
    """
    s_ub = tla.alloc((1, K), tl.float32, space="ub")
    win_idx_ub = tla.alloc((1, WIN_LEN), tl.int32, space="ub")
    cmp_idx_ub = tla.alloc((1, K - WIN_LEN), tl.int32, space="ub")
    sink_ub = tla.alloc((1, 1), tl.float32, space="ub")
    score_ub = tla.alloc((K,), tl.float32, space="ub")
    p_ub = tla.alloc((K,), tl.float32, space="ub")
    pb_ub = tla.alloc((1, K), tl.bfloat16, space="ub")

    vec_id = block * 2 + tla.sync.sub_block_id()
    vec_num = nblk * 2
    for item in range(vec_id, TOKENS * HEADS, vec_num):
        tok = item // HEADS
        row = item % HEADS

        # tile_slice addresses trailing dimensions; flatten T/H before selecting a token.
        s_tok = tla.subview(s_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, K])
        p_tok = tla.subview(p_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, K])

        tla.copy(
            tla.subview(indices_in_kvcache_gm, offsets=[tok, 0], sizes=[1, WIN_LEN]),
            win_idx_ub,
        )
        tla.copy(
            tla.subview(extra_indices_in_kvcache_gm, offsets=[tok, 0],
                        sizes=[1, K - WIN_LEN]),
            cmp_idx_ub,
        )
        tla.load_gm_to_ub(
            tla.subview(s_tok, offsets=[row, 0], sizes=[1, K]),
            s_ub,
            params=tla.LoadGM2UBParams(eviction_policy=tla.EvictionPolicy.EvictFirst),
        )
        # Load sink for this head
        tla.copy(
            tla.subview(attn_sink_gm, offsets=[0, row], sizes=[1, 1]),
            sink_ub,
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)

        if WIN_LENGTH_STRIDE > 0:
            mask_index_prefix(
                win_idx_ub, WIN_LEN,
                tl.cast(topk_length_gm[tok // WIN_LENGTH_STRIDE], tl.int32),
            )
        if CMP_LENGTH_STRIDE > 0:
            mask_index_prefix(
                cmp_idx_ub, K - WIN_LEN,
                tl.cast(extra_topk_length_gm[tok // CMP_LENGTH_STRIDE], tl.int32),
            )

        # Reuse the multi-row function with row_count=1
        compute_masked_sink_softmax_rows(
            s_ub, win_idx_ub, cmp_idx_ub, sink_ub,
            score_ub, p_ub, pb_ub, 0, 1, K, WIN_LEN,
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)

        tla.store_ub_to_gm(
            pb_ub, tla.subview(p_tok, offsets=[row, 0], sizes=[1, K])
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# Softmax dispatch: per-token (non-dual-AIV path)
# ===========================================================================
@tla.jit
def run_softmax_by_token(
    s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
    attn_sink_gm, p_gm,
    topk_length_gm, extra_topk_length_gm,
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """Token-wise masked attn_sink softmax from Cube scores to BF16 P[H,K].

    Port of MegaMlaKernel._run_softmax_by_token.
    """
    s_ub = tla.alloc((1, K), tl.float32, space="ub")
    win_idx_ub = tla.alloc((1, WIN_LEN), tl.int32, space="ub")
    cmp_idx_ub = tla.alloc((1, K - WIN_LEN), tl.int32, space="ub")
    sink_ub = tla.alloc((1, 1), tl.float32, space="ub")
    score_ub = tla.alloc((K,), tl.float32, space="ub")
    p_ub = tla.alloc((K,), tl.float32, space="ub")
    denom_ub = tla.alloc((1, 1), tl.float32, space="ub")
    pb_ub = tla.alloc((1, K), tl.bfloat16, space="ub")

    for tok in range(TOKENS):
        s_tok = tla.subview(s_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, K])
        p_tok = tla.subview(p_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, K])

        # idx is shared by every head row of a token: load it once.
        tla.copy(
            tla.subview(indices_in_kvcache_gm, offsets=[tok, 0], sizes=[1, WIN_LEN]),
            win_idx_ub,
        )
        tla.copy(
            tla.subview(extra_indices_in_kvcache_gm, offsets=[tok, 0],
                        sizes=[1, K - WIN_LEN]),
            cmp_idx_ub,
        )

        if WIN_LENGTH_STRIDE > 0 or CMP_LENGTH_STRIDE > 0:
            tla.sync.barrier(tla.sync.PIPE.ALL)
        if WIN_LENGTH_STRIDE > 0:
            mask_index_prefix(
                win_idx_ub, WIN_LEN,
                tl.cast(topk_length_gm[tok // WIN_LENGTH_STRIDE], tl.int32),
            )
        if CMP_LENGTH_STRIDE > 0:
            mask_index_prefix(
                cmp_idx_ub, K - WIN_LEN,
                tl.cast(extra_topk_length_gm[tok // CMP_LENGTH_STRIDE], tl.int32),
            )

        for row in range(block, HEADS, nblk):
            tla.load_gm_to_ub(
                tla.subview(s_tok, offsets=[row, 0], sizes=[1, K]),
                s_ub,
                params=tla.LoadGM2UBParams(
                    eviction_policy=tla.EvictionPolicy.EvictFirst),
            )
            tla.copy(
                tla.subview(attn_sink_gm, offsets=[0, row], sizes=[1, 1]),
                sink_ub,
            )
            tla.sync.barrier(tla.sync.PIPE.ALL)

            mask_attention_scores(s_ub, win_idx_ub, cmp_idx_ub, score_ub,
                                  K, WIN_LEN)

            # Online softmax for T=16, two-pass for other shapes
            # T=16 uses the online recurrence; others use the original two-pass
            compute_sink_softmax_statistics_online(score_ub, sink_ub, p_ub,
                                                   denom_ub, K)

            # BF16 normalization: P = exp / denom
            with tla.scope.simd():
                denom_b = tla.simd.vlds(denom_ub[0, 0])
                denom_b = tla.simd.vdupv(denom_b)
                for c in range(K // VECTOR_CHUNK_ELEMENTS):
                    off = c * VECTOR_CHUNK_ELEMENTS
                    p = tla.simd.vdiv(
                        tla.simd.vlds(p_ub[off]),
                        denom_b,
                    )
                    p_bf16 = tla.simd.vtruncf(p, tl.bfloat16)
                    tla.simd.vsts(pb_ub[0, off], p_bf16)
                tla.simd.membar("VST_VLD")

            tla.sync.barrier(tla.sync.PIPE.ALL)
            tla.store_ub_to_gm(
                pb_ub, tla.subview(p_tok, offsets=[row, 0], sizes=[1, K])
            )
            tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# Softmax dispatch: choose path based on config
# ===========================================================================
@tla.jit
def run_attention_softmax(
    s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
    attn_sink_gm, p_gm,
    topk_length_gm, extra_topk_length_gm,
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
    GRID: tl.constexpr,
    DUAL_AIV: tl.constexpr,
    SOFTMAX_BATCH_ROWS: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """Softmax dispatch: batched, all-AIV, or per-token.

    Port of MegaMlaKernel._run_attention_softmax.
    """
    if DUAL_AIV and SOFTMAX_BATCH_ROWS > 0:
        run_softmax_batched_heads(
            s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
            attn_sink_gm, p_gm,
            topk_length_gm, extra_topk_length_gm,
            block, nblk,
            TOKENS, HEADS, K, WIN_LEN, GRID, SOFTMAX_BATCH_ROWS,
            WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
        )
    elif DUAL_AIV:
        run_softmax_all_aivs(
            s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
            attn_sink_gm, p_gm,
            topk_length_gm, extra_topk_length_gm,
            block, nblk,
            TOKENS, HEADS, K, WIN_LEN, WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
        )
    else:
        run_softmax_by_token(
            s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
            attn_sink_gm, p_gm,
            topk_length_gm, extra_topk_length_gm,
            block, nblk,
            TOKENS, HEADS, K, WIN_LEN, WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
        )


# ===========================================================================
# QK matmul: flat tile dispatch (BF16 Cube)
# ===========================================================================
@tla.jit
def qk_compute_flat_tiles(
    s_gm,       # GM (T*H, K) f32 output scores
    q_gm,       # GM (T*H, D) bf16 query
    k_gm,       # GM (T*K, D) bf16 key
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    BASE_N: tl.constexpr = 64,
    BASE_K: tl.constexpr = 128,
):
    """S[m,n] = Q[m,k] @ K[n,k]^T flat dispatch over (token, N-tile).

    Port of QueryKeyMatmulKernel.compute_flat_tiles.
    """
    n_tiles = K // BASE_N
    with tla.scope.cube():
        # NZ layout: A(Q)[M=HEADS, K=BASE_K] → (K//16, M//16, 16, 16)
        # B(K)[N=BASE_N, K=BASE_K] with transpose_b=True → (K//16, N//16, 16, 16)
        # C(S)[M=HEADS, N=BASE_N] → (N//16, M//16, 16, 16)
        q_l1 = tla.alloc((BASE_K // 16, HEADS // 16, 16, 16), tl.bfloat16, space="l1", versions=2)
        k_l1 = tla.alloc((BASE_K // 16, BASE_N // 16, 16, 16), tl.bfloat16, space="l1", versions=2)
        acc = tla.alloc((BASE_N // 16, HEADS // 16, 16, 16), tl.float32, space="l0c")

        for flat_id in range(block, TOKENS * n_tiles, nblk):
            tok = flat_id // n_tiles
            nt = flat_id % n_tiles

            # Load Q for this token (all heads)
            q_tok = tla.subview(q_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, D])
            k_tok = tla.subview(k_gm, offsets=[tok * K, nt * BASE_N], sizes=[BASE_N, D])

            # First K-tile: clear accumulator
            tla.load_gm_to_l1(
                tla.subview(q_tok, offsets=[0, 0], sizes=[HEADS, BASE_K]),
                q_l1,
            )
            tla.load_gm_to_l1(
                tla.subview(k_tok, offsets=[0, 0], sizes=[BASE_N, BASE_K]),
                k_l1,
            )
            tla.cube.mmad(
                q_l1, k_l1, acc,
                transpose_a=False, transpose_b=True,
                clear_accum=True,
            )

            # Remaining K-tiles: accumulate
            for ki in range(1, D // BASE_K):
                tla.load_gm_to_l1(
                    tla.subview(q_tok, offsets=[0, ki * BASE_K], sizes=[HEADS, BASE_K]),
                    q_l1,
                )
                tla.load_gm_to_l1(
                    tla.subview(k_tok, offsets=[0, ki * BASE_K], sizes=[BASE_N, BASE_K]),
                    k_l1,
                )
                tla.cube.mmad(
                    q_l1, k_l1, acc,
                    transpose_a=False, transpose_b=True,
                    clear_accum=False,
                )

            # Fixpipe writeback
            s_tile = tla.subview(s_gm, offsets=[tok * HEADS, nt * BASE_N],
                                 sizes=[HEADS, BASE_N])
            tla.store_l0c_to_gm(
                acc, s_tile,
                params=tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND),
            )


# ===========================================================================
# QK matmul: token-local owner schedule
# ===========================================================================
@tla.jit
def qk_compute_owned_tiles(
    s_gm, q_gm, k_gm,
    block,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    BPT: tl.constexpr,       # blocks per token
    NT_PER: tl.constexpr,    # N-tiles per owner
    BASE_N: tl.constexpr = 64,
    BASE_K: tl.constexpr = 128,
):
    """Token-local QK dispatch. Block b owns token b//bpt.

    Port of QueryKeyMatmulKernel.compute_owned_tiles.
    """
    n_tiles = K // BASE_N
    tok = block // BPT
    sub = block % BPT
    nt_start = sub * NT_PER
    nt_end = tla.minimum((sub + 1) * NT_PER, n_tiles)

    with tla.scope.cube():
        q_l1 = tla.alloc((BASE_K // 16, HEADS // 16, 16, 16), tl.bfloat16, space="l1", versions=2)
        k_l1 = tla.alloc((BASE_K // 16, BASE_N // 16, 16, 16), tl.bfloat16, space="l1", versions=2)
        acc = tla.alloc((BASE_N // 16, HEADS // 16, 16, 16), tl.float32, space="l0c")

        if tok < TOKENS:
            q_tok = tla.subview(q_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, D])
            k_tok_base = tla.subview(k_gm, offsets=[tok * K, 0], sizes=[K, D])

            for nt in range(nt_start, nt_end):
                # First K-tile: clear
                tla.load_gm_to_l1(
                    tla.subview(q_tok, offsets=[0, 0], sizes=[HEADS, BASE_K]),
                    q_l1,
                )
                tla.load_gm_to_l1(
                    tla.subview(k_tok_base,
                                offsets=[nt * BASE_N, 0],
                                sizes=[BASE_N, BASE_K]),
                    k_l1,
                )
                tla.cube.mmad(
                    q_l1, k_l1, acc,
                    transpose_a=False, transpose_b=True,
                    clear_accum=True,
                )
                # Remaining K-tiles: accumulate
                for ki in range(1, D // BASE_K):
                    tla.load_gm_to_l1(
                        tla.subview(q_tok, offsets=[0, ki * BASE_K],
                                    sizes=[HEADS, BASE_K]),
                        q_l1,
                    )
                    tla.load_gm_to_l1(
                        tla.subview(k_tok_base,
                                    offsets=[nt * BASE_N, ki * BASE_K],
                                    sizes=[BASE_N, BASE_K]),
                        k_l1,
                    )
                    tla.cube.mmad(
                        q_l1, k_l1, acc,
                        transpose_a=False, transpose_b=True,
                        clear_accum=False,
                    )
                s_tile = tla.subview(s_gm, offsets=[tok * HEADS, nt * BASE_N],
                                     sizes=[HEADS, BASE_N])
                tla.store_l0c_to_gm(
                    acc, s_tile,
                    params=tla.FixpipeL0C2GMParams(
                        dma_mode=tla.FixpipeDMAMode.NZ2ND),
                )


# ===========================================================================
# PV matmul: flat dispatch  O[T,H,D] = P[T,H,K] @ V[T,K,D]
# ===========================================================================
@tla.jit
def pv_compute_flat_tiles(
    out_gm,     # GM (T*H, D) bf16 output
    p_gm,       # GM (T*H, K) bf16 probability
    v_gm,       # GM (T*K, D) bf16 value
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    BASE_N: tl.constexpr = 64,
    BASE_K: tl.constexpr = 128,
):
    """BF16 Cube PV O[T,H,D] = P[T,H,K] @ V[T,K,D], all tokens.

    Port of ProbabilityValueMatmulKernel.compute_flat_tiles.
    """
    n_tiles = D // BASE_N
    with tla.scope.cube():
        # PV: O[H,BN] = P[H,BK] @ V[BK,BN], transpose_b=False
        # A(P)[M=HEADS, K=BASE_K] → NZ (K//16, M//16, 16, 16)
        # B(V)[K=BASE_K, N=BASE_N] transpose_b=False → NZ (N//16, K//16, 16, 16)
        # C(O)[M=HEADS, N=BASE_N] → NZ (N//16, M//16, 16, 16)
        p_l1 = tla.alloc((BASE_K // 16, HEADS // 16, 16, 16), tl.bfloat16, space="l1", versions=2)
        v_l1 = tla.alloc((BASE_N // 16, BASE_K // 16, 16, 16), tl.bfloat16, space="l1", versions=2)
        acc = tla.alloc((BASE_N // 16, HEADS // 16, 16, 16), tl.float32, space="l0c")

        for flat_id in range(block, TOKENS * n_tiles, nblk):
            tok = flat_id // n_tiles
            nt = flat_id % n_tiles

            p_tok = tla.subview(p_gm, offsets=[tok * HEADS, 0], sizes=[HEADS, K])
            v_tok = tla.subview(v_gm, offsets=[tok * K, nt * BASE_N],
                                sizes=[K, BASE_N])

            # First K-tile: clear
            tla.load_gm_to_l1(
                tla.subview(p_tok, offsets=[0, 0], sizes=[HEADS, BASE_K]),
                p_l1,
            )
            tla.load_gm_to_l1(
                tla.subview(v_tok, offsets=[0, 0], sizes=[BASE_K, BASE_N]),
                v_l1,
            )
            tla.cube.mmad(
                p_l1, v_l1, acc,
                transpose_a=False, transpose_b=False,
                clear_accum=True,
            )

            # Remaining K-tiles: accumulate
            for ki in range(1, K // BASE_K):
                tla.load_gm_to_l1(
                    tla.subview(p_tok, offsets=[0, ki * BASE_K],
                                sizes=[HEADS, BASE_K]),
                    p_l1,
                )
                tla.load_gm_to_l1(
                    tla.subview(v_tok, offsets=[ki * BASE_K, 0],
                                sizes=[BASE_K, BASE_N]),
                    v_l1,
                )
                tla.cube.mmad(
                    p_l1, v_l1, acc,
                    transpose_a=False, transpose_b=False,
                    clear_accum=False,
                )

            out_tile = tla.subview(out_gm, offsets=[tok * HEADS, nt * BASE_N],
                                   sizes=[HEADS, BASE_N])
            tla.store_l0c_to_gm(
                acc, out_tile,
                params=tla.FixpipeL0C2GMParams(
                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                    pre_quant=tla.FixpipePreQuantMode.F322BF16,
                ),
            )


# ===========================================================================
# PV matmul dispatch
# ===========================================================================
@tla.jit
def run_probability_value_matmul(
    out_gm, p_gm, v_gm,
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    GRID: tl.constexpr,
    FLAT_ATTENTION_TILES: tl.constexpr,
    PV_BASE_N: tl.constexpr = 64,
    PV_BASE_K: tl.constexpr = 128,
):
    """BF16 Cube PV dispatch, shared across the full launch grid.

    Port of MegaMlaKernel._run_probability_value_matmul.
    """
    tla.sync.barrier(tla.sync.PIPE.ALL)
    # TLA_TODO: onchip_rewind(reset_sync_id=True)
    n_tiles_pv = D // PV_BASE_N
    if TOKENS <= GRID and not FLAT_ATTENTION_TILES:
        bpt_pv = tla.maximum(1, GRID // TOKENS)
        nt_per_pv = tla.cdiv(n_tiles_pv, bpt_pv)
        # TLA_TODO: call pv_compute_owned_tiles (omitted for brevity,
        # mirrors qk_compute_owned_tiles with PV dimensions)
        pass
    else:
        pv_compute_flat_tiles(
            out_gm, p_gm, v_gm, block, nblk,
            TOKENS, HEADS, K, PV_BASE_N, PV_BASE_K,
        )


# ===========================================================================
# Small-K attention pipeline dispatch
# ===========================================================================
@tla.jit
def run_small_k_attention_pipeline(
    qb_gm, kg_gm,
    indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
    attn_sink_gm, out_gm,
    topk_length_gm, extra_topk_length_gm,
    block, nblk,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    GRID: tl.constexpr,
    WIN_LEN: tl.constexpr,
    CV_TILE_M: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """H64 aligned CV template dispatch.

    Port of MegaMlaKernel._run_small_k_attention_pipeline.
    The small-K pipeline uses token-local online QK/Softmax/PV with
    M32/M64 head tiles and L0 double buffering.

    TLA_TODO: The full _SmallKAttentionPipeline with its 4-stage
    pipeline (QK compute, softmax, PV compute, output accumulate)
    requires extensive L0/L1 ring buffer management. Outlined here
    as dispatch logic; the inner pipeline body needs dedicated porting.
    """
    if TOKENS == 16 and GRID == 32:
        # Balanced two-token layout: block//2 = token, block%2 = head-M half
        tok = block // 2
        hm = block % 2
        h0 = hm * CV_TILE_M
        q_tok = tla.subview(qb_gm, offsets=[tok * HEADS + h0, 0],
                            sizes=[CV_TILE_M, D])
        kv_tok = tla.subview(kg_gm, offsets=[tok * K, 0], sizes=[K, D])
        win_idx = tla.subview(indices_in_kvcache_gm, offsets=[tok, 0],
                              sizes=[1, WIN_LEN])
        cmp_idx = tla.subview(extra_indices_in_kvcache_gm, offsets=[tok, 0],
                              sizes=[1, K - WIN_LEN])
        out_tok = tla.subview(out_gm, offsets=[tok * HEADS + h0, 0],
                              sizes=[CV_TILE_M, D])

        # TLA_TODO: call _SmallKAttentionPipeline.run_attention_head_tile
        # with q_tok, kv_tok, win_idx, cmp_idx, win/cmp valid lengths,
        # attn_sink_gm, out_tok, h0
        pass
    else:
        for tok in range(block, TOKENS, nblk):
            q_tok = tla.subview(qb_gm, offsets=[tok * HEADS, 0],
                                sizes=[HEADS, D])
            kv_tok = tla.subview(kg_gm, offsets=[tok * K, 0], sizes=[K, D])
            win_idx = tla.subview(indices_in_kvcache_gm, offsets=[tok, 0],
                                  sizes=[1, WIN_LEN])
            cmp_idx = tla.subview(extra_indices_in_kvcache_gm, offsets=[tok, 0],
                                  sizes=[1, K - WIN_LEN])
            out_tok = tla.subview(out_gm, offsets=[tok * HEADS, 0],
                                  sizes=[HEADS, D])

            # TLA_TODO: iterate over head-M tiles, call pipeline for each
            for hm in range(HEADS // CV_TILE_M):
                h0 = hm * CV_TILE_M
                # TLA_TODO: call _SmallKAttentionPipeline.run_attention_head_tile
                pass


# ===========================================================================
# Q RoPE + KV gather (prepare_query_and_gather_kv stub)
# ===========================================================================
@tla.jit
def prepare_query_and_gather_kv(
    qfull_gm, rope_cos_gm, rope_sin_gm,
    wkc_gm, wvc_gm, block_table_gm, ckc_gm, wsc_gm,
    extra_block_table_gm, cmp_dec_gm,
    indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
    topk_length_gm, extra_topk_length_gm,
    zero_gm, qb_gm, kg_gm,
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    WIN_LEN: tl.constexpr,
    GRID: tl.constexpr,
    DUAL_AIV: tl.constexpr,
    QKV_PACKED: tl.constexpr,
    QKV_QPIPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    CMP_DECODE_INDEXED: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
    CACHE_BATCH_STRIDE: tl.constexpr,
    ROPE_DIM: tl.constexpr = 64,
    GATHER_BATCH: tl.constexpr = 16,
):
    """Q partial RoPE + 1/sqrt(d) scaling, then single K/V gather for all tokens.

    Port of MegaMlaKernel._prepare_query_and_gather_kv.

    This is the largest single function in the kernel (~400 lines in the
    original). It handles:
    - Per-head Q RoPE with 1/sqrt(D) scaling
    - Window cache FP8 record gather + native decode
    - Compressed cache BF16 gather (predecode or indexed strategy)
    - Dual-AIV work distribution
    - Ping-pong double buffering for compressed rows

    TLA_TODO: Full implementation requires porting the inner loops with:
    - fp8_cache_decode from cache_ops.py
    - paged_cache_page from cache_ops.py
    - apply_query_rope_scale_paired from rotary_embedding.py
    - Extensive tla.alloc for UB/ring buffers
    - Complex token/owner work distribution logic
    """
    inv_sqrt_d = 1.0 / (D ** 0.5)
    gather_k = WIN_LEN if CMP_DECODE_INDEXED else K
    batch = 16 if QKV_PACKED else GATHER_BATCH

    # 临时分配
    cos_ub = tla.alloc((1, ROPE_DIM), tl.float32, space="ub")
    sin_ub = tla.alloc((1, ROPE_DIM), tl.float32, space="ub")
    qb_ub = tla.alloc((1, D), tl.bfloat16, space="ub", versions=2 if QKV_QPIPE else 1)
    q_f_ub = tla.alloc((1, D), tl.float32, space="ub")
    qb_rot_ub = tla.alloc((1, D), tl.bfloat16, space="ub", versions=2 if QKV_QPIPE else 1)
    kb_ub = tla.alloc((batch, D), tl.bfloat16, space="ub")
    win_idx_ub = tla.alloc((1, WIN_LEN), tl.int32, space="ub")

    # Small T uses both physical AIVs with unique row ownership
    vec_id = block
    vec_num = nblk
    if DUAL_AIV:
        vec_id = block * 2 + tla.sync.sub_block_id()
        vec_num = nblk * 2

    # Token ownership logic (balancing across AIVs)
    # TLA_TODO: port the full tok_begin/tok_end/tok_stride/part/parts logic
    # from the original. Simplified version below for the common T < vec_num case.
    for tok in range(vec_id, TOKENS, vec_num):
        # Load RoPE coefficients
        tla.copy(
            tla.subview(rope_cos_gm, offsets=[tok, 0], sizes=[1, 32]),
            tla.subview(cos_ub, offsets=[0, 0], sizes=[1, 32]),
        )
        tla.copy(
            tla.subview(rope_sin_gm, offsets=[tok, 0], sizes=[1, 32]),
            tla.subview(sin_ub, offsets=[0, 0], sizes=[1, 32]),
        )

        # Load indices
        tla.copy(
            tla.subview(indices_in_kvcache_gm, offsets=[tok, 0], sizes=[1, WIN_LEN]),
            win_idx_ub,
        )
        tla.sync.barrier(tla.sync.PIPE.ALL)

        # Apply length masking
        if WIN_LENGTH_STRIDE > 0:
            mask_index_prefix(
                win_idx_ub, WIN_LEN,
                tl.cast(topk_length_gm[tok // WIN_LENGTH_STRIDE], tl.int32),
            )

        # (6) Q partial RoPE + 1/sqrt(d) -> bf16 q
        # TLA_TODO: port the full per-head Q RoPE loop using
        # apply_query_rope_scale_paired from rotary_embedding.py

        # (9/12) K/V gather: window records
        # TLA_TODO: port the QKV_PACKED window cache decode loop
        # using fp8_cache_decode and paged_cache_page from cache_ops.py

        # Compressed rows gather (predecode or indexed)
        # TLA_TODO: port the compressed cache ping-pong gather

        tla.sync.barrier(tla.sync.PIPE.ALL)
