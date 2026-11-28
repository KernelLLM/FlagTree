# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TLA port of cannbot-dsl mega_mla_kernel.py.

Device projection/cache/gather/attention scheduling for mega_mla.

P0 initializes projection accumulators and the zero record; P1 distributes
28 logical split-K tasks per M tile over the launch grid. P2 normalizes Q/KV,
applies KV RoPE and writes the window cache; P3 produces full Q while spare
AIVs decode compressed history. Gather prepares Q/KV for attention.
Small-K attention uses the token-local online QK/Softmax/PV pipeline;
other shapes retain the flat QK, vector Softmax and PV paths.

Original: cannbot-dsl/samples/attention/mega_mla/mega_mla_kernel.py
Target:   TLA (triton-ascend low-level API for Ascend NPU)
"""

import triton
import triton.language as tl
import triton.language.extra.tla as tla

# ---------------------------------------------------------------------------
# Module-level imports from sibling TLA modules
# (These are ported from the respective cannbot-dsl helper modules)
# ---------------------------------------------------------------------------
from qkv_projection import (
    quantize_mxfp8_rows, rms_norm_quantize_query,
    rms_norm_key_value, compute_wqb_output_tile,
)
from attention_ops import (
    prepare_query_and_gather_kv, run_attention_softmax,
    run_probability_value_matmul, run_small_k_attention_pipeline,
    qk_compute_flat_tiles, qk_compute_owned_tiles,
    publish_qk_to_paired_aivs,
)
from cache_ops import (
    decode_full_compressed_cache, rms_norm_rope_quantize_window_cache,
    compressed_cache_gather_run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 512
D = HEAD_DIM
HID = 5120
QLORA = 1280
KVD = 512
BN1 = 128
BN2 = 256
BK1 = 256
BK2 = 128
KL1_1 = 512
KL1_2 = 256
SKL1_2 = 1280
L2_CACHE_DISABLE = 4
COMPRESSED_DECODE_UNROLL = 1
COMPRESSED_DECODE_DEPTH = 2
SOFTMAX_UB_RESERVE_BYTES = 16 * 1024
VECTOR_CHUNK_ELEMENTS = 64
NEG_INF = -1.7e38


# ===========================================================================
# Workspace binding: create GM views from a flat workspace pointer
# ===========================================================================
def workspace_region_specs(tokens, heads, k, strategy, win_len=128):
    """Return workspace region layout as list of (name, shape, dtype, width).

    Matches AttentionWorkspaceLayout from cannbot-dsl/attention_workspace.py.
    All regions are 512-byte aligned.
    """
    decoded = (1, 1, 512) if strategy == "indexed" else (1, 1, 512)
    return [
        ("qr", (tokens, 1280), "float32", 4),
        ("kv", (tokens, 512), "float32", 4),
        ("qfull", (tokens, heads * 512), "bfloat16", 2),
        ("x8", (tokens, 5120), "float8_e4m3fn", 1),
        ("sx", (tokens, 80, 2), "float8_e8m0", 1),
        ("qn8", (tokens, 1280), "float8_e4m3fn", 1),
        ("sqn", (tokens, 20, 2), "float8_e8m0", 1),
        ("kvn", (tokens, 512), "bfloat16", 2),
        ("qb", (tokens, heads, 512), "bfloat16", 2),
        ("kg", (tokens, k, 512), "bfloat16", 2),
        ("s", (tokens, heads, k), "float32", 4),
        ("p", (tokens, heads, k), "bfloat16", 2),
        ("cmp_dec", decoded, "bfloat16", 2),
        ("zero", (1, 512), "bfloat16", 2),
    ]


# ===========================================================================
# Sync helper: publish QK scores to paired AIVs
# ===========================================================================
@tla.jit
def publish_qk_to_paired_aivs():
    """All QK AIC complete, then each AIC releases its paired 1:2 AIV."""
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
# Projection: clear FP32 accumulators
# ===========================================================================
@tla.jit
def clear_projection_buffers(
    qr_gm,     # GM: (tokens, QLORA) float32
    kv_gm,     # GM: (tokens, KVD) float32
    zero_gm,   # GM: (1, D//2) float32 → immutable zero record
    block,
    nblk,
    TOKENS: tl.constexpr,
):
    """Share immutable zeros and one drain across independent P0 outputs.

    Port of MegaMlaKernel._clear_projection_buffers.
    """
    z = tla.alloc((1, 128), tl.float32, space="ub")

    with tla.scope.simd():
        zero = tla.simd.vdups(0.0, tl.float32)
        tla.simd.vsts(z[0, 0], zero)
        tla.simd.vsts(z[0, 64], zero)
        tla.simd.membar("VST_VLD")

    for it in range(block, TOKENS * (QLORA // 128), nblk):
        tla.copy(z, tla.subview(qr_gm, offsets=[it // (QLORA // 128), (it % (QLORA // 128)) * 128], sizes=[1, 128]))

    for it in range(block, TOKENS * (KVD // 128), nblk):
        tla.copy(z, tla.subview(kv_gm, offsets=[it // (KVD // 128), (it % (KVD // 128)) * 128], sizes=[1, 128]))

    # The BF16 invalid record has the same 1024 bytes as this FP32 view.
    # The zero buffer is immutable until every MTE3 consumer has finished.
    for it in range(block, (D // 2) // 128, nblk):
        tla.copy(z, tla.subview(zero_gm, offsets=[0, it * 128], sizes=[1, 128]))

    tla.sync.barrier(tla.sync.PIPE.ALL)


# ===========================================================================
# Projection: split-K MXfp8 tile (P1)
# ===========================================================================
@tla.jit
def compute_split_k_projection_tile(
    out_gm,   # GM: (m, N) float32 — atomic accumulator
    a_gm,     # GM: (m, K) fp8e4m3
    b_gm,     # GM: (N_total, K) fp8e4m3 (FRACTAL_NZ)
    sa_gm,    # GM: (m, K//64, 2) E8M0
    sb_gm,    # GM: (N_total, K//64, 2) E8M0
    mi,       # M tile index
    ni,       # N tile index (within block's N assignment)
    split,    # split-K index (0 or 1)
    P1_M: tl.constexpr,
    VALID_M: tl.constexpr,
):
    """Two owners accumulate separate K2560 halves into one FP32 tile.

    Port of MegaMlaKernel._compute_split_k_projection_tile.
    Uses MXfp8 matmul with atomic add fixpipe writeback.
    """
    with tla.scope.cube():
        # NZ layout for MXfp8 split-K projection
        # A[P1_M, KL1_1] → (KL1_1//16, P1_M//16, 16, 16)
        # B[BN1, KL1_1] with transpose_b=True → (KL1_1//16, BN1//16, 16, 16)
        # L0A[P1_M, BK1] → (BK1//16, P1_M//16, 16, 16)
        # L0B[BN1, BK1] → (BK1//16, BN1//16, 16, 16)
        # acc[P1_M, BN1] → (BN1//16, P1_M//16, 16, 16)
        l1_a = tla.alloc((KL1_1 // 16, P1_M // 16, 16, 16), tl.float8e4nv, space="l1", versions=4)
        l1_b = tla.alloc((KL1_1 // 16, BN1 // 16, 16, 16), tl.float8e4nv, space="l1", versions=4)
        # TLA_TODO: MX scale layout may differ from data NZ; load_gm_to_l1_mx_scale handles it
        l1_sa = tla.alloc((P1_M, HID // 64), tl.uint8, space="l1", versions=2)
        l1_sb = tla.alloc((HID // 64, BN1), tl.uint8, space="l1", versions=2)
        l0a = tla.alloc((BK1 // 16, P1_M // 16, 16, 16), tl.float8e4nv, space="l0a", versions=2)
        l0b = tla.alloc((BK1 // 16, BN1 // 16, 16, 16), tl.float8e4nv, space="l0b", versions=2)
        acc = tla.alloc((BN1 // 16, P1_M // 16, 16, 16), tl.float32, space="l0c", versions=2)

        # Load scales for this split-K half
        # K=5120 -> two disjoint 2560 slices, each 5x512 / 10x256
        tla.load_gm_to_l1_mx_scale(
            tla.subview(sa_gm, offsets=[mi * P1_M, split * 40, 0], sizes=[P1_M, 40, 2]),
            l1_sa,
        )
        tla.load_gm_to_l1_mx_scale(
            tla.subview(sb_gm, offsets=[ni * BN1, split * 40, 0], sizes=[BN1, 40, 2]),
            l1_sb,
        )

        for ki in range(5):  # 5 L1 K-tiles per split half
            gki = split * 5 + ki
            # Load A and B tiles
            tla.load_gm_to_l1(
                tla.subview(a_gm, offsets=[mi * P1_M, gki * KL1_1], sizes=[P1_M, KL1_1]),
                l1_a,
            )
            tla.load_gm_to_l1(
                tla.subview(b_gm, offsets=[ni * BN1, gki * KL1_1], sizes=[BN1, KL1_1]),
                l1_b,
            )

            for kk in range(KL1_1 // BK1):  # 2 L0 K-tiles per L1 K-tile
                si = ki * (KL1_1 // BK1) + kk

                # Copy L1 -> L0A/L0B with MX scale
                # TLA_TODO: copy_options(mx_scale=...) → need subview of scale buffers
                tla.copy_l1_to_l0(
                    tla.subview(l1_a, offsets=[0, kk * BK1], sizes=[P1_M, BK1]),
                    l0a,
                )
                tla.copy_l1_to_l0(
                    tla.subview(l1_b, offsets=[0, kk * BK1], sizes=[BN1, BK1]),
                    l0b,
                )

                # MXfp8 matmul
                tla.cube.mmad_mx(
                    l0a, l0b, l1_sa, l1_sb, acc,
                    format_a=tla.MxFormat.E4M3,
                    format_b=tla.MxFormat.E4M3,
                    clear_accum=(si == 0),
                    unit_flag_ctrl=(
                        tla.UnitFlagMode.ENABLED_WITH_UPDATE if si == 9
                        else tla.UnitFlagMode.ENABLED_WITHOUT_UPDATE
                    ),
                    unit_flag_group_id=0,
                )

        # Fixpipe atomic add writeback to FP32 GM accumulator
        tla.store_l0c_to_gm(
            acc,
            tla.subview(out_gm, offsets=[mi * P1_M, ni * BN1], sizes=[P1_M, BN1]),
            params=tla.FixpipeL0C2GMParams(
                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITH_UPDATE,
                unit_flag_group_id=0,
            ),
        )


# ===========================================================================
# Projection: dispatch split-K tasks (P1)
# ===========================================================================
@tla.jit
def run_qa_kv_projection(
    qr_gm, kv_gm, x8_gm,
    query_down_proj_weight_gm, kv_proj_weight_gm,
    sx_gm, query_down_proj_scale_gm, kv_proj_scale_gm,
    block,
    P1_M: tl.constexpr,
    P1_M_TILES: tl.constexpr,
    N1_BLOCKS: tl.constexpr,
    N3_BLOCKS: tl.constexpr,
    GRID: tl.constexpr,
):
    """P1: distribute 28 split-K matmul tasks over the AIC grid.

    Port of MegaMlaKernel._run_qa_kv_projection.
    仅 P1 的 FP8 L0B 双缓冲；N128×K256 每份 32 KiB。
    """
    for mi in range(P1_M_TILES):
        if GRID >= 2 * (N1_BLOCKS + N3_BLOCKS):
            # Full grid: each block owns exactly one task
            task = block
            if task < 2 * N1_BLOCKS:
                compute_split_k_projection_tile(
                    qr_gm, x8_gm, query_down_proj_weight_gm,
                    sx_gm, query_down_proj_scale_gm,
                    mi, task // 2, task % 2,
                    P1_M, P1_M,
                )
            elif task < 2 * (N1_BLOCKS + N3_BLOCKS):
                p = task - 2 * N1_BLOCKS
                compute_split_k_projection_tile(
                    kv_gm, x8_gm, kv_proj_weight_gm,
                    sx_gm, kv_proj_scale_gm,
                    mi, p // 2, p % 2,
                    P1_M, P1_M,
                )
        else:
            # Smaller grids cycle over all 28 logical split-K tasks
            for task in range(block, 2 * (N1_BLOCKS + N3_BLOCKS), GRID):
                if task < 2 * N1_BLOCKS:
                    compute_split_k_projection_tile(
                        qr_gm, x8_gm, query_down_proj_weight_gm,
                        sx_gm, query_down_proj_scale_gm,
                        mi, task // 2, task % 2,
                        P1_M, P1_M,
                    )
                elif task < 2 * (N1_BLOCKS + N3_BLOCKS):
                    p = task - 2 * N1_BLOCKS
                    compute_split_k_projection_tile(
                        kv_gm, x8_gm, kv_proj_weight_gm,
                        sx_gm, kv_proj_scale_gm,
                        mi, p // 2, p % 2,
                        P1_M, P1_M,
                    )


# ===========================================================================
# Projection: full pipeline P0-P3
# ===========================================================================
@tla.jit
def run_projection_stages(
    qr_gm, kv_gm, qfull_gm,
    hidden_states_gm, query_norm_weight_gm, kv_norm_weight_gm,
    query_down_proj_weight_gm, kv_proj_weight_gm, query_up_proj_weight_gm,
    x8_gm, sx_gm, qn8_gm, sqn_gm,
    query_down_proj_scale_gm, kv_proj_scale_gm, query_up_proj_scale_gm,
    kvn_gm,
    ckc_gm, cmp_dec_gm,
    extra_indices_in_kvcache_gm, cmp_lengths_gm,
    zero_gm, kg_gm, extra_block_table_gm,
    rope_cos_gm, rope_sin_gm,
    window_cache_slots_gm, wkc_gm, wvc_gm, wsc_gm,
    block, nblk,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    P1_M: tl.constexpr,
    P1_M_TILES: tl.constexpr,
    N1_BLOCKS: tl.constexpr,
    N3_BLOCKS: tl.constexpr,
    N2_BLOCKS: tl.constexpr,
    M_TILES: tl.constexpr,
    GRID: tl.constexpr,
    MODE: tl.constexpr,
    CMP_DECODE_INDEXED: tl.constexpr,
    CMP_P1_ITEMS: tl.constexpr,
    CMP_P3_ITEMS: tl.constexpr,
    BASE_M_K2: tl.constexpr,
    K_ATTN: tl.constexpr,
    WIN_LEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
    CACHE_BATCH_STRIDE: tl.constexpr,
    N2_BLOCKS: tl.constexpr,
    M_TILES: tl.constexpr,
    GRID: tl.constexpr,
    MODE: tl.constexpr,
    CMP_DECODE_INDEXED: tl.constexpr,
    CMP_P1_ITEMS: tl.constexpr,
    CMP_P3_ITEMS: tl.constexpr,
    BASE_M_K2: tl.constexpr,
):
    """Full projection pipeline P0 → P1 → P2 → P3.

    Port of MegaMlaKernel._run_projection_stages.
    """
    sub_block = tla.sync.sub_block_id()

    # --- P0: AIV0 quantizes x; AIV1 clears atomic destinations ---
    # These buffers are independent; the common B0 barrier publishes both.
    if sub_block == 0:
        quantize_mxfp8_rows(hidden_states_gm, x8_gm, sx_gm, block, nblk, TOKENS, HID)
    else:
        clear_projection_buffers(qr_gm, kv_gm, zero_gm, block, nblk, TOKENS)

    # Publish quantization and clear completion before P1 atomic stores.
    tla.sync.sync_all(True)

    # --- P1: split-K MXfp8 projection (AIC) ---
    run_qa_kv_projection(
        qr_gm, kv_gm, x8_gm,
        query_down_proj_weight_gm, kv_proj_weight_gm,
        sx_gm, query_down_proj_scale_gm, kv_proj_scale_gm,
        block,
        P1_M, P1_M_TILES, N1_BLOCKS, N3_BLOCKS, GRID,
    )

    if MODE != "proj" and not CMP_DECODE_INDEXED:
        # All 2*nblk AIVs decode physical compressed rows while the
        # P1 AIC owners compute. This existing barrier also publishes
        # the decoded BF16 workspace before the later indexed gather.
        decode_full_compressed_cache(
            ckc_gm, v_gm, block, nblk, COMPRESSED_PAGE_SIZE)

    if MODE != "proj" and CMP_DECODE_INDEXED and CMP_P1_ITEMS > 0:
        compressed_cache_gather_run(
            ckc_gm, extra_indices_in_kvcache_gm, extra_topk_length_gm,
            zero_gm, kg_gm,
            block * 2 + sub_block, nblk * 2, extra_block_table_gm,
            TOKENS, K_ATTN - WIN_LEN, COMPRESSED_PAGE_SIZE, WIN_LEN,
            CMP_LENGTH_STRIDE, CACHE_BATCH_STRIDE,
            0, CMP_P1_ITEMS,
        )

    # All P1 atomic FIXPIPE stores complete before any norm load.
    tla.sync.sync_all(True)

    # --- P2: B2 each token's two AIV partners work independently after B1 ---
    # AIV0 prepares Q for WqB; AIV1 keeps KV norm/RoPE/quantization in UB
    # and writes the finished window record. B2 publishes both branches.
    if sub_block == 0:
        rms_norm_quantize_query(
            qr_gm, query_norm_weight_gm, qn8_gm, sqn_gm,
            block, nblk, TOKENS, QLORA)
    else:
        if MODE == "proj":
            rms_norm_key_value(
                kv_gm, kv_norm_weight_gm, kvn_gm,
                block, nblk, TOKENS, KVD)
        else:
            rms_norm_rope_quantize_window_cache(
                kv_gm, kv_norm_weight_gm,
                rope_cos_gm, rope_sin_gm,
                window_cache_slots_gm, wkc_gm,
                block, nblk, TOKENS, D, PAGE_SIZE)

    tla.sync.barrier(tla.sync.PIPE.ALL)
    tla.sync.sync_all(True)

    # TLA_TODO: onchip_rewind — no direct TLA equivalent
    # onchip_rewind(reset_sync_id=True)

    # --- P3: WqB (AIC) with block-stride N scheduling ---
    with tla.scope.cube():
        for it in range(block, M_TILES * N2_BLOCKS, nblk):
            m_idx = it // N2_BLOCKS
            n_idx = it % N2_BLOCKS
            compute_wqb_output_tile(
                qfull_gm, qn8_gm, query_up_proj_weight_gm,
                sqn_gm, query_up_proj_scale_gm,
                m_idx, n_idx, BASE_M_K2, BN2, BK2, KL1_2, SKL1_2)

    if MODE != "proj" and CMP_DECODE_INDEXED and CMP_P3_ITEMS > CMP_P1_ITEMS:
        # AIC执行WqB期间，AIV从独立的压缩cache读取并写最终kg压缩区。
        compressed_cache_gather_run(
            ckc_gm, extra_indices_in_kvcache_gm, extra_topk_length_gm,
            zero_gm, kg_gm,
            block * 2 + sub_block, nblk * 2, extra_block_table_gm,
            TOKENS, K_ATTN - WIN_LEN, COMPRESSED_PAGE_SIZE, WIN_LEN,
            CMP_LENGTH_STRIDE, CACHE_BATCH_STRIDE,
            CMP_P1_ITEMS, CMP_P3_ITEMS,
        )


# ===========================================================================
# Main kernel body
# ===========================================================================
@tla.jit
def mega_mla_body(
    # --- GM tensor pointers (bound from host) ---
    hidden_states_gm,
    query_norm_weight_gm, kv_norm_weight_gm,
    query_down_proj_weight_gm, kv_proj_weight_gm, query_up_proj_weight_gm,
    query_down_proj_scale_gm, kv_proj_scale_gm, query_up_proj_scale_gm,
    k_cache_gm, extra_k_cache_gm,
    indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
    attn_sink_gm,
    rope_cos_gm, rope_sin_gm,
    window_cache_slots_gm,
    block_table_gm, extra_block_table_gm,
    topk_length_gm, extra_topk_length_gm,
    out_gm,
    workspace_gm, workspace_bf16_gm,
    workspace_fp8_gm, workspace_e8m0_gm, workspace_int32_gm,
    # --- Compile-time config ---
    K_ATTN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIN_LEN: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    GRID: tl.constexpr,
    MODE: tl.constexpr,            # "attn" | "proj" | "cache"
    CMP_DECODE_INDEXED: tl.constexpr,
    CMP_P1_ITEMS: tl.constexpr,
    CMP_P3_ITEMS: tl.constexpr,
    CV_TILE_M: tl.constexpr,
    FLAT_ATTENTION_TILES: tl.constexpr,
    DUAL_AIV: tl.constexpr,
    SOFTMAX_BATCH_ROWS: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    QKV_PACKED: tl.constexpr,
    SMALL_TOKEN_VECTOR_SCHEDULE: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_BATCH_STRIDE: tl.constexpr,
):
    """Core mega_mla device body.

    Port of MegaMlaKernel._mega_mla_body.
    All private GM intermediates are disjoint views of one allocation.
    Existing phase barriers publish them across their producer/consumer cores.
    """
    # ---------------------------------------------------------------
    # Workspace binding: carve typed GM views from the flat allocation.
    #
    # The host passes five typed aliases of a single byte-allocation:
    #   workspace_gm      → float32 view  (elem width 4)
    #   workspace_bf16_gm → bfloat16 view (elem width 2)
    #   workspace_fp8_gm  → fp8e4m3 view  (elem width 1)
    #   workspace_e8m0_gm → uint8 view    (elem width 1)
    #   workspace_int32_gm→ int32 view    (elem width 4)
    #
    # Region byte offsets are 512-byte aligned and monotonically
    # increasing, matching AttentionWorkspaceLayout exactly.
    # Each tla.view slices the appropriately-typed pointer alias
    # at   elem_offset = byte_offset // elem_width   elements.
    #
    # All quantities below are tl.constexpr (TOKENS, HEADS, K_ATTN).
    # ---------------------------------------------------------------
    _A: tl.constexpr = 512  # alignment in bytes

    # Helper: align-up to _A bytes (constexpr)
    # _a(v) = ceil(v / 512) * 512
    # Written inline because tla.jit does not support nested def.

    # Region 0: qr — (TOKENS, 1280) float32
    _off_qr: tl.constexpr = 0
    _sz_qr: tl.constexpr = TOKENS * QLORA * 4
    qr_gm = tla.view(workspace_gm,
                      shape=(TOKENS, QLORA),
                      offsets=[_off_qr // 4],
                      space="gm")

    # Region 1: kv — (TOKENS, 512) float32
    _off_kv: tl.constexpr = ((_off_qr + _sz_qr + _A - 1) // _A) * _A
    _sz_kv: tl.constexpr = TOKENS * KVD * 4
    kv_gm = tla.view(workspace_gm,
                      shape=(TOKENS, KVD),
                      offsets=[_off_kv // 4],
                      space="gm")

    # Region 2: qfull — (TOKENS, HEADS*D) bfloat16
    _off_qfull: tl.constexpr = ((_off_kv + _sz_kv + _A - 1) // _A) * _A
    _sz_qfull: tl.constexpr = TOKENS * HEADS * D * 2
    qfull_gm = tla.view(workspace_bf16_gm,
                         shape=(TOKENS, HEADS * D),
                         offsets=[_off_qfull // 2],
                         space="gm")

    # Region 3: x8 — (TOKENS, 5120) float8_e4m3fn
    _off_x8: tl.constexpr = ((_off_qfull + _sz_qfull + _A - 1) // _A) * _A
    _sz_x8: tl.constexpr = TOKENS * HID * 1
    x8_gm = tla.view(workspace_fp8_gm,
                      shape=(TOKENS, HID),
                      offsets=[_off_x8 // 1],
                      space="gm")

    # Region 4: sx — (TOKENS, 80, 2) float8_e8m0  (stored as uint8)
    _off_sx: tl.constexpr = ((_off_x8 + _sz_x8 + _A - 1) // _A) * _A
    _sz_sx: tl.constexpr = TOKENS * 80 * 2 * 1
    sx_gm = tla.view(workspace_e8m0_gm,
                      shape=(TOKENS, 80, 2),
                      offsets=[_off_sx // 1],
                      space="gm")

    # Region 5: qn8 — (TOKENS, 1280) float8_e4m3fn
    _off_qn8: tl.constexpr = ((_off_sx + _sz_sx + _A - 1) // _A) * _A
    _sz_qn8: tl.constexpr = TOKENS * QLORA * 1
    qn8_gm = tla.view(workspace_fp8_gm,
                       shape=(TOKENS, QLORA),
                       offsets=[_off_qn8 // 1],
                       space="gm")

    # Region 6: sqn — (TOKENS, 20, 2) float8_e8m0
    _off_sqn: tl.constexpr = ((_off_qn8 + _sz_qn8 + _A - 1) // _A) * _A
    _sz_sqn: tl.constexpr = TOKENS * 20 * 2 * 1
    sqn_gm = tla.view(workspace_e8m0_gm,
                       shape=(TOKENS, 20, 2),
                       offsets=[_off_sqn // 1],
                       space="gm")

    # Region 7: kvn — (TOKENS, 512) bfloat16
    _off_kvn: tl.constexpr = ((_off_sqn + _sz_sqn + _A - 1) // _A) * _A
    _sz_kvn: tl.constexpr = TOKENS * KVD * 2
    kvn_gm = tla.view(workspace_bf16_gm,
                       shape=(TOKENS, KVD),
                       offsets=[_off_kvn // 2],
                       space="gm")

    # Region 8: qb — (TOKENS, HEADS, D) bfloat16
    _off_qb: tl.constexpr = ((_off_kvn + _sz_kvn + _A - 1) // _A) * _A
    _sz_qb: tl.constexpr = TOKENS * HEADS * D * 2
    qb_gm = tla.view(workspace_bf16_gm,
                      shape=(TOKENS, HEADS, D),
                      offsets=[_off_qb // 2],
                      space="gm")

    # Region 9: kg — (TOKENS, K_ATTN, D) bfloat16
    _off_kg: tl.constexpr = ((_off_qb + _sz_qb + _A - 1) // _A) * _A
    _sz_kg: tl.constexpr = TOKENS * K_ATTN * D * 2
    kg_gm = tla.view(workspace_bf16_gm,
                      shape=(TOKENS, K_ATTN, D),
                      offsets=[_off_kg // 2],
                      space="gm")

    # Region 10: s — (TOKENS, HEADS, K_ATTN) float32
    _off_s: tl.constexpr = ((_off_kg + _sz_kg + _A - 1) // _A) * _A
    _sz_s: tl.constexpr = TOKENS * HEADS * K_ATTN * 4
    s_gm = tla.view(workspace_gm,
                     shape=(TOKENS, HEADS, K_ATTN),
                     offsets=[_off_s // 4],
                     space="gm")

    # Region 11: p — (TOKENS, HEADS, K_ATTN) bfloat16
    _off_p: tl.constexpr = ((_off_s + _sz_s + _A - 1) // _A) * _A
    _sz_p: tl.constexpr = TOKENS * HEADS * K_ATTN * 2
    p_gm = tla.view(workspace_bf16_gm,
                     shape=(TOKENS, HEADS, K_ATTN),
                     offsets=[_off_p // 2],
                     space="gm")

    # Region 12: cmp_dec — bfloat16 decoded compressed cache
    # For the indexed strategy this is a minimal (1,1,D) placeholder;
    # for predecode it is (compressed_pages, page_size, D).
    # The shape depends on host-side strategy selection; use a small
    # constant here, matching the indexed default.
    _off_cmp: tl.constexpr = ((_off_p + _sz_p + _A - 1) // _A) * _A
    _sz_cmp: tl.constexpr = 1 * 1 * D * 2  # indexed strategy minimum
    v_gm = tla.view(workspace_bf16_gm,
                     shape=(1, 1, D),
                     offsets=[_off_cmp // 2],
                     space="gm")

    # Region 13: zero — (1, D) bfloat16 immutable zero record
    _off_zero: tl.constexpr = ((_off_cmp + _sz_cmp + _A - 1) // _A) * _A
    zero_gm = tla.view(workspace_bf16_gm,
                        shape=(1, D),
                        offsets=[_off_zero // 2],
                        space="gm")

    # ---------------------------------------------------------------
    # Window cache byte views
    # The host k_cache tensor has shape [N_pages, PAGE_SIZE, 1, 544]
    # with a page stride >= PAGE_SIZE * 544 bytes.
    # wkc_gm: int8 view  for byte-level record access
    # wvc_gm: uint64 view for 8-byte store-bypass writes
    # wsc_gm: uint32 view for 4-byte scale writes
    # In TLA, these are dtype-reinterpreted views of the same pointer.
    # ---------------------------------------------------------------
    ps = PAGE_SIZE
    wkc_gm = tla.view(k_cache_gm, shape=(k_cache_gm.shape[0], ps * 544),
                       space="gm")
    # wvc_gm and wsc_gm share the same underlying data with different
    # element types. TLA view reinterpretation:
    wvc_gm = wkc_gm  # uint64 reinterpretation used only by vec_store_bypass
    wsc_gm = wkc_gm  # uint32 reinterpretation used only for scale writes

    # Compressed cache: flatten page-local axes into (pages, ps, 320)
    ckc_gm = tla.view(extra_k_cache_gm,
                       shape=(extra_k_cache_gm.shape[0], COMPRESSED_PAGE_SIZE, 320),
                       space="gm")

    block = tl.program_id(0)
    nblk = GRID

    # Derived projection constants
    P1_M: tl.constexpr = min(64, TOKENS)
    P1_M_TILES: tl.constexpr = (TOKENS + P1_M - 1) // P1_M
    N1_BLOCKS: tl.constexpr = QLORA // BN1    # 10
    N3_BLOCKS: tl.constexpr = KVD // BN1      # 4
    Q_WIDTH: tl.constexpr = HEADS * HEAD_DIM
    N2_BLOCKS: tl.constexpr = Q_WIDTH // BN2
    M_TILES: tl.constexpr = (TOKENS + 15) // 16
    BASE_M_K2: tl.constexpr = min(16, TOKENS)

    # One shared flag triple, reused by every barrier (manual id reuse).
    # This kernel owns one reusable global-barrier triple.

    # --- Projection stages P0-P3 ---
    run_projection_stages(
        qr_gm, kv_gm, qfull_gm,
        hidden_states_gm, query_norm_weight_gm, kv_norm_weight_gm,
        query_down_proj_weight_gm, kv_proj_weight_gm, query_up_proj_weight_gm,
        x8_gm, sx_gm, qn8_gm, sqn_gm,
        query_down_proj_scale_gm, kv_proj_scale_gm, query_up_proj_scale_gm,
        kvn_gm,
        ckc_gm, v_gm,
        extra_indices_in_kvcache_gm, extra_topk_length_gm,
        zero_gm, kg_gm, extra_block_table_gm,
        rope_cos_gm, rope_sin_gm,
        window_cache_slots_gm, wkc_gm, wvc_gm, wsc_gm,
        block, nblk,
        TOKENS, HEADS, P1_M, P1_M_TILES,
        N1_BLOCKS, N3_BLOCKS, N2_BLOCKS, M_TILES, GRID,
        MODE, CMP_DECODE_INDEXED, CMP_P1_ITEMS, CMP_P3_ITEMS, BASE_M_K2,
        K_ATTN, WIN_LEN, PAGE_SIZE, COMPRESSED_PAGE_SIZE,
        CMP_LENGTH_STRIDE, CACHE_BATCH_STRIDE,
    )

    if MODE != "proj":
        # QK consumers see completed WqB and quantized window scatter.
        tla.sync.sync_all(True)

        # (6)+(9/12) Q RoPE + single K/V gather for all tokens
        prepare_query_and_gather_kv(
            qfull_gm, rope_cos_gm, rope_sin_gm,
            wkc_gm, wvc_gm, block_table_gm, ckc_gm, wsc_gm,
            extra_block_table_gm, v_gm,
            indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
            topk_length_gm, extra_topk_length_gm,
            zero_gm, qb_gm, kg_gm, block, nblk,
            TOKENS, HEADS, K_ATTN, WIN_LEN, GRID,
            DUAL_AIV, QKV_PACKED, QKV_PACKED,  # QKV_QPIPE = QKV_PACKED
            PAGE_SIZE, COMPRESSED_PAGE_SIZE,
            CMP_DECODE_INDEXED, WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
            CACHE_BATCH_STRIDE,
        )

        if CMP_DECODE_INDEXED and CMP_P3_ITEMS < TOKENS * (K_ATTN - WIN_LEN):
            compressed_cache_gather_run(
                ckc_gm, extra_indices_in_kvcache_gm, extra_topk_length_gm,
                zero_gm, kg_gm,
                block * 2 + tla.sync.sub_block_id(), nblk * 2,
                extra_block_table_gm,
                TOKENS, K_ATTN - WIN_LEN, COMPRESSED_PAGE_SIZE, WIN_LEN,
                CMP_LENGTH_STRIDE, CACHE_BATCH_STRIDE,
                CMP_P3_ITEMS, TOKENS * (K_ATTN - WIN_LEN),
            )

        tla.sync.sync_all(True)

        if MODE != "cache":
            # CV split-M is valid only for H64. H8/H16 keep logical M
            # and the existing GM bridge, with no padded head computation.
            if CV_TILE_M > 0:
                tla.sync.barrier(tla.sync.PIPE.ALL)
                # TLA_TODO: onchip_rewind
                run_small_k_attention_pipeline(
                    qb_gm, kg_gm,
                    indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
                    attn_sink_gm, out_gm,
                    topk_length_gm, extra_topk_length_gm,
                    block, nblk,
                    K_ATTN, TOKENS, HEADS, GRID, WIN_LEN,
                    CV_TILE_M, WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
                )
            else:
                # QK Cube, then mask/sink softmax AIV, then PV Cube.
                tla.sync.barrier(tla.sync.PIPE.ALL)
                # TLA_TODO: onchip_rewind

                # --- QK matmul ---
                n_tiles_qk: tl.constexpr = K_ATTN // 64
                if TOKENS <= GRID and not FLAT_ATTENTION_TILES:
                    bpt_qk: tl.constexpr = max(1, GRID // TOKENS)
                    nt_per_qk: tl.constexpr = (n_tiles_qk + bpt_qk - 1) // bpt_qk
                    qk_compute_owned_tiles(
                        s_gm, qb_gm, kg_gm, block,
                        TOKENS, HEADS, K_ATTN, bpt_qk, nt_per_qk,
                    )
                else:
                    qk_compute_flat_tiles(
                        s_gm, qb_gm, kg_gm, block, nblk,
                        TOKENS, HEADS, K_ATTN,
                    )

                if GRID < TOKENS <= 2 * GRID:
                    publish_qk_to_paired_aivs()
                else:
                    tla.sync.sync_all(True)

                # --- Softmax ---
                run_attention_softmax(
                    s_gm, indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
                    attn_sink_gm, p_gm,
                    topk_length_gm, extra_topk_length_gm,
                    block, nblk,
                    TOKENS, HEADS, K_ATTN, WIN_LEN, GRID,
                    DUAL_AIV, SOFTMAX_BATCH_ROWS,
                    WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
                )

                tla.sync.sync_all(True)

                # --- PV matmul ---
                run_probability_value_matmul(
                    out_gm, p_gm, kg_gm, block, nblk,
                    TOKENS, HEADS, K_ATTN, GRID, FLAT_ATTENTION_TILES,
                )


# ===========================================================================
# Kernel entry: mega_mla
# ===========================================================================
@tla.jit
def mega_mla(
    # All GM tensor pointers from the host ABI
    ptr_hidden_states,
    ptr_query_norm_weight, ptr_kv_norm_weight,
    ptr_query_down_proj_weight, ptr_kv_proj_weight, ptr_query_up_proj_weight,
    ptr_query_down_proj_scale, ptr_kv_proj_scale, ptr_query_up_proj_scale,
    ptr_k_cache, ptr_extra_k_cache,
    ptr_indices_in_kvcache, ptr_extra_indices_in_kvcache,
    ptr_attn_sink,
    ptr_rope_cos, ptr_rope_sin,
    ptr_window_cache_slots,
    ptr_block_table, ptr_extra_block_table,
    ptr_topk_length, ptr_extra_topk_length,
    ptr_out,
    ptr_workspace, ptr_workspace_bf16,
    ptr_workspace_fp8, ptr_workspace_e8m0, ptr_workspace_int32,
    # Compile-time geometry
    K_ATTN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIN_LEN: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    GRID: tl.constexpr,
    MODE: tl.constexpr,
    CMP_DECODE_INDEXED: tl.constexpr,
    CMP_P1_ITEMS: tl.constexpr,
    CMP_P3_ITEMS: tl.constexpr,
    CV_TILE_M: tl.constexpr,
    FLAT_ATTENTION_TILES: tl.constexpr,
    DUAL_AIV: tl.constexpr,
    SOFTMAX_BATCH_ROWS: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    QKV_PACKED: tl.constexpr,
    SMALL_TOKEN_VECTOR_SCHEDULE: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """Kernel entry for mega_mla.

    Port of MegaMlaKernel.mega_mla (@kernel).
    """
    # Bind GM pointers to shaped buffers
    hidden_states_gm = tla.view(ptr_hidden_states, shape=(TOKENS, HID), space="gm")
    query_norm_weight_gm = tla.view(ptr_query_norm_weight, shape=(1, QLORA), space="gm")
    kv_norm_weight_gm = tla.view(ptr_kv_norm_weight, shape=(1, KVD), space="gm")
    # TLA_TODO: weight shapes depend on FRACTAL_NZ layout
    query_down_proj_weight_gm = tla.view(ptr_query_down_proj_weight, shape=(QLORA, HID), space="gm")
    kv_proj_weight_gm = tla.view(ptr_kv_proj_weight, shape=(KVD, HID), space="gm")
    query_up_proj_weight_gm = tla.view(ptr_query_up_proj_weight, shape=(HEADS * D, QLORA), space="gm")
    query_down_proj_scale_gm = tla.view(ptr_query_down_proj_scale, shape=(QLORA, HID // 64, 2), space="gm")
    kv_proj_scale_gm = tla.view(ptr_kv_proj_scale, shape=(KVD, HID // 64, 2), space="gm")
    query_up_proj_scale_gm = tla.view(ptr_query_up_proj_scale, shape=(HEADS * D, QLORA // 64, 2), space="gm")
    # TLA_TODO: k_cache shape depends on page layout [N_pages, PAGE_SIZE, 1, record_bytes]
    k_cache_gm = tla.view(ptr_k_cache, shape=(1, PAGE_SIZE * 544), space="gm")
    extra_k_cache_gm = tla.view(ptr_extra_k_cache, shape=(1, COMPRESSED_PAGE_SIZE, 320), space="gm")
    indices_in_kvcache_gm = tla.view(ptr_indices_in_kvcache, shape=(TOKENS, WIN_LEN), space="gm")
    extra_indices_in_kvcache_gm = tla.view(ptr_extra_indices_in_kvcache, shape=(TOKENS, K_ATTN - WIN_LEN), space="gm")
    attn_sink_gm = tla.view(ptr_attn_sink, shape=(1, HEADS), space="gm")
    rope_cos_gm = tla.view(ptr_rope_cos, shape=(TOKENS, 32), space="gm")
    rope_sin_gm = tla.view(ptr_rope_sin, shape=(TOKENS, 32), space="gm")
    window_cache_slots_gm = tla.view(ptr_window_cache_slots, shape=(TOKENS,), space="gm")
    block_table_gm = tla.view(ptr_block_table, shape=(1, 1), space="gm")  # TLA_TODO: proper shape
    extra_block_table_gm = tla.view(ptr_extra_block_table, shape=(1, 1), space="gm")  # TLA_TODO
    topk_length_gm = tla.view(ptr_topk_length, shape=(TOKENS,), space="gm")
    extra_topk_length_gm = tla.view(ptr_extra_topk_length, shape=(TOKENS,), space="gm")
    out_gm = tla.view(ptr_out, shape=(TOKENS, HEADS, D), space="gm")
    workspace_gm = tla.view(ptr_workspace, shape=(1,), space="gm")  # TLA_TODO: proper flat size
    workspace_bf16_gm = tla.view(ptr_workspace_bf16, shape=(1,), space="gm")
    workspace_fp8_gm = tla.view(ptr_workspace_fp8, shape=(1,), space="gm")
    workspace_e8m0_gm = tla.view(ptr_workspace_e8m0, shape=(1,), space="gm")
    workspace_int32_gm = tla.view(ptr_workspace_int32, shape=(1,), space="gm")

    mega_mla_body(
        hidden_states_gm,
        query_norm_weight_gm, kv_norm_weight_gm,
        query_down_proj_weight_gm, kv_proj_weight_gm, query_up_proj_weight_gm,
        query_down_proj_scale_gm, kv_proj_scale_gm, query_up_proj_scale_gm,
        k_cache_gm, extra_k_cache_gm,
        indices_in_kvcache_gm, extra_indices_in_kvcache_gm,
        attn_sink_gm,
        rope_cos_gm, rope_sin_gm,
        window_cache_slots_gm,
        block_table_gm, extra_block_table_gm,
        topk_length_gm, extra_topk_length_gm,
        out_gm,
        workspace_gm, workspace_bf16_gm,
        workspace_fp8_gm, workspace_e8m0_gm, workspace_int32_gm,
        K_ATTN, PAGE_SIZE, WIN_LEN, TOKENS, HEADS, GRID, MODE,
        CMP_DECODE_INDEXED, CMP_P1_ITEMS, CMP_P3_ITEMS,
        CV_TILE_M, FLAT_ATTENTION_TILES, DUAL_AIV, SOFTMAX_BATCH_ROWS,
        COMPRESSED_PAGE_SIZE, QKV_PACKED, SMALL_TOKEN_VECTOR_SCHEDULE,
        WIN_LENGTH_STRIDE, CMP_LENGTH_STRIDE,
        PAGE_SIZE, 0,  # CACHE_BATCH_STRIDE=0 for non-batched path
    )
@tla.jit
def mega_mla_with_seqlens(
    # Same GM pointers as mega_mla, plus sequence metadata
    ptr_hidden_states,
    ptr_query_norm_weight, ptr_kv_norm_weight,
    ptr_query_down_proj_weight, ptr_kv_proj_weight, ptr_query_up_proj_weight,
    ptr_query_down_proj_scale, ptr_kv_proj_scale, ptr_query_up_proj_scale,
    ptr_k_cache, ptr_extra_k_cache,
    ptr_indices_in_kvcache, ptr_extra_indices_in_kvcache,
    ptr_attn_sink,
    ptr_rope_cos, ptr_rope_sin,
    ptr_window_cache_slots,
    ptr_block_table, ptr_extra_block_table,
    ptr_topk_length, ptr_extra_topk_length,
    ptr_out,
    ptr_workspace, ptr_workspace_bf16,
    ptr_workspace_fp8, ptr_workspace_e8m0, ptr_workspace_int32,
    ptr_cu_seqlens_qo, ptr_seqused_q,
    ptr_cache_seqlens, ptr_extra_cache_seqlens,
    # Compile-time config — same as mega_mla
    K_ATTN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIN_LEN: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    GRID: tl.constexpr,
    MODE: tl.constexpr,
    CMP_DECODE_INDEXED: tl.constexpr,
    CMP_P1_ITEMS: tl.constexpr,
    CMP_P3_ITEMS: tl.constexpr,
    CV_TILE_M: tl.constexpr,
    FLAT_ATTENTION_TILES: tl.constexpr,
    DUAL_AIV: tl.constexpr,
    SOFTMAX_BATCH_ROWS: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    QKV_PACKED: tl.constexpr,
    SMALL_TOKEN_VECTOR_SCHEDULE: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """Kernel entry with per-sequence index filtering.

    Port of MegaMlaKernel.mega_mla_with_seqlens (@kernel).
    Runs sequence_filter first to remap indices, then calls mega_mla_body.
    """
    # TLA_TODO: bind workspace and sequence metadata GM views
    # TLA_TODO: call sequence_filter.run to produce seq_win_idx, seq_cmp_idx, seq_slot
    # TLA_TODO: drain local metadata writes
    tla.sync.barrier(tla.sync.PIPE.ALL)
    # TLA_TODO: onchip_rewind
    if not SMALL_TOKEN_VECTOR_SCHEDULE:
        tla.sync.sync_all(True)

    # TLA_TODO: call mega_mla_body with filtered indices
    # mega_mla_body(..., seq_win_idx, seq_cmp_idx, ..., seq_slot, ...)
    pass


# ===========================================================================
# Kernel entry: mega_mla_with_metadata
# ===========================================================================
@tla.jit
def mega_mla_with_metadata(
    # Same as mega_mla_with_seqlens, plus query_metadata_gm
    ptr_hidden_states,
    ptr_query_norm_weight, ptr_kv_norm_weight,
    ptr_query_down_proj_weight, ptr_kv_proj_weight, ptr_query_up_proj_weight,
    ptr_query_down_proj_scale, ptr_kv_proj_scale, ptr_query_up_proj_scale,
    ptr_k_cache, ptr_extra_k_cache,
    ptr_indices_in_kvcache, ptr_extra_indices_in_kvcache,
    ptr_attn_sink,
    ptr_rope_cos, ptr_rope_sin,
    ptr_window_cache_slots,
    ptr_block_table, ptr_extra_block_table,
    ptr_topk_length, ptr_extra_topk_length,
    ptr_out,
    ptr_workspace, ptr_workspace_bf16,
    ptr_workspace_fp8, ptr_workspace_e8m0, ptr_workspace_int32,
    ptr_cu_seqlens_qo, ptr_seqused_q,
    ptr_cache_seqlens, ptr_extra_cache_seqlens,
    ptr_query_metadata,
    # Compile-time config
    K_ATTN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIN_LEN: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    GRID: tl.constexpr,
    MODE: tl.constexpr,
    CMP_DECODE_INDEXED: tl.constexpr,
    CMP_P1_ITEMS: tl.constexpr,
    CMP_P3_ITEMS: tl.constexpr,
    CV_TILE_M: tl.constexpr,
    FLAT_ATTENTION_TILES: tl.constexpr,
    DUAL_AIV: tl.constexpr,
    SOFTMAX_BATCH_ROWS: tl.constexpr,
    COMPRESSED_PAGE_SIZE: tl.constexpr,
    QKV_PACKED: tl.constexpr,
    SMALL_TOKEN_VECTOR_SCHEDULE: tl.constexpr,
    WIN_LENGTH_STRIDE: tl.constexpr,
    CMP_LENGTH_STRIDE: tl.constexpr,
):
    """Kernel entry with sequence filtering and query metadata.

    Port of MegaMlaKernel.mega_mla_with_metadata (@kernel).
    """
    # TLA_TODO: bind workspace and sequence/metadata GM views
    # TLA_TODO: call sequence_filter.run with query_metadata_gm
    tla.sync.barrier(tla.sync.PIPE.ALL)
    # TLA_TODO: onchip_rewind
    if not SMALL_TOKEN_VECTOR_SCHEDULE:
        tla.sync.sync_all(True)

    # TLA_TODO: call mega_mla_body with filtered indices
    pass
