import argparse
import os

import torch
import triton
import triton.language as tl

# =============================================================================
#  arch22 "3-task" (Cube 双发) FlashAttention schedule, ported to Triton/TileIR
#  following tilelang_3_task_fa.py.
#
#   * 3-task pipeline (taskId % 3): MM1(g) dual-issues with MM2(g-1); Vec1(g)
#     softmax dual-issues with Vec2(g-1) rescale+accumulate.
#   * GLOBAL pipeline: (tile x KV) loops flattened into one task stream.
#   * nRatio (NR): one task covers NR consecutive KV blocks; single cross-core
#     handshake per task cuts C-V syncs by NR.
#   * Full online-softmax state: neg_sm ping-pong, r_factors, sumexp_is,
#     acc_o; finalises at the last KV block of each output tile.
#   * Each vector sub-core (vid 0/1) owns HALF_M = 64 rows.
# =============================================================================

import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa.core import (
    tile_alloc,
    tile_copy,
    tile_to_tensor,
    tile_pipe_barrier,
    tensor_to_tile,
)
from triton.experimental.tle.language.dsa.ascend import (  # noqa: F401
    PIPE, sync_block_set, sync_block_wait,
    L1, L0A, L0B, L0C,
)
import triton.language.extra.cann.extension as al

# ---- TileIR Pipe ids -------------------------------------------------------
PIPE_M, PIPE_V, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_FIX, PIPE_S = 0,1,2,3,4,5,6

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_CORES = 24
BLOCK_M   = 128
BLOCK_N   = 128
DIM       = 128
HALF_M    = BLOCK_M // 2   # each vector sub-core owns half the rows

CBM  = tl.constexpr(BLOCK_M)
CBHM = tl.constexpr(HALF_M)
CBN  = tl.constexpr(BLOCK_N)
CD   = tl.constexpr(DIM)

# ---- arch22 "3-task" schedule constants -----------------------------------
RING = 3   # depth of the task ring  (the "3-task" of the schedule)

# ---- intra-core signal IDs (Cube scope) ------------------------------------
SIG_K_L1 = 0
SIG_P_L1 = 1
SIG_V_L1 = 2
SIG_L0AB = 3   # double-buffer base; slot s -> SIG_L0AB+s  (3,4)
SIG_L0C  = 5   # double-buffer base; slot s -> SIG_L0C+s   (5,6)
SIG_Q    = 7   # resident-Q guard across tiles

# ---- intra-core signal IDs (Vector scope) ----------------------------------
SIG_IO_UB  = 0
SIG_S_HALF = 1

# ---- cross-core semaphore IDs ----------------------------------------------
SEM_S_READY  = 0   # C->V : workspace_s (S)   has data
SEM_S_FREE   = 1   # V->C : workspace_s slot  free
SEM_P_READY  = 2   # V->C : workspace_p (P)   has data
SEM_P_FREE   = 3   # C->V : workspace_p slot  free
SEM_PV_READY = 4   # C->V : workspace_pv (PV) has data
SEM_PV_FREE  = 5   # V->C : workspace_pv slot  free


# =============================================================================
#  Step sub-functions (each called from the main kernel; decorated @triton.jit
#  so the compiler sees them as inlinable device functions).
# =============================================================================

@triton.jit
def _mm1_qkt(
    # inputs
    Q, K,
    q_l1, k_l1, mm1_l0a, mm1_l0b,
    workspace_s,
    # task geometry
    cid, task_in_tile, tile_row, batch_idx, head_idx, kv_head_idx,
    ring_slot, tasks_per_tile,
    # strides
    sQb, sQh, sQs, sQd,
    sKb, sKh, sKs, sKd,
    S, NR: tl.constexpr,
):
    """MM1: compute S = Q * K^T for NR KV blocks and store into workspace_s.

    Handles resident-Q reload (first task of a tile) and the full L1/L0
    ping-pong DMA + MMA sequence for each KV block.
    """
    # wait workspace_s[ring_slot] task slot free (Vector released after Vec1)
    sync_block_wait("vector", "cube", SEM_S_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

    # reload resident Q at the first task of each output tile
    if task_in_tile == 0:
        q_bp = tl.make_block_ptr(
            Q + batch_idx * sQb + head_idx * sQh, (S, DIM), (sQs, sQd),
            (tile_row * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
        tile_copy(q_bp, q_l1, [CBM, CD])

    for nr in range(NR):
        kv_block_idx = task_in_tile * NR + nr
        l0_slot      = nr % 2

        k_bp = tl.make_block_ptr(
            K + batch_idx * sKb + kv_head_idx * sKh, (S, DIM), (sKs, sKd),
            (kv_block_idx * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
        tile_copy(k_bp, k_l1, [CBN, CD])

        tile_copy(q_l1, mm1_l0a, [CBM, CD])

        tile_copy(k_l1, mm1_l0b, [CBN, CD])  # NOTE: no transpose flag yet

        # attn_score = Q * K^T  (synchronous MMA stand-in for tile.cube_launch)
        attn_score = tl.dot(tile_to_tensor(mm1_l0a, writable=False),
                            tile_to_tensor(mm1_l0b, writable=False))

        score_store_bp = tl.make_block_ptr(
            workspace_s + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                           + ring_slot  * (NR * BLOCK_M * BLOCK_N)
                           + nr  * (BLOCK_M * BLOCK_N)),
            (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
        tl.store(score_store_bp, attn_score)

    # all NR S-blocks written -> notify Vec1
    sync_block_set("cube", "vector", SEM_S_READY, PIPE.PIPE_FIX, PIPE.PIPE_V)


@triton.jit
def _mm2_pv(
    # inputs
    V,
    v_l1, p_l1, mm2_l0a, mm2_l0b,
    workspace_p, workspace_pv,
    # task geometry
    cid, prev_task_in_tile, prev_batch_idx, kv_prev_head_idx,
    prev_ring_slot,
    # strides
    sKb, sKh, sKs, sKd,
    S, NR: tl.constexpr,
):
    """MM2: compute O_part = P * V for NR blocks and store into workspace_pv.

    Loads P from workspace_p (written by Vec1), loads V from GM, performs
    MMA and writes the partial output to workspace_pv for Vec2.
    """
    # wait workspace_p[prev_ring_slot] (P from Vec1) ready
    sync_block_wait("vector", "cube", SEM_P_READY, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    for nr in range(NR):
        prev_kv_block_idx = prev_task_in_tile * NR + nr
        prev_l0_slot      = nr % 2

        v_bp = tl.make_block_ptr(
            V + prev_batch_idx * sKb + kv_prev_head_idx * sKh, (S, DIM), (sKs, sKd),
            (prev_kv_block_idx * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
        tile_copy(v_bp, v_l1, [CBN, CD])

        prob_load_bp = tl.make_block_ptr(
            workspace_p + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                           + prev_ring_slot  * (NR * BLOCK_M * BLOCK_N)
                           + nr  * (BLOCK_M * BLOCK_N)),
            (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
        tile_copy(prob_load_bp, p_l1, [CBM, CBN])

        tile_copy(v_l1, mm2_l0b, [CBN, CD])

        tile_copy(p_l1, mm2_l0a, [CBM, CBN])

        # pv_part = P * V  (synchronous MMA stand-in for tile.cube_launch)
        pv_part = tl.dot(tile_to_tensor(mm2_l0a, writable=False),
                         tile_to_tensor(mm2_l0b, writable=False))

        pv_store_bp = tl.make_block_ptr(
            workspace_pv + (cid * (RING * NR * BLOCK_M * DIM)
                            + prev_ring_slot  * (NR * BLOCK_M * DIM)
                            + nr  * (BLOCK_M * DIM)),
            (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
        tl.store(pv_store_bp, pv_part)

    # all NR P*V blocks done -> notify Vec2; release workspace_p[prev_ring_slot]
    sync_block_set("cube", "vector", SEM_PV_READY, PIPE.PIPE_FIX, PIPE.PIPE_V)
    sync_block_set("cube", "vector", SEM_P_FREE,  PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)


@triton.jit
def _vec1_softmax(
    workspace_s, workspace_p, workspace_rescale, workspace_expsum,
    cid, vid, task_in_tile, ring_slot,
    neg_max_even, neg_max_odd,
    sm_scale,
    # causal mask inputs
    IS_CAUSAL: tl.constexpr,
    tile_start, tasks_per_tile, num_seq_blocks,
    g,
    NR: tl.constexpr,
):
    """Vec1: online softmax over NR score blocks -> workspace_p + rescale/expsum.

    Reads workspace_s[ring_slot], applies causal mask if needed, computes
    stabilised softmax probabilities, and stores P, rescale, and block_expsum
    to their respective GM ring-buffers for MM2 and Vec2.

    Returns updated (neg_max_even, neg_max_odd).
    """
    # wait workspace_s[ring_slot] (all NR score blocks) ready from MM1
    sync_block_wait("cube", "vector", SEM_S_READY, PIPE.PIPE_FIX, PIPE.PIPE_V)

    # reset running max at the first task of each output tile
    if task_in_tile == 0:
        neg_max_even = tl.full((HALF_M, 1), 2**30, tl.float32)
        neg_max_odd  = tl.full((HALF_M, 1), 2**30, tl.float32)

    for nr in range(NR):
        kv_block_idx = task_in_tile * NR + nr
        cur_parity   = kv_block_idx % 2
        prv_parity   = 1 - cur_parity

        # load score[vid*HALF_M:(vid+1)*HALF_M, :] from workspace_s GM -> UB
        score_load_bp = tl.make_block_ptr(
            workspace_s + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                           + ring_slot  * (NR * BLOCK_M * BLOCK_N)
                           + nr  * (BLOCK_M * BLOCK_N)
                           + vid * HALF_M * BLOCK_N),
            (HALF_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (HALF_M, BLOCK_N), (1, 0))
        attn_score_tile = tl.load(score_load_bp).to(tl.float32)

        if IS_CAUSAL:
            tile_seq_idx   = g // tasks_per_tile
            global_tile_id = tile_start + tile_seq_idx
            q_tile_row     = global_tile_id % num_seq_blocks
            q_row_idx      = q_tile_row * BLOCK_M + vid * HALF_M + tl.arange(0, HALF_M)
            kv_col_idx     = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            causal_mask    = q_row_idx[:, None] >= kv_col_idx[None, :]
            attn_score_tile = tl.where(causal_mask, attn_score_tile, float("-inf"))

        # online softmax: compute new running -max*scale (ping-pong)
        block_row_max = tl.max(attn_score_tile, axis=-1, keep_dims=True)
        neg_max_new = tl.minimum(-block_row_max * sm_scale,
                                 tl.where(cur_parity == 0, neg_max_even, neg_max_odd))
        neg_max_prv = tl.where(cur_parity == 0, neg_max_odd, neg_max_even)

        # softmax_p = exp(sm_scale * score + neg_max_new)
        softmax_p = tl.exp(sm_scale * attn_score_tile + neg_max_new)

        # rescale = exp(neg_max_new - neg_max_prv): correction factor for Vec2
        rescale = tl.exp(neg_max_new - neg_max_prv)
        # block_expsum: partial row-sum contributed by this KV block
        block_expsum = tl.sum(softmax_p, axis=-1, keep_dims=True)

        # store rescale and block_expsum into GM ring-buffers for Vec2
        # layout: [NUM_CORES, RING, NR, 2 sub-cores, HALF_M]
        rescale_offset = (cid * (RING * NR * 2 * HALF_M)
                          + ring_slot  * (NR * 2 * HALF_M)
                          + nr  * (2 * HALF_M)
                          + vid * HALF_M)
        rescale_store_bp = tl.make_block_ptr(
            workspace_rescale + rescale_offset,
            (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
        tl.store(rescale_store_bp, rescale)
        expsum_store_bp = tl.make_block_ptr(
            workspace_expsum + rescale_offset,
            (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
        tl.store(expsum_store_bp, block_expsum)

        # update running max ping-pong
        if cur_parity == 0:
            neg_max_even = neg_max_new
        else:
            neg_max_odd = neg_max_new

        # softmax_p -> workspace_p GM (MTE3): sub-core owns HALF_M rows
        prob_store_bp = tl.make_block_ptr(
            workspace_p + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                           + ring_slot  * (NR * BLOCK_M * BLOCK_N)
                           + nr  * (BLOCK_M * BLOCK_N)
                           + vid * HALF_M * BLOCK_N),
            (HALF_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (HALF_M, BLOCK_N), (1, 0))
        tl.store(prob_store_bp, softmax_p.to(workspace_p.dtype.element_ty))

    # all NR P-blocks written -> release workspace_s[ring_slot]; notify MM2
    sync_block_set("vector", "cube", SEM_S_FREE,  PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
    sync_block_set("vector", "cube", SEM_P_READY, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

    return neg_max_even, neg_max_odd


@triton.jit
def _vec2_accumulate(
    Out,
    workspace_pv, workspace_rescale, workspace_expsum,
    cid, vid,
    prev_task_in_tile, prev_tile_row, prev_batch_idx, prev_head_idx,
    prev_ring_slot,
    acc_o, softmax_denom,
    sOb, sOh, sOs, sOd,
    S, NR: tl.constexpr, NUM_ITERS: tl.constexpr,
):
    """Vec2: rescale acc_o with each new KV block's P*V; finalize on last block.

    Reads pv_acc from workspace_pv and (rescale, block_expsum) from
    workspace_rescale/expsum (both written by Vec1), accumulates into acc_o
    and softmax_denom, and writes the final output row on the last KV block.

    Returns updated (acc_o, softmax_denom).
    """
    # wait workspace_pv[prev_ring_slot] (all NR P*V blocks) ready from MM2
    sync_block_wait("cube", "vector", SEM_PV_READY, PIPE.PIPE_FIX, PIPE.PIPE_V)

    for nr in range(NR):
        prev_kv_block_idx = prev_task_in_tile * NR + nr

        # load pv_acc[vid*HALF_M:(vid+1)*HALF_M, :] from workspace_pv (MTE2)
        pv_load_bp = tl.make_block_ptr(
            workspace_pv + (cid * (RING * NR * BLOCK_M * DIM)
                            + prev_ring_slot  * (NR * BLOCK_M * DIM)
                            + nr  * (BLOCK_M * DIM)
                            + vid * HALF_M * DIM),
            (HALF_M, DIM), (DIM, 1), (0, 0), (HALF_M, DIM), (1, 0))
        pv_acc = tl.load(pv_load_bp).to(tl.float32)

        # load rescale and block_expsum written by Vec1 for this slot
        prev_rescale_offset = (cid * (RING * NR * 2 * HALF_M)
                           + prev_ring_slot  * (NR * 2 * HALF_M)
                           + nr  * (2 * HALF_M)
                           + vid * HALF_M)
        rescale_load_bp = tl.make_block_ptr(
            workspace_rescale + prev_rescale_offset,
            (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
        rescale = tl.load(rescale_load_bp).to(tl.float32)
        expsum_load_bp = tl.make_block_ptr(
            workspace_expsum + prev_rescale_offset,
            (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
        block_expsum = tl.load(expsum_load_bp).to(tl.float32)

        if prev_kv_block_idx == 0:
            # first KV block: init acc_o and softmax_denom directly
            acc_o         = pv_acc
            softmax_denom = block_expsum
        else:
            # rescale acc_o and accumulate
            rescale_bc    = tl.broadcast_to(rescale, (HALF_M, DIM))
            acc_o         = acc_o * rescale_bc + pv_acc
            softmax_denom = softmax_denom * rescale + block_expsum

        if prev_kv_block_idx == NUM_ITERS - 1:
            # last KV block: divide by softmax denominator and write output
            denom_bc    = tl.broadcast_to(softmax_denom, (HALF_M, DIM))
            output_tile = (acc_o / denom_bc).to(Out.dtype.element_ty)
            o_bp = tl.make_block_ptr(
                Out + prev_batch_idx * sOb + prev_head_idx * sOh, (S, DIM), (sOs, sOd),
                (prev_tile_row * BLOCK_M + vid * HALF_M, 0), (HALF_M, DIM), (1, 0))
            tl.store(o_bp, output_tile)

    # all NR blocks consumed -> release workspace_pv[prev_ring_slot]
    sync_block_set("vector", "cube", SEM_PV_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

    return acc_o, softmax_denom


# =============================================================================
#  The single-stream 3-task scheduler kernel (TileIR / tle.dsa form)
#
#  grid = (NUM_CORES,).  Each program drives one Cube + one Vector engine.
#  On-chip staging allocated with tile.alloc; DMA uses tile.copy; cross-engine
#  ordering uses intra_set/wait_flag (intra-core cross-pipe) + sync_block_set/wait (cross-core).
# =============================================================================
@triton.jit
def flash_attention_fwd_3task_kernel(
    Q, K, V, Out,
    workspace_s,    # [NUM_CORES, RING, NR, BLOCK_M, BLOCK_N]  fp16  S        (MM1 out)
    workspace_p,    # [NUM_CORES, RING, NR, BLOCK_M, BLOCK_N]  fp16  P        (Vec1 out)
    workspace_pv,   # [NUM_CORES, RING, NR, BLOCK_M, DIM]      fp16  P*V      (MM2 out)
    workspace_rescale, # [NUM_CORES, RING, NR, 2, HALF_M, 1]      fp32  exp(m_old-m_new) (Vec1->Vec2)
    workspace_expsum,  # [NUM_CORES, RING, NR, 2, HALF_M, 1]      fp32  sum(exp(s-m))    (Vec1->Vec2)
    sm_scale,
    B, Hq, Hkv, S,
    sQb, sQh, sQs, sQd,
    sKb, sKh, sKs, sKd,
    sOb, sOh, sOs, sOd,
    num_seq_blocks, heads_q, gqa_group,
    n_iters,           # KV blocks per output tile  (= seq_len // BLOCK_N)
    tasks_per_tile,               # tasks per output tile      (= n_iters // NR)
    tiles_per_core, extra_tiles,
    NR:        tl.constexpr,   # KV blocks per task (n_ratio)
    NUM_ITERS: tl.constexpr,   # = n_iters  (used in causal mask)
    IS_CAUSAL: tl.constexpr,
):
    cid = tl.program_id(0)
    vid = al.sub_vec_id()   # 0 or 1 — vector sub-core index, used in both scopes

    # ---- static task distribution  (== AICPU GetFASectionInfo metadata) ----
    tile_start          = cid * tiles_per_core + tl.where(cid < extra_tiles, cid, extra_tiles)
    tile_count          = tiles_per_core + tl.where(cid < extra_tiles, 1, 0)
    num_global_tasks  = tile_count * tasks_per_tile   # total pipelined tasks on this core

    # =========================================================================
    #  On-chip buffers  (tile.alloc -> explicit memory hierarchy)
    # =========================================================================
    # -- Cube side: L1 staging + L0 double-buffer (slot 0=MM1, slot 1=MM2) --
    q_l1 = tile_alloc([BLOCK_M, DIM],     Q.dtype.element_ty, L1)
    k_l1 = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L1)
    v_l1 = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L1)
    p_l1 = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L1)

    mm1_l0a = tile_alloc([BLOCK_M, DIM],     Q.dtype.element_ty, L0A)  # MM1 Q
    mm1_l0b = tile_alloc([DIM,     BLOCK_N], Q.dtype.element_ty, L0B)  # MM1 K
    mm1_l0c = tile_alloc([BLOCK_M, BLOCK_N], tl.float32,         L0C)  # MM1 out
    mm2_l0a = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L0A)  # MM2 P
    mm2_l0b = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L0B)  # MM2 V
    mm2_l0c = tile_alloc([BLOCK_M, DIM],     tl.float32,         L0C)  # MM2 out

    # -- Vector side (UB registers): full online-softmax state ---------------
    # acc_o uses HALF_M rows; state per (ring_slot, nr) stored as flat GM-backed
    # workspaces; running max uses a ping-pong register pair (one per kv parity).
    acc_o  = tl.zeros((HALF_M, DIM), tl.float32)
    softmax_denom = tl.zeros((HALF_M, 1), tl.float32)   # running denominator l
    # neg_max_even/odd: running -max*scale for even/odd kv index, reset each tile
    neg_max_even = tl.full((HALF_M, 1), 2**30, tl.float32)
    neg_max_odd  = tl.full((HALF_M, 1), 2**30, tl.float32)

    # =========================================================================
    #  CUBE scope: MM1(g) and MM2(g-1) dual-issued; each task = NR MMAs.
    # =========================================================================
    with al.scope(core_mode="cube"):
        # ---- init: 3 ws2-FREE tokens + pre-arm intra-core signals ----
        # RING ws2-FREE tokens (one per slot) so MM2 pipeline can start
        sync_block_set("cube", "vector", SEM_P_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("cube", "vector", SEM_P_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("cube", "vector", SEM_P_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

        for g in range(num_global_tasks + 1):

            # ===== MM1(g): S = Q*K^T for NR KV blocks -> workspace_s[cid, g%RING, :] =====
            if g < num_global_tasks:
                task_in_tile   = g % tasks_per_tile
                tile_idx       = g // tasks_per_tile
                output_tile_id = tile_start + tile_idx
                tile_row       = output_tile_id % num_seq_blocks
                head_idx       = (output_tile_id // num_seq_blocks) % heads_q
                batch_idx      = output_tile_id // (num_seq_blocks * heads_q)
                kv_head_idx    = head_idx // gqa_group
                ring_slot      = g % RING

                _mm1_qkt(
                    Q, K,
                    q_l1, k_l1, mm1_l0a, mm1_l0b,
                    workspace_s,
                    cid, task_in_tile, tile_row, batch_idx, head_idx, kv_head_idx,
                    ring_slot, tasks_per_tile,
                    sQb, sQh, sQs, sQd,
                    sKb, sKh, sKs, sKd,
                    S, NR,
                )

            # ===== MM2(g-1): O_part = P*V for NR blocks -> workspace_pv[cid,(g-1)%RING,:] =====
            if g >= 1:
                prev_g           = g - 1
                prev_task_in_tile    = prev_g % tasks_per_tile
                prev_tile_idx        = prev_g // tasks_per_tile
                prev_output_tile_id  = tile_start + prev_tile_idx
                prev_head_idx        = (prev_output_tile_id // num_seq_blocks) % heads_q
                prev_batch_idx       = prev_output_tile_id // (num_seq_blocks * heads_q)
                kv_prev_head_idx     = prev_head_idx // gqa_group
                prev_ring_slot       = prev_g % RING

                _mm2_pv(
                    V,
                    v_l1, p_l1, mm2_l0a, mm2_l0b,
                    workspace_p, workspace_pv,
                    cid, prev_task_in_tile, prev_batch_idx, kv_prev_head_idx,
                    prev_ring_slot,
                    sKb, sKh, sKs, sKd,
                    S, NR,
                )

        # ---- destroy: consume outstanding init-direction signals ----

    # =========================================================================
    #  VECTOR scope: Vec1(g) online-softmax, Vec2(g-1) rescale+accumulate.
    # =========================================================================
    with al.scope(core_mode="vector"):
        # ---- init: 3 ws1-FREE + 3 ws3-FREE tokens + pre-arm intra-core signals ----
        sync_block_set("vector", "cube", SEM_S_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_S_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_S_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_PV_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_PV_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_PV_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

        for g in range(num_global_tasks + 1):

            # ===== Vec1(g): softmax(workspace_s[g%RING]) -> workspace_p[g%RING] =====
            if g < num_global_tasks:
                task_in_tile = g % tasks_per_tile
                ring_slot    = g % RING

                neg_max_even, neg_max_odd = _vec1_softmax(
                    workspace_s, workspace_p, workspace_rescale, workspace_expsum,
                    cid, vid, task_in_tile, ring_slot,
                    neg_max_even, neg_max_odd,
                    sm_scale,
                    IS_CAUSAL, tile_start, tasks_per_tile, num_seq_blocks,
                    g, NR,
                )

            # ===== Vec2(g-1): acc_o = acc_o*rescale + pv_acc; finalize at last KV block =====
            if g >= 1:
                prev_g           = g - 1
                prev_task_in_tile    = prev_g % tasks_per_tile
                prev_tile_idx        = prev_g // tasks_per_tile
                prev_output_tile_id  = tile_start + prev_tile_idx
                prev_tile_row        = prev_output_tile_id % num_seq_blocks
                prev_head_idx        = (prev_output_tile_id // num_seq_blocks) % heads_q
                prev_batch_idx       = prev_output_tile_id // (num_seq_blocks * heads_q)
                prev_ring_slot       = prev_g % RING

                acc_o, softmax_denom = _vec2_accumulate(
                    Out,
                    workspace_pv, workspace_rescale, workspace_expsum,
                    cid, vid, prev_task_in_tile, prev_tile_row, prev_batch_idx, prev_head_idx,
                    prev_ring_slot,
                    acc_o, softmax_denom,
                    sOb, sOh, sOs, sOd,
                    S, NR, NUM_ITERS,
                )

        # ---- destroy: consume outstanding init-direction signals ----



# =============================================================================
#  Intermediate-TileIR dump (no device required)
# =============================================================================
class _DumpOptions:
    num_warps = 4
    num_stages = 1
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    allowed_dot_input_precisions = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default = 0
    default_dot_input_precision = "ieee"


def _dump_signature(nr):
    ptr = {"Q": "*fp16", "K": "*fp16", "V": "*fp16", "Out": "*fp16",
           "workspace_s": "*fp16", "workspace_p": "*fp16", "workspace_pv": "*fp16",
           "workspace_rescale": "*fp32", "workspace_expsum": "*fp32"}
    i32s = ["B", "Hq", "Hkv", "S",
            "sQb", "sQh", "sQs", "sQd",
            "sKb", "sKh", "sKs", "sKd",
            "sOb", "sOh", "sOs", "sOd",
            "num_seq_blocks", "heads_q", "gqa_group",
            "n_iters", "tasks_per_tile", "tiles_per_core", "extra_tiles"]
    sig = dict(ptr)
    sig["sm_scale"] = "fp32"
    for n in i32s:
        sig[n] = "i32"
    sig["NR"]        = "constexpr"
    sig["NUM_ITERS"] = "constexpr"
    sig["IS_CAUSAL"] = "constexpr"
    return sig


def dump_tileir(path=None, ttir_path=None, num_iters=32, is_causal=False):
    """Compile the kernel to TTIR (containing tile.* ops) and write it to `path`.

    Also runs the TileIR→HIVM pass to lower tile.* ops and dumps the resulting
    pure TTIR to `ttir_path`. Requires no NPU/GPU — pure front-end compilation.

    Returns the TileIR MLIR string. The TTIR dump is written as a side effect.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch.mlir")
    if ttir_path is None:
        ttir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch_ttir.mlir")

    nr = n_ratio
    tasks_per_tile = num_iters // nr
    signature = _dump_signature(nr)
    constants = {"NR": nr, "NUM_ITERS": num_iters, "IS_CAUSAL": is_causal}

    src = ASTSource(flash_attention_fwd_3task_kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(flash_attention_fwd_3task_kernel, src, context, _DumpOptions(), codegen_fns, {})

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_tileir: module.verify() failed -- IR is not legal")

    # ---- dump TileIR (TTIR + tile.* ops) ----
    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_tileir] module.verify() = {ok}; wrote legal TileIR to {path}")

    # ---- dump TTIR (TileIR→HIVM + TTIR optimization passes) ----
    from triton._C.libtriton import passes as ir_passes
    pm = ir.pass_manager(context)
    pm.enable_debug()

    # Phase 1: TileIR→HIVM — lower tile.* ops, producing pure TTIR/HIVM IR.
    try:
        from triton._C.libtriton.ascend import passes as ascend_passes
        ascend_passes.ttir.add_tileir_to_hivm(pm)
    except Exception:
        pass
    pm.run(module)
    print(f"[dump_tileir] after TileIR→HIVM: verify={module.verify()}", flush=True)

    # Phase 2: TTIR optimization passes (mirrors compiler.py make_ttir).
    pm2 = ir.pass_manager(context)
    pm2.enable_debug()
    ir_passes.common.add_inliner(pm2)
    ir_passes.ttir.add_combine(pm2)
    ir_passes.common.add_canonicalizer(pm2)
    ir_passes.ttir.add_reorder_broadcast(pm2)
    ir_passes.common.add_cse(pm2)
    ir_passes.common.add_licm(pm2)
    ir_passes.common.add_symbol_dce(pm2)
    ir_passes.ttir.add_loop_unroll(pm2)
    pm2.run(module)
    print(f"[dump_tileir] after TTIR opt passes: verify={module.verify()}", flush=True)

    ttir_ok = module.verify()
    if not ttir_ok:
        print(f"[dump_tileir] WARNING: module.verify() failed after TTIR optimization — IR may be illegal")
    else:
        print(f"[dump_tileir] TTIR optimization complete: verify={ttir_ok}")

    ttir_mlir = str(module)
    with open(ttir_path, "w") as f:
        f.write(ttir_mlir)
    print(f"[dump_tileir] wrote optimized TTIR to {ttir_path}")

    return mlir


def dump_hivm(path=None, num_iters=32, is_causal=False):
    """Compile the kernel to TTIR, then lower through TileIR→HIVM pipeline to HIVM IR.

    Pipeline (matches compiler.py ttir_to_linalg):
      1. add_triton_to_structure_incubated
      2. add_discrete_mask_access_conversion
      3. add_triton_to_unstructure_incubated
      4. add_triton_to_hivm          (Triton CustomOp → HIVM SyncOp)
      5. add_triton_to_hfusion       (Triton → HFusion)
      6. add_tileir_to_hivm          (TileIR → HIVM)          ← our pass
      7. add_triton_to_llvm          (Triton → LLVM)
      8. add_bubble_up_operation
      9. add_triton_to_structure_incubated (second round)
     10. add_triton_to_linalg_incubated    (TTIR compute → Linalg)

    Returns the MLIR string. Requires no NPU/GPU — pure front-end + pass pipeline.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR (TileIR)
    tileir_mlir = dump_tileir(path=None, num_iters=num_iters, is_causal=is_causal)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_triton_arch_hivm.mlir")

    # Step 2: parse the TileIR module and run the pass pipeline
    context = ir.context()
    ir.load_dialects(context)
    from triton._C.libtriton import tle as tle_ir
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    # Write TileIR to temp file and parse it back into a module
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(tileir_mlir)
        tmp_path = f.name
    module = ir.parse_mlir_module(tmp_path, context)
    os.unlink(tmp_path)

    # Phase 1: TileIR → HIVM (first pass) — converts alloc/to_tensor/buf-to-buf copy/sync.
    #   tile.copy with !tt.ptr source is skipped (needs ptr→memref first).
    pm1 = ir.pass_manager(context)
    pm1.enable_debug()
    ascend.passes.ttir.add_tileir_to_hivm(pm1)
    pm1.run(module)
    print(f"[dump_hivm] after tileir (pass 1): verify={module.verify()}", flush=True)

    # Phase 2: Triton CustomOp → HIVM SyncOp
    pm2 = ir.pass_manager(context)
    pm2.enable_debug()
    ascend.passes.ttir.add_triton_to_hivm(pm2)
    pm2.run(module)
    print(f"[dump_hivm] after triton_to_hivm: verify={module.verify()}", flush=True)

    # Phase 3: Inline + canonicalize to clean up the IR.
    pm3 = ir.pass_manager(context)
    pm3.enable_debug()
    passes.common.add_inliner(pm3)
    passes.common.add_canonicalizer(pm3)
    pm3.run(module)
    print(f"[dump_hivm] after canonicalize: verify={module.verify()}", flush=True)

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_hivm: module.verify() failed after pipeline — IR is not legal")

    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_hivm] module.verify() = {ok}; wrote HIVM IR to {path}")
    return mlir


# =============================================================================
#  Host launcher
# =============================================================================
def flash_attention_fwd(q, k, v, is_causal=False, n_ratio=8):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and S % BLOCK_N == 0 and Hq % Hkv == 0

    num_seq_blocks = S // BLOCK_M
    block_num      = num_seq_blocks * Hq * B
    n_iters        = S // BLOCK_N   # KV blocks per output tile

    NR = n_ratio
    if n_iters < NR:
        NR = n_iters
    assert n_iters % NR == 0, f"n_iters ({n_iters}) must be divisible by n_ratio ({NR})"
    tasks_per_tile = n_iters // NR   # tasks per output tile

    tiles_per_core = block_num // NUM_CORES
    extra_tiles = block_num % NUM_CORES

    out = torch.empty_like(q)
    # GM workspaces
    workspace_s = torch.empty((NUM_CORES, RING, NR, BLOCK_M, BLOCK_N),
                              dtype=q.dtype, device=q.device)   # S
    workspace_p = torch.empty((NUM_CORES, RING, NR, BLOCK_M, BLOCK_N),
                              dtype=q.dtype, device=q.device)   # P
    workspace_pv = torch.empty((NUM_CORES, RING, NR, BLOCK_M, DIM),
                              dtype=q.dtype, device=q.device)   # P*V
    # [NUM_CORES, RING, NR, 2 sub-cores, HALF_M, 1] — written by Vec1, read by Vec2
    workspace_rescale = torch.empty((NUM_CORES, RING, NR, 2, HALF_M, 1),
                              dtype=torch.float32, device=q.device)  # exp(m_old - m_new)
    workspace_expsum = torch.empty((NUM_CORES, RING, NR, 2, HALF_M, 1),
                              dtype=torch.float32, device=q.device)  # sum(exp(s - m))
    sm_scale = (1.0 / D) ** 0.5

    grid = (NUM_CORES,)
    flash_attention_fwd_3task_kernel[grid](
        q, k, v, out,
        workspace_s, workspace_p, workspace_pv, workspace_rescale, workspace_expsum,
        sm_scale,
        B, Hq, Hkv, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_seq_blocks, Hq, Hq // Hkv,
        n_iters, tasks_per_tile, tiles_per_core, extra_tiles,
        NR=NR,
        NUM_ITERS=n_iters,
        IS_CAUSAL=is_causal,
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--n-ratio", type=int, default=8,
                        help="KV blocks per task (arch22 nRatio)")
    parser.add_argument("--dump-mlir", nargs="?", const="", default=None,
                        help="Dump intermediate TileIR to PATH (default skill/op/fa_triton_arch.mlir) and exit; no device needed.")
    parser.add_argument("--dump-ir", nargs="?", const="", default=None,
                        help="Dump HIVM IR (after TileIR→HIVM lowering) to PATH and exit; no device needed.")
    args = parser.parse_args()

    if args.dump_mlir is not None:
        B, S, H, D = args.B, args.S, args.H, args.D
        n_iters = S // BLOCK_N
        dump_tileir(path=(args.dump_mlir or None), num_iters=n_iters,
                    n_ratio=args.n_ratio, is_causal=args.causal)
        raise SystemExit(0)

    # ---- dump HIVM IR after full lowering pipeline (no device required) ----
    if args.dump_ir is not None:
        B, S, H, D = args.B, args.S, args.H, args.D
        n_iters = S // BLOCK_N
        dump_hivm(path=(args.dump_ir or None), num_iters=n_iters, is_causal=args.causal)
        raise SystemExit(0)

    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    q = torch.randn((B, Q_H, S, D), dtype=torch.float16, device=device)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16, device=device)

    out = flash_attention_fwd(q, k, v, is_causal=args.causal, n_ratio=args.n_ratio)

    if not args.no_check:
        def ref(q, k, v):
            if k.shape[1] != q.shape[1]:
                gqa_rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(gqa_rep, dim=1)
                v = v.repeat_interleave(gqa_rep, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=args.causal).to(torch.float16)

        torch.testing.assert_close(ref(q, k, v), out, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
