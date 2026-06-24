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
SEM_WS1_READY = 0   # C->V : ws1 (S)   has data
SEM_WS1_FREE  = 1   # V->C : ws1 slot  free
SEM_WS2_READY = 2   # V->C : ws2 (P)   has data
SEM_WS2_FREE  = 3   # C->V : ws2 slot  free
SEM_WS3_READY = 4   # C->V : ws3 (P*V) has data
SEM_WS3_FREE  = 5   # V->C : ws3 slot  free


# =============================================================================
#  The single-stream 3-task scheduler kernel (TileIR / tle.dsa form)
#
#  grid = (NUM_CORES,).  Each program drives one Cube + one Vector engine.
#  On-chip staging allocated with tile.alloc; DMA uses tile.copy; cross-engine
#  ordering uses tile_pipe_barrier + sync_block_set/wait.
# =============================================================================
@triton.jit
def flash_attention_fwd_3task_kernel(
    Q, K, V, Out,
    workspace_1,   # [NUM_CORES, RING, NR, BLOCK_M, BLOCK_N]  fp16   S
    workspace_2,   # [NUM_CORES, RING, NR, BLOCK_M, BLOCK_N]  fp16   P
    workspace_3,   # [NUM_CORES, RING, NR, BLOCK_M, DIM]      fp16   P*V
    workspace_4,   # [NUM_CORES, RING, NR, 2, HALF_M, 1]      fp32   r_fac (Vec1->Vec2)
    workspace_5,   # [NUM_CORES, RING, NR, 2, HALF_M, 1]      fp32   row_sum (Vec1->Vec2)
    sm_scale,
    B, Hq, Hkv, S,
    sQb, sQh, sQs, sQd,
    sKb, sKh, sKs, sKd,
    sOb, sOh, sOs, sOd,
    num_seq_blocks, heads_q, gqa_group,
    n_iters,           # KV blocks per output tile  (= seq_len // BLOCK_N)
    tpt,               # tasks per output tile      (= n_iters // NR)
    q_tasks, r_tasks,
    NR:        tl.constexpr,   # KV blocks per task (n_ratio)
    NUM_ITERS: tl.constexpr,   # = n_iters  (used in causal mask)
    IS_CAUSAL: tl.constexpr,
):
    cid = tl.program_id(0)
    vid = al.sub_vec_id()   # 0 or 1 — vector sub-core index, used in both scopes

    # ---- static task distribution  (== AICPU GetFASectionInfo metadata) ----
    my_start = cid * q_tasks + tl.where(cid < r_tasks, cid, r_tasks)
    my_count = q_tasks + tl.where(cid < r_tasks, 1, 0)
    GT = my_count * tpt   # total global tasks on this core

    # =========================================================================
    #  On-chip buffers  (tile.alloc -> explicit memory hierarchy)
    # =========================================================================
    # -- Cube side: L1 staging + L0 double-buffer (slot 0=MM1, slot 1=MM2) --
    q_l1 = tile_alloc([BLOCK_M, DIM],     Q.dtype.element_ty, L1)
    k_l1 = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L1)
    v_l1 = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L1)
    p_l1 = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L1)

    l0a0 = tile_alloc([BLOCK_M, DIM],     Q.dtype.element_ty, L0A)  # MM1 Q
    l0b0 = tile_alloc([DIM,     BLOCK_N], Q.dtype.element_ty, L0B)  # MM1 K
    l0c0 = tile_alloc([BLOCK_M, BLOCK_N], tl.float32,         L0C)  # MM1 out
    l0a1 = tile_alloc([BLOCK_M, BLOCK_N], Q.dtype.element_ty, L0A)  # MM2 P
    l0b1 = tile_alloc([BLOCK_N, DIM],     Q.dtype.element_ty, L0B)  # MM2 V
    l0c1 = tile_alloc([BLOCK_M, DIM],     tl.float32,         L0C)  # MM2 out

    # -- Vector side (UB registers): full online-softmax state ---------------
    # acc_o uses HALF_M rows; state per (ring_slot, nr) stored as flat GM-backed
    # workspaces; running max uses a ping-pong register pair (one per kv parity).
    acc_o  = tl.zeros((HALF_M, DIM), tl.float32)
    sumexp = tl.zeros((HALF_M, 1),   tl.float32)  # running denominator l
    # neg_sm[0/1]: running -m*scale for even/odd kv, reset each tile
    neg_sm_0 = tl.full((HALF_M, 1), 2**30, tl.float32)
    neg_sm_1 = tl.full((HALF_M, 1), 2**30, tl.float32)

    # =========================================================================
    #  CUBE scope: MM1(g) and MM2(g-1) dual-issued; each task = NR MMAs.
    # =========================================================================
    with al.scope(core_mode="cube"):
        # ---- init: 3 ws2-FREE tokens + pre-arm intra-core signals ----
        # RING ws2-FREE tokens (one per slot) so MM2 pipeline can start
        sync_block_set("cube", "vector", SEM_WS2_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("cube", "vector", SEM_WS2_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("cube", "vector", SEM_WS2_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        tile_pipe_barrier(PIPE_MTE1)   # SIG_K_L1 free
        tile_pipe_barrier(PIPE_MTE1)   # SIG_P_L1 free
        tile_pipe_barrier(PIPE_MTE1)   # SIG_V_L1 free
        tile_pipe_barrier(PIPE_M)      # SIG_L0AB slot 0 free
        tile_pipe_barrier(PIPE_M)      # SIG_L0AB slot 1 free
        tile_pipe_barrier(PIPE_FIX)    # SIG_L0C  slot 0 free
        tile_pipe_barrier(PIPE_FIX)    # SIG_L0C  slot 1 free
        tile_pipe_barrier(PIPE_MTE1)   # SIG_Q free

        for g in range(GT + 1):

            # ===== MM1(g): S = Q*K^T for NR KV blocks -> workspace_1[cid, g%RING, :] =====
            if g < GT:
                tit     = g % tpt
                tl_idx  = g // tpt
                task_id = my_start + tl_idx
                bx      = task_id % num_seq_blocks
                by      = (task_id // num_seq_blocks) % heads_q
                bz      = task_id // (num_seq_blocks * heads_q)
                kv_by   = by // gqa_group
                r1      = g % RING

                # wait ws1[r1] task slot free (Vector released after Vec1)
                sync_block_wait("vector", "cube", SEM_WS1_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

                # reload resident Q at the first task of each output tile
                if tit == 0:
                    tile_pipe_barrier(PIPE_MTE1)   # wait SIG_Q
                    q_bp = tl.make_block_ptr(
                        Q + bz * sQb + by * sQh, (S, DIM), (sQs, sQd),
                        (bx * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
                    tile_copy(q_bp, q_l1, [CBM, CD])
                    tile_pipe_barrier(PIPE_MTE2)   # set SIG_Q  (Q in L1)
                    tile_pipe_barrier(PIPE_MTE2)   # wait SIG_Q ack

                for nr in range(NR):
                    kv = tit * NR + nr
                    s1 = nr % 2

                    tile_pipe_barrier(PIPE_MTE1)   # wait SIG_K_L1 free
                    k_bp = tl.make_block_ptr(
                        K + bz * sKb + kv_by * sKh, (S, DIM), (sKs, sKd),
                        (kv * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
                    tile_copy(k_bp, k_l1, [CBN, CD])
                    tile_pipe_barrier(PIPE_MTE2)   # set SIG_K_L1

                    tile_pipe_barrier(PIPE_M)      # wait SIG_L0AB+s1
                    tile_copy(q_l1, l0a0, [CBM, CD])

                    tile_pipe_barrier(PIPE_MTE2)   # wait SIG_K_L1 ack
                    tile_copy(k_l1, l0b0, [CBN, CD])  # NOTE: no transpose flag yet
                    tile_pipe_barrier(PIPE_MTE1)   # set SIG_K_L1 free
                    tile_pipe_barrier(PIPE_MTE1)   # set SIG_L0AB+s1

                    tile_pipe_barrier(PIPE_MTE1)   # wait SIG_L0AB+s1
                    tile_pipe_barrier(PIPE_FIX)    # wait SIG_L0C+s1
                    # S = Q*K^T  (synchronous MMA stand-in for tile.cube_launch)
                    s_mat = tl.dot(tile_to_tensor(l0a0, writable=False),
                                   tile_to_tensor(l0b0, writable=False))
                    tile_pipe_barrier(PIPE_M)      # set SIG_L0AB+s1 free
                    tile_pipe_barrier(PIPE_M)      # set SIG_L0C+s1

                    tile_pipe_barrier(PIPE_M)      # wait SIG_L0C+s1
                    ws1_bp = tl.make_block_ptr(
                        workspace_1 + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                                       + r1  * (NR * BLOCK_M * BLOCK_N)
                                       + nr  * (BLOCK_M * BLOCK_N)),
                        (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                    tl.store(ws1_bp, s_mat)
                    tile_pipe_barrier(PIPE_FIX)    # set SIG_L0C+s1 free

                # all NR S-blocks written -> notify Vec1
                sync_block_set("cube", "vector", SEM_WS1_READY, PIPE.PIPE_FIX, PIPE.PIPE_V)

                # release resident Q at the last task of the tile
                if tit == tpt - 1:
                    tile_pipe_barrier(PIPE_MTE1)   # set SIG_Q free

            # ===== MM2(g-1): O_part = P*V for NR blocks -> workspace_3[cid,(g-1)%RING,:] =====
            if g >= 1:
                gm       = g - 1
                tit2     = gm % tpt
                tl_idx2  = gm // tpt
                task_id2 = my_start + tl_idx2
                by2      = (task_id2 // num_seq_blocks) % heads_q
                bz2      = task_id2 // (num_seq_blocks * heads_q)
                kv_by2   = by2 // gqa_group
                r2       = gm % RING

                # wait ws2[r2] (P from Vec1) ready
                sync_block_wait("vector", "cube", SEM_WS2_READY, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

                for nr in range(NR):
                    kv2 = tit2 * NR + nr
                    s2  = nr % 2

                    tile_pipe_barrier(PIPE_MTE1)   # wait SIG_V_L1 free
                    v_bp = tl.make_block_ptr(
                        V + bz2 * sKb + kv_by2 * sKh, (S, DIM), (sKs, sKd),
                        (kv2 * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
                    tile_copy(v_bp, v_l1, [CBN, CD])
                    tile_pipe_barrier(PIPE_MTE2)   # set SIG_V_L1

                    tile_pipe_barrier(PIPE_MTE1)   # wait SIG_P_L1 free
                    ws2_bp = tl.make_block_ptr(
                        workspace_2 + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                                       + r2  * (NR * BLOCK_M * BLOCK_N)
                                       + nr  * (BLOCK_M * BLOCK_N)),
                        (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                    tile_copy(ws2_bp, p_l1, [CBM, CBN])
                    tile_pipe_barrier(PIPE_MTE2)   # set SIG_P_L1

                    tile_pipe_barrier(PIPE_MTE2)   # wait SIG_V_L1 ack
                    tile_pipe_barrier(PIPE_M)      # wait SIG_L0AB+s2
                    tile_copy(v_l1, l0b1, [CBN, CD])
                    tile_pipe_barrier(PIPE_MTE1)   # set SIG_V_L1 free

                    tile_pipe_barrier(PIPE_MTE2)   # wait SIG_P_L1 ack
                    tile_copy(p_l1, l0a1, [CBM, CBN])
                    tile_pipe_barrier(PIPE_MTE1)   # set SIG_P_L1 free
                    tile_pipe_barrier(PIPE_MTE1)   # set SIG_L0AB+s2

                    tile_pipe_barrier(PIPE_MTE1)   # wait SIG_L0AB+s2
                    tile_pipe_barrier(PIPE_FIX)    # wait SIG_L0C+s2
                    # O_part = P*V  (synchronous MMA stand-in)
                    o_mat = tl.dot(tile_to_tensor(l0a1, writable=False),
                                   tile_to_tensor(l0b1, writable=False))
                    tile_pipe_barrier(PIPE_M)      # set SIG_L0AB+s2 free
                    tile_pipe_barrier(PIPE_M)      # set SIG_L0C+s2

                    tile_pipe_barrier(PIPE_M)      # wait SIG_L0C+s2
                    ws3_bp = tl.make_block_ptr(
                        workspace_3 + (cid * (RING * NR * BLOCK_M * DIM)
                                       + r2  * (NR * BLOCK_M * DIM)
                                       + nr  * (BLOCK_M * DIM)),
                        (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
                    tl.store(ws3_bp, o_mat)
                    tile_pipe_barrier(PIPE_FIX)    # set SIG_L0C+s2 free

                # all NR P*V blocks done -> notify Vec2; release ws2[r2]
                sync_block_set("cube", "vector", SEM_WS3_READY, PIPE.PIPE_FIX, PIPE.PIPE_V)
                sync_block_set("cube", "vector", SEM_WS2_FREE,  PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

        # ---- destroy: consume outstanding init-direction signals ----
        tile_pipe_barrier(PIPE_MTE1)   # SIG_K_L1
        tile_pipe_barrier(PIPE_MTE1)   # SIG_P_L1
        tile_pipe_barrier(PIPE_MTE1)   # SIG_V_L1
        tile_pipe_barrier(PIPE_M)      # SIG_L0AB slot 0
        tile_pipe_barrier(PIPE_M)      # SIG_L0AB slot 1
        tile_pipe_barrier(PIPE_FIX)    # SIG_L0C  slot 0
        tile_pipe_barrier(PIPE_FIX)    # SIG_L0C  slot 1
        tile_pipe_barrier(PIPE_MTE1)   # SIG_Q

    # =========================================================================
    #  VECTOR scope: Vec1(g) online-softmax, Vec2(g-1) rescale+accumulate.
    # =========================================================================
    with al.scope(core_mode="vector"):
        # ---- init: 3 ws1-FREE + 3 ws3-FREE tokens + pre-arm intra-core signals ----
        sync_block_set("vector", "cube", SEM_WS1_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_WS1_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_WS1_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_WS3_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_WS3_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        sync_block_set("vector", "cube", SEM_WS3_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
        tile_pipe_barrier(PIPE_V)      # SIG_IO_UB free
        tile_pipe_barrier(PIPE_MTE3)   # SIG_S_HALF free

        for g in range(GT + 1):

            q_bp = tl.make_block_ptr(Q + bz * sQb + by * sQh, (S, DIM), (sQs, sQd),
                                     (bx * BLOCK_M, 0), (BLOCK_M, DIM), (1, 0))
            if j == 0:                              # new output tile: (re)load Q into L1
                tile_copy(tensor_to_tile(q_bp), q_l1, [CBM, CD])
            # K[j] -> k_l1 -> L0 slot 0
            k_bp = tl.make_block_ptr(K + bz * sKb + kv_by * sKh, (S, DIM), (sKs, sKd),
                                     (j * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
            tile_copy(tensor_to_tile(k_bp), k_l1, [CBN, CD])
            tile_copy(q_l1, l0a0, [CBM, CD])
            tile_copy(k_l1, l0b0, [CBN, CD])   # NOTE: tile.copy has no transpose flag yet
            # S = Q·Kᵀ : matmul stand-in for tile.cube_launch (no DSA binding yet).
            # NOTE: tt.dot requires standard ranked tensors — a !tile.tensor (from
            # tile.to_tensor) is rejected by the verifier — so the dot runs on tt
            # tensors loaded from GM, not on the !tile.* staging buffers above.
            m1_bp = tl.make_block_ptr(mm1Res + cid * (PP * BLOCK_M * BLOCK_N) + cur_pp * (BLOCK_M * BLOCK_N),
                                      (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
            s = tl.dot(tile_to_tensor(l0a0, writable=False), tile_to_tensor(l0b0, writable=False))                           
            # s = tl.dot(tl.load(q_bp), tl.trans(tl.load(k_bp)))
            tl.store(m1_bp, s)

                # wait ws1[r1] (all NR S-blocks) ready from MM1
                sync_block_wait("cube", "vector", SEM_WS1_READY, PIPE.PIPE_V, PIPE.PIPE_V)

                # reset running max at the first task of each output tile
                if tit == 0:
                    neg_sm_0 = tl.full((HALF_M, 1), 2**30, tl.float32)
                    neg_sm_1 = tl.full((HALF_M, 1), 2**30, tl.float32)

                for nr in range(NR):
                    kv  = tit * NR + nr
                    cur = kv % 2
                    prv = 1 - cur

                    # load S[vid*HALF_M:(vid+1)*HALF_M, :] from ws1 GM -> UB (MTE2)
                    tile_pipe_barrier(PIPE_V)      # wait SIG_IO_UB free
                    ws1r_bp = tl.make_block_ptr(
                        workspace_1 + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                                       + r1  * (NR * BLOCK_M * BLOCK_N)
                                       + nr  * (BLOCK_M * BLOCK_N)
                                       + vid * HALF_M * BLOCK_N),
                        (HALF_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (HALF_M, BLOCK_N), (1, 0))
                    s_tile = tl.load(ws1r_bp).to(tl.float32)
                    tile_pipe_barrier(PIPE_MTE2)   # set SIG_IO_UB
                    tile_pipe_barrier(PIPE_MTE2)   # wait SIG_IO_UB ack
                    tile_pipe_barrier(PIPE_V)      # set SIG_IO_UB free

                    if IS_CAUSAL:
                        t1  = g // tpt
                        tid = my_start + t1
                        bx1 = tid % num_seq_blocks
                        q_idx  = bx1 * BLOCK_M + vid * HALF_M + tl.arange(0, HALF_M)
                        kv_idx = kv * BLOCK_N + tl.arange(0, BLOCK_N)
                        valid  = q_idx[:, None] >= kv_idx[None, :]
                        s_tile = tl.where(valid, s_tile, float("-inf"))

                    # online softmax: compute new running -m*scale (ping-pong)
                    row_max = tl.max(s_tile, axis=-1, keep_dims=True)
                    neg_m_new = tl.minimum(-row_max * sm_scale,
                                           tl.where(cur == 0, neg_sm_0, neg_sm_1))
                    neg_m_prv = tl.where(cur == 0, neg_sm_1, neg_sm_0)

                    # P = exp(sm_scale * S + neg_m_new)
                    p_tile = tl.exp(sm_scale * s_tile + neg_m_new)

                # P from stage1Res[pp1] -> p_l1 -> L0 slot 1
                sync_block_wait(EVT_MTE3_MTE2[0], EVT_MTE3_MTE2[1], EVT_MTE3_MTE2[2], EVT_MTE3_MTE2[3], EVT_MTE3_MTE2[4])
                s1r_bp = tl.make_block_ptr(stage1Res + cid * (PP * BLOCK_M * BLOCK_N) + pp1 * (BLOCK_M * BLOCK_N),
                                           (BLOCK_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (BLOCK_M, BLOCK_N), (1, 0))
                tile_copy(tensor_to_tile(s1r_bp), p_l1, [CBM, CBN])
                # V[j1] -> v_l1 -> L0 slot 1
                v_bp = tl.make_block_ptr(V + bz1 * sKb + kv_by1 * sKh, (S, DIM), (sKs, sKd),
                                         (j1 * BLOCK_N, 0), (BLOCK_N, DIM), (1, 0))
                tile_copy(tensor_to_tile(v_bp), v_l1, [CBN, CD])
                tile_copy(p_l1, l0a1, [CBM, CBN])
                tile_copy(v_l1, l0b1, [CBN, CD])
                # O_part = P·V : mma -> l0c1 -> FIX -> mm2Res[pp1]  (synchronous stand-in).
                m2_bp = tl.make_block_ptr(mm2Res + cid * (PP * BLOCK_M * DIM) + pp1 * (BLOCK_M * DIM),
                                          (BLOCK_M, DIM), (DIM, 1), (0, 0), (BLOCK_M, DIM), (1, 0))
                o = tl.dot(tile_to_tensor(l0a1, writable=False), tile_to_tensor(l0b1, writable=False))
                tl.store(m2_bp, o)

                    # store r_fac and row_sum into GM ring-buffers for Vec2
                    # layout: [NUM_CORES, RING, NR, 2, HALF_M, 1]  (dim-3: vid 0/1)
                    ws_rfac_base = (cid * (RING * NR * 2 * HALF_M)
                                    + r1  * (NR * 2 * HALF_M)
                                    + nr  * (2 * HALF_M)
                                    + vid * HALF_M)
                    rfac_bp = tl.make_block_ptr(
                        workspace_4 + ws_rfac_base,
                        (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
                    tl.store(rfac_bp, r_fac)
                    rsum_bp = tl.make_block_ptr(
                        workspace_5 + ws_rfac_base,
                        (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
                    tl.store(rsum_bp, row_sum)

                    # update running max ping-pong
                    if cur == 0:
                        neg_sm_0 = neg_m_new
                    else:
                        neg_sm_1 = neg_m_new

                    # P -> ws2 GM (MTE3): sub-core owns HALF_M rows
                    tile_pipe_barrier(PIPE_MTE3)   # wait SIG_S_HALF free
                    ws2w_bp = tl.make_block_ptr(
                        workspace_2 + (cid * (RING * NR * BLOCK_M * BLOCK_N)
                                       + r1  * (NR * BLOCK_M * BLOCK_N)
                                       + nr  * (BLOCK_M * BLOCK_N)
                                       + vid * HALF_M * BLOCK_N),
                        (HALF_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (HALF_M, BLOCK_N), (1, 0))
                    tl.store(ws2w_bp, p_tile.to(workspace_2.dtype.element_ty))
                    tile_pipe_barrier(PIPE_V)      # set SIG_S_HALF
                    tile_pipe_barrier(PIPE_V)      # wait SIG_S_HALF ack
                    tile_pipe_barrier(PIPE_MTE3)   # set SIG_S_HALF free

                # all NR P-blocks written -> release ws1[r1]; notify MM2 ws2[r1] ready
                sync_block_set("vector", "cube", SEM_WS1_FREE,  PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)
                sync_block_set("vector", "cube", SEM_WS2_READY, PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)

            # ===== Vec2(g-1): acc_o = acc_o*r + P*V; finalize at last KV block =====
            if g >= 1:
                gm   = g - 1
                tit2 = gm % tpt
                tl_idx2  = gm // tpt
                task_id2 = my_start + tl_idx2
                bx2  = task_id2 % num_seq_blocks
                by2  = (task_id2 // num_seq_blocks) % heads_q
                bz2  = task_id2 // (num_seq_blocks * heads_q)
                r2   = gm % RING

                # wait ws3[r2] (all NR P*V blocks) ready from MM2
                sync_block_wait("cube", "vector", SEM_WS3_READY, PIPE.PIPE_V, PIPE.PIPE_V)

                for nr in range(NR):
                    kv2 = tit2 * NR + nr

                    # load P*V[vid*HALF_M:(vid+1)*HALF_M, :] from ws3 (MTE2)
                    tile_pipe_barrier(PIPE_V)      # wait SIG_IO_UB free
                    ws3r_bp = tl.make_block_ptr(
                        workspace_3 + (cid * (RING * NR * BLOCK_M * DIM)
                                       + r2  * (NR * BLOCK_M * DIM)
                                       + nr  * (BLOCK_M * DIM)
                                       + vid * HALF_M * DIM),
                        (HALF_M, DIM), (DIM, 1), (0, 0), (HALF_M, DIM), (1, 0))
                    pv_tile = tl.load(ws3r_bp).to(tl.float32)
                    tile_pipe_barrier(PIPE_MTE2)   # set SIG_IO_UB
                    tile_pipe_barrier(PIPE_MTE2)   # wait SIG_IO_UB ack
                    tile_pipe_barrier(PIPE_V)      # set SIG_IO_UB free

                    # load r_fac and row_sum written by Vec1 for this (r2, nr, vid) slot
                    ws_rfac_base2 = (cid * (RING * NR * 2 * HALF_M)
                                     + r2  * (NR * 2 * HALF_M)
                                     + nr  * (2 * HALF_M)
                                     + vid * HALF_M)
                    rfac_r_bp = tl.make_block_ptr(
                        workspace_4 + ws_rfac_base2,
                        (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
                    r_fac = tl.load(rfac_r_bp).to(tl.float32)
                    rsum_r_bp = tl.make_block_ptr(
                        workspace_5 + ws_rfac_base2,
                        (HALF_M, 1), (1, 1), (0, 0), (HALF_M, 1), (1, 0))
                    row_sum = tl.load(rsum_r_bp).to(tl.float32)

                    if kv2 == 0:
                        # first KV block: init acc_o and sumexp directly
                        acc_o  = pv_tile
                        sumexp = row_sum
                    else:
                        # rescale acc_o and accumulate
                        r_fac_bc = tl.broadcast_to(r_fac, (HALF_M, DIM))
                        acc_o    = acc_o * r_fac_bc + pv_tile
                        sumexp   = sumexp * r_fac + row_sum

                    if kv2 == NUM_ITERS - 1:
                        # last KV block: divide by l and write Output
                        l_bc    = tl.broadcast_to(sumexp, (HALF_M, DIM))
                        out_tile = (acc_o / l_bc).to(Out.dtype.element_ty)
                        o_bp = tl.make_block_ptr(
                            Out + bz2 * sOb + by2 * sOh, (S, DIM), (sOs, sOd),
                            (bx2 * BLOCK_M + vid * HALF_M, 0), (HALF_M, DIM), (1, 0))
                        tile_pipe_barrier(PIPE_V)      # wait SIG_IO_UB free (MTE3)
                        tl.store(o_bp, out_tile)
                        tile_pipe_barrier(PIPE_MTE3)   # set done

                # all NR blocks consumed -> release ws3[r2]
                sync_block_set("vector", "cube", SEM_WS3_FREE, PIPE.PIPE_MTE2, PIPE.PIPE_MTE2)

        # ---- destroy: consume outstanding init-direction signals ----
        tile_pipe_barrier(PIPE_V)      # SIG_IO_UB
        tile_pipe_barrier(PIPE_MTE3)   # SIG_S_HALF



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
           "workspace_1": "*fp16", "workspace_2": "*fp16", "workspace_3": "*fp16",
           "workspace_4": "*fp32", "workspace_5": "*fp32"}
    i32s = ["B", "Hq", "Hkv", "S",
            "sQb", "sQh", "sQs", "sQd",
            "sKb", "sKh", "sKs", "sKd",
            "sOb", "sOh", "sOs", "sOd",
            "num_seq_blocks", "heads_q", "gqa_group",
            "n_iters", "tpt", "q_tasks", "r_tasks"]
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
    tpt = num_iters // nr
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
    tpt = n_iters // NR   # tasks per output tile

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    out = torch.empty_like(q)
    # GM workspaces
    workspace_1 = torch.empty((NUM_CORES, RING, NR, BLOCK_M, BLOCK_N),
                              dtype=q.dtype, device=q.device)   # S
    workspace_2 = torch.empty((NUM_CORES, RING, NR, BLOCK_M, BLOCK_N),
                              dtype=q.dtype, device=q.device)   # P
    workspace_3 = torch.empty((NUM_CORES, RING, NR, BLOCK_M, DIM),
                              dtype=q.dtype, device=q.device)   # P*V
    # [NUM_CORES, RING, NR, 2 sub-cores, HALF_M, 1]  — r_fac and row_sum from Vec1
    workspace_4 = torch.empty((NUM_CORES, RING, NR, 2, HALF_M, 1),
                              dtype=torch.float32, device=q.device)  # r_fac
    workspace_5 = torch.empty((NUM_CORES, RING, NR, 2, HALF_M, 1),
                              dtype=torch.float32, device=q.device)  # row_sum
    sm_scale = (1.0 / D) ** 0.5

    grid = (NUM_CORES,)
    flash_attention_fwd_3task_kernel[grid](
        q, k, v, out,
        workspace_1, workspace_2, workspace_3, workspace_4, workspace_5,
        sm_scale,
        B, Hq, Hkv, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_seq_blocks, Hq, Hq // Hkv,
        n_iters, tpt, q_tasks, r_tasks,
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
                n_rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=args.causal).to(torch.float16)

        torch.testing.assert_close(ref(q, k, v), out, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
