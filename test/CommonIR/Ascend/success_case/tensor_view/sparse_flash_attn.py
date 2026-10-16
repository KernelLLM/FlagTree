"""
Sparse Flash Attention (DeepSeek V3.2 DSA) written with TensorView
==================================================================

TensorView rewrite of `python/tutorials/tle/dsa/01-sparse-flash-attn-tle.py`.
The compute (online-softmax attention) is unchanged; the kernels that address
memory by data-dependent indices express it through an explicit
`tl.tensor_view` instead of pointer arithmetic, a scalar loop, or
`tle.dsa.insert_slice`.

Access patterns in this operator and how they are expressed:

  kernel                        original access                     TensorView form
  ----------------------------  ----------------------------------  -----------------------------------------
  fused_pa_rope_to_sparse       sparse_indices[b, 0, t]  (scalar)   partition_view, one TOPK tile per load
                                block_table[b, idx // BS] (scalar)  gather_scatter_view, sparse_dim=[1]
                                                                    -> fully discrete: every lane is an
                                                                       independent address
                                K/K_rope/V row at a data-dependent  gather_scatter_view, sparse_dim=[0]
                                (block, offset) (scalar loop)       -> discrete rows, contiguous inside a row
                                insert_slice(K, K_rope) + store     two partition_view stores into disjoint
                                                                    column tiles of the same output
  gather_kv_bnsd_vec            K/V[b, 0, idx, :] (scalar loop)     gather_scatter_view, sparse_dim=[1]
  trans_tnd_to_bsnd_fused       make_block_ptr + insert_slice       partition_view (data-dependent tile index)
  _attn_fwd                     make_block_ptr + advance            unchanged: dense tiles, see the kernel

Two properties of the discrete views replace code that the original writes by
hand:
  * The original visits one TOPK index per loop iteration; here each load moves
    a whole TOPK_TILE of indices, and the lowering chooses per-row block DMA
    when a row is contiguous (innermost stride == 1).
  * Out-of-range indices are handled by the view: loads return the padding
    value (zero) and stores are dropped. `fused_pa_rope_to_sparse_tv_kernel`
    uses this to support the DSA convention that `sparse_indices == -1` marks
    an unused slot, without any mask.

Constraints of the current TensorView frontend that shape this file:
  * A view base must be a kernel pointer argument, never `ptr + offset`; the
    batch/token position is selected through `index=` instead.
  * Once a pointer argument backs a view, the kernel accesses that pointer only
    through views.
  * Regular-dimension indices are int32 scalars; sparse-dimension indices are
    rank-1 integer tensors whose length equals the tile size.
  * Block DMA for a gather needs a non-sparse dimension with a static stride
    of 1, so innermost strides are written as the literal `1` (the host checks
    this). Static strides require the `Preserve static TensorView strides` fix
    in flir.
"""

import argparse
import importlib.util
import os
import sys
from datetime import datetime

import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu

DEVICE = "npu"
DEVICE_ID = 0
torch.manual_seed(20)
torch_npu.npu.set_device(int(DEVICE_ID))
torch.set_printoptions(sci_mode=False, precision=4, linewidth=300)

ascend_aiv_core_nums = triton.language.constexpr(24)

# Number of TOPK entries moved by one gather (overridable with --topk-tile).
DEFAULT_TOPK_TILE = 16

ORIGINAL_TUTORIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "python",
                                 "tutorials", "tle", "dsa", "01-sparse-flash-attn-tle.py")


# =============================================================================
#  PA_BSND + rope concat -> BNSD sparse K/V (two-level indirect gather)
# =============================================================================
@triton.jit
def fused_pa_rope_to_sparse_tv_kernel(
    k_pa_ptr,
    k_rope_pa_ptr,
    v_pa_ptr,  # PA_BSND input [block_num, block_size, 1, d]
    block_table_ptr,  # [B, max_blocks]
    sparse_indices_ptr,  # [B, 1, TOPK]
    k_sparse_out_ptr,  # [B, 1, TOPK, dk + d_rope]
    v_sparse_out_ptr,  # [B, 1, TOPK, dv]
    total_rows,  # block_num * block_size
    max_blocks,
    stride_k_pa_bs,
    stride_k_rope_pa_bs,
    stride_v_pa_bs,  # row (token slot) strides of the PA caches
    stride_bt_b,
    stride_si_b,
    stride_out_b,
    stride_out_topk,
    stride_v_b,
    stride_v_topk,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DK_ROPE: tl.constexpr,  # 0 if no rope
    TOPK: tl.constexpr,
    TOPK_TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    """
    For each (b, TOPK tile):
      sparse_idx[TOPK_TILE] = sparse_indices[b, 0, tile]            (partition view)
      page[TOPK_TILE]       = block_table[b, sparse_idx // BLOCK_SIZE]  (fully discrete gather)
      row[TOPK_TILE]        = page * BLOCK_SIZE + sparse_idx % BLOCK_SIZE
      K/V[TOPK_TILE, d]     = cache[row, :]                         (row gather, block DMA per row)
    The PA caches are viewed as a flat [block_num * block_size, d] row space,
    so the (page, offset) pair collapses into a single sparse coordinate.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    NUM_TOPK_TILES: tl.constexpr = (TOPK + TOPK_TILE - 1) // TOPK_TILE

    # Regular views: indices in, BNSD (N == 1) outputs.
    sparse_idx_tiles = tl.make_partition_view(sparse_indices_ptr, shape=[B, TOPK], strides=[stride_si_b, 1],
                                              tile=[1, TOPK_TILE])
    k_out_tiles = tl.make_partition_view(k_sparse_out_ptr, shape=[B, TOPK, BLOCK_DK + BLOCK_DK_ROPE],
                                         strides=[stride_out_b, stride_out_topk, 1], tile=[1, TOPK_TILE, BLOCK_DK])
    v_out_tiles = tl.make_partition_view(v_sparse_out_ptr, shape=[B, TOPK, BLOCK_DV],
                                         strides=[stride_v_b, stride_v_topk, 1], tile=[1, TOPK_TILE, BLOCK_DV])

    # Discrete views.
    # Level 1: one independent block-table entry per lane.
    page_gather = tl.make_gather_scatter_view(block_table_ptr, shape=[B, max_blocks], strides=[stride_bt_b, 1],
                                              tile=[1, TOPK_TILE], sparse_dim=[1])
    # Level 2: one contiguous cache row per lane.
    k_rows = tl.make_gather_scatter_view(k_pa_ptr, shape=[total_rows, BLOCK_DK], strides=[stride_k_pa_bs, 1],
                                         tile=[TOPK_TILE, BLOCK_DK], sparse_dim=[0])
    v_rows = tl.make_gather_scatter_view(v_pa_ptr, shape=[total_rows, BLOCK_DV], strides=[stride_v_pa_bs, 1],
                                         tile=[TOPK_TILE, BLOCK_DV], sparse_dim=[0])
    if BLOCK_DK_ROPE > 0:
        tl.static_assert(BLOCK_DK % BLOCK_DK_ROPE == 0, "rope columns must start on a rope-tile boundary")
        k_rope_rows = tl.make_gather_scatter_view(k_rope_pa_ptr, shape=[total_rows, BLOCK_DK_ROPE],
                                                  strides=[stride_k_rope_pa_bs, 1], tile=[TOPK_TILE, BLOCK_DK_ROPE],
                                                  sparse_dim=[0])
        # Same output tensor, rope columns [BLOCK_DK, BLOCK_DK + BLOCK_DK_ROPE).
        k_rope_out_tiles = tl.make_partition_view(k_sparse_out_ptr, shape=[B, TOPK, BLOCK_DK + BLOCK_DK_ROPE],
                                                  strides=[stride_out_b, stride_out_topk, 1],
                                                  tile=[1, TOPK_TILE, BLOCK_DK_ROPE])

    for b in range(B):
        for t in range(pid, NUM_TOPK_TILES, num_programs):
            sparse_idx = tl.reshape(tl.load(sparse_idx_tiles, index=(b, t)), (TOPK_TILE, ))
            valid = sparse_idx >= 0

            # Level 1: logical block -> physical page.
            page = tl.reshape(tl.load(page_gather, index=(b, sparse_idx // BLOCK_SIZE)), (TOPK_TILE, ))
            # Invalid slots get an out-of-range row, so the row gather pads them with zeros.
            row = tl.where(valid, page * BLOCK_SIZE + sparse_idx % BLOCK_SIZE, -1)

            # Level 2: gather rows, store into column tiles of the BNSD outputs.
            k_vec = tl.load(k_rows, index=(row, 0))  # [TOPK_TILE, BLOCK_DK]
            tl.store(k_out_tiles, k_vec[None, :, :], index=(b, t, 0))
            if BLOCK_DK_ROPE > 0:
                k_rope_vec = tl.load(k_rope_rows, index=(row, 0))  # [TOPK_TILE, BLOCK_DK_ROPE]
                tl.store(k_rope_out_tiles, k_rope_vec[None, :, :], index=(b, t, BLOCK_DK // BLOCK_DK_ROPE))

            v_vec = tl.load(v_rows, index=(row, 0))  # [TOPK_TILE, BLOCK_DV]
            tl.store(v_out_tiles, v_vec[None, :, :], index=(b, t, 0))


def _check_pa_cache(cache, name):
    block_num, block_size, n, _ = cache.shape
    assert n == 1, f"{name}: KV_N must be 1"
    assert cache.stride(-1) == 1, f"{name}: innermost dimension must be contiguous"
    assert cache.stride(0) == block_size * cache.stride(1), \
        f"{name}: pages must be contiguous so rows form a flat [block_num * block_size, d] space"


def triton_fused_pa_rope_to_sparse(k_pa, k_rope_pa, v_pa, block_table, sparse_indices, block_size,
                                   topk_tile=None):
    """
    Fused PA_BSND + rope concat -> BNSD sparse conversion.

    Args:
        k_pa: [block_num, block_size, 1, dk]
        k_rope_pa: [block_num, block_size, 1, d_rope], or None
        v_pa: [block_num, block_size, 1, dv]
        block_table: [B, max_blocks]
        sparse_indices: [B, 1, TOPK] or [B, TOPK]; -1 marks an unused slot
    Returns:
        k_sparse [B, 1, TOPK, dk + d_rope], v_sparse [B, 1, TOPK, dv]; unused slots are zero.
    """
    block_num, _, _, dk = k_pa.shape
    dv = v_pa.shape[-1]
    B, max_blocks = block_table.shape
    TOPK = sparse_indices.size(-1)

    has_rope = k_rope_pa is not None
    dk_rope = k_rope_pa.shape[-1] if has_rope else 0
    k_rope_pa_input = k_rope_pa if has_rope else k_pa
    for cache, name in ((k_pa, "k_pa"), (k_rope_pa_input, "k_rope_pa"), (v_pa, "v_pa")):
        _check_pa_cache(cache, name)
    assert block_table.stride(1) == 1

    if sparse_indices.dim() == 3:
        assert sparse_indices.shape[1] == 1, "KV_N must be 1"
        sparse_indices = sparse_indices[:, 0, :]
    assert sparse_indices.stride(-1) == 1

    topk_tile = topk_tile or DEFAULT_TOPK_TILE
    k_sparse = torch.empty((B, 1, TOPK, dk + dk_rope), dtype=k_pa.dtype, device=DEVICE)
    v_sparse = torch.empty((B, 1, TOPK, dv), dtype=v_pa.dtype, device=DEVICE)

    grid = (min(48, triton.cdiv(TOPK, topk_tile)), )
    fused_pa_rope_to_sparse_tv_kernel[grid](
        k_pa, k_rope_pa_input, v_pa, block_table, sparse_indices, k_sparse, v_sparse,
        block_num * block_size, max_blocks,
        k_pa.stride(1), k_rope_pa_input.stride(1), v_pa.stride(1),
        block_table.stride(0), sparse_indices.stride(0),
        k_sparse.stride(0), k_sparse.stride(2), v_sparse.stride(0), v_sparse.stride(2),
        BLOCK_DK=dk, BLOCK_DV=dv, BLOCK_DK_ROPE=dk_rope, TOPK=TOPK, TOPK_TILE=topk_tile,
        BLOCK_SIZE=block_size, B=B)
    return k_sparse, v_sparse


# =============================================================================
#  BNSD K/V gather along the sequence dimension (non-PA layout)
# =============================================================================
@triton.jit
def gather_kv_bnsd_vec_tv_kernel(
    k_ptr,
    v_ptr,
    ind_ptr,  # [TOPK], shared by every batch (same as the original kernel)
    k_out_ptr,
    v_out_ptr,
    SK,
    stride_kb,
    stride_ks,
    stride_vb,
    stride_vs,
    stride_ob,
    stride_os,
    stride_ovb,
    stride_ovs,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_TILE: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    NUM_TOPK_TILES: tl.constexpr = (TOPK + TOPK_TILE - 1) // TOPK_TILE

    index_tiles = tl.make_partition_view(ind_ptr, shape=[TOPK], strides=[1], tile=[TOPK_TILE])
    # Sequence dimension is sparse; each gathered position is a contiguous row.
    k_src = tl.make_gather_scatter_view(k_ptr, shape=[B, SK, BLOCK_DK], strides=[stride_kb, stride_ks, 1],
                                        tile=[1, TOPK_TILE, BLOCK_DK], sparse_dim=[1])
    v_src = tl.make_gather_scatter_view(v_ptr, shape=[B, SK, BLOCK_DV], strides=[stride_vb, stride_vs, 1],
                                        tile=[1, TOPK_TILE, BLOCK_DV], sparse_dim=[1])
    k_dst = tl.make_partition_view(k_out_ptr, shape=[B, TOPK, BLOCK_DK], strides=[stride_ob, stride_os, 1],
                                   tile=[1, TOPK_TILE, BLOCK_DK])
    v_dst = tl.make_partition_view(v_out_ptr, shape=[B, TOPK, BLOCK_DV], strides=[stride_ovb, stride_ovs, 1],
                                   tile=[1, TOPK_TILE, BLOCK_DV])

    for b in range(B):
        for t in range(pid, NUM_TOPK_TILES, num_programs):
            idx = tl.load(index_tiles, index=(t, ))  # [TOPK_TILE]
            tl.store(k_dst, tl.load(k_src, index=(b, idx, 0)), index=(b, t, 0))
            tl.store(v_dst, tl.load(v_src, index=(b, idx, 0)), index=(b, t, 0))


def triton_gather_kv_bnsd_vec(k, v, indices, topk_tile=None):
    B, N, SK, Dk = k.shape
    Dv = v.shape[-1]
    assert N == 1, "KV_N must be 1"
    assert k.stride(-1) == 1 and v.stride(-1) == 1
    TOPK = indices.size(-1)
    # Same index source as the original kernel: the first TOPK entries.
    ind = indices.reshape(-1)[:TOPK].contiguous()

    topk_tile = topk_tile or DEFAULT_TOPK_TILE
    k_sparse = torch.empty((B, N, TOPK, Dk), dtype=k.dtype, device=DEVICE)
    v_sparse = torch.empty((B, N, TOPK, Dv), dtype=v.dtype, device=DEVICE)

    grid = (min(48, triton.cdiv(TOPK, topk_tile)), )
    gather_kv_bnsd_vec_tv_kernel[grid](k, v, ind, k_sparse, v_sparse, SK, k.stride(0), k.stride(2), v.stride(0),
                                       v.stride(2), k_sparse.stride(0), k_sparse.stride(2), v_sparse.stride(0),
                                       v_sparse.stride(2), BLOCK_DK=Dk, BLOCK_DV=Dv, TOPK=TOPK, TOPK_TILE=topk_tile,
                                       B=B)
    return k_sparse, v_sparse


# =============================================================================
#  Attention forward (replaces _attn_fwd and _attn_fwd_fused_bsnd_to_tnd)
#
#  This is the tutorial's attention body, unchanged apart from folding the two
#  output layouts into one kernel. Q/K/V here are read as whole dense tiles, so
#  there is no discrete access for a TensorView to describe, and block pointers
#  keep the loop in the shape the cube/vector pipeliner expects: a view_load
#  lowers to an `scf.if` bounds guard around the transfer, and the pipeliner
#  rejects that conditional inside the K loop (`LLVM ERROR: Error in
#  cv-pipelining`).
# =============================================================================
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    O,
    scale_value,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,  # 0 on the TND path, where the output has no S axis
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    B: tl.constexpr,
    Q_N: tl.constexpr,
    Q_D: tl.constexpr,
    Q_S: tl.constexpr,
    KV_S: tl.constexpr,
    K_D: tl.constexpr,
    V_D: tl.constexpr,
    O_N: tl.constexpr,
    O_D: tl.constexpr,
    actual_seq_lengths_query,
    blk_size: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
):
    BLOCK_QN_NUM = Q_N // Q_BLOCK_SIZE
    NUM_BLOCKS = B * Q_S * BLOCK_QN_NUM
    pid = tl.program_id(0)
    num_cores = min(ascend_aiv_core_nums, NUM_BLOCKS)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        off_b = (block_idx // (Q_S * BLOCK_QN_NUM)).to(tl.int32)
        off_s = ((block_idx // BLOCK_QN_NUM) % Q_S).to(tl.int32)
        off_n = (block_idx % BLOCK_QN_NUM).to(tl.int32)

        q_offset = off_b * stride_qb + off_s * stride_qs
        o_offset = off_b * stride_ob + off_s * stride_os
        k_offset = off_b * stride_kb  # KV_N = 1
        v_offset = off_b * stride_vb

        cur_act_s_q = tl.load(actual_seq_lengths_query + off_b)

        for i in range(cur_act_s_q):
            cur_max = tl.full((Q_BLOCK_SIZE, ), float('-inf'), dtype=tl.float32)
            logSum = tl.zeros((Q_BLOCK_SIZE, ), dtype=tl.float32)
            acc = tl.zeros((Q_BLOCK_SIZE, V_D), dtype=tl.float32)

            q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(Q_N, Q_D), strides=(stride_qn, stride_qd),
                                            offsets=(off_n * Q_BLOCK_SIZE, 0), block_shape=(Q_BLOCK_SIZE, Q_D),
                                            order=(1, 0))
            q_vec = tl.load(q_block_ptr, boundary_check=(0, 1))
            k_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(KV_S, K_D), strides=(stride_ks, stride_kd),
                                            offsets=(0, 0), block_shape=(blk_size, K_D), order=(1, 0))
            v_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(KV_S, V_D), strides=(stride_vs, stride_vd),
                                            offsets=(0, 0), block_shape=(blk_size, V_D), order=(1, 0))

            for k_idx in range(KV_S // blk_size):
                k_vec = tl.load(k_block_ptr, boundary_check=(0, 1))

                qk = tl.dot(q_vec.to(tl.float16), tl.trans(k_vec).to(tl.float16)) * scale_value
                block_max = tl.max(qk, axis=1)
                new_max = tl.maximum(cur_max, block_max)
                coeff = tl.math.exp(cur_max - new_max)
                p = tl.math.exp(qk - new_max[:, None])
                logSum = logSum * coeff + tl.sum(p, axis=1)

                v_vec = tl.load(v_block_ptr, boundary_check=(0, 1))
                pv = tl.dot(p.to(tl.float16), v_vec)
                acc = acc * coeff[:, None] + pv
                cur_max = new_max

                k_block_ptr = k_block_ptr.advance((blk_size, 0))
                v_block_ptr = v_block_ptr.advance((blk_size, 0))

            o_block_ptr = tl.make_block_ptr(base=O + o_offset, shape=(O_N, O_D), strides=(stride_on, stride_od),
                                            offsets=(off_n * Q_BLOCK_SIZE, 0), block_shape=(Q_BLOCK_SIZE, O_D),
                                            order=(1, 0))
            acc = acc / logSum[:, None]
            tl.store(o_block_ptr, acc)


# =============================================================================
#  TND -> BSND query layout conversion with rope concat
# =============================================================================
@triton.jit
def trans_tnd_to_bsnd_fused_tv_kernel(
    query_ptr,
    query_rope_ptr,
    sparse_ptr,
    query_out_ptr,  # rope concatenated
    sparse_out_ptr,
    act_s,
    T,
    S_OUT,
    stride_q_t,
    stride_q_tn,
    stride_qr_t,
    stride_qr_tn,
    stride_s_t,
    stride_s_tn,
    stride_qob,
    stride_qobs,
    stride_qon,
    stride_sb,
    stride_sbs,
    stride_sbn,
    B: tl.constexpr,
    N: tl.constexpr,
    D_QUERY: tl.constexpr,
    D_ROPE: tl.constexpr,
    D_SPARSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(D_ROPE > 0, "the TND path requires a rope query")
    tl.static_assert(D_QUERY % D_ROPE == 0, "rope columns must start on a rope-tile boundary")
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_head_blocks: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    D_OUT: tl.constexpr = D_QUERY + D_ROPE

    q_in = tl.make_partition_view(query_ptr, shape=[T, N, D_QUERY], strides=[stride_q_t, stride_q_tn, 1],
                                  tile=[1, BLOCK_N, D_QUERY])
    q_rope_in = tl.make_partition_view(query_rope_ptr, shape=[T, N, D_ROPE], strides=[stride_qr_t, stride_qr_tn, 1],
                                       tile=[1, BLOCK_N, D_ROPE])
    # Two column tiles of the same output replace the insert_slice concat.
    q_out = tl.make_partition_view(query_out_ptr, shape=[B, S_OUT, N, D_OUT], strides=[stride_qob, stride_qobs,
                                                                                         stride_qon, 1],
                                   tile=[1, 1, BLOCK_N, D_QUERY])
    q_rope_out = tl.make_partition_view(query_out_ptr, shape=[B, S_OUT, N, D_OUT], strides=[stride_qob, stride_qobs,
                                                                                              stride_qon, 1],
                                        tile=[1, 1, BLOCK_N, D_ROPE])
    sparse_in = tl.make_partition_view(sparse_ptr, shape=[T, 1, D_SPARSE], strides=[stride_s_t, stride_s_tn, 1],
                                       tile=[1, 1, D_SPARSE])
    sparse_out = tl.make_partition_view(sparse_out_ptr, shape=[B, S_OUT, 1, D_SPARSE],
                                        strides=[stride_sb, stride_sbs, stride_sbn, 1], tile=[1, 1, 1, D_SPARSE])

    t_idx = tl.full((), 0, dtype=tl.int32)  # TODO (same as the original): exact token mapping
    for tn_id in range(B):
        # sparse_indices has a single head; copy it once.
        if pid == 0:
            sparse = tl.load(sparse_in, index=(t_idx, 0, 0))
            tl.store(sparse_out, tl.reshape(sparse, (1, 1, 1, D_SPARSE)), index=(t_idx, 0, 0, 0))

        for head_block_id in range(pid, num_head_blocks, num_programs):
            q = tl.load(q_in, index=(t_idx, head_block_id, 0))
            q_ro = tl.load(q_rope_in, index=(t_idx, head_block_id, 0))
            tl.store(q_out, tl.reshape(q, (1, 1, BLOCK_N, D_QUERY)), index=(t_idx, 0, head_block_id, 0))
            tl.store(q_rope_out, tl.reshape(q_ro, (1, 1, BLOCK_N, D_ROPE)),
                     index=(t_idx, 0, head_block_id, D_QUERY // D_ROPE))
        t_idx = t_idx + tl.load(act_s + tn_id).to(tl.int32)


def trans_tnd_to_bsnd_fused(query, query_rope, sparse_indices, shape, act_seq):
    t, n, d_query = shape
    b = len(act_seq)
    # `act_seq` is a device tensor, so keep the extent a Python int: the views
    # below pass it to the kernel as a scalar shape argument.
    s = int(max(act_seq))
    assert query_rope is not None, "rope query is required on the TND path"
    for x in (query, query_rope, sparse_indices):
        assert x.stride(-1) == 1
    d_rope = query_rope.shape[2]
    d_sparse = sparse_indices.shape[2]
    assert sparse_indices.shape[1] == 1, "sparse_indices second dim must be 1 when MLA"

    query_out = torch.empty((b, s, n, d_query + d_rope), dtype=query.dtype, device=query.device)
    sparse_out = torch.empty((b, s, 1, d_sparse), dtype=sparse_indices.dtype, device=sparse_indices.device)

    block_n = min(16, n)
    num_head_blocks = (n + block_n - 1) // block_n
    num_programs = min(ascend_aiv_core_nums, num_head_blocks)

    trans_tnd_to_bsnd_fused_tv_kernel[(num_programs, )](
        query, query_rope, sparse_indices, query_out, sparse_out, act_seq, t, s,
        query.stride(0), query.stride(1), query_rope.stride(0), query_rope.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        query_out.stride(0), query_out.stride(1), query_out.stride(2),
        sparse_out.stride(0), sparse_out.stride(1), sparse_out.stride(2),
        B=b, N=n, D_QUERY=d_query, D_ROPE=d_rope, D_SPARSE=d_sparse, BLOCK_N=block_n)
    return query_out, sparse_out


def trans_tnd_actseq(seq):
    if isinstance(seq, torch.Tensor):
        seq = seq.cpu().tolist()
    output = [seq[0]]
    total_len = seq[0]
    for i in range(len(seq) - 1):
        new_item = seq[i + 1] - seq[i]
        if new_item >= 0:
            output.append(new_item)
            total_len += new_item
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{seq}, in loop {i}, item {new_item} < 0")
    return torch.tensor(output).to(DEVICE), total_len


def sparse_attention(query, key, value, sparse_indices, scale_value, sparse_block_size=1, actual_seq_lengths_query=None,
                     actual_seq_lengths_kv=None, query_rope=None, key_rope=None, layout_query='BSND', layout_kv='BSND',
                     sparse_mode=0, block_table=None):
    sparse_indices_orig = sparse_indices
    total_len = 0
    if layout_query == 'TND':
        actual_seq_lengths_query, total_len = trans_tnd_actseq(actual_seq_lengths_query)
        query, sparse_indices = trans_tnd_to_bsnd_fused(query, query_rope, sparse_indices, query.shape,
                                                        actual_seq_lengths_query)
    elif query_rope is not None:
        query = torch.cat([query, query_rope], dim=-1)

    if layout_kv == 'PA_BSND':
        block_size = key.shape[1]
        k_sparse, v_sparse = triton_fused_pa_rope_to_sparse(key, key_rope, value, block_table, sparse_indices_orig,
                                                            block_size)
    else:
        if key_rope is not None:
            key = torch.cat([key, key_rope], dim=-1)
        key_bnsd = key.permute(0, 2, 1, 3).contiguous()
        value_bnsd = value.permute(0, 2, 1, 3).contiguous()
        sparse_indices_bnsd = sparse_indices.permute(0, 2, 1, 3).contiguous()
        k_sparse, v_sparse = triton_gather_kv_bnsd_vec(key_bnsd, value_bnsd, sparse_indices_bnsd)

    out_shape_bsnd = list(query.shape)
    if query_rope is not None:
        out_shape_bsnd[-1] = out_shape_bsnd[-1] - query_rope.shape[-1]
    B, Q_S, Q_N, Q_D = query.shape
    _, _, KV_S, K_D = k_sparse.shape

    out_tnd = layout_query == 'TND'
    if out_tnd:
        output = torch.empty((total_len, out_shape_bsnd[2], out_shape_bsnd[3]), device=query.device,
                             dtype=torch.float32)
    else:
        output = torch.empty(out_shape_bsnd, device=query.device, dtype=torch.float32)

    # The TND output has no S axis, so a zero S stride keeps one offset formula.
    stride_os = 0 if out_tnd else output.stride(1)
    _attn_fwd[(ascend_aiv_core_nums, )](
        query, k_sparse, v_sparse, output, scale_value, query.stride(0), query.stride(1), query.stride(2),
        query.stride(3), k_sparse.stride(0), k_sparse.stride(2), k_sparse.stride(3), v_sparse.stride(0),
        v_sparse.stride(2), v_sparse.stride(3), output.stride(0), stride_os, output.stride(-2), output.stride(-1),
        B=B, Q_N=Q_N, Q_D=Q_D, Q_S=Q_S, KV_S=KV_S, K_D=K_D, V_D=v_sparse.shape[3], O_N=output.shape[-2],
        O_D=output.shape[-1], actual_seq_lengths_query=actual_seq_lengths_query, blk_size=128, Q_BLOCK_SIZE=16,
        limit_auto_multi_buffer_only_for_local_buffer=False, limit_auto_multi_buffer_of_local_buffer="no-limit")

    if not out_tnd:
        output = output.permute(0, 2, 1, 3).contiguous()
    return output


# =============================================================================
#  Tests
# =============================================================================
def torch_pa_gather_ref(cache, block_table, sparse_indices, block_size):
    """Reference for the two-level gather; -1 entries produce zero rows."""
    idx = sparse_indices.reshape(sparse_indices.shape[0], -1).long()  # [B, TOPK]
    valid = idx >= 0
    safe = idx.clamp(min=0)
    page = torch.gather(block_table.long(), 1, safe // block_size)
    rows = page * block_size + safe % block_size
    out = cache[:, :, 0, :].reshape(-1, cache.shape[-1])[rows]  # [B, TOPK, d]
    out[~valid] = 0
    return out.unsqueeze(1)


def make_random_pa_inputs(B, KV_S, D, D_rope, block_size, topk, invalid_ratio=0.0, dtype=torch.float16):
    """PA caches with a shuffled block table and random, unsorted sparse indices."""
    blocks_per_seq = KV_S // block_size
    block_num = B * blocks_per_seq
    key = torch.randn((block_num, block_size, 1, D), dtype=dtype, device=DEVICE)
    value = torch.randn_like(key)
    key_rope = torch.randn((block_num, block_size, 1, D_rope), dtype=dtype, device=DEVICE) if D_rope else None
    block_table = torch.randperm(block_num, device=DEVICE).to(torch.int32).reshape(B, blocks_per_seq)
    sparse_indices = torch.stack([torch.randperm(KV_S, device=DEVICE)[:topk] for _ in range(B)])
    sparse_indices = sparse_indices.to(torch.int32).unsqueeze(1)  # [B, 1, TOPK]
    if invalid_ratio > 0:
        drop = torch.rand(sparse_indices.shape, device=DEVICE) < invalid_ratio
        sparse_indices[drop] = -1
    return key, key_rope, value, block_table, sparse_indices


def check_close(name, actual, expected, rtol, atol, raise_on_fail=True):
    """Print the max abs/rel difference and whether it is within tolerance."""
    actual = actual.float()
    expected = expected.to(actual.device).float()
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp(min=1e-6)
    ok = torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    print(f"  {name:<32} max_abs={diff.max().item():.3e} max_rel={rel.max().item():.3e} "
          f"(rtol={rtol}, atol={atol}) {'OK' if ok else 'MISMATCH'}")
    if raise_on_fail:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    return ok


def bench_pair(name, subject_fn, original_fn, label):
    """Time the kernel under test against the original one, if available."""
    subject_time = do_bench_npu(subject_fn, clear_l2_cache=True)
    print(f"  {name:<32} {label}: {subject_time:.4f} us")
    if original_fn is None:
        return
    original_time = do_bench_npu(original_fn, clear_l2_cache=True)
    print(f"  {name:<32} original: {original_time:.4f} us "
          f"({original_time / subject_time:.2f}x vs {label})")


def test_pa_gather(B=2, KV_S=2560, D=512, D_rope=64, block_size=128, topk=2048, invalid_ratio=0.1, original=None,
                   subject=None, label="TensorView", bench=False):
    """Check the two-level gather alone, including -1 (unused) slots.

    `subject` is the module under test (this file by default, or the original
    tutorial with --only-original); `original` is an optional second module
    compared against it.
    """
    subject = subject or sys.modules[__name__]
    print(f"[PA gather] {label} B={B} KV_S={KV_S} TOPK={topk} invalid_ratio={invalid_ratio}")
    key, key_rope, value, block_table, sparse_indices = make_random_pa_inputs(B, KV_S, D, D_rope, block_size, topk,
                                                                              invalid_ratio)
    k_sparse, v_sparse = subject.triton_fused_pa_rope_to_sparse(key, key_rope, value, block_table, sparse_indices,
                                                                block_size)

    k_ref = torch_pa_gather_ref(key, block_table, sparse_indices, block_size)
    if key_rope is not None:
        k_ref = torch.cat([k_ref, torch_pa_gather_ref(key_rope, block_table, sparse_indices, block_size)], dim=-1)
    v_ref = torch_pa_gather_ref(value, block_table, sparse_indices, block_size)
    check_close(f"K  {label} vs torch", k_sparse, k_ref, rtol=0, atol=0)
    check_close(f"V  {label} vs torch", v_sparse, v_ref, rtol=0, atol=0)

    if original is not None:
        if invalid_ratio > 0:
            # The original kernel has no handling for -1 slots.
            print("  (original kernel skipped: it does not support -1 indices)")
        else:
            k_orig, v_orig = original.triton_fused_pa_rope_to_sparse(key, key_rope, value, block_table,
                                                                     sparse_indices, block_size)
            check_close("K  TensorView vs original", k_sparse, k_orig, rtol=0, atol=0)
            check_close("V  TensorView vs original", v_sparse, v_orig, rtol=0, atol=0)
    print("  [PASSED]")

    if bench:
        gather_args = (key, key_rope, value, block_table, sparse_indices, block_size)
        bench_pair(
            "PA gather", lambda: subject.triton_fused_pa_rope_to_sparse(*gather_args),
            None if original is None or invalid_ratio > 0 else
            (lambda: original.triton_fused_pa_rope_to_sparse(*gather_args)), label)


def test_bnsd_gather(B=1, SK=4096, D=576, topk=2048, original=None, subject=None, label="TensorView", bench=False):
    subject = subject or sys.modules[__name__]
    print(f"[BNSD gather] {label} SK={SK} TOPK={topk}")
    k = torch.randn((B, 1, SK, D), dtype=torch.float16, device=DEVICE)
    v = torch.randn((B, 1, SK, D - 64), dtype=torch.float16, device=DEVICE)
    indices = torch.randperm(SK, device=DEVICE)[:topk].to(torch.int32).reshape(1, 1, 1, topk)
    k_sparse, v_sparse = subject.triton_gather_kv_bnsd_vec(k, v, indices)
    idx = indices.reshape(-1).long()
    check_close(f"K  {label} vs torch", k_sparse, k[:, :, idx, :], rtol=0, atol=0)
    check_close(f"V  {label} vs torch", v_sparse, v[:, :, idx, :], rtol=0, atol=0)
    if original is not None:
        k_orig, v_orig = original.triton_gather_kv_bnsd_vec(k, v, indices)
        check_close("K  TensorView vs original", k_sparse, k_orig, rtol=0, atol=0)
        check_close("V  TensorView vs original", v_sparse, v_orig, rtol=0, atol=0)
    print("  [PASSED]")

    if bench:
        bench_pair("BNSD gather", lambda: subject.triton_gather_kv_bnsd_vec(k, v, indices),
                   None if original is None else (lambda: original.triton_gather_kv_bnsd_vec(k, v, indices)), label)


def _load_original_tutorial(path=ORIGINAL_TUTORIAL):
    path = os.path.abspath(path)
    print(f"loading original tutorial from {path}")
    spec = importlib.util.spec_from_file_location("sparse_flash_attn_tle_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_op(T, B, KV_S, Q_N, KV_N, D, D_rope, sparse_size, scale_value, sparse_block_size, sparse_mode, block_size,
            act_kv_s, random_indices=True, bench=True, original=None, subject=None, label="TensorView"):
    subject = subject or sys.modules[__name__]
    assert sparse_size <= KV_S
    assert KV_N == 1
    assert sparse_block_size == 1
    assert (B * KV_S) % block_size == 0
    assert D == 512
    assert D_rope in (0, 64)
    qkv_dtype = torch.float16

    query = torch.empty((T, Q_N, D), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    query_rope = torch.empty((T, Q_N, D_rope), dtype=qkv_dtype, device=DEVICE).normal_(mean=0.0,
                                                                                       std=0.5) if D_rope else None
    if random_indices:
        # Shuffled pages and unsorted indices exercise the discrete paths.
        key, key_rope, value, block_table, sparse_indices = make_random_pa_inputs(B, KV_S, D, D_rope, block_size,
                                                                                  sparse_size)
        value = key.clone()
    else:
        # Original tutorial inputs: identity block table, indices 0..sparse_size-1.
        key = torch.empty((B * KV_S // block_size, block_size, KV_N, D), dtype=qkv_dtype,
                          device=DEVICE).normal_(mean=0.0, std=0.5)
        value = key.clone()
        key_rope = torch.empty((B * KV_S // block_size, block_size, KV_N, D_rope), dtype=qkv_dtype,
                               device=DEVICE).normal_(mean=0.0, std=0.5) if D_rope else None
        block_table = torch.tensor([range(B * KV_S // block_size)], dtype=torch.int32, device=DEVICE).reshape(B, -1)
        sparse_indices = torch.arange(sparse_size, device=DEVICE, dtype=torch.int32).view(1, 1, -1).expand(
            T, KV_N, -1).contiguous()

    actual_seq_lengths_query = torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE)
    actual_seq_lengths_kv = torch.tensor([act_kv_s] * B, dtype=torch.int32, device=DEVICE)
    args = dict(query=query, key=key, value=value, sparse_indices=sparse_indices, scale_value=scale_value,
                sparse_block_size=sparse_block_size, actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv, query_rope=query_rope, key_rope=key_rope,
                layout_query='TND', layout_kv='PA_BSND', sparse_mode=sparse_mode, block_table=block_table)

    print(f"[SFA] {label} KV_S={KV_S} TOPK={sparse_size} random_indices={random_indices}")
    tv_out = subject.sparse_attention(**args)
    npu_out, _, _ = torch_npu.npu_sparse_flash_attention(**args, attention_mode=2)
    # Print every comparison before failing, so one run shows which side diverges.
    results = [check_close(f"{label} vs torch_npu", tv_out, npu_out, rtol=1e-2, atol=1e-2, raise_on_fail=False)]
    if original is not None:
        orig_out = original.sparse_attention(**args)
        results.append(
            check_close("original   vs torch_npu", orig_out, npu_out, rtol=1e-2, atol=1e-2, raise_on_fail=False))
        # Same arithmetic on both sides; only the data movement differs.
        results.append(
            check_close("TensorView vs original", tv_out, orig_out, rtol=1e-3, atol=1e-3, raise_on_fail=False))
    assert all(results), f"SFA mismatch for KV_S={KV_S}"
    print("  [PASSED]")

    if not bench:
        return
    tv_time = do_bench_npu(lambda: subject.sparse_attention(**args), clear_l2_cache=True)
    print(f"[{label} SFA] Time: {tv_time:.4f} us")
    if original is not None:
        orig_time = do_bench_npu(lambda: original.sparse_attention(**args), clear_l2_cache=True)
        print(f"[Original TLE SFA] Time: {orig_time:.4f} us")
    npu_time = do_bench_npu(lambda: torch_npu.npu_sparse_flash_attention(**args, attention_mode=2),
                            clear_l2_cache=True)
    print(f"[Torch-NPU SFA] Time: {npu_time:.4f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--no-bench", action="store_true", help="only check correctness")
    parser.add_argument("--compare-original", action="store_true",
                        help="run the original TLE tutorial on the same inputs and compare results and time")
    parser.add_argument("--only-original", action="store_true",
                        help="run the same checks on the original tutorial only (TensorView kernels are not compiled)")
    parser.add_argument("--topk-tile", type=int, default=DEFAULT_TOPK_TILE,
                        help="TOPK entries per TensorView gather (UB usage scales with it)")
    parser.add_argument("--original-path", default=ORIGINAL_TUTORIAL,
                        help="path to 01-sparse-flash-attn-tle.py")
    parser.add_argument("--original-inputs", action="store_true",
                        help="use the tutorial's identity block table and sorted indices")
    parser.add_argument("--kv-s", type=int, nargs="+", default=[2560, 5120, 10240, 20480],
                        help="KV_S values for the end-to-end cases")
    parser.add_argument("--skip-gather", action="store_true", help="skip the standalone gather checks")
    parser.add_argument("--skip-sfa", action="store_true", help="skip the end-to-end attention cases")
    cli = parser.parse_args()

    print(torch_npu.__version__)
    print(f"time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    DEFAULT_TOPK_TILE = cli.topk_tile
    original = _load_original_tutorial(cli.original_path) if cli.compare_original or cli.only_original else None
    subject, label = None, f"TensorView(tile={cli.topk_tile})"
    if cli.only_original:
        # The original kernel has no -1 handling, so the invalid-slot case is skipped.
        subject, label, original = original, "original", None

    bench = not cli.no_bench
    if not cli.skip_gather:
        if not cli.only_original:
            test_pa_gather(original=original, subject=subject, label=label)
        test_pa_gather(B=1, invalid_ratio=0.0, original=original, subject=subject, label=label, bench=bench)
        test_bnsd_gather(original=original, subject=subject, label=label, bench=bench)
    for kv_s in ([] if cli.skip_sfa else cli.kv_s):
        test_op(T=1, B=1, KV_S=kv_s, Q_N=128, KV_N=1, D=512, D_rope=64, sparse_size=2048, scale_value=0.5,
                sparse_block_size=1, sparse_mode=0, block_size=128, act_kv_s=kv_s,
                random_indices=not cli.original_inputs, bench=not cli.no_bench, original=original, subject=subject,
                label=label)
