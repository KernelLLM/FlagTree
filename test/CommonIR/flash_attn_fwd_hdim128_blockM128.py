"""
Flash Attention Forward - hdim128, blockM=128, blockN=64
=========================================================
Triton Gluon 实现，对标沐曦 xcore1000 kernel `flash_fwd_kernel_hdim128_blockM128.h`。
使用显式 layout 描述分块和计算调度，适配 NVIDIA Ampere (sm_80) mma.sync m16n8k16。

=== 原始 CUDA kernel 分析 ===

【参数】
  kBlockM  = 128  (Q 的行分块)
  kBlockN  = 64   (K/V 的行分块，即 attention score 的列分块)
  kHeadDim = 128  (head dimension)
  kNWarps  = 4    (xcore warp=64 threads, 总 256 threads)
  Share_Q_K_smem = true  (Q 和 K 共用 SMEM)
  Is_Q_in_regs = true    (Q 常驻寄存器)

【SMEM 分配】
  sQ/sK: [128, 128] fp16 = 32KB (共享，Q load 完毕后 K 复用同一空间)
  sV:    [64, 128] fp16  = 16KB (transposed layout for P@V gemm)
  sO:    [128, 128] fp16 = 32KB (epilogue 写回时复用 sQ/sK 空间)
  SmemLayout: Swizzle<3, 3, 3> 即 8-row xor pattern, 16x64 atom, tiled to full block

【寄存器分配】
  tQrQ: kRegSize = kSmemSize/4/kNThreads = 32KB/4/256 = 32 uint32 → Q 常驻 RF
  tKrK: kRegSize/2 = 16 uint32 → K 先 load 到 RF 再写入 SMEM
  tVrV: kRegSize/2 = 16 uint32 → V 先 load 到 RF 再写入 SMEM (transposed)
  acc_s: [BLOCK_M, BLOCK_N] fp32 = MMA accumulator for S = Q@K^T
  acc_o: [BLOCK_M, HEAD_DIM] fp32 = MMA accumulator for O = P@V

【计算调度 (每次 N-block 迭代)】
  1. copy_global_to_reg(K)         — GMEM → tKrK (128-bit vector load, boundary predicate)
  2. copy_global_to_reg_V(V)       — GMEM → tVrV (128-bit vector load, custom row mapping)
  3. copy_reg_to_share(tKrK, sK)   — RF → SMEM (bank-conflict-free swizzled store)
  4. __syncthreads()
  5. gemm(acc_s, Q_rf, K_smem)     — MMA: [128,64] = [128,128] @ [128,64]^T
                                      Inner loop: 128/16=8 K-tiles, each m16n8k16
  6. apply_mask / softcap / causal mask
  7. copy_reg_to_share_V(tVrV, sV) — RF → SMEM (transposed, permuted for MMA-B)
  8. softmax_rescale_o              — online softmax: max, exp, sum, rescale acc_o
  9. convert fp32→fp16(acc_s → rP)
  10. gemm_rs(acc_o, rP, V_smem)   — MMA: [128,128] += [128,64] @ [64,128]
                                      Inner loop: 64/16=4 K-tiles
  11. Advance K/V pointers, repeat

=== Triton Gluon 实现 ===

Layout 映射 (NVIDIA Ampere, warpSize=32):
  - MMA instruction: m16n8k16 for fp16
  - For S=Q@K^T [128,64]: NVMMADistributedLayout(version=[2,0], warps_per_cta=[4,1], instr_shape=[16,8])
    → 4 warps tile M: each warp handles 128/4=32 rows (2 MMA tiles of 16 rows)
    → N=64: 64/8=8 MMA tiles along N
    → K reduction: 128/16=8 steps
  - For O=P@V [128,128]: same MMA layout
    → 4 warps tile M: 128/4=32 rows per warp
    → N=128: 128/8=16 MMA tiles along N
    → K reduction: 64/16=4 steps
  - SMEM for K: SwizzledSharedLayout(vec=8, per_phase=4, max_phase=2, order=[1,0])
    对应 Swizzle<3,3,3> → 128-byte swizzle for fp16, row-major
  - SMEM for V (transposed): NVMMASharedLayout for operand B (transposed=True)

Register 分布:
  - Q_rf: DotOperandLayout(operand_index=0, parent=mma_layout_S, k_width=8)
    每个 thread 持有 Q 矩阵的 fragment, 在 S gemm 中作为 operand A
  - acc_s: mma_layout_S 分布, [128,64] fp32
  - acc_o: mma_layout_O 分布, [128,128] fp32
"""

import math

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import (
    mma_v2,
    async_copy,
)

# ============================================================
# Layout 定义
# ============================================================

# --- 常量 ---
BLOCK_M = 128
BLOCK_N = 64
HEAD_DIM = 128
NUM_WARPS = 4  # Ampere: 4 warps * 32 threads = 128 threads


@gluon.constexpr_function
def make_mma_layout_S():
    """
    S = Q @ K^T 的 MMA accumulator layout.
    Shape: [BLOCK_M=128, BLOCK_N=64]
    mma.sync m16n8k16 fp16:
      - version [2,0] = Ampere mma.sync
      - warps_per_cta [4,1]: 4 warps 沿 M 分布
      - instr_shape [16,8]: 每条 MMA 指令计算 16x8 输出

    每个 warp 负责 128/4=32 行, 即 2 个 m16 tile.
    N=64 被分为 64/8=8 个 n8 tile, 每个 warp 处理所有 8 列 tile.
    总计每 warp: 2*8=16 个 MMA 指令(per K-step), K 有 128/16=8 steps.
    """
    return gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 8],
    )


@gluon.constexpr_function
def make_mma_layout_O():
    """
    O = P @ V 的 MMA accumulator layout.
    Shape: [BLOCK_M=128, HEAD_DIM=128]
    与 S 的 MMA 相同基础 layout, 但 N 维度更大 (128 vs 64).
      - warps_per_cta [4,1]: 4 warps 沿 M 分布
      - instr_shape [16,8]: m16n8k16

    每个 warp 负责 32 行, N=128 → 128/8=16 个 n8 tile.
    K reduction: BLOCK_N/16 = 64/16 = 4 steps.
    """
    return gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 8],
    )


@gluon.constexpr_function
def make_q_dot_layout(mma_layout):
    """
    Q 作为 S=Q@K^T 的 operand A (LHS).
    DotOperandLayout: operand_index=0, k_width=8 (fp16: 32bit/16bit=2, 但 mma k16 → 8 elements per thread)

    对应原始 kernel 的 tSrQ = thr_mma.partition_fragment_A(sQ)
    Q [128, 128] 分布到 threads:
      每个 thread 持有 Q fragment 用于 8 个 K-step 的 MMA.
    """
    return gl.DotOperandLayout(operand_index=0, parent=mma_layout, k_width=8)


@gluon.constexpr_function
def make_k_dot_layout(mma_layout):
    """
    K^T 作为 S=Q@K^T 的 operand B (RHS).
    从 SMEM 读取, DotOperandLayout operand_index=1.
    k_width=8: fp16, mma m16n8k16 → B operand 每 thread 8 elements along K.

    对应原始 kernel 的 tSrK = thr_mma.partition_fragment_B(sK)
    K [64, 128] 在 SMEM 中 row-major (行=N序列位置, 列=head_dim),
    MMA 需要将其视为 [K=128, N=64] 的转置形式.
    """
    return gl.DotOperandLayout(operand_index=1, parent=mma_layout, k_width=8)


@gluon.constexpr_function
def make_p_dot_layout(mma_layout_o):
    """
    P (softmax output) 作为 O=P@V 的 operand A.
    Shape [128, 64], fp16.
    DotOperandLayout operand_index=0, k_width=8.

    对应原始 kernel 的 rP (convert from acc_s fp32 → fp16).
    """
    return gl.DotOperandLayout(operand_index=0, parent=mma_layout_o, k_width=8)


@gluon.constexpr_function
def make_v_dot_layout(mma_layout_o):
    """
    V^T 作为 O=P@V 的 operand B.
    V 原始 shape [64, 128], 转置后 [128, 64] 作为 B operand.
    k_width=8.

    对应原始 kernel 的 tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle)
    SMEM layout: SmemLayoutVtransposedNoSwizzle = [HeadDim=128, BlockN=64] col-major.
    """
    return gl.DotOperandLayout(operand_index=1, parent=mma_layout_o, k_width=8)


@gluon.constexpr_function
def make_smem_k_layout():
    """
    K 的 SMEM layout: [BLOCK_N=64, HEAD_DIM=128] fp16, swizzled.

    原始 kernel: SmemLayoutKV = tile_to_shape(
        composition(Swizzle<3, 3, 3>{}, Layout<Shape<16, 64>, Stride<64, 1>>{}),
        Shape<64, 128>{})

    对应 128-byte swizzle (swizzle_byte_width=128):
      - 每行 128 fp16 = 256 bytes
      - Swizzle<3,3,3>: vec=8(2^3), per_phase=8(2^3), max_phase=8(2^3)
      - 但实际 tile atom 是 16x64, kBlockKSmem=64

    在 Gluon 中用 NVMMASharedLayout (自动推导 swizzle):
      swizzle_byte_width=128, element_bitwidth=16, rank=2, transposed=False
    """
    return gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=16,
        rank=2,
        transposed=False,
    )


@gluon.constexpr_function
def make_smem_v_layout():
    """
    V 的 SMEM layout: [BLOCK_N=64, HEAD_DIM=128] fp16, transposed for MMA operand B.

    原始 kernel: SmemLayoutVtransposedNoSwizzle = [HeadDim=128, BlockN=64]
    实际以 transposed 形式存储, MMA-B 直接读取.

    swizzle_byte_width=128, transposed=True:
      contiguous dim (after transpose) = BLOCK_N=64 → 64*2=128 bytes → 128-byte swizzle
    """
    return gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=16,
        rank=2,
        transposed=True,
    )


@gluon.constexpr_function
def make_load_layout():
    """
    Global memory load layout (coalesced blocked).
    用于 Q/K/V 从 GMEM 加载到 registers.

    原始 kernel 的 GmemTiledCopyQKV:
      - 128 threads (在 NVIDIA 上是 4*32=128)
      - 每 thread load 128-bit = 8 fp16
      - Layout: threads_per_warp=[32,1], warps=[4,1]
      - 每 thread 处理 size_per_thread=[1,8]

    对于 [BLOCK_N=64, HEAD_DIM=128]:
      - 64*128/128 threads/8 elems = 64 loads per thread (分多次)
      - 实际: 64 rows * 128 cols / (128 threads * 8 cols/thread) = 8 行/thread 每次
    """
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[32, 1],
        warps_per_cta=[4, 1],
        order=[1, 0],  # 列优先 coalesced
    )


@gluon.constexpr_function
def make_store_layout():
    """
    Output store layout. 与 load_layout 相同结构.
    128 threads, 每 thread store 8 fp16 per trip.
    """
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[32, 1],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )


# ============================================================
# Kernel 实现
# ============================================================

@gluon.jit
def flash_attn_fwd_inner(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb: gl.constexpr, stride_qh: gl.constexpr, stride_qm: gl.constexpr,
    stride_kb: gl.constexpr, stride_kh: gl.constexpr, stride_kn: gl.constexpr,
    stride_vb: gl.constexpr, stride_vh: gl.constexpr, stride_vn: gl.constexpr,
    stride_ob: gl.constexpr, stride_oh: gl.constexpr, stride_om: gl.constexpr,
    seqlen_q: gl.constexpr, seqlen_k: gl.constexpr,
    scale: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    nheads_q: gl.constexpr,
    nheads_k: gl.constexpr,
    DTYPE: gl.constexpr,
):
    """
    Flash Attention Forward kernel (single CTA).
    Grid: (ceil(seqlen_q / BLOCK_M), batch * nheads_q)

    计算调度:
    =========
    Phase 1: Load Q → SMEM → RF (常驻)
    Phase 2: 逆序遍历 N blocks:
      for n_block = n_block_max-1 ... 0:
        2a. Load K[n_block] → RF → SMEM
        2b. GEMM: acc_s[128,64] = Q_rf[128,128] @ K_smem[64,128]^T
            - 8 个 K-reduction steps (128/16=8)
            - 每步: 4 warps 并行, 每 warp 2*8=16 条 mma.sync m16n8k16
        2c. Causal mask + scale
        2d. Online softmax: m_new, l_new, rescale acc_o
        2e. Load V[n_block] → RF → SMEM (transposed)
        2f. Convert P = softmax(S) → fp16
        2g. GEMM: acc_o[128,128] += P[128,64] @ V_smem[64,128]
            - 4 个 K-reduction steps (64/16=4)
            - 每步: 4 warps 并行, 每 warp 2*16=32 条 mma.sync m16n8k16
    Phase 3: Normalize acc_o, write O and LSE
    """
    # --- Program ID ---
    pid_m = gl.program_id(0)
    pid_bh = gl.program_id(1)

    # Batch/head decomposition
    h_ratio = nheads_q // nheads_k  # GQA ratio
    pid_batch = pid_bh // nheads_q
    pid_head_q = pid_bh % nheads_q
    pid_head_k = pid_head_q // h_ratio

    m_start = pid_m * BLOCK_M

    # --- Layout instances ---
    mma_layout_s: gl.constexpr = make_mma_layout_S()
    mma_layout_o: gl.constexpr = make_mma_layout_O()
    q_dot_layout: gl.constexpr = make_q_dot_layout(mma_layout_s)
    k_dot_layout: gl.constexpr = make_k_dot_layout(mma_layout_s)
    p_dot_layout: gl.constexpr = make_p_dot_layout(mma_layout_o)
    v_dot_layout: gl.constexpr = make_v_dot_layout(mma_layout_o)
    load_layout: gl.constexpr = make_load_layout()
    smem_k_layout: gl.constexpr = make_smem_k_layout()
    smem_v_layout: gl.constexpr = make_smem_v_layout()

    # --- Allocate SMEM ---
    # sQ: [BLOCK_M, HEAD_DIM] — Q 暂存，加载后读入 RF
    # sK: [HEAD_DIM, BLOCK_N] — K 转置形式暂存（K^T for Q@K^T MMA）
    # sV: [BLOCK_N, HEAD_DIM] — V 以 transposed layout 暂存（P@V gemm 的 operand B）
    smem_q = gl.allocate_shared_memory(DTYPE, [BLOCK_M, HEAD_DIM], smem_k_layout)
    smem_k = gl.allocate_shared_memory(DTYPE, [HEAD_DIM, BLOCK_N], smem_v_layout)
    smem_v = gl.allocate_shared_memory(DTYPE, [BLOCK_N, HEAD_DIM], smem_v_layout)

    # load_layout_kv: 用于加载 [BLOCK_N, HEAD_DIM] 或 [HEAD_DIM, BLOCK_N] 形状的数据
    # load_layout 是 [threads_per_warp=[32,1], warps=[4,1], spt=[1,8]]
    # 适用于列数为 HEAD_DIM=128（32*1*... 列）的矩阵；
    # 对于 [HEAD_DIM=128, BLOCK_N=64]，行=128，列=64，
    # 需要 [threads_per_warp=[8,4], warps=[4,1], spt=[1,8]]:
    #   每 warp 8 thread × 4 thread = 32 threads，8 列/thread → 32 cols/warp（需要 2 次覆盖 64 列）
    # 为简便，直接用 blocked layout [spt=[1,8], tpw=[8,4], wpg=[4,1]]:
    #   cols: 8*8*1=64 ✓  rows: 1*4*4=16, 128/16=8 trips → 8*16=128 rows ✓
    load_layout_kT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 8],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    # --- Phase 1: Load Q into registers ---
    # Q shape: [BLOCK_M, HEAD_DIM] = [128, 128]
    # 使用 blocked load layout: 128 threads, 每 thread 8 fp16/trip
    # 需要 128*128/(128*8) = 16 trips → 全部加载到 SMEM 再读到 RF
    q_offset = pid_batch * stride_qb + pid_head_q * stride_qh + m_start * stride_qm
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=load_layout))
    offs_k = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(dim=0, parent=load_layout))
    # expand_dims: offs_m [M] → [M,1], offs_k [D] → [1,D]，然后相加得到 [M,D] 指针矩阵
    q_ptrs = Q_ptr + q_offset + gl.expand_dims(offs_m, 1) * stride_qm + gl.expand_dims(offs_k, 0)

    # Load Q → SMEM
    q_mask = gl.expand_dims((m_start + offs_m) < seqlen_q, 1)
    q_vals = gl.load(q_ptrs, mask=q_mask)
    smem_q.store(q_vals)
    gl.thread_barrier()

    # Load Q from SMEM → distributed RF in MMA-A layout (常驻整个 kernel)
    q_rf = smem_q.load(q_dot_layout)

    # --- Initialize accumulators ---
    # acc_o: [BLOCK_M, HEAD_DIM] fp32, MMA layout
    acc_o = gl.zeros((BLOCK_M, HEAD_DIM), dtype=gl.float32, layout=mma_layout_o)
    # Online softmax state: per-row max and sum
    # 每 warp 处理 32 行, 每 thread 在 MMA 中负责若干行
    # 用 SliceLayout 从 mma_layout_o 取行维度
    row_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout_o)
    # 初始化为大负数而非 -inf，避免 -inf - (-inf) = NaN 的情况
    # (当整个 n_block 都被 causal mask 掉时)
    m_i = gl.full((BLOCK_M,), -1e6, dtype=gl.float32, layout=row_layout)
    l_i = gl.zeros((BLOCK_M,), dtype=gl.float32, layout=row_layout)

    # --- N-block iteration range ---
    n_block_max = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        n_block_max = gl.minimum(
            n_block_max,
            (m_start + BLOCK_M + seqlen_k - seqlen_q + BLOCK_N - 1) // BLOCK_N
        )

    # --- Phase 2: Main loop (reverse iteration) ---
    k_offset_base = pid_batch * stride_kb + pid_head_k * stride_kh
    v_offset_base = pid_batch * stride_vb + pid_head_k * stride_vh

    # offs_n / offs_d 用于 V 加载（[BLOCK_N, HEAD_DIM] 形状）
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=load_layout))
    offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(dim=0, parent=load_layout))
    # offs_d_row / offs_n_col 用于 K 转置加载（[HEAD_DIM, BLOCK_N] 形状）
    offs_d_row = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(dim=1, parent=load_layout_kT))
    offs_n_col = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=load_layout_kT))

    scale_log2 = scale * 1.44269504  # log2(e) * scale

    # --- Phase 2: N-block loop, split in two ---
    # Phase A (last block, index n_block_max-1): may be a partial block when
    # seqlen_k is not a multiple of BLOCK_N → apply K-boundary mask.
    # Phase B (blocks n_block_max-2 .. 0): always full → no boundary mask.
    # This avoids a runtime branch inside the hot loop.

    # ── Phase A: last N-block ─────────────────────────────────────────────
    n_block = n_block_max - 1
    n_start = n_block * BLOCK_N

    k_ptrs = K_ptr + k_offset_base + gl.expand_dims(offs_d_row, 1) + gl.expand_dims(n_start + offs_n_col, 0) * stride_kn
    k_mask = gl.expand_dims((n_start + offs_n_col) < seqlen_k, 0)
    k_vals = gl.load(k_ptrs, mask=k_mask)
    smem_k.store(k_vals)
    gl.thread_barrier()

    k_from_smem = smem_k.load(k_dot_layout)
    acc_s = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout_s)
    acc_s = mma_v2(q_rf, k_from_smem, acc_s)
    acc_s = acc_s * scale

    col_idx = n_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=mma_layout_s))
    # K-boundary mask: out-of-bounds columns were zero-padded by gl.load, not -inf
    kbound_mask = col_idx[None, :] < seqlen_k
    kbound_mask = gl.convert_layout(kbound_mask, mma_layout_s)
    acc_s = gl.where(kbound_mask, acc_s, float("-inf"))

    if IS_CAUSAL:
        row_idx = m_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=mma_layout_s))
        causal_mask = row_idx[:, None] - (seqlen_q - seqlen_k) >= col_idx[None, :]
        causal_mask = gl.convert_layout(causal_mask, mma_layout_s)
        acc_s = gl.where(causal_mask, acc_s, float("-inf"))

    row_max = gl.max(acc_s, axis=1)
    m_new = gl.maximum(m_i, row_max)
    alpha = gl.exp2((m_i - m_new) * 1.44269504)
    acc_o = acc_o * alpha[:, None]
    l_i = l_i * alpha
    p = gl.exp2((acc_s - m_new[:, None]) * 1.44269504)
    l_i = l_i + gl.sum(p, axis=1)
    m_i = m_new

    v_ptrs = V_ptr + v_offset_base + gl.expand_dims(n_start + offs_n, 1) * stride_vn + gl.expand_dims(offs_d, 0)
    v_mask = gl.expand_dims((n_start + offs_n) < seqlen_k, 1)
    v_vals = gl.load(v_ptrs, mask=v_mask)
    smem_v.store(v_vals)
    gl.thread_barrier()

    p_fp16 = gl.cast(p, DTYPE)
    p_a = gl.convert_layout(p_fp16, p_dot_layout)
    v_from_smem = smem_v.load(v_dot_layout)
    acc_o = mma_v2(p_a, v_from_smem, acc_o)
    gl.thread_barrier()

    # ── Phase B: remaining full N-blocks ──────────────────────────────────
    for n_block in range(n_block_max - 2, -1, -1):
        n_start = n_block * BLOCK_N

        k_ptrs = K_ptr + k_offset_base + gl.expand_dims(offs_d_row, 1) + gl.expand_dims(n_start + offs_n_col, 0) * stride_kn
        k_vals = gl.load(k_ptrs)  # full block, no boundary mask
        smem_k.store(k_vals)
        gl.thread_barrier()

        k_from_smem = smem_k.load(k_dot_layout)
        acc_s = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout_s)
        acc_s = mma_v2(q_rf, k_from_smem, acc_s)
        acc_s = acc_s * scale

        if IS_CAUSAL:
            col_idx = n_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=mma_layout_s))
            row_idx = m_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=mma_layout_s))
            causal_mask = row_idx[:, None] - (seqlen_q - seqlen_k) >= col_idx[None, :]
            causal_mask = gl.convert_layout(causal_mask, mma_layout_s)
            acc_s = gl.where(causal_mask, acc_s, float("-inf"))

        row_max = gl.max(acc_s, axis=1)
        m_new = gl.maximum(m_i, row_max)
        alpha = gl.exp2((m_i - m_new) * 1.44269504)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha
        p = gl.exp2((acc_s - m_new[:, None]) * 1.44269504)
        l_i = l_i + gl.sum(p, axis=1)
        m_i = m_new

        v_ptrs = V_ptr + v_offset_base + gl.expand_dims(n_start + offs_n, 1) * stride_vn + gl.expand_dims(offs_d, 0)
        v_vals = gl.load(v_ptrs)  # full block, no boundary mask
        smem_v.store(v_vals)
        gl.thread_barrier()

        p_fp16 = gl.cast(p, DTYPE)
        p_a = gl.convert_layout(p_fp16, p_dot_layout)
        v_from_smem = smem_v.load(v_dot_layout)
        acc_o = mma_v2(p_a, v_from_smem, acc_o)
        gl.thread_barrier()
    # --- Phase 3: Epilogue ---
    # Normalize: O = acc_o / l_i
    l_inv = gl.fdiv(1.0, l_i)
    acc_o = acc_o * l_inv[:, None]

    # LSE = m_i + log(l_i) (log2 scale already applied via scale_log2)
    lse = m_i + gl.log2(l_i) / 1.44269504  # convert back from log2 to ln

    # Convert to output dtype and store
    o_fp16 = gl.cast(acc_o, DTYPE)

    # Store O
    o_offset = pid_batch * stride_ob + pid_head_q * stride_oh + m_start * stride_om
    o_ptrs = O_ptr + o_offset + gl.expand_dims(offs_m, 1) * stride_om + gl.expand_dims(offs_k, 0)
    o_store_layout: gl.constexpr = make_store_layout()
    o_out = gl.convert_layout(o_fp16, o_store_layout)
    o_mask = gl.expand_dims((m_start + offs_m) < seqlen_q, 1)
    gl.store(o_ptrs, o_out, mask=o_mask)

    # Store LSE
    lse_offset = pid_batch * (nheads_q * seqlen_q) + pid_head_q * seqlen_q + m_start
    lse_ptrs = LSE_ptr + lse_offset + gl.arange(0, BLOCK_M, layout=row_layout)
    lse_mask = (m_start + gl.arange(0, BLOCK_M, layout=row_layout)) < seqlen_q
    gl.store(lse_ptrs, lse, mask=lse_mask)


# ============================================================
# Host wrapper
# ============================================================

def flash_attn_fwd(q, k, v, causal=False, scale=None):
    """
    Flash Attention Forward.

    Args:
        q: [batch, seqlen_q, nheads_q, head_dim] fp16 / bf16
        k: [batch, seqlen_k, nheads_k, head_dim] fp16 / bf16
        v: [batch, seqlen_k, nheads_k, head_dim] fp16 / bf16
        causal: bool
        scale: float, default 1/sqrt(head_dim)

    Returns:
        o: [batch, seqlen_q, nheads_q, head_dim] same dtype as q
        lse: [batch, nheads_q, seqlen_q] fp32
    """
    batch, seqlen_q_val, nheads_q_val, head_dim = q.shape
    _, seqlen_k_val, nheads_k_val, _ = k.shape
    assert head_dim == HEAD_DIM, f"This kernel only supports head_dim={HEAD_DIM}"

    if scale is None:
        scale = head_dim ** -0.5

    # Reshape to [batch, nheads, seqlen, head_dim] for stride computation
    q = q.transpose(1, 2).contiguous()  # [B, H_q, M, D]
    k = k.transpose(1, 2).contiguous()  # [B, H_k, N, D]
    v = v.transpose(1, 2).contiguous()  # [B, H_k, N, D]

    o = torch.empty_like(q)
    lse = torch.empty(batch, nheads_q_val, seqlen_q_val, dtype=torch.float32, device=q.device)

    grid = (
        (seqlen_q_val + BLOCK_M - 1) // BLOCK_M,
        batch * nheads_q_val,
    )

    flash_attn_fwd_inner[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2),   # stride_qb, stride_qh, stride_qm
        k.stride(0), k.stride(1), k.stride(2),   # stride_kb, stride_kh, stride_kn
        v.stride(0), v.stride(1), v.stride(2),   # stride_vb, stride_vh, stride_vn
        o.stride(0), o.stride(1), o.stride(2),   # stride_ob, stride_oh, stride_om
        seqlen_q_val, seqlen_k_val,
        scale,
        causal,
        BLOCK_M, BLOCK_N, HEAD_DIM,
        nheads_q_val, nheads_k_val,
        gl.float16 if q.dtype == torch.float16 else gl.bfloat16,
        num_warps=NUM_WARPS,
        num_stages=2,
    )

    o = o.transpose(1, 2)  # back to [B, M, H, D]
    return o, lse


# ============================================================
# flash_attention_forward-compatible public interface
# ============================================================

def flash_attention_forward(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    softcap=0.0,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
    disable_splitkv=False,
):
    """Drop-in replacement for flag_gems.flash_attention_forward.

    Implements the same call signature and 5-tuple return value
    (out, lse, philox_seed, philox_offset, debug_attn_mask) so it
    can be swapped in transparently wherever flag_gems or
    torch.ops.aten._flash_attention_forward is used.

    This kernel's supported subset:
      - head_dim == 128  (hard constraint of the Triton Gluon implementation)
      - Non-varlen: cumulative_sequence_length_q/k must be None
      - No dropout:  dropout_p must be 0.0
      - No alibi slopes
      - No sliding window attention
      - No softcap
      - GQA (nheads_q != nheads_k) is fully supported

    Inputs follow the flash-attention convention:
      query  : [batch, seqlen_q, nheads_q, head_dim]  fp16 / bf16
      key    : [batch, seqlen_k, nheads_k, head_dim]
      value  : [batch, seqlen_k, nheads_k, head_dim]
    """
    if cumulative_sequence_length_q is not None or cumulative_sequence_length_k is not None:
        raise NotImplementedError("varlen is not supported by this kernel")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported by this kernel")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi slopes are not supported by this kernel")
    if window_size_left is not None or window_size_right is not None:
        raise NotImplementedError("sliding window attention is not supported by this kernel")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported by this kernel")

    head_dim = query.shape[-1]
    if head_dim != HEAD_DIM:
        raise NotImplementedError(
            f"This kernel only supports head_dim={HEAD_DIM}, got {head_dim}"
        )

    if scale is None:
        scale = head_dim ** -0.5

    out, lse = flash_attn_fwd(query, key, value, causal=is_causal, scale=scale)

    # Dummy philox tensors (no dropout → no random state)
    philox_seed = torch.zeros(2, dtype=torch.int64, device=query.device)
    philox_offset = torch.zeros((), dtype=torch.int64, device=query.device)
    # Empty debug mask (return_debug_mask not supported)
    debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    return out, lse, philox_seed, philox_offset, debug_mask
