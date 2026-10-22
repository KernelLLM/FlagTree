"""TLE/DSA implementation of commonir_op_sparse_attn_sharedkv.

SparseAttnSharedKV (SWA / CFA / SCFA) on Ascend910_9382.

Semantics follow the shipped reference exactly:
  * shared KV: an ori_kv / cmp_kv entry is used as both key and value;
  * query position  p = kv_len - q_len + t ;
  * window = logical positions [p - 127, p] clamped to [0, kv_len);
  * CFA adds the contiguous compressed prefix [0, (p + 1) // cmp_ratio);
  * SCFA adds the compressed positions named by cmp_sparse_indices[t, 0, :],
    keeping only 0 <= idx < (p + 1) // cmp_ratio ;
  * one softmax spans sink + window + compressed.

Kernels
-------
_depage_kernel   pure TLE: dsa.copy moves every PA_ND page into a dense
                 [logical_token, HEAD_DIM] buffer -- the operator's bulk,
                 in-bounds data movement.
_gather_kernel   materialises the compressed rows named by cmp_sparse_indices
                 for one query chunk into a dense [chunk, topk, HEAD_DIM]
                 buffer.  A gathered tile that feeds tl.dot is miscompiled or
                 faults on this toolchain, so the gather is kept away from every
                 matmul and the attention kernel reads the result affinely.
_attn_kernel     flash-attention style: one program per (query token, head
                 block); HEAD_BLOCK query heads share the single KV head, and
                 the visible set is walked in N chunks with an online softmax so
                 the [HEAD_BLOCK, HEAD_DIM] fp32 accumulator is the only large
                 resident buffer.

Three toolchain constraints drive the shapes:
  * A clamped (tl.where) *index* tensor used as a row base makes BiShengIR
    allocate a dynamic gather, which hivm rejects ("Unknown core type:
    llvm.func @malloc"), so both dense buffers carry one page of front padding
    and out-of-range rows are removed by masking scores instead.
  * A gathered tile feeding tl.dot is miscompiled, hence _gather_kernel.
  * A single launch spanning 8192 query rows (512 MiB of q and of out) faults
    the MTE engine, so the query axis is walked in _QUERY_CHUNK-token launches
    over pointer-offset views.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

Q_HEADS = 64
KV_HEADS = 1
HEAD_DIM = 512
PAGE_SIZE = 128

# 32 heads per program fits the UB budget for the sliding-window (SWA) path.
# The compressed paths (CFA / SCFA) additionally keep the cmp chunk tiles live
# and then overflow it: `ub overflow, requires 2230272 bits while 1572864
# available` (272 KiB vs 192 KiB), so they could not be compiled at all.  At 16
# they build and match the counterpart to one bf16 ULP; the accumulator
# [HEAD_BLOCK, 512] fp32 is what dominates, so halving the head block is what
# buys the room.  Kept shape-dependent so SWA keeps its tuned 32.
_HEAD_BLOCK = 32
_HEAD_BLOCK_CMP = 16
# width of the head_dim slice each window QK dot contracts over
_N_CHUNK = 128
_CMP_CHUNK = 128
_TOPK = 512
_DEPAGE_CHUNK = 16384
_QUERY_CHUNK = 4096
_SPARSE_CHUNK = 1024


@triton.jit
def _depage_kernel(
    ori_kv_ptr,
    ori_bt_ptr,
    ori_out_ptr,
    cmp_kv_ptr,
    cmp_bt_ptr,
    cmp_out_ptr,
    ori_np,
    cmp_np,
    HAS_CMP: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Copy every logical PA_ND page of ori_kv / cmp_kv into a dense buffer.

    One program per (logical page, batch).  A physical page is a contiguous
    PAGE_SIZE x D block, so the transfer is an in-bounds bulk move executed in
    CHUNK-element dsa.copy steps through one pre-allocated UB buffer.
    """
    p = tl.program_id(0)
    b = tl.program_id(1)

    buf = tle.dsa.alloc([CHUNK], tl.bfloat16, tle.dsa.ascend.UB)
    phys = tl.load(ori_bt_ptr + b * ori_np + p)
    for c in tl.static_range(0, (128 * D) // CHUNK):
        off = c * CHUNK + tl.arange(0, CHUNK)
        tle.dsa.copy(ori_kv_ptr + phys * (128 * D) + off, buf, [CHUNK])
        tle.dsa.copy(buf, ori_out_ptr + (b * ori_np * 128 + p * 128) * D + off, [CHUNK])
        if HAS_CMP:
            if p < cmp_np:
                cphys = tl.load(cmp_bt_ptr + b * cmp_np + p)
                tle.dsa.copy(cmp_kv_ptr + cphys * (128 * D) + off, buf, [CHUNK])
                tle.dsa.copy(buf, cmp_out_ptr + (b * cmp_np * 128 + p * 128) * D + off,
                             [CHUNK])


@triton.jit
def _gather_kernel(
    cmp_ptr,
    sparse_ptr,
    cu_ptr,
    seqused_ptr,
    kg_ptr,
    row0,
    cmp_bs,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    CMP_RATIO: tl.constexpr,
):
    """kg[r, i, :] = cmp_dense[idx[row0 + r, i], :] for the rows of this chunk."""
    b = tl.program_id(0)
    r = tl.program_id(1)

    q_start = tl.load(cu_ptr + b)
    q_len = tl.load(cu_ptr + b + 1) - q_start
    if r >= q_len:
        return
    kv_len = tl.load(seqused_ptr + b)
    thr = (kv_len - q_len + r + 1) // CMP_RATIO
    g = row0 + r

    offs = tl.arange(0, D)
    for c in tl.static_range(0, TOPK // CHUNK):
        i0 = c * CHUNK + tl.arange(0, CHUNK)
        idx = tl.load(sparse_ptr + g * TOPK + i0)
        k = tl.load(cmp_ptr + (b * cmp_bs + idx)[:, None] * D + offs[None, :])
        tl.store(kg_ptr + (r * TOPK + i0)[:, None] * D + offs[None, :], k)


@triton.jit
def _attn_kernel(
    q_ptr,
    out_ptr,
    ori_ptr,
    cmp_ptr,
    sparse_ptr,
    cu_ptr,
    seqused_ptr,
    sink_ptr,
    scale,
    ori_batch_stride,
    cmp_batch_stride,
    cmp_t_stride,
    sparse_row0,
    D: tl.constexpr,
    QH: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    N_ORI: tl.constexpr,
    N_CHUNK: tl.constexpr,
    CMP_CHUNK: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_CMP: tl.constexpr,
    HAS_SPARSE: tl.constexpr,
    CMP_RATIO: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    hb = tl.program_id(2)

    q_start = tl.load(cu_ptr + b)
    q_len = tl.load(cu_ptr + b + 1) - q_start
    if t >= q_len:
        return
    kv_len = tl.load(seqused_ptr + b)
    p_pos = kv_len - q_len + t
    g = q_start + t

    h_offs = hb * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    hq = tl.arange(0, HEAD_BLOCK)
    d_offs = tl.arange(0, D)
    q_off = hq[:, None] * D + d_offs[None, :]
    q_base = q_ptr + g * (QH * D) + hb * (HEAD_BLOCK * D)

    sink = tl.load(sink_ptr + h_offs).to(tl.float32)

    ori_row0 = b * ori_batch_stride
    m_i = tl.full([HEAD_BLOCK], -1.0e30, tl.float32)
    l_i = tl.zeros([HEAD_BLOCK], tl.float32)
    acc = tl.zeros([HEAD_BLOCK, D], tl.float32)

    # ---- compressed contribution (CMP_CHUNK-wide chunks, one rescale each) ----
    if HAS_CMP:
        cmp_row0 = b * cmp_batch_stride + t * cmp_t_stride
        thr = (p_pos + 1) // CMP_RATIO
        if HAS_SPARSE:
            n_cmp = TOPK // CMP_CHUNK
        else:
            n_cmp = tl.cdiv(thr, CMP_CHUNK)
        for c in range(0, n_cmp):
            ii = c * CMP_CHUNK + tl.arange(0, CMP_CHUNK)
            if HAS_SPARSE:
                idx = tl.load(sparse_ptr + (sparse_row0 + t) * TOPK + ii)
                ok = (idx >= 0) & (idx < thr)
            else:
                ok = ii < thr
            crows = (cmp_row0 + ii)[:, None] * D
            s = tl.zeros([HEAD_BLOCK, CMP_CHUNK], tl.float32)
            for dc in tl.static_range(0, D // N_CHUNK):
                dd = dc * N_CHUNK + tl.arange(0, N_CHUNK)
                qc = tl.load(q_base + hq[:, None] * D + dd[None, :])
                kc = tl.load(cmp_ptr + crows + dd[None, :])
                s += tl.dot(qc, tl.trans(kc))
            s = tl.where(ok[None, :], s * scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            kk = tl.load(cmp_ptr + crows + d_offs[None, :])
            acc = tl.dot(p.to(tl.bfloat16), kk, acc * alpha[:, None])
            m_i = m_new

    # ---- sliding window: one N_ORI-wide pass.  The whole row of window scores
    # is built once (QK accumulated over head_dim slices), reduced once, and the
    # accumulator is rescaled once, instead of once per 32-wide step.
    jj = tl.arange(0, N_ORI)
    opos = p_pos - (N_ORI - 1) + jj
    ok = (opos >= 0) & (opos < kv_len)
    ori_rows = (ori_row0 + opos)[:, None] * D
    s = tl.zeros([HEAD_BLOCK, N_ORI], tl.float32)
    for dc in tl.static_range(0, D // N_CHUNK):
        dd = dc * N_CHUNK + tl.arange(0, N_CHUNK)
        qc = tl.load(q_base + hq[:, None] * D + dd[None, :])
        kc = tl.load(ori_ptr + ori_rows + dd[None, :])
        s += tl.dot(qc, tl.trans(kc))
    s = tl.where(ok[None, :], s * scale, float("-inf"))
    m_f = tl.maximum(tl.maximum(m_i, tl.max(s, 1)), sink)
    alpha = tl.exp(m_i - m_f)
    p = tl.exp(s - m_f[:, None])
    denom = l_i * alpha + tl.sum(p, 1) + tl.exp(sink - m_f)
    kw = tl.load(ori_ptr + ori_rows + d_offs[None, :])
    acc = tl.dot(p.to(tl.bfloat16), kw, acc * alpha[:, None])

    res = (acc / denom[:, None]).to(tl.bfloat16)
    tl.store(out_ptr + g * (QH * D) + hb * (HEAD_BLOCK * D) + q_off, res)


def run(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_kv=None,
    sinks=None,
    softmax_scale=0.04419417,
    cmp_ratio=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
):
    if layout_q != "TND" or layout_kv != "PA_ND":
        raise NotImplementedError("only layout_q=TND and layout_kv=PA_ND are supported")
    if ori_mask_mode != 4 or cmp_mask_mode != 3:
        raise ValueError("ori_mask_mode and cmp_mask_mode must be 4 and 3")
    if ori_win_left != 127 or ori_win_right != 0:
        raise ValueError("ori_win_left and ori_win_right must be 127 and 0")
    if q.dim() != 3 or tuple(q.shape[1:]) != (Q_HEADS, HEAD_DIM):
        raise ValueError("q must have shape [T, 64, 512]")

    device = q.device
    total_q = int(q.shape[0])
    q_heads = int(q.shape[1])
    out_dtype = q.dtype
    out = torch.empty((total_q, q_heads, HEAD_DIM), device=device, dtype=torch.bfloat16)
    empty_lse = torch.empty((0,), dtype=torch.float32, device=device)

    batch = int(cu_seqlens_q.numel()) - 1
    if batch <= 0:
        return out.to(out_dtype), empty_lse

    ratio = int(cmp_ratio)
    has_cmp = (cmp_kv is not None) and ratio > 0
    has_sparse = has_cmp and (cmp_sparse_indices is not None)

    if sinks is None:
        sinks = torch.zeros((Q_HEADS,), device=device, dtype=torch.float32)
    if ori_block_table is None:
        ori_block_table = torch.zeros((batch, 1), device=device, dtype=torch.int32)

    def _as_bf16(x):
        return x if x is None or x.dtype == torch.bfloat16 else x.to(torch.bfloat16)

    qk = _as_bf16(q)
    okv = _as_bf16(ori_kv)
    ckv = _as_bf16(cmp_kv)

    ori_np = int(ori_block_table.shape[1])
    cmp_np = int(cmp_block_table.shape[1]) if cmp_block_table is not None else 1

    # PA_ND -> dense [logical token, HEAD_DIM].  One page of front pad per dense
    # buffer keeps every window start and every -1 sparse index affine.
    ori_buf = torch.zeros(
        (PAGE_SIZE + batch * ori_np * PAGE_SIZE, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    ori_view = ori_buf[PAGE_SIZE:]
    if has_cmp:
        cmp_buf = torch.zeros(
            (PAGE_SIZE + batch * cmp_np * PAGE_SIZE + _CMP_CHUNK, HEAD_DIM),
            device=device,
            dtype=torch.bfloat16,
        )
        cmp_view = cmp_buf[PAGE_SIZE:]
    else:
        cmp_view = ori_view

    _depage_kernel[(ori_np, batch)](
        okv,
        ori_block_table,
        ori_view,
        ckv if has_cmp else okv,
        cmp_block_table if has_cmp else ori_block_table,
        cmp_view,
        ori_np,
        cmp_np,
        HAS_CMP=has_cmp,
        D=HEAD_DIM,
        CHUNK=_DEPAGE_CHUNK,
    )

    sparse_ptr = cmp_sparse_indices if has_sparse else qk
    scale = float(softmax_scale)
    head_block = _HEAD_BLOCK_CMP if has_cmp else _HEAD_BLOCK
    hb_blocks = q_heads // head_block
    ori_stride = ori_np * PAGE_SIZE

    if has_sparse:
        _run_sparse(
            qk, out, ori_view, cmp_view, sparse_ptr, cu_seqlens_q, seqused_kv,
            sinks, scale, ori_stride, cmp_np * PAGE_SIZE, ratio, batch, total_q,
            hb_blocks,
        )
    elif total_q <= _QUERY_CHUNK:
        _attn_kernel[(batch, total_q, hb_blocks)](
            qk, out, ori_view, cmp_view, sparse_ptr, cu_seqlens_q, seqused_kv,
            sinks, scale, ori_stride, cmp_np * PAGE_SIZE, 0, 0,
            D=HEAD_DIM, QH=q_heads, HEAD_BLOCK=head_block, N_ORI=PAGE_SIZE,
            N_CHUNK=_N_CHUNK, CMP_CHUNK=_CMP_CHUNK, TOPK=_TOPK, HAS_CMP=has_cmp,
            HAS_SPARSE=False, CMP_RATIO=ratio if ratio > 0 else 1,
        )
    else:
        cu_host = [int(v) for v in cu_seqlens_q.detach().to("cpu").tolist()]
        kv_host = [int(v) for v in seqused_kv.detach().to("cpu").tolist()]
        slices = []
        for b in range(batch):
            q_start, q_end = cu_host[b], cu_host[b + 1]
            q_len, kvl = q_end - q_start, kv_host[b]
            for lo in range(0, q_len, _QUERY_CHUNK):
                n = min(_QUERY_CHUNK, q_len - lo)
                slices.append((q_start + lo, n, kvl - q_len + lo + n))
        cu_t = torch.zeros((len(slices), 2), dtype=torch.int32, device=device)
        seq_t = torch.zeros((len(slices),), dtype=torch.int32, device=device)
        for i, (_, n, s_used) in enumerate(slices):
            cu_t[i, 1] = n
            seq_t[i] = s_used
        for i, (start, n, _) in enumerate(slices):
            _attn_kernel[(1, n, hb_blocks)](
                qk[start:start + n], out[start:start + n], ori_view, cmp_view,
                qk, cu_t[i], seq_t[i:i + 1], sinks, scale, ori_stride,
                cmp_np * PAGE_SIZE, 0, 0,
                D=HEAD_DIM, QH=q_heads, HEAD_BLOCK=head_block, N_ORI=PAGE_SIZE,
                N_CHUNK=_N_CHUNK, CMP_CHUNK=_CMP_CHUNK, TOPK=_TOPK, HAS_CMP=has_cmp,
                HAS_SPARSE=False, CMP_RATIO=ratio if ratio > 0 else 1,
            )

    if out_dtype != torch.bfloat16:
        out = out.to(out_dtype)
    return out, empty_lse


def _run_sparse(
    qk, out, ori_view, cmp_view, sparse_ptr, cu_seqlens_q, seqused_kv, sinks,
    scale, ori_stride, cmp_stride, ratio, batch, total_q, hb_blocks,
):
    """SCFA: pre-gather the named compressed rows, then run the attention kernel."""
    device = qk.device
    simple = (batch == 1) and (total_q <= _SPARSE_CHUNK)

    if simple:
        chunks = [(0, total_q, 0, cu_seqlens_q, seqused_kv)]
    else:
        cu_host = [int(v) for v in cu_seqlens_q.detach().to("cpu").tolist()]
        kv_host = [int(v) for v in seqused_kv.detach().to("cpu").tolist()]
        plain = []
        for b in range(batch):
            q_start, q_end = cu_host[b], cu_host[b + 1]
            q_len, kvl = q_end - q_start, kv_host[b]
            for lo in range(0, q_len, _SPARSE_CHUNK):
                n = min(_SPARSE_CHUNK, q_len - lo)
                plain.append((q_start + lo, n, kvl - q_len + lo + n, q_start + lo))
        cu_t = torch.zeros((len(plain), 2), dtype=torch.int32, device=device)
        seq_t = torch.zeros((len(plain),), dtype=torch.int32, device=device)
        for i, (_, n, s_used, _) in enumerate(plain):
            cu_t[i, 1] = n
            seq_t[i] = s_used
        chunks = [(start, n, row0, cu_t[i], seq_t[i:i + 1])
                  for i, (start, n, _, row0) in enumerate(plain)]

    for start, n, row0, cu1, seq1 in chunks:
        kg = torch.empty((n, _TOPK, HEAD_DIM), device=device, dtype=torch.bfloat16)
        _gather_kernel[(1, n)](
            cmp_view, sparse_ptr, cu1, seq1, kg, row0, cmp_stride,
            D=HEAD_DIM, TOPK=_TOPK, CHUNK=_CMP_CHUNK,
            CMP_RATIO=ratio,
        )
        _attn_kernel[(1, n, hb_blocks)](
            qk[start:start + n], out[start:start + n], ori_view, kg, sparse_ptr,
            cu1, seq1, sinks, scale, ori_stride, 0, _TOPK, row0,
            D=HEAD_DIM, QH=qk.shape[1], HEAD_BLOCK=_HEAD_BLOCK_CMP,
            N_ORI=PAGE_SIZE,
            N_CHUNK=_N_CHUNK, CMP_CHUNK=_CMP_CHUNK, TOPK=_TOPK, HAS_CMP=True,
            HAS_SPARSE=True, CMP_RATIO=ratio,
        )
