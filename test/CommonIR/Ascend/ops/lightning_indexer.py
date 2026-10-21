"""Lightning Indexer (TND / PA_BSND / sparse_mode=3) implemented with TLE.

Reference semantics reproduced exactly:

    per request r with cumulative query range [qb, qe) and key length K:
        active(t) = K - (qe - t) + 1          # causal cutoff for token t
        if active(t) <= 0                -> row stays -1
        elif active(t) <= sparse_count   -> row = arange(active(t)), tail stays -1
        else -> scores[n] = sum_h relu(q[t,h,:] . k[n,:]) * w[t,h]
                row = the top-`sparse_count` scores' key indices (row order free)

Three device kernels, no host/device round trip:

  _li_scores_kernel    grid (R, cdiv(T, BM), NCH)
      phys = block_table[r, j]                    # one paged key block per n tile
      qk   = dot(q[m, h, :], k_tile^T)            # Cube, bf16 x bf16 -> fp32
      acc += relu(qk) * w[m, h]                   # head-loop reduction
      lanes outside [0, active) are written as -3e38 (never selectable)

  _li_select_kernel    grid (T,)
      -1 fill / arange() for the degenerate windows, and for the regular
      window an exact threshold bisection over the fp32 score row staged into
      UB with tle.dsa.copy, then a masked compaction of the selected indices.

  _li_select_long_kernel  grid (R,)
      only the (rare) tokens whose causal window exceeds the register-resident
      row bound; streams the row in CX-wide chunks with the same bisection.

TLE placement note (established by target probes): the Ascend compiler rejects
every `tle.dsa` buffer that is consumed by `tl.dot` (and `to_buffer` of a
dot-derived accumulator) with "Unsupported copy from cbuf to ub", so the
dot-bound score kernel uses register tile loads while the bulk staged score row
of the selection kernels is moved with tle.dsa.copy.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

_INT64_MAX = 9223372036854775807

# scores kernel
_BM = 64      # query tokens per program
_NCH = 8      # key-block chunks per request
# selection kernels
_NN = 8192    # register resident score row bound (verified UB fit on target)
_CX = 2048    # chunk width of the oversized-row streaming path
_CXS = 512    # compaction chunk width of the register-resident fast path
_ITERS = 40   # threshold bisection iterations
_NEG = -3.0e38
_BIG = 3.0e38


@triton.jit
def _li_scores_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    bt_ptr,
    asq_ptr,
    ask_ptr,
    sc_ptr,
    sc_cnt,
    BTW,
    KP,
    RSTRIDE,
    HQ: tl.constexpr,
    KD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NCH: tl.constexpr,
    NEG: tl.constexpr,
):
    r = tl.program_id(0)
    mb = tl.program_id(1)
    ch = tl.program_id(2)

    klen = tl.load(ask_ptr + r)
    if klen > sc_cnt:
        qe = tl.load(asq_ptr + r)
        qb = tl.load(asq_ptr + tl.maximum(r - 1, 0))
        qb = tl.where(r > 0, qb, 0)
        t0 = qb + mb * BM
        if t0 < qe:
            nblk = tl.cdiv(klen, BN)
            per = tl.cdiv(nblk, NCH)
            j0 = ch * per
            j1 = tl.minimum(j0 + per, nblk)
            if j0 < j1:
                rows = tl.arange(0, BM)
                tg = t0 + rows
                rmask = tg < qe
                active = klen - qe + tg + 1
                dd = tl.arange(0, KD)
                nn = tl.arange(0, BN)
                for j in range(j0, j1):
                    phys = tl.load(bt_ptr + r * BTW + j)
                    kt = tl.load(
                        k_ptr
                        + phys * (BN * RSTRIDE)
                        + nn[:, None] * RSTRIDE
                        + dd[None, :]
                    )
                    ktt = tl.trans(kt)
                    acc = tl.zeros([BM, BN], dtype=tl.float32)
                    for h in range(0, HQ):
                        qoff = tg[:, None] * (HQ * KD) + h * KD + dd[None, :]
                        qt = tl.load(q_ptr + qoff, mask=rmask[:, None], other=0.0)
                        qk = tl.dot(qt, ktt, out_dtype=tl.float32)
                        qk = tl.maximum(qk, 0.0)
                        wv = tl.load(w_ptr + tg * HQ + h, mask=rmask, other=0.0).to(
                            tl.float32
                        )
                        acc = acc + qk * wv[:, None]
                    n = j * BN + nn
                    ok = rmask[:, None] & (n[None, :] < active[:, None])
                    acc = tl.where(ok, acc, NEG)
                    # only valid query rows are stored (tail rows of the tile have
                    # no scratch row of their own)
                    tl.store(
                        sc_ptr + tg[:, None] * KP + n[None, :], acc, mask=rmask[:, None]
                    )


@triton.jit
def _li_scores_hm_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    bt_ptr,
    asq_ptr,
    ask_ptr,
    sc_ptr,
    sc_cnt,
    BTW,
    KP,
    RSTRIDE,
    HQ: tl.constexpr,
    KD: tl.constexpr,
    BN: tl.constexpr,
    NCH: tl.constexpr,
    NEG: tl.constexpr,
):
    """Single-token variant: the whole query of one token (all heads) becomes the
    M dimension of one dot per key block, so the per-head weighting is a
    reduction along M.  Compared with a per-token row tile that issues one dot
    per (key block, head), this issues HQ times fewer dots and HQ times less
    query traffic, which is what dominates the cube on single-token requests.

    The grid stays (R, 1, NCH): each program walks its request's tokens
    internally (skipping the degenerate windows), so the block count and the
    associated empty-program dispatch cost are unchanged.
    """
    r = tl.program_id(0)
    ch = tl.program_id(2)
    klen = tl.load(ask_ptr + r)
    if klen > sc_cnt:
        qe = tl.load(asq_ptr + r)
        qb = tl.load(asq_ptr + tl.maximum(r - 1, 0))
        qb = tl.where(r > 0, qb, 0)
        nblk = tl.cdiv(klen, BN)
        per = tl.cdiv(nblk, NCH)
        j0 = ch * per
        j1 = tl.minimum(j0 + per, nblk)
        if j0 < j1:
            hh = tl.arange(0, HQ)
            dd = tl.arange(0, KD)
            nn = tl.arange(0, BN)
            for t in range(qb, qe):
                active = klen - (qe - t) + 1
                if active > sc_cnt:
                    q_tile = tl.load(
                        q_ptr + t * (HQ * KD) + hh[:, None] * KD + dd[None, :]
                    )
                    wv = tl.load(w_ptr + t * HQ + hh).to(tl.float32)
                    for j in range(j0, j1):
                        phys = tl.load(bt_ptr + r * BTW + j)
                        kt = tl.load(
                            k_ptr
                            + phys * (BN * RSTRIDE)
                            + nn[:, None] * RSTRIDE
                            + dd[None, :]
                        )
                        qk = tl.dot(q_tile, tl.trans(kt), out_dtype=tl.float32)
                        p = tl.maximum(qk, 0.0)
                        s = tl.sum(p * wv[:, None], axis=0)
                        n = j * BN + nn
                        s = tl.where(n < active, s, NEG)
                        tl.store(sc_ptr + t * KP + n, s)


@triton.jit
def _li_select_kernel(
    sc_ptr,
    asq_ptr,
    ask_ptr,
    out_ptr,
    sc_cnt,
    s_out,
    R,
    KP,
    NN: tl.constexpr,
    CXS: tl.constexpr,
    SPR: tl.constexpr,
    ITERS: tl.constexpr,
    NEG: tl.constexpr,
    BIG: tl.constexpr,
):
    t = tl.program_id(0)

    # -1 fill of this token's whole row
    nfill = tl.cdiv(s_out, SPR)
    for i in range(0, nfill):
        off = i * SPR + tl.arange(0, SPR)
        tl.store(
            out_ptr + t * s_out + off, tl.full([SPR], -1, tl.int32), mask=off < s_out
        )

    # request lookup: req = #{r : asq[r] <= t}
    req = 0
    for r in range(0, R - 1):
        req += (t >= tl.load(asq_ptr + r)).to(tl.int32)
    qe = tl.load(asq_ptr + req)
    qb = tl.load(asq_ptr + tl.maximum(req - 1, 0))
    qb = tl.where(req > 0, qb, 0)
    klen = tl.load(ask_ptr + req)
    active = klen - (qe - t) + 1

    if active > 0:
        if active <= sc_cnt:
            idx = tl.arange(0, SPR)
            tl.store(out_ptr + t * s_out + idx, idx, mask=idx < active)
        elif active <= NN:
            offs = tl.arange(0, NN)
            rb = tle.dsa.alloc([NN], tl.float32, tle.dsa.ascend.UB)
            tle.dsa.copy(sc_ptr + t * KP + offs, rb, [NN])
            raw = tle.dsa.to_tensor(rb)
            m = offs < active
            v = tl.where(m, raw, NEG)
            lo = tl.min(tl.where(m, v, BIG), axis=0)
            hi = tl.max(v, axis=0)
            for _ in range(0, ITERS):
                mid = (lo + hi) * 0.5
                c = tl.sum((v > mid).to(tl.int32), axis=0)
                lo = tl.where(c >= sc_cnt, mid, lo)
                hi = tl.where(c >= sc_cnt, hi, mid)
            # Compaction: the rank of a selected key is accumulated chunk by
            # chunk (base + within-chunk prefix).  Emitting the output from a
            # full-width cumsum keeps an extra NN-wide tensor live next to the
            # staged row in UB and measurably slows this kernel down.
            coff = tl.arange(0, CXS)
            base = 0
            for c in range(0, tl.cdiv(active, CXS)):
                o = c * CXS + coff
                vc = tl.load(sc_ptr + t * KP + o, mask=o < active, other=NEG)
                s_ = vc > lo
                rk = base + tl.cumsum(s_.to(tl.int32), axis=0) - 1
                tl.store(out_ptr + t * s_out + rk, o, mask=s_ & (rk < s_out))
                base += tl.sum(s_.to(tl.int32), axis=0)


@triton.jit
def _li_select_long_kernel(
    sc_ptr,
    asq_ptr,
    ask_ptr,
    out_ptr,
    sc_cnt,
    s_out,
    KP,
    NN: tl.constexpr,
    CX: tl.constexpr,
    ITERS: tl.constexpr,
    NEG: tl.constexpr,
    BIG: tl.constexpr,
):
    """Causal windows wider than NN: stream the row in CX-wide chunks."""
    r = tl.program_id(0)
    klen = tl.load(ask_ptr + r)
    if klen > NN:
        qe = tl.load(asq_ptr + r)
        qb = tl.load(asq_ptr + tl.maximum(r - 1, 0))
        qb = tl.where(r > 0, qb, 0)
        t0 = tl.maximum(qb, NN + qe - klen)
        if t0 < qe:
            offs = tl.arange(0, CX)
            for t in range(t0, qe):
                active = klen - (qe - t) + 1
                nseg = tl.cdiv(active, CX)
                lo = BIG
                hi = -BIG
                for c in range(0, nseg):
                    o = c * CX + offs
                    mm = o < active
                    v = tl.load(sc_ptr + t * KP + o, mask=mm, other=NEG)
                    lo = tl.minimum(lo, tl.min(tl.where(mm, v, BIG), axis=0))
                    hi = tl.maximum(hi, tl.max(v, axis=0))
                for _ in range(0, ITERS):
                    mid = (lo + hi) * 0.5
                    cnt = 0
                    for c in range(0, nseg):
                        o = c * CX + offs
                        v = tl.load(sc_ptr + t * KP + o, mask=o < active, other=NEG)
                        cnt += tl.sum((v > mid).to(tl.int32), axis=0)
                    lo = tl.where(cnt >= sc_cnt, mid, lo)
                    hi = tl.where(cnt >= sc_cnt, hi, mid)
                base = 0
                for c in range(0, nseg):
                    o = c * CX + offs
                    v = tl.load(sc_ptr + t * KP + o, mask=o < active, other=NEG)
                    sel = v > lo
                    rk = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
                    tl.store(
                        out_ptr + t * s_out + rk,
                        o,
                        mask=sel & (rk < s_out),
                    )
                    base += tl.sum(sel.to(tl.int32), axis=0)


def run(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    pre_tokens=_INT64_MAX,
    next_tokens=_INT64_MAX,
    return_value=False,
):
    if layout_query != "TND" or layout_key != "PA_BSND" or sparse_mode != 3:
        raise NotImplementedError("only TND / PA_BSND / sparse_mode 3 are covered")
    if pre_tokens != _INT64_MAX or next_tokens != _INT64_MAX:
        raise NotImplementedError("only the default token windows are covered")

    T = query.shape[0]
    R = actual_seq_lengths_query.numel()
    HQ = query.shape[1]
    KD = query.shape[2]
    BN = key.shape[1]
    RSTRIDE = key.shape[2] * key.shape[3]
    BTW = block_table.shape[1]
    s = int(sparse_count)

    # Launch-shape dispatch (host-visible shapes only, no device round trip).
    #
    # Key-block chunks per request.  An immediately-exiting program still costs
    # about 0.22 us of block dispatch (round-2 measurement), and the
    # many-request workloads of this operator are the degenerate ones (every
    # causal window <= sparse_count, so the score kernel has nothing to
    # compute), which makes the grid size itself their latency.  Requests with
    # few tokens keep the full key-block split; larger request counts, which
    # only appear with small key lengths, use fewer chunks.
    if R >= 8:
        NCH = 2
    elif R >= 3:
        NCH = 4
    else:
        NCH = _NCH

    # Per-program tile shape.  Single-token requests evaluate 63/64 idle query
    # rows at BM=64 and their causal window never needs the full
    # register-resident row, so for T <= 8 they use BM=16 (cdiv(T, 16) ==
    # cdiv(T, 64) == 1, i.e. the same program count) and a narrower staged row.
    # Mid-sized batches keep BM but stage a narrower row as well: every
    # registered case with 8 < T <= 256 has a causal window of at most 2048
    # keys, and wider windows are still handled exactly by the chunked
    # continuation kernel.
    if T <= 8:
        BM, NN = 16, 4096
    elif T <= 256:
        BM, NN = _BM, 2048
    else:
        BM, NN = _BM, _NN

    KP = max(BTW * BN, NN)

    scratch = torch.empty((T, KP), dtype=torch.float32, device=query.device)
    indices = torch.empty((T, 1, s), dtype=torch.int32, device=query.device)

    if T <= 8 and HQ >= 16:
        _li_scores_hm_kernel[(R, 1, NCH)](
            query,
            key,
            weights,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            scratch,
            s,
            BTW,
            KP,
            RSTRIDE,
            HQ=HQ,
            KD=KD,
            BN=BN,
            NCH=NCH,
            NEG=_NEG,
        )
    else:
        _li_scores_kernel[(R, triton.cdiv(T, BM), NCH)](
            query,
            key,
            weights,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            scratch,
            s,
            BTW,
            KP,
            RSTRIDE,
            HQ=HQ,
            KD=KD,
            BM=BM,
            BN=BN,
            NCH=NCH,
            NEG=_NEG,
        )

    _li_select_kernel[(T,)](
        scratch,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        indices,
        s,
        s,
        R,
        KP,
        NN=NN,
        CXS=_CXS,
        SPR=triton.next_power_of_2(s),
        ITERS=_ITERS,
        NEG=_NEG,
        BIG=_BIG,
    )

    _li_select_long_kernel[(R,)](
        scratch,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        indices,
        s,
        s,
        KP,
        NN=NN,
        CX=_CX,
        ITERS=_ITERS,
        NEG=_NEG,
        BIG=_BIG,
    )

    return indices, torch.empty((0,), dtype=query.dtype, device=query.device)
