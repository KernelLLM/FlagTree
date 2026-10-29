"""TLE (Triton Language Extensions / DSA) implementation of the FlagGems-vllm
``compressor`` operator.

Three device kernels, all of them on the real data path:

``_token_meta_kernel``
    Per-token gather metadata (cache row index, ``pos % cmp_ratio``, validity),
    computed **once** per token into a small scratch buffer.  Computing it inline
    in the projection kernel was both redundant (once per column tile) and slow:
    on this backend a data-dependent gather that feeds a *store address* is
    scalarised (~2 us/element), whereas a row index read from memory is emitted as
    a per-row DMA.  The block size is a fixed 128 elements: sizing it from the
    token count overran the 192 KB UB budget for mid-size cases.

``_proj_scatter_kernel``
    ``kv = x @ wkv.T`` / ``gate = x @ wgate.T`` (bf16 operands, fp32
    accumulation) plus the cache scatter.  Every address is affine: the token
    index is ``arange``-based (offset by ``base = min(pid*BM, T-BM)`` so that the
    last token block *overlaps* its predecessor instead of running past the end)
    and the cache row comes from an affine load of the metadata buffer.

    The overlapping block layout means every program covers only real tokens, so
    the scatter needs no mask at all.  That matters: a store mask that is a
    *loaded* vector costs ~5.5 ns/element against 0.7 ns/element for an
    arange-derived mask, and a mask on a loaded-row store after a ``tl.dot``
    miscompiles this backend outright (``LLVM ERROR: Failed to obtain op buffer
    shape size which should be static``).  Only the ``T < BM`` case (a single
    token block) still needs the loaded validity mask, and it is selected by the
    ``MASKED`` constexpr so the hot specializations stay mask-free.

``_compress_kernel``
    One program per compressed group: cache row windows are staged GM -> UB with
    ``tle.dsa.copy``, the softmax weighted sum runs as an online softmax over
    k-chunks, and the RoPE'd row is written back with ``tle.dsa.to_buffer`` +
    ``tle.dsa.copy``.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

try:  # the device tensors handed to run() are NPU tensors
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    pass

_LARGE = 3.4028234663852886e38
_META_BLOCK = 128

# Projection tiling, measured on Ascend910B4 (all bit-exact vs. the non-CommonIR
# counterpart).  BM/BN/BK = 128/128/128 leaves L1 at 25% and was 5.234 ms on
# prefill0_b2_rank3; BN=256 (the lever: it halves the A-tile reloads) with
# BK=256 gives 3.547 ms (-31.5%).  BK alone does not help - matching the
# counterpart's BK=512 at BN=128 recovers only 11% because that already fills
# L1.  L0C (128 KiB) caps the fp32 accumulator at BM*BN <= 32768, so BN=512
# needs BM=64 (5% better again, but it regresses the small shapes).  Below
# ~2k tokens the 128-wide tile still measured better, which mirrors the
# counterpart's own `block_n = 256 if wide_projection else 128` rule.
_PROJ_BM_MAX = 128
_PROJ_BN = 256
_PROJ_BK = 256
_PROJ_WIDE_MIN_TOKENS = 2048


# --------------------------------------------------------------------------- #
# Kernel 0: per-token gather metadata (mirrors _cache_block_id)
# --------------------------------------------------------------------------- #
@triton.jit
def _token_meta_kernel(
    cum_ptr, start_ptr, table_ptr, row_ptr, ai_ptr, vm_ptr,
    T, BS, SB, MB, NB1,
    RANK3: tl.constexpr, S: tl.constexpr, R: tl.constexpr,
    BM: tl.constexpr, PADB: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BM + tl.arange(0, BM)
    tmask = t < T
    if RANK3:
        b = t // S
        local = t - b * S
        tmask2 = tmask
    else:
        jb = tl.arange(0, PADB)
        cj = tl.load(cum_ptr + jb, mask=jb < NB1, other=2147483647)
        b = tl.sum((cj[None, :] <= t[:, None]).to(tl.int32), axis=1) - 1
        tmask2 = tmask & (b >= 0)
        base = tl.load(cum_ptr + b, mask=tmask2, other=0)
        local = t - base

    pos = tl.load(start_ptr + b, mask=tmask2, other=0) + local
    pv = tmask2 & (pos >= 0)
    ppos = tl.where(pv, pos, 0)
    blk = ppos // BS
    valid = pv & (blk < MB)
    cblk = tl.load(table_ptr + b * MB + blk, mask=valid, other=0)
    valid = valid & (cblk > 0) & (cblk < SB)
    row = tl.where(valid, cblk * BS + ppos % BS, 0)
    tl.store(row_ptr + t, row, mask=tmask)
    tl.store(ai_ptr + t, tl.where(valid, ppos % R, 0), mask=tmask)
    tl.store(vm_ptr + t, valid.to(tl.int32), mask=tmask)


# --------------------------------------------------------------------------- #
# Kernel 1: projection + cache scatter
# --------------------------------------------------------------------------- #
@triton.jit
def _proj_scatter_kernel(
    x_ptr, w_ptr, cache_ptr, ape_ptr, row_ptr, ai_ptr, vm_ptr,
    T, TBASE, P, W2, COLOFF,
    IS_GATE: tl.constexpr, MASKED: tl.constexpr, S: tl.constexpr,
    H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """One half of ``[kv | gate]``; ``IS_GATE`` selects the weight matrix, the
    destination column offset and the ``ape`` epilogue.  A runtime scalar branch
    around a ``tl.dot`` accumulator miscompiles on this backend, hence the
    constexpr split into two launches."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    base = tl.minimum(pid_m * BM, TBASE)
    t = base + tl.arange(0, BM)
    wc = pid_n * BN + tl.arange(0, BN)
    cols = wc + COLOFF
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if MASKED:
        tmask = t < T
    for k0 in range(0, H, BK):
        ko = (k0 + rk)[None, :]
        if MASKED:
            xt = tl.load(x_ptr + t[:, None] * H + ko, mask=tmask[:, None],
                         other=0.0)
        else:
            xt = tl.load(x_ptr + t[:, None] * H + ko)
        wt = tl.load(w_ptr + wc[:, None] * H + ko)
        acc = tl.dot(xt, tl.trans(wt), acc, out_dtype=tl.float32)
    row = tl.load(row_ptr + t)
    if IS_GATE:
        ai = tl.load(ai_ptr + t)
        acc = acc + tl.load(ape_ptr + ai[:, None] * P + wc[None, :])
    dst = cache_ptr + row[:, None] * W2 + cols[None, :]
    if MASKED:
        vm = tl.load(vm_ptr + t)
        tl.store(dst, acc, mask=(vm[:, None] > 0))
    else:
        tl.store(dst, acc)


# --------------------------------------------------------------------------- #
# Kernel 2: compression (softmax weighted sum + RMS norm + RoPE)
# --------------------------------------------------------------------------- #
@triton.jit
def _compress_kernel(
    cache_ptr, table_ptr, cum_ptr, start_ptr, norm_ptr, rs_ptr, rc_ptr, out_ptr,
    P, R, BS, SB, MB, NB1, W2, ROWS, OPT, SZ, EPS,
    RANK3: tl.constexpr, PADB: tl.constexpr, COFF: tl.constexpr,
    HD: tl.constexpr, BKC: tl.constexpr, NCH: tl.constexpr,
    RD: tl.constexpr, MODE1: tl.constexpr, LARGE: tl.constexpr,
):
    gid = tl.program_id(0)

    # ---- per-request group geometry (prefix sum over the batch) ----
    jb = tl.arange(0, PADB)
    jm = jb < (NB1 - 1)
    s_b = tl.load(start_ptr + jb, mask=jm, other=0)
    if RANK3:
        u_b = tl.where(jm, SZ, 0)
    else:
        ca = tl.load(cum_ptr + jb, mask=jm, other=0)
        cb = tl.load(cum_ptr + jb + 1, mask=jm, other=0)
        u_b = cb - ca
    s_b = tl.maximum(s_b, 0)
    cg = (s_b + u_b) // R - s_b // R
    cg = tl.where(jm, tl.maximum(cg, 0), 0)
    pre = tl.cumsum(cg, axis=0) - cg
    if RANK3:
        gg = tl.minimum(cg, OPT)
    else:
        gg = tl.maximum(tl.minimum(cg, ROWS - pre), 0)
    total = tl.sum(gg, axis=0)
    if gid >= total:
        return
    pre2 = tl.cumsum(gg, axis=0) - gg
    b = tl.sum((pre2 <= gid).to(tl.int32), axis=0) - 1
    lg = gid - tl.sum(tl.where(jb == b, pre2, 0), axis=0)

    s0 = tl.maximum(tl.load(start_ptr + b), 0)
    gstart = (s0 // R + lg) * R
    if RANK3:
        orow = b * OPT + lg
    else:
        orow = gid

    hd = tl.arange(0, HD)
    kvb = tle.dsa.alloc([BKC, HD], tl.float32, tle.dsa.ascend.UB)
    gab = tle.dsa.alloc([BKC, HD], tl.float32, tle.dsa.ascend.UB)

    m_ = tl.full((HD,), -LARGE, dtype=tl.float32)
    l_ = tl.zeros((HD,), dtype=tl.float32)
    a_ = tl.zeros((HD,), dtype=tl.float32)

    for seg in tl.static_range(COFF):
        dim_start = seg * HD
        base = (seg - COFF + 1) * R
        for c in tl.static_range(NCH):
            kk = c * BKC + tl.arange(0, BKC)
            kin = kk < R
            pos = gstart + base + kk
            pv = kin & (pos >= 0)
            blk = tl.where(pv, pos, 0) // BS
            valid = pv & (blk < MB)
            cblk = tl.load(table_ptr + b * MB + blk, mask=valid, other=0)
            valid = valid & (cblk > 0) & (cblk < SB)
            row = tl.where(valid, cblk * BS + tl.where(pv, pos, 0) % BS, 0)
            tle.dsa.copy(
                cache_ptr + row[:, None] * W2 + (dim_start + hd)[None, :],
                kvb, [BKC, HD])
            tle.dsa.copy(
                cache_ptr + row[:, None] * W2 + (P + dim_start + hd)[None, :],
                gab, [BKC, HD])
            kv = tl.where(valid[:, None], tle.dsa.to_tensor(kvb), 0.0)
            ga = tl.where(valid[:, None], tle.dsa.to_tensor(gab), -LARGE)
            m_new = tl.maximum(m_, tl.max(ga, axis=0))
            alpha = tl.exp(m_ - m_new)
            p_ = tl.exp(ga - m_new[None, :])
            l_ = l_ * alpha + tl.sum(p_, axis=0)
            a_ = a_ * alpha + tl.sum(p_ * kv, axis=0)
            m_ = m_new

    comp = a_ / l_
    v = comp * tl.rsqrt(tl.sum(comp * comp, axis=0) / HD + EPS)
    v = v * tl.load(norm_ptr + hd).to(tl.float32)

    # ---- RoPE on the trailing RD lanes (interleaved or half-split) ----
    start = HD - RD
    inr = hd >= start
    rel = tl.where(inr, hd - start, 0)
    if MODE1:
        half = RD // 2
        fh = hd < (start + half)
        partner = tl.where(fh, hd + half, hd - half)
        sgn = tl.where(fh, -1.0, 1.0)
    else:
        reve = (rel % 2) == 0
        partner = tl.where(reve, hd + 1, hd - 1)
        sgn = tl.where(reve, -1.0, 1.0)
    partner = tl.where(inr, partner, hd)
    rot = tl.gather(v, partner, axis=0) * sgn
    cc = tl.where(inr, tl.load(rc_ptr + orow * RD + rel).to(tl.float32), 1.0)
    ss = tl.where(inr, tl.load(rs_ptr + orow * RD + rel).to(tl.float32), 0.0)
    res = (v * cc + rot * ss).to(tl.bfloat16)
    ob = tle.dsa.to_buffer(res, tle.dsa.ascend.UB)
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(ob, out_ptr + orow * HD + hd, [HD])


# --------------------------------------------------------------------------- #
# Host wrapper
# --------------------------------------------------------------------------- #
def _as_i32(value, length, default, device):
    if value is None:
        return torch.full((length,), default, dtype=torch.int32, device=device)
    if torch.is_tensor(value):
        flat = value.detach().reshape(-1).to(torch.int32)
    elif isinstance(value, (list, tuple)):
        flat = torch.tensor([int(v) for v in value], dtype=torch.int32)
    else:
        flat = torch.full((length,), int(value), dtype=torch.int32)
    if length is not None and flat.numel() != length:
        pad = torch.full((length,), default, dtype=torch.int32)
        n = min(flat.numel(), length)
        pad[:n] = flat[:n].cpu()
        flat = pad
    return flat.to(device)


def _pow2_ceil(n):
    n = max(int(n), 1)
    p = 1
    while p < n:
        p <<= 1
    return p


def _pow2_floor(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def run(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    rope_head_dim=64,
    cmp_ratio=128,
    coff=1,
    norm_eps=1e-6,
    rotary_mode=2,
    cache_mode=1,
):
    """Compressor, vllm-ascend / FlagGems semantics (see module docstring)."""
    del cache_mode  # the reference ignores it; only the paged layout is used.

    hidden = int(x.shape[-1])
    head_dim = int(norm_weight.shape[0])
    proj_dim = int(coff) * head_dim
    ratio = int(cmp_ratio)
    coff_i = int(coff)
    rd = int(rope_head_dim)
    rank3 = x.dim() == 3
    device = x.device

    block_size = int(state_cache.shape[1])
    state_blocks = int(state_cache.shape[0])
    row_span = int(state_cache.shape[-1])

    if rank3:
        batch, seq_len, _ = x.shape
        total_tokens = batch * seq_len
        opt = (seq_len + ratio - 1) // ratio
        cum = torch.zeros(2, dtype=torch.int32, device=device)
    else:
        total_tokens = int(x.shape[0])
        seq_len = 0
        if cu_seqlens is not None:
            cum = _as_i32(cu_seqlens, None, 0, device)
            batch = max(int(cum.numel()) - 1, 0)
        else:
            batch = 1
            cum = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
        opt = 1

    if state_block_table is None:
        table = torch.zeros((max(batch, 1), 1), dtype=torch.int32, device=device)
        max_blocks = 1
    else:
        table = state_block_table
        max_blocks = int(table.shape[1])

    starts = _as_i32(start_pos, batch, 0, device)
    cum = _as_i32(cum, None, 0, device)
    del seqused  # the reference never passes it

    rows = batch * opt if rank3 else int(rope_sin.shape[0])
    out_shape = (batch, opt, head_dim) if rank3 else (rows, head_dim)
    out = torch.zeros(out_shape, dtype=torch.bfloat16, device=device)

    # ---- projection geometry ------------------------------------------------
    # Every program must cover only real tokens, so the last block slides back
    # and overlaps its predecessor (idempotent: identical values are rewritten).
    # The 128x128x128 tile measured best on the production prefill shape
    # (128 TFLOPS against 60 for 64x128x64); the overlap trick requires T >= BM,
    # hence the descending fallbacks.
    token_cap = max(total_tokens, 1)
    if token_cap >= 128:
        bm = _PROJ_BM_MAX
        masked = False
        tbase = total_tokens - bm
    elif token_cap >= 64:
        bm = 64
        masked = False
        tbase = total_tokens - bm
    elif token_cap >= 16:
        bm = _pow2_floor(token_cap)
        masked = False
        tbase = total_tokens - bm
    else:
        bm = 16
        masked = True
        tbase = 0
    num_m = (total_tokens + bm - 1) // bm
    meta_n = ((num_m * bm + _META_BLOCK - 1) // _META_BLOCK) * _META_BLOCK

    # ---- kernel 0: per-token gather metadata (scratch, not returned) ----
    row_buf = torch.zeros(meta_n, dtype=torch.int32, device=device)
    ai_buf = torch.zeros(meta_n, dtype=torch.int32, device=device)
    vm_buf = torch.zeros(meta_n, dtype=torch.int32, device=device)
    padb1 = _pow2_ceil(batch + 1)
    _token_meta_kernel[(meta_n // _META_BLOCK,)](
        cum, starts, table, row_buf, ai_buf, vm_buf,
        total_tokens, block_size, state_blocks, max_blocks, batch + 1,
        RANK3=rank3, S=int(seq_len), R=ratio, BM=_META_BLOCK, PADB=padb1,
    )

    # ---- kernel 1: projection + scatter (two constexpr halves) ----
    bn = _PROJ_BN if total_tokens >= _PROJ_WIDE_MIN_TOKENS else min(_PROJ_BN, 128)
    while bn > 16 and proj_dim % bn:
        bn >>= 1
    if proj_dim % bn:
        raise ValueError("projection width is not tileable")
    bk = _PROJ_BK
    while bk > 16 and hidden % bk:
        bk >>= 1
    if hidden % bk:
        raise ValueError("hidden size is not tileable")
    grid1 = ((proj_dim + bn - 1) // bn, num_m)
    _proj_scatter_kernel[grid1](
        x, wkv, state_cache, ape, row_buf, ai_buf, vm_buf,
        total_tokens, tbase, proj_dim, row_span, 0,
        IS_GATE=False, MASKED=masked, S=int(seq_len), H=hidden,
        BM=bm, BN=bn, BK=bk,
    )
    _proj_scatter_kernel[grid1](
        x, wgate, state_cache, ape, row_buf, ai_buf, vm_buf,
        total_tokens, tbase, proj_dim, row_span, proj_dim,
        IS_GATE=True, MASKED=masked, S=int(seq_len), H=hidden,
        BM=bm, BN=bn, BK=bk,
    )

    # ---- kernel 2: compression ----
    hd = _pow2_ceil(head_dim)
    bkc = 1
    lim = max(4096 // max(hd, 1), 1)
    while bkc * 2 <= min(lim, ratio):
        bkc *= 2
    nch = (ratio + bkc - 1) // bkc
    padb2 = _pow2_ceil(max(batch, 1))
    _compress_kernel[(max(rows, 1),)](
        state_cache, table, cum, starts, norm_weight, rope_sin, rope_cos, out,
        proj_dim, ratio, block_size, state_blocks, max_blocks, batch + 1,
        row_span, rows, opt, seq_len, float(norm_eps),
        RANK3=rank3, PADB=padb2, COFF=coff_i, HD=hd, BKC=bkc, NCH=nch,
        RD=rd, MODE1=(int(rotary_mode) == 1), LARGE=_LARGE,
    )
    return out
