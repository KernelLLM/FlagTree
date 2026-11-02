"""Varlen causal conv1d (prefill) for Ascend NPU, implemented with TLE/DSA.

Semantics follow the delivery reference exactly:

  * sequence i spans ``x[:, query_start_loc[i]:query_start_loc[i+1]]``;
  * a sequence whose ``cache_indices[i] == pad_slot_id`` contributes neither an
    output column nor a state write;
  * the convolution of sequence i is evaluated over
    ``[conv_states[slot], x[:, start:end]]`` -- the ``width-1`` leading columns
    come from the state when ``has_initial_state[i]``, otherwise from implicit
    zero padding -- and the trailing ``width-1`` columns of that concatenation
    are written back into ``conv_states[slot]`` in place;
  * the output is the concatenation, along the token axis, of the per-sequence
    outputs.

Tiling: one program per ``(sequence, dim-block)``.  A program is split into

  * a *head* covering the first ``min(nv, width-1)`` tokens.  Its sources are the
    ``width-1`` state columns and the first ``width-1`` ``x`` columns, so the head
    is written as ``width-1`` fully unrolled output columns, each a single
    ``(dim, )`` vector that loads its ``width`` tap sources as contiguous rows.
    The state rows are staged through UB with one bulk ``tle.dsa.copy`` and the
    final-state write is a single contiguous store per column, ordered after every
    state read in the same program;
  * a *body* covering ``[min(nv, width-1), nv)``, whose sources are purely ``x``
    columns.  Bodies with a full ``BLOCK_T`` chunk available use ``tle.dsa.copy``
    for every in-bounds tile load and the tile store; shorter bodies -- and
    boundary dim blocks -- use masked ``tl.load``/``tl.store``, which
    ``tle.dsa.copy`` cannot express.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def _silu(v):
    return v * (1.0 / (1.0 + tl.exp(-v)))


@triton.jit
def _cc1d_kernel(
    x_ptr, w_ptr, b_ptr, cs_ptr, qsl_ptr, ci_ptr, hi_ptr, o_ptr,
    xs0, xs1,                       # x strides (dim, token)
    cs0, cs1, cs2,                  # conv_states strides (slot, dim, width-1)
    dim, L, pad_slot_id,
    WIDTH: tl.constexpr, WP: tl.constexpr, TH: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    NSPLIT: tl.constexpr, MAXSEQ: tl.constexpr, MINSPAN: tl.constexpr,
    XDT: tl.constexpr, WDT: tl.constexpr, ODT: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_CACHE: tl.constexpr,
    HAS_INIT: tl.constexpr, SILU: tl.constexpr, HEAD_DSA: tl.constexpr,
):
    H: tl.constexpr = WIDTH - 1
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    part = tl.program_id(2)

    start = tl.load(qsl_ptr + pid_s)
    nv = tl.load(qsl_ptr + pid_s + 1) - start
    if HAS_CACHE:
        slot = tl.load(ci_ptr + pid_s)
    else:
        slot = pid_s
    if slot == pad_slot_id:
        return

    # output column offset: number of live tokens preceding this sequence
    jv = tl.arange(0, MAXSEQ)
    jm = jv < pid_s
    qa = tl.load(qsl_ptr + jv, mask=jm, other=0)
    qb = tl.load(qsl_ptr + jv + 1, mask=jm, other=0)
    if HAS_CACHE:
        cj = tl.load(ci_ptr + jv, mask=jm, other=pad_slot_id)
        live = cj != pad_slot_id
    else:
        live = jv >= 0
    off = tl.sum(tl.where(jm & live, qb - qa, 0))

    if HAS_INIT:
        initf = tl.where(tl.load(hi_ptr + pid_s) != 0, 1.0, 0.0)
    else:
        initf = 0.0

    dcol = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    kt = tl.arange(0, BLOCK_T)
    kh = tl.arange(0, TH)
    dmask = dcol < dim
    xd = dcol[:, None] * xs0
    csd = dcol[:, None] * cs1

    xbuf = tle.dsa.alloc([BLOCK_D, BLOCK_T], XDT, tle.dsa.ascend.UB)
    wbuf = tle.dsa.alloc([WP, BLOCK_D], WDT, tle.dsa.ascend.UB)
    if HAS_BIAS:
        bbuf = tle.dsa.alloc([BLOCK_D], XDT, tle.dsa.ascend.UB)

    head_hi = tl.minimum(nv, H)
    body_lo = head_hi
    span = nv - body_lo

    if part == 0:
        # ------------------------------------------------------------------
        # head: first min(nv, WIDTH-1) tokens, one unrolled column at a time
        # ------------------------------------------------------------------
        if HEAD_DSA and H > 0:
            # Tap weights staged with one bulk transfer; the rows are then read
            # out of UB, so the head costs no strided weight gathers.
            tle.dsa.copy(
                w_ptr + (tl.arange(0, WP)[:, None] + dcol[None, :] * WIDTH),
                wbuf, [WP, BLOCK_D])
            wt = tle.dsa.to_tensor(wbuf)
            if HAS_BIAS:
                bv = tl.load(b_ptr + dcol, mask=dmask, other=0.0).to(tl.float32)
            else:
                bv = tl.zeros([BLOCK_D], tl.float32)
            for k in tl.static_range(TH):
                if k < H:
                    acc = bv
                    for j in tl.static_range(WIDTH):
                        wj = tl.reshape(
                            tle.dsa.extract_slice(wt, [j, 0], [1, BLOCK_D], [BLOCK_D, 1]),
                            [BLOCK_D]).to(tl.float32)
                        if k + j >= H:
                            # source is x column start + (k + j - H)
                            xr = tl.load(x_ptr + (start + k + j - H) * xs1 + dcol,
                                         mask=dmask, other=0.0).to(tl.float32)
                            acc = acc + wj * xr
                        else:
                            # source is state column k + j
                            sr = tl.load(cs_ptr + slot * cs0 + (k + j) * cs2 + dcol,
                                         mask=dmask, other=0.0).to(tl.float32)
                            acc = acc + wj * (sr * initf)
                    res = _silu(acc) if SILU else acc
                    tl.store(o_ptr + dcol * L + (off + k), res.to(ODT),
                             mask=dmask & (k < nv))

            # final state: new_state[s] = concat(state, x)[nv + s]
            for s in tl.static_range(TH):
                if s < H:
                    u0 = nv - H + s
                    fx = tl.load(x_ptr + (start + u0) * xs1 + dcol,
                                 mask=dmask & (u0 >= 0), other=0.0).to(tl.float32)
                    fs = tl.load(cs_ptr + slot * cs0 + (u0 + H) * cs2 + dcol,
                                 mask=dmask & (u0 < 0), other=0.0).to(tl.float32)
                    tl.store(cs_ptr + slot * cs0 + s * cs2 + dcol,
                             tl.where(u0 >= 0, fx, fs * initf).to(XDT),
                             mask=dmask)

        if not HEAD_DSA and H > 0:
            hh = kh < head_hi
            acc2 = tl.zeros([BLOCK_D, TH], tl.float32)
            if HAS_BIAS:
                acc2 += tl.load(b_ptr + dcol, mask=dmask, other=0.0).to(tl.float32)[:, None]
            for j in range(0, WIDTH):
                u = kh + (j - H)
                xv2 = tl.load(x_ptr + xd + (start + u)[None, :] * xs1,
                              mask=((u >= 0) & hh)[None, :] & dmask[:, None],
                              other=0.0).to(tl.float32)
                hs = (u < 0) & hh
                sv2 = tl.load(cs_ptr + slot * cs0 + csd + (u + H)[None, :] * cs2,
                              mask=hs[None, :] & dmask[:, None],
                              other=0.0).to(tl.float32)
                val = tl.where(hs[None, :], sv2 * initf, xv2)
                wj2 = tl.load(w_ptr + dcol * WIDTH + j, mask=dmask,
                              other=0.0).to(tl.float32)
                acc2 += val * wj2[:, None]
            res2 = _silu(acc2) if SILU else acc2
            tl.store(o_ptr + dcol[:, None] * L + (off + kh)[None, :], res2.to(ODT),
                     mask=hh[None, :] & dmask[:, None])

            u0 = nv - H + kh
            j0 = kh < H
            f_lo = j0 & (u0 < 0)
            f_hi = j0 & (u0 >= 0)
            fx = tl.load(x_ptr + xd + (start + u0)[None, :] * xs1,
                         mask=f_hi[None, :] & dmask[:, None],
                         other=0.0).to(tl.float32)
            fs = tl.load(cs_ptr + slot * cs0 + csd + (u0 + H)[None, :] * cs2,
                         mask=f_lo[None, :] & dmask[:, None],
                         other=0.0).to(tl.float32)
            fv = tl.where(f_lo[None, :], fs * initf, fx)
            tl.store(cs_ptr + slot * cs0 + csd + kh[None, :] * cs2,
                     fv.to(XDT), mask=j0[None, :] & dmask[:, None])

    if span >= MINSPAN:
        # ---------------- bulk body: in-bounds full BLOCK_T chunks ------------
        tle.dsa.copy(w_ptr + (dcol[None, :] * WIDTH + tl.arange(0, WP)[:, None]),
                     wbuf, [WP, BLOCK_D])
        wt2 = tle.dsa.to_tensor(wbuf)
        if HAS_BIAS:
            tle.dsa.copy(b_ptr + dcol, bbuf, [BLOCK_D])
            bv2 = tl.reshape(tle.dsa.to_tensor(bbuf).to(tl.float32), [BLOCK_D])
        nparts = tl.minimum(NSPLIT, span // BLOCK_T)
        if part < nparts:
            bs = body_lo + (span * part) // nparts
            be = body_lo + (span * (part + 1)) // nparts
            nch = (be - bs + BLOCK_T - 1) // BLOCK_T
            for m in range(0, nch):
                c = tl.maximum(tl.minimum(bs + m * BLOCK_T, be - BLOCK_T), bs)
                acc = tl.zeros([BLOCK_D, BLOCK_T], tl.float32)
                if HAS_BIAS:
                    acc += bv2[:, None]
                for j in range(0, WIDTH):
                    tle.dsa.copy(
                        x_ptr + xd + (start + c - H + j + kt)[None, :] * xs1,
                        xbuf, [BLOCK_D, BLOCK_T])
                    xv = tle.dsa.to_tensor(xbuf).to(tl.float32)
                    wj = tl.reshape(
                        tle.dsa.extract_slice(wt2, [j, 0], [1, BLOCK_D], [BLOCK_D, 1]),
                        [BLOCK_D])
                    acc += xv * wj[:, None]
                res = _silu(acc) if SILU else acc
                rbuf = tle.dsa.to_buffer(res.to(ODT), tle.dsa.ascend.UB)
                with tle.dsa.hint(inter_no_alias=True):
                    tle.dsa.copy(rbuf, o_ptr + dcol[:, None] * L
                                 + (off + c + kt)[None, :], [BLOCK_D, BLOCK_T])
    else:
        # ---------------- masked body (short spans / partial dim blocks) ------
        # body tokens are >= min(nv, H), so every source column is inside x
        bs = body_lo + (span * part) // NSPLIT
        be = body_lo + (span * (part + 1)) // NSPLIT
        nch = (be - bs + BLOCK_T - 1) // BLOCK_T
        for m in range(0, nch):
            c = tl.maximum(tl.minimum(bs + m * BLOCK_T, be - BLOCK_T), bs)
            kk = c + kt
            rowm = (kk < be)[None, :] & dmask[:, None]
            acc = tl.zeros([BLOCK_D, BLOCK_T], tl.float32)
            if HAS_BIAS:
                acc += tl.load(b_ptr + dcol, mask=dmask, other=0.0).to(tl.float32)[:, None]
            for j in range(0, WIDTH):
                xv = tl.load(x_ptr + xd + (start + kk + (j - H))[None, :] * xs1,
                             mask=rowm, other=0.0).to(tl.float32)
                wj = tl.load(w_ptr + dcol * WIDTH + j, mask=dmask, other=0.0).to(tl.float32)
                acc += xv * wj[:, None]
            res = _silu(acc) if SILU else acc
            tl.store(o_ptr + dcol[:, None] * L + (off + kk)[None, :], res.to(ODT),
                     mask=rowm)


_TORCH_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


# Tiling knobs, hoisted to module scope so they can be swept from
# bench_ops.py (`--const _BLOCK_T=64`) without editing the kernel.  Values are
# unchanged defaults.  NOTE: no benchmarked improvement was found for this
# operator - the body reads x WIDTH times, which is 2.5x the bytes of the
# non-CommonIR counterpart at comparable achieved bandwidth and accounts for the
# whole gap; _BLOCK_T=64 is ~8% better (accuracy verified) but within run-to-run
# noise, and both attempts at a shared-strip formulation faulted the MTE engine.
_MAX_BLOCK_D = 128
_BLOCK_T = 32
_HEAD_DSA_AVG_LIMIT = 256.0


def _pick_block_d(dim):
    """Widest dim block that divides dim, up to ``_MAX_BLOCK_D``.

    The kernel pays a fixed per-program cost, so for large dim it is better to
    cover twice as many dims per program (half as many programs dispatched) than
    to keep the per-program tile small.
    """
    bd = _MAX_BLOCK_D
    while bd > 16 and (bd > dim or dim % bd):
        bd //= 2
    return bd


def run(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
):
    dim, total_len = x.shape
    width = weight.shape[1]
    pad_slot_id = int(pad_slot_id)

    qsl = query_start_loc.detach().cpu().tolist()
    nseq = len(qsl) - 1
    if nseq <= 0:
        return x.new_empty(dim, 0)
    if cache_indices is not None:
        slots = cache_indices.detach().cpu().tolist()
    else:
        slots = list(range(nseq))

    total_out = 0
    live_any = False
    for i in range(nseq):
        if slots[i] != pad_slot_id:
            total_out += int(qsl[i + 1]) - int(qsl[i])
            live_any = True
    if not live_any:
        return x.new_empty(dim, 0)

    out = torch.empty(dim, total_out, dtype=x.dtype, device=x.device)

    block_d = _pick_block_d(dim)
    # BLOCK_T also fixes the shortest body that can use the unmasked DSA path.
    block_t = _BLOCK_T
    hh = width - 1
    th = max(4, triton.next_power_of_2(hh))
    wp = triton.next_power_of_2(width)
    dim_whole = (dim % block_d == 0)
    # The head's bulk staging and contiguous row accesses assume the
    # (slots, dim, width-1) layout with a unit dim stride, which is what the
    # ABI's transpose(1, 2) construction produces.
    head_ok = (dim_whole and (weight.stride(0) == width) and (weight.stride(1) == 1)
               and (x.stride(0) == 1) and (conv_states.stride(1) == 1))
    # The per-column head pays off while the boundary columns are a large share
    # of the work, i.e. for short sequences; on long ones its extra code makes
    # the bulk body slower, so the masked head is kept there.
    head_dsa = head_ok and (total_len / max(1, nseq)) < _HEAD_DSA_AVG_LIMIT
    dsa_ok = (width == wp) and dim_whole
    minspan = hh + block_t if dsa_ok else (1 << 30)
    nsplit = 1
    maxseq = max(1, triton.next_power_of_2(nseq))

    grid = (nseq, triton.cdiv(dim, block_d), nsplit)
    _cc1d_kernel[grid](
        x, weight,
        bias if bias is not None else x,
        conv_states,
        query_start_loc,
        cache_indices if cache_indices is not None else x,
        has_initial_state if has_initial_state is not None else x,
        out,
        x.stride(0), x.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        dim, total_out, pad_slot_id,
        WIDTH=width, WP=wp, TH=th,
        BLOCK_T=block_t, BLOCK_D=block_d,
        NSPLIT=nsplit, MAXSEQ=maxseq, MINSPAN=minspan,
        XDT=_TORCH_DTYPE[x.dtype], WDT=_TORCH_DTYPE[weight.dtype], ODT=_TORCH_DTYPE[x.dtype],
        HAS_BIAS=bias is not None,
        HAS_CACHE=cache_indices is not None,
        HAS_INIT=has_initial_state is not None,
        SILU=activation is not None,
        HEAD_DSA=head_dsa,
        num_warps=1,
    )
    return out
