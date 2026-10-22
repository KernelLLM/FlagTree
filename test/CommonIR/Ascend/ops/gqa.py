"""Grouped-query attention forward (BSND) — TLE / Triton implementation.

Semantics follow the reference golden:

* ``scaleValue <= 0`` selects ``1/sqrt(D)``; the scale multiplies the QK scores
  *before* the causal mask and the softmax.
* the causal mask is bottom-right aligned: column ``j`` is visible from row ``i``
  iff ``j <= i + (S_kv - S)``.  Requires ``S <= S_kv``.
* query head ``h`` reads KV head ``h // (N_q // N_kv)``.
* accumulation is float32, the output is cast once to ``query``'s dtype.

The kernel walks the KV axis in ``BLOCK_N`` tiles with an online softmax, so the
score tile never has to be materialised in global memory.  The output tile
writeback is staged through an explicit UB buffer with ``tle.dsa.to_buffer`` /
``tle.dsa.copy``; boundary tiles fall back to a masked ``tl.store`` because
``dsa.copy`` has no mask.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle


# 128x128 tiles keep 17 UB buffers of [BLOCK_M, BLOCK_D] live; at head_dim 128
# that is 557 KiB against a 192 KiB budget and the kernel does not compile
# (`ub overflow, requires 2387968 bits while 1572864 available`).  head_dim 64
# fits at 128x128 (and is the faster choice there), so gate on the head dim.
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_M_WIDE_D = 64
_BLOCK_N_WIDE_D = 64


@triton.jit
def _gqa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale,
    S, S_KV, N_Q, N_KV,
    sqb, sqs, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sob, sos, soh,
    IS_CAUSAL: tl.constexpr,
    M_FULL: tl.constexpr,
    N_FULL: tl.constexpr,
    NEED_BOUNDS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    NEG_BIG: tl.constexpr = -1.0e30

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // N_Q
    hq = pid_bh % N_Q
    hkv = hq // (N_Q // N_KV)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + b * sqb + hq * sqh + offs_m[:, None] * sqs + offs_d[None, :]
    k_ptrs = k_ptr + b * skb + hkv * skh + offs_n[:, None] * sks + offs_d[None, :]
    v_ptrs = v_ptr + b * svb + hkv * svh + offs_n[:, None] * svs + offs_d[None, :]

    # ---- Q tile (loaded once, reused by every KV tile) ----
    if D == BLOCK_D:
        if M_FULL:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
            other=0.0,
        )

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], NEG_BIG, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    n_blocks = tl.cdiv(S_KV, BLOCK_N)
    if IS_CAUSAL:
        # Only KV blocks up to the last column visible from this row block can
        # contribute: j_max = (pid_m+1)*BLOCK_M - 1 + (S_KV - S).
        n_blocks = tl.minimum(
            n_blocks,
            tl.cdiv((pid_m + 1) * BLOCK_M + (S_KV - S), BLOCK_N),
        )

    for n in range(0, n_blocks):
        if D == BLOCK_D:
            if N_FULL:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                kv_ok = (offs_n + n * BLOCK_N)[:, None] < S_KV
                k = tl.load(k_ptrs, mask=kv_ok, other=0.0)
                v = tl.load(v_ptrs, mask=kv_ok, other=0.0)
        else:
            kv_ok = ((offs_n + n * BLOCK_N)[:, None] < S_KV) & (offs_d[None, :] < D)
            k = tl.load(k_ptrs, mask=kv_ok, other=0.0)
            v = tl.load(v_ptrs, mask=kv_ok, other=0.0)

        # ---- scores: [BLOCK_M, BLOCK_N] ----
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        if IS_CAUSAL:
            # A tile whose last column is <= the first row's limit holds only
            # visible columns, so it needs no mask at all.
            if (n + 1) * BLOCK_N - 1 > pid_m * BLOCK_M + (S_KV - S):
                n_idx = offs_n[None, :] + n * BLOCK_N
                limit = offs_m[:, None] + (S_KV - S)
                if NEED_BOUNDS:
                    valid = (n_idx < S_KV) & (offs_m[:, None] < S) & (n_idx <= limit)
                else:
                    valid = n_idx <= limit
                s = tl.where(valid, s, NEG_BIG)
        else:
            if NEED_BOUNDS:
                n_idx = offs_n[None, :] + n * BLOCK_N
                valid = (offs_m[:, None] < S) & (n_idx < S_KV)
                s = tl.where(valid, s, NEG_BIG)

        # ---- online softmax ----
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc, out_dtype=tl.float32)
        m_i = m_new

        k_ptrs += BLOCK_N * sks
        v_ptrs += BLOCK_N * svs

    # Rows that ended up fully masked keep weight 0 (matches the golden's guard).
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(q.dtype)

    o_ptrs = o_ptr + b * sob + hq * soh + offs_m[:, None] * sos + offs_d[None, :]
    if D == BLOCK_D:
        with tle.dsa.hint(inter_no_alias=True):
            tl.store(o_ptrs, o, mask=offs_m[:, None] < S)
    else:
        with tle.dsa.hint(inter_no_alias=True):
            tl.store(
                o_ptrs,
                o,
                mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
            )


def run(query, key, value, scaleValue=-1.0, is_causal=False):
    """Grouped-query attention forward pass over BSND tensors."""
    batch, seq_q, heads_q, head_dim = query.shape
    seq_kv = key.shape[1]
    heads_kv = key.shape[2]
    if heads_q % heads_kv != 0:
        raise AssertionError(
            f"N_q ({heads_q}) must be divisible by N_kv ({heads_kv})"
        )
    if key.shape != value.shape:
        raise AssertionError("key and value must have the same shape")
    if is_causal and seq_q > seq_kv:
        raise AssertionError(
            f"is_causal requires S ({seq_q}) <= S_kv ({seq_kv})"
        )

    scale = float(scaleValue) if scaleValue > 0 else 1.0 / (head_dim ** 0.5)

    out = torch.empty(
        (batch, seq_q, heads_q, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    if batch * heads_q * seq_q == 0:
        return out

    if head_dim <= 64:
        block_m, block_n = _BLOCK_M, _BLOCK_N
    else:
        block_m, block_n = _BLOCK_M_WIDE_D, _BLOCK_N_WIDE_D
    block_d = triton.next_power_of_2(head_dim)
    m_full = (seq_q % block_m) == 0
    n_full = (seq_kv % block_n) == 0

    grid = (triton.cdiv(seq_q, block_m), batch * heads_q)
    _gqa_fwd_kernel[grid](
        query, key, value, out,
        scale,
        seq_q, seq_kv, heads_q, heads_kv,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        IS_CAUSAL=bool(is_causal),
        M_FULL=m_full,
        N_FULL=n_full,
        NEED_BOUNDS=not (m_full and n_full),
        D=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        multibuffer=True,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        limit_auto_multi_buffer_of_local_buffer='no-limit',
    )
    return out
