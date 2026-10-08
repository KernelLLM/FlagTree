"""GLM KDA recurrent kernel — H=32 optimized for Ascend 910B2.

Optimization approach: single-kernel with auto-tuned BV, masked loads,
hoisted invariants, and varlen support. Masks are required on Ascend —
the compiler generates incorrect code for unmasked loads at BV=64 state sizes.

Strategy:
  - Auto-tuned BV ∈ {16, 32, 64, 128} × num_warps × num_stages (via @triton.autotune)
  - Masked loads/stores (required for correct Ascend compiler codegen)
  - Hoisted invariants: decay_base, dt_bias loaded once per program
  - Supports varlen (multiple packed sequences) via grid dim 1
  - Falls back to BV=64 reference kernel for non-qualifying cases

Autotuning control:
  - GLM53_KDA_AUTOTUNE=0  → disable autotune, use fixed BV from GLM53_KDA_BV_OPT
  - GLM53_KDA_AUTOTUNE=1  → enable autotune (default)
  - GLM53_KDA_BV_OPT=64   → fixed BV when autotune is off (default: 64)
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_BV_OPT = int(os.getenv("GLM53_KDA_BV_OPT", "64"))
_OPT_MIN_HEADS = 4  # minimum head count for optimized path


# ---------------------------------------------------------------------------
# Autotuning configuration
# ---------------------------------------------------------------------------
_AUTOTUNE = os.getenv("GLM53_KDA_AUTOTUNE", "1") != "0"


def _opt_autotune_configs():
    """Generate autotuning configs for the optimized recurrent kernel.

    Search space:
      - BV ∈ {16, 32, 64, 128}: tile size along value dim
        (128 may hit UB overflow on some Ascend configs — the autotuner
        catches compilation failures gracefully and skips those configs)
      - num_warps ∈ {1, 2, 4}: warp-level parallelism
      - num_stages ∈ {1, 3}: software pipeline depth

    When autotune is disabled, returns a single config using the env-specified BV.
    """
    if not _AUTOTUNE:
        return [triton.Config({"BV": _DEFAULT_BV_OPT}, num_warps=1, num_stages=1)]
    configs = []
    for bv in (16, 32, 64, 128):
        for num_warps in (1, 2, 4):
            for num_stages in (1, 3):
                configs.append(
                    triton.Config({"BV": bv}, num_warps=num_warps, num_stages=num_stages)
                )
    return configs


# ---------------------------------------------------------------------------
# Optimized single-kernel: auto-tuned recurrence with masked loads
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_opt_autotune_configs(),
    key=["H", "D", "TRACK_STATE"],
    restore_value=["STATE"],
)
@triton.jit
def _opt_recurrent_kernel(
    Q, K, V, A, B, ALOG, BIAS,
    OUT, STATE, INDICES, STARTS,
    TRACK_INDICES, TRACK_LENS,
    scale, lower_bound,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
):
    """Optimized per-token recurrence with hoisted invariants and masks.

    Grid: (D // BV, num_seqs, H)

    Each program handles one V-tile of one head across the full sequence.
    Per-head constants (A_log, dt_bias) loaded once outside the token loop.
    Masks are mandatory on Ascend for correct compiler codegen at BV>=64.
    """
    iv = tl.program_id(0)
    i_n = tl.program_id(1)
    head = tl.program_id(2)

    bos = tl.load(STARTS + i_n).to(tl.int64)
    eos = tl.load(STARTS + i_n + 1).to(tl.int64)
    seq_len = eos - bos

    ks = tl.arange(0, D)
    vs = iv * BV + tl.arange(0, BV)
    mask_k = ks < D
    mask_v = vs < D
    mask_h = mask_v[:, None] & mask_k[None, :]

    # Per-head constants — loaded once, reused for all T tokens
    b_alog = tl.load(ALOG + head).to(tl.float32)
    b_bias = tl.load(BIAS + head * D + ks, mask=mask_k, other=0.0).to(tl.float32)
    b_decay_base = tl.exp(b_alog)

    # Load initial state [BV, D]
    slot = tl.load(INDICES + i_n)
    pstate = STATE + slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
    state = tl.zeros((BV, D), tl.float32)
    if slot > 0:
        state = tl.load(pstate, mask=mask_h, other=0.0).to(tl.float32)

    if TRACK_STATE:
        track_slot = tl.load(TRACK_INDICES + i_n)
        track_len = tl.load(TRACK_LENS + i_n)
        ptrack = STATE + track_slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
        if track_slot > 0 and track_len == 0:
            tl.store(ptrack, state, mask=mask_h)

    # Hoisted pointer bases — only token offset varies in the loop
    q_base = Q + head * D + ks
    k_base = K + head * D + ks
    v_base = V + head * D + vs
    a_base = A + head * D + ks
    b_base = B + head
    o_base = OUT + head * D + vs
    stride_t = H * D
    stride_t_b = H

    for i in range(seq_len):
        t_off = (bos + i) * stride_t
        t_off_b = (bos + i) * stride_t_b

        # Per-token loads with masks (mandatory for Ascend correctness)
        q = tl.load(q_base + t_off, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_base + t_off, mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_base + t_off, mask=mask_v, other=0.0).to(tl.float32)
        a = tl.load(a_base + t_off, mask=mask_k, other=0.0).to(tl.float32)
        b = tl.load(b_base + t_off_b).to(tl.float32)

        # Inline precomputation — same math as reference
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= scale

        x = a + b_bias
        gate = lower_bound / (1.0 + tl.exp(-(b_decay_base * x)))
        decay = tl.exp(gate)
        beta = 1.0 / (1.0 + tl.exp(-b))

        # State recurrence
        state *= decay[None, :]
        v -= tl.sum(state * k[None, :], axis=1)
        v *= beta
        state += k[None, :] * v[:, None]

        if TRACK_STATE:
            if track_slot > 0 and i + 1 == track_len:
                tl.store(ptrack, state, mask=mask_h)

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(o_base + t_off, out.to(OUT.dtype.element_ty), mask=mask_v)

    if slot > 0:
        tl.store(pstate, state, mask=mask_h)


# ---------------------------------------------------------------------------
# Fallback: reference kernel (BV=64, bitwise-identical to upstream)
# ---------------------------------------------------------------------------
@triton.jit
def _glm_kda_varlen_recurrent_kernel(
    A_log, a, dt_bias, q, k, v, b, o,
    h0_source, h0_indices, cu_seqlens,
    scale, lower_bound,
    intermediate, track_indices, track_lens,
    S0: tl.constexpr, S1: tl.constexpr,
    SK: tl.constexpr, SV: tl.constexpr,
    M0: tl.constexpr, M1: tl.constexpr,
    M2: tl.constexpr, MK: tl.constexpr, MV: tl.constexpr,
    STEPS: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BHV: tl.constexpr,
):
    i_v, i_n, i_nhv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    seq_len = eos - bos

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    for i_bhv in range(BHV):
        i_hv = i_nhv * BHV + i_bhv
        i_h = i_hv // (HV // H)
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_b = b + bos * HV + i_hv
        p_o = o + (bos * HV + i_hv) * V + o_v
        p_A_log = A_log + i_hv
        p_dt_bias = dt_bias + i_hv * K + o_k

        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias, mask=mask_k).to(tl.float32)
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        state_index = tl.load(h0_indices + i_n)
        p_h0 = (
            h0_source + state_index * S0 + i_hv * S1
            + o_k[None, :] * SK + o_v[:, None] * SV
        )
        if state_index > 0:
            b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

        if TRACK_STATE:
            track_index = tl.load(track_indices + i_n)
            track_len = tl.load(track_lens + i_n)
            p_track = (
                h0_source + track_index * S0 + i_hv * S1
                + o_k[None, :] * SK + o_v[:, None] * SV
            )
            if track_index > 0 and track_len == 0:
                tl.store(p_track, b_h, mask=mask_h)

        for i in range(seq_len):
            b_q = tl.load(p_q + i * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_k = tl.load(p_k + i * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v + i * HV * V, mask=mask_v, other=0.0).to(tl.float32)
            b_a = tl.load(p_a + i * HV * K, mask=mask_k, other=0.0).to(tl.float32)
            b_beta = tl.load(p_b + i * HV).to(tl.float32)

            decay = tl.exp(b_A_log)
            x = b_a + b_dt_bias
            b_g = lower_bound / (1.0 + tl.exp(-(decay * x)))
            b_beta = 1.0 / (1.0 + tl.exp(-b_beta))

            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q *= scale

            b_h *= tl.exp(b_g[None, :])
            b_v -= tl.sum(b_h * b_k[None, :], axis=1)
            b_v *= b_beta
            b_h += b_k[None, :] * b_v[:, None]

            if TRACK_STATE:
                if track_index > 0 and i + 1 == track_len:
                    tl.store(p_track, b_h, mask=mask_h)
            if SAVE_INTERMEDIATE:
                p_mid = (
                    intermediate + i_n * M0 + i * M1 + i_hv * M2
                    + o_k[None, :] * MK + o_v[:, None] * MV
                )
                tl.store(p_mid, b_h, mask=mask_h & (state_index > 0))
            b_o = tl.sum(b_h * b_q[None, :], axis=1)
            tl.store(
                p_o + i * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v,
            )

        if not SAVE_INTERMEDIATE and state_index > 0:
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)



def _can_use_optimized_path(
    q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens,
    intermediate_state, track_state_indices, prefill,
    num_q_heads, num_value_heads, key_dim, value_dim,
):
    """Check whether the optimized path is safe to use."""
    if not prefill:
        return False
    if intermediate_state is not None:
        return False
    if num_value_heads != num_q_heads:
        return False
    if key_dim != 128 or value_dim != 128:
        return False
    if not all(t.dtype == torch.bfloat16 for t in (q, k, v, a, b)):
        return False
    if initial_state_source.dtype != torch.float32:
        return False
    if initial_state_source.ndim != 4:
        return False
    if initial_state_source.shape[1:] != (num_value_heads, key_dim, value_dim):
        return False
    if initial_state_source.device != q.device:
        return False
    if initial_state_indices.device != q.device:
        return False
    if initial_state_indices.dtype not in (torch.int32, torch.int64):
        return False
    if track_state_indices is not None:
        if track_state_indices.device != q.device:
            return False
        if track_state_indices.dtype not in (torch.int32, torch.int64):
            return False
    # Minimum tokens for overhead amortization
    if q.shape[0] < 64:
        return False
    return True


def _run_optimized(
    *, q, k, v, a, b,
    A_log, dt_bias, output, state, indices, starts,
    scale, lower_bound,
    track_indices, track_lens,
    key_dim, num_seqs,
):
    """Optimized single-kernel path with autotuned BV and masks."""
    heads = q.shape[1]
    D = key_dim
    track = track_indices is not None
    # Grid depends on BV which is selected by autotune — use lambda META
    grid = lambda META: (D // META["BV"], num_seqs, heads)
    _opt_recurrent_kernel[grid](
        q, k, v, a, b,
        A_log, dt_bias,
        output, state, indices, starts,
        track_indices if track else indices,
        track_lens if track else starts,
        scale, lower_bound,
        TRACK_STATE=track,
        H=heads, D=D,
        S0=state.stride(0), S1=state.stride(1),
        SK=state.stride(2), SV=state.stride(3),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def glm_kda_varlen_recurrent_npu(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: Optional[float],
    scale: Optional[float] = None,
    intermediate_state: Optional[torch.Tensor] = None,
    prefill: bool = False,
    track_state_indices: Optional[torch.Tensor] = None,
    track_state_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run packed varlen KDA with the optimized or reference kernel.

    Drop-in replacement for the upstream kda_recurrent_npu.py. Uses the
    optimized BV-tuned path when structural preconditions are met.
    Falls back to the BV=64 reference kernel otherwise.
    """
    # --- upstream input validation ---
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or a.ndim != 4:
        raise ValueError("GLM KDA varlen tensors must have shape [1,T,H,D]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or a.shape[0] != 1:
        raise ValueError("GLM KDA prefill expects one packed token batch")
    if (
        q.shape[:2] != k.shape[:2]
        or q.shape[1] != v.shape[1]
        or q.shape[1] != a.shape[1]
    ):
        raise ValueError("GLM KDA q/k/v/a token dimensions must match")
    if b.shape != v.shape[:-1]:
        raise ValueError("GLM KDA beta must have shape [1,T,HV]")

    num_q_heads, key_dim = k.shape[2:]
    num_value_heads, value_dim = v.shape[2:]
    if a.shape[2:] != (num_value_heads, key_dim):
        raise ValueError("GLM KDA gate must have shape [1,T,HV,K]")
    if num_value_heads % num_q_heads or num_value_heads % 2:
        raise ValueError("GLM KDA requires even HV divisible by H")
    if initial_state_indices.numel() != cu_seqlens.numel() - 1:
        raise ValueError("GLM KDA needs one state index per varlen sequence")
    if lower_bound is None or lower_bound >= 0:
        raise ValueError("GLM KDA requires a negative lower_bound")

    if (track_state_indices is None) != (track_state_lens is None):
        raise ValueError("GLM KDA prefix tracking needs both slots and lengths")
    if track_state_indices is not None:
        if not prefill or intermediate_state is not None:
            raise ValueError("GLM KDA prefix checkpoints are prefill-only")
        if (
            initial_state_source.ndim != 4
            or initial_state_source.shape[1:]
            != (num_value_heads, key_dim, value_dim)
            or initial_state_source.dtype != torch.float32
            or initial_state_source.device != q.device
            or initial_state_indices.device != q.device
        ):
            raise ValueError("GLM KDA prefix checkpoints need the native FP32 state pool")
        for t in (track_state_indices, track_state_lens):
            if (
                t.ndim != 1
                or t.numel() != initial_state_indices.numel()
                or t.dtype not in (torch.int32, torch.int64)
                or t.device != initial_state_indices.device
                or not t.is_contiguous()
            ):
                raise ValueError("Invalid GLM KDA prefix checkpoint metadata")

    # --- squeeze to [T, H, D] ---
    q, k, v, a, b = (t.squeeze(0) for t in (q, k, v, a, b))
    num_seqs = cu_seqlens.numel() - 1

    if scale is None:
        scale = key_dim**-0.5

    q, k, v, a, b = (
        t if t.is_contiguous() else t.contiguous() for t in (q, k, v, a, b)
    )

    # --- try optimized path ---
    if _can_use_optimized_path(
        q, k, v, a, b, initial_state_source, initial_state_indices,
        cu_seqlens, intermediate_state, track_state_indices, prefill,
        num_q_heads, num_value_heads, key_dim, value_dim,
    ):
        output = torch.empty_like(v)
        _run_optimized(
            q=q, k=k, v=v, a=a, b=b,
            A_log=A_log, dt_bias=dt_bias,
            output=output, state=initial_state_source,
            indices=initial_state_indices, starts=cu_seqlens,
            scale=scale, lower_bound=lower_bound,
            track_indices=track_state_indices,
            track_lens=track_state_lens,
            key_dim=key_dim, num_seqs=num_seqs,
        )
        return output.unsqueeze(0)

    # --- fallback: reference kernel (BV=64) ---
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 64)
    if triton.cdiv(key_dim, block_k) != 1:
        raise NotImplementedError(
            "GLM KDA key dimensions spanning tiles are unsupported"
        )

    if intermediate_state is not None:
        if (
            intermediate_state.ndim != 5
            or intermediate_state.shape[0] < initial_state_indices.numel()
            or intermediate_state.shape[2:] != (num_value_heads, key_dim, value_dim)
            or intermediate_state.dtype != initial_state_source.dtype
        ):
            raise ValueError("Invalid GLM KDA per-token intermediate state layout")

    output = torch.empty_like(v)
    heads_per_program = 1
    block_value_count = triton.cdiv(value_dim, block_v)
    grid = (
        block_value_count,
        num_seqs,
        num_value_heads // heads_per_program,
    )
    if track_state_indices is not None:
        effective_track_indices = track_state_indices
        effective_track_lens = track_state_lens
    else:
        effective_track_indices = torch.zeros_like(initial_state_indices)
        effective_track_lens = torch.zeros(
            initial_state_indices.numel(), dtype=torch.int32,
            device=initial_state_indices.device,
        )
    _glm_kda_varlen_recurrent_kernel[grid](
        A_log=A_log, a=a, dt_bias=dt_bias,
        q=q, k=k, v=v, b=b, o=output,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale, lower_bound=lower_bound,
        intermediate=(
            intermediate_state if intermediate_state is not None
            else initial_state_source
        ),
        track_indices=effective_track_indices,
        track_lens=effective_track_lens,
        S0=initial_state_source.stride(0),
        S1=initial_state_source.stride(1),
        SK=initial_state_source.stride(2),
        SV=initial_state_source.stride(3),
        M0=intermediate_state.stride(0) if intermediate_state is not None else 0,
        M1=intermediate_state.stride(1) if intermediate_state is not None else 0,
        M2=intermediate_state.stride(2) if intermediate_state is not None else 0,
        MK=intermediate_state.stride(3) if intermediate_state is not None else 0,
        MV=intermediate_state.stride(4) if intermediate_state is not None else 0,
        STEPS=intermediate_state.shape[1] if intermediate_state is not None else 0,
        SAVE_INTERMEDIATE=intermediate_state is not None,
        TRACK_STATE=True,
        H=num_q_heads, HV=num_value_heads,
        K=key_dim, V=value_dim,
        BK=block_k, BV=block_v, BHV=heads_per_program,
        num_warps=1, num_stages=3, multibuffer=False,
    )
    return output.unsqueeze(0)
