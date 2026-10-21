"""Fused RMSNorm + RoPE + paged KV-cache scatter implemented in TLE (Triton Language Extensions).

Geometry (DeepSeek MLA): per token the fused row is [ rms(512) | rope(64) ].

Reference semantics reproduced here:
  * RMS half   : var = sum(x*x) / 512 ; rrms = sqrt(var + eps) ; y = x / rrms * gamma
  * RoPE half  : x1 = kv[..., 512::2], x2 = kv[..., 512+1::2] (de-interleaved lanes),
                 first  = x1 * cos[:32] - x2 * sin[:32]
                 second = x2 * cos[32:] + x1 * sin[32:]
                 k_rope = cat([first, second])
  * Cache      : index is a flat token -> slot map; k_cache / ckv_cache are written in
                 place as flat (num_slots, last_dim) arrays at those slots.

TLE/DSA usage: the bulk, in-bounds, contiguous (BLOCK_M, 512) RMS tile of kv is staged
into UB with tle.dsa.copy, the two RoPE halves are concatenated into one 64-wide row with
tle.dsa.insert_slice, and (on full tiles) the contiguous k_rope / c_kv outputs are written
back through tle.dsa.to_buffer + tle.dsa.copy.  The paged-cache update is a data-dependent
gather scatter whose destination rows come from `index`, which a shape-based dsa.copy
cannot express, so it uses masked tl.store.

Layout facts that drove this structure (both measured on the target):
  * a stride-2 load pair makes the TritonToLinalgIncubated pass abort in
    InterleaveOptimization.cpp ("deinterleave offset" assertion), so the even/odd RoPE
    lanes are extracted from one contiguous load with tl.reshape + tl.split;
  * a (BLOCK_M, 32) masked store costs ~40 % more than a (BLOCK_M, 64) store of the same
    bytes on this target (every 64 B row segment becomes its own DMA transfer), hence the
    single concatenated 64-wide rope tile used for both the cache and the k_rope output.
    tl.join/tl.permute based concatenation is ~30x slower than tle.dsa.insert_slice here.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

RMS_SIZE = 512
ROPE_SIZE = 64
HALF_ROPE = ROPE_SIZE // 2
TOTAL_SIZE = RMS_SIZE + ROPE_SIZE


@triton.jit
def _kv_rmsnorm_rope_cache_kernel(
    kv_ptr,
    gamma_ptr,
    cos_ptr,
    sin_ptr,
    index_ptr,
    kcache_ptr,
    ckvcache_ptr,
    krope_ptr,
    ckv_ptr,
    num_tokens,
    epsilon,
    BLOCK_M: tl.constexpr,
    RMS: tl.constexpr,
    ROPE: tl.constexpr,
    HALF: tl.constexpr,
    TOT: tl.constexpr,
    IS_OUT: tl.constexpr,
    FULL: tl.constexpr,
    DT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < num_tokens
    rows_safe = tl.minimum(rows, num_tokens - 1)
    cmask = rmask[:, None]
    rc = tl.arange(0, RMS)
    hh = tl.arange(0, HALF)

    # ---- bulk in-bounds load of the RMS tile through TLE DSA (GM -> UB) ----
    rms_buf = tle.dsa.alloc([BLOCK_M, RMS], DT, tle.dsa.ascend.UB)
    tle.dsa.copy(kv_ptr + rows_safe[:, None] * TOT + rc[None, :], rms_buf, [BLOCK_M, RMS])

    # ---- every independent GM read is issued before any vector work, so the MTE can keep
    #      several transfers in flight instead of exposing each latency in program order ----
    rp = tl.load(
        kv_ptr + rows[:, None] * TOT + RMS + tl.arange(0, ROPE)[None, :], mask=cmask, other=0.0
    )
    c1 = tl.load(cos_ptr + rows[:, None] * ROPE + hh[None, :], mask=cmask, other=0.0)
    c2 = tl.load(cos_ptr + rows[:, None] * ROPE + HALF + hh[None, :], mask=cmask, other=0.0)
    s1 = tl.load(sin_ptr + rows[:, None] * ROPE + hh[None, :], mask=cmask, other=0.0)
    s2 = tl.load(sin_ptr + rows[:, None] * ROPE + HALF + hh[None, :], mask=cmask, other=0.0)
    gamma = tl.load(gamma_ptr + rc)
    slot = tl.load(index_ptr + rows, mask=rmask, other=0).to(tl.int32)

    xr = tle.dsa.to_tensor(rms_buf).to(tl.float32)
    gf = gamma.to(tl.float32)
    var = tl.sum(xr * xr, axis=1) * (1.0 / RMS)
    rrms = tl.sqrt(var + epsilon)
    y_dt = ((xr / rrms[:, None]) * gf[None, :]).to(DT)

    # ---- RoPE half: contiguous load, de-interleave even/odd lanes in registers ----
    x1, x2 = tl.split(tl.reshape(rp.to(tl.float32), (BLOCK_M, HALF, 2)))
    fh = (x1 * c1.to(tl.float32) - x2 * s1.to(tl.float32)).to(DT)
    sh = (x2 * c2.to(tl.float32) + x1 * s2.to(tl.float32)).to(DT)

    # one concatenated 64-wide rope row: [ first_half | second_half ]
    rope_tile = tle.dsa.insert_slice(
        tl.zeros([BLOCK_M, ROPE], dtype=DT), fh, [0, 0], [BLOCK_M, HALF], [1, 1]
    )
    rope_tile = tle.dsa.insert_slice(rope_tile, sh, [0, HALF], [BLOCK_M, HALF], [1, 1])

    # ---- contiguous outputs: bulk tile stores via TLE DSA on full tiles ----
    if IS_OUT:
        if FULL:
            kb = tle.dsa.to_buffer(rope_tile, tle.dsa.ascend.UB)
            tle.dsa.copy(kb, krope_ptr + rows[:, None] * ROPE + tl.arange(0, ROPE)[None, :],
                         [BLOCK_M, ROPE])
            yb = tle.dsa.to_buffer(y_dt, tle.dsa.ascend.UB)
            tle.dsa.copy(yb, ckv_ptr + rows[:, None] * RMS + rc[None, :], [BLOCK_M, RMS])
        else:
            tl.store(krope_ptr + rows[:, None] * ROPE + tl.arange(0, ROPE)[None, :], rope_tile,
                     mask=cmask)
            tl.store(ckv_ptr + rows[:, None] * RMS + rc[None, :], y_dt, mask=cmask)

    # ---- paged cache update: data-dependent slot rows -> masked scatter stores ----
    tl.store(kcache_ptr + slot[:, None] * ROPE + tl.arange(0, ROPE)[None, :], rope_tile,
             mask=cmask)
    tl.store(ckvcache_ptr + slot[:, None] * RMS + rc[None, :], y_dt, mask=cmask)


def _pick_block_m(num_tokens):
    return 16


def run(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    k_rope_scale,
    c_kv_scale,
    k_rope_offset,
    c_kv_offset,
    epsilon,
    cache_mode,
    is_output_kv,
):
    if kv.shape[-1] != TOTAL_SIZE:
        raise ValueError(f"kv last dimension must be {TOTAL_SIZE}, got {kv.shape[-1]}")
    if cache_mode not in ("PA", "PA_BNSD"):
        raise NotImplementedError(f"unsupported cache_mode {cache_mode!r}")
    if any(t is not None for t in (k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset)):
        raise NotImplementedError("quantized scale/offset path is not implemented")

    out_dtype = kv.dtype
    if out_dtype == torch.float16:
        dt = tl.float16
    elif out_dtype == torch.bfloat16:
        dt = tl.bfloat16
    else:
        raise NotImplementedError(f"unsupported dtype {out_dtype}")

    kc = k_cache.view(-1, ROPE_SIZE)
    cc = ckv_cache.view(-1, RMS_SIZE)

    num_tokens = kv.numel() // TOTAL_SIZE
    lead = kv.shape[:-1]
    if is_output_kv:
        k_rope = torch.empty(lead + (ROPE_SIZE,), dtype=out_dtype, device=kv.device)
        c_kv = torch.empty(lead + (RMS_SIZE,), dtype=out_dtype, device=kv.device)
        krope_arg = k_rope
        ckv_arg = c_kv
    else:
        k_rope = None
        c_kv = None
        krope_arg = kc
        ckv_arg = cc

    block_m = _pick_block_m(num_tokens)
    grid = (triton.cdiv(num_tokens, block_m),)
    _kv_rmsnorm_rope_cache_kernel[grid](
        kv,
        gamma,
        cos,
        sin,
        index,
        kc,
        cc,
        krope_arg,
        ckv_arg,
        num_tokens,
        float(epsilon),
        BLOCK_M=block_m,
        RMS=RMS_SIZE,
        ROPE=ROPE_SIZE,
        HALF=HALF_ROPE,
        TOT=TOTAL_SIZE,
        IS_OUT=bool(is_output_kv),
        FULL=(num_tokens % block_m == 0),
        DT=dt,
        multibuffer=True,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    return k_cache, ckv_cache, k_rope, c_kv
