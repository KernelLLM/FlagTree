#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_ops.py -- standalone latency benchmark for the six CommonIR Ascend ops.

No reference implementation is required.  The script measures each operator on
its own, and is built so that a *before/after* comparison of a kernel change is
the primary use case:

    python bench_ops.py run --outdir out/before --label before
    # ... apply the change / check out the fixed tree ...
    python bench_ops.py run --outdir out/after  --label after
    python bench_ops.py compare --before out/before --after out/after

Three timing modes, because they answer different questions:

    loop        wall clock of ``run(**kwargs)`` -- what a caller sees,
                host bookkeeping included
    npu-graph   capture once, time ``graph.replay()`` -- device only.  Fails for
                ops that synchronise the host inside ``run()`` (then it reports
                ``graph_capture_failed`` and falls back to loop)
    host-split  wall clock, plus the same call with every Triton launch stubbed
                out; the difference is host cost (including any D2H sync)

Numerical regression guard without a reference: every run records an fp64
digest (sum, sum of squares, absmax, non-finite count) of each output tensor.
``compare`` reports whether the digest matches, so a change that alters the
numbers is visible next to the timings.  Tile changes legitimately perturb the
fp32 reduction order, so a digest is reported as identical / delta, not
pass/fail.

Every measurement runs in its own subprocess: a kernel that faults the device
(``aivec error``) or hangs cannot take the rest of the benchmark with it.

Usage examples
--------------
    python bench_ops.py list
    python bench_ops.py selfcheck                       # builds every case on CPU
    python bench_ops.py run --label before --outdir out/b
    python bench_ops.py run --ops gqa,compressor --cases default --measure-mode npu-graph
    python bench_ops.py run --const _BLOCK_T=64          # ad-hoc A/B on one knob
    python bench_ops.py compare --before out/b --after out/a
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)

OP_ORDER = (
    "causal_conv1d_fn",
    "compressor",
    "gqa",
    "kv_rmsnorm_rope_cache",
    "lightning_indexer",
    "sparse_attn_sharedkv",
)

# =========================================================================== #
# Case registry
# =========================================================================== #
#
# The shapes are the ones that showed a measurable difference while the rewrite
# was compared against its non-CommonIR counterpart, so they are the ones worth
# tracking across a kernel change.  `quick` is for a smoke run, `full` adds the
# heavier production shapes.

CASES = {
    "causal_conv1d_fn": {
        "quick": ("b4_d64_s8_pad", "b4_d4096_s249"),
        "default": (
            "b4_d64_s8_pad",
            "b4_d4096_s249",
            "b10_d4096_s1024_pad_fp16",
            "b2_d256_s512_nopad",
        ),
        "full": (
            "b4_d64_s8_pad",
            "b4_d4096_s249",
            "b10_d4096_s1024_pad_fp16",
            "b2_d256_s512_nopad",
            "b10_d4096_s4096_pad_fp16",
            "b2_d128_s2048_avg_long",
        ),
    },
    "compressor": {
        "quick": ("prefill_small", "decode3"),
        "default": (
            "prefill_small",
            "decode3",
            "prefill0_b2_rank3",
            "prefill1_hd128",
            "decode5_c128",
        ),
        "full": (
            "prefill_small",
            "decode3",
            "prefill0",
            "prefill0_b2_rank3",
            "prefill1_hd128",
            "prefill2_c128",
            "decode0_start8195",
            "decode5_c128",
        ),
    },
    "gqa": {
        "quick": ("b1_s128_h16x8_d128_causal", "b1_s1024_h8x2_d64_noncausal"),
        "default": (
            "b1_s128_h16x8_d128_causal",
            "b1_s1024_h8x2_d64_noncausal",
            "b2_s256x512_h32x8_d128_causal",
            "b4_s512_h32x4_d64_causal_fp16",
        ),
        "full": (
            "b1_s128_h16x8_d128_causal",
            "b1_s1024_h8x2_d64_noncausal",
            "b2_s256x512_h32x8_d128_causal",
            "b4_s512_h32x4_d64_causal_fp16",
            "b1_s2048_h64x8_d128_causal",
            "b1_s1x1024_h64x1_d128_decode",
        ),
    },
    "kv_rmsnorm_rope_cache": {
        "quick": ("decode_unaligned_fp16", "prefill_fp16"),
        "default": ("decode_unaligned_fp16", "prefill_fp16", "decode_bf16"),
        "full": (
            "decode_aligned_fp16",
            "decode_unaligned_fp16",
            "prefill_fp16",
            "decode_bf16",
        ),
    },
    "lightning_indexer": {
        "quick": ("R1", "R3"),
        "default": ("R1", "R3", "R6", "R8", "R11"),
        "full": ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10", "R11"),
    },
    "sparse_attn_sharedkv": {
        "quick": ("scfa_decode", "swa_decode", "cfa_decode"),
        "default": (
            "scfa_decode",
            "swa_decode",
            "cfa_decode",
            "swa_prefill_small",
            "cfa_prefill_small",
        ),
        "full": (
            "scfa_decode",
            "swa_decode",
            "cfa_decode",
            "swa_prefill",
            "cfa_prefill",
        ),
    },
}
for _op in OP_ORDER:
    CASES[_op].setdefault("heavy", ())

# causal_conv1d_fn: (batch, dim, total_len, with_pad, dtype, has_bias, silu)
CC1D = {
    "b4_d64_s8_pad": (4, 64, 8, True, "bfloat16", True, True),
    "b4_d4096_s249": (4, 4096, 249, False, "bfloat16", True, True),
    "b10_d4096_s1024_pad_fp16": (10, 4096, 1024, True, "float16", True, True),
    "b2_d256_s512_nopad": (2, 256, 512, False, "bfloat16", True, True),
    "b10_d4096_s4096_pad_fp16": (10, 4096, 4096, True, "float16", True, True),
    "b2_d128_s2048_avg_long": (2, 128, 2048, True, "bfloat16", True, True),
}
# compressor: (batch, hidden, seq, head_dim, block_size, rope_dim, ratio, coff, start)
COMPRESSOR = {
    "prefill0": (1, 4096, 8192, 512, 128, 64, 4, 2, 0),
    "prefill0_b2": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "prefill0_b2_rank3": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "prefill1_hd128": (1, 4096, 8192, 128, 128, 64, 4, 2, 0),
    "prefill2_c128": (1, 4096, 8192, 512, 128, 64, 128, 1, 0),
    "decode0_start8195": (1, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "decode3": (8, 4096, 3, 512, 128, 64, 4, 2, 8193),
    "decode5_c128": (8, 4096, 3, 512, 128, 64, 128, 1, 8193),
    "prefill_small": (1, 2048, 1024, 512, 128, 64, 4, 2, 0),
}
COMPRESSOR_RANK3 = {"prefill0_b2_rank3"}
# gqa: (batch, seq_q, seq_kv, heads_q, heads_kv, head_dim, is_causal, dtype)
GQA = {
    "b1_s128_h16x8_d128_causal": (1, 128, 128, 16, 8, 128, True, "bfloat16"),
    "b1_s1024_h8x2_d64_noncausal": (1, 1024, 1024, 8, 2, 64, False, "bfloat16"),
    "b2_s256x512_h32x8_d128_causal": (2, 256, 512, 32, 8, 128, True, "bfloat16"),
    "b4_s512_h32x4_d64_causal_fp16": (4, 512, 512, 32, 4, 64, True, "float16"),
    "b1_s2048_h64x8_d128_causal": (1, 2048, 2048, 64, 8, 128, True, "bfloat16"),
    "b1_s1x1024_h64x1_d128_decode": (1, 1, 1024, 64, 1, 128, True, "bfloat16"),
}
# kv_rmsnorm_rope_cache: (batch, seq_len, dtype)
KVRMS = {
    "decode_aligned_fp16": (16384, 1, "float16"),
    "decode_unaligned_fp16": (3, 1, "float16"),
    "prefill_fp16": (64, 256, "float16"),
    "decode_bf16": (4096, 1, "bfloat16"),
}
# lightning_indexer: (query_t, local_q_lengths, key_lengths)
LI = {
    "R1": (1, (1,), (8,)),
    "R2": (16, (16,), (128,)),
    "R3": (256, (256,), (2048,)),
    "R4": (1024, (1024,), (8192,)),
    "R5": (1, (1, 0, 0, 0), (130, 0, 0, 0)),
    "R6": (1, (1, 0, 0, 0), (2050, 0, 0, 0)),
    "R7": (1, (1, 0, 0, 0), (8194, 0, 0, 0)),
    "R8": (33, (1, 16, 16), (130, 128, 128)),
    "R10": (2, (1, 1, 0, 0, 0, 0, 0, 0), (134, 132, 132, 130, 130, 130, 130, 130)),
    "R11": (2, (1, 1, 0, 0, 0, 0, 0, 0), (136, 136, 136, 136, 136, 0, 0, 0)),
}
# sparse_attn_sharedkv: (mode, q_len, kv_len, cmp_ratio, topk)
SAS = {
    "scfa_decode": ("SCFA", 1, 8193, 4, 512),
    "swa_decode": ("SWA", 1, 8193, 0, 0),
    "cfa_decode": ("CFA", 1, 8193, 128, 0),
    "scfa_prefill": ("SCFA", 8192, 8192, 4, 512),
    "swa_prefill": ("SWA", 8192, 8192, 0, 0),
    "cfa_prefill": ("CFA", 8192, 8192, 128, 0),
    "swa_prefill_small": ("SWA", 512, 512, 0, 0),
    "cfa_prefill_small": ("CFA", 512, 512, 128, 0),
}

# Constants that reproduce the pre-port configuration of the operators, so the
# "before" side of a comparison can be measured on the current tree without a
# git checkout.  This is exact because the port is *only* module constants plus
# gates whose conditions read the operator's inputs, and the functional diff
# against the merge commit is exactly: 128/128 (compressor projection), 64/64
# (gqa narrow tile) and 32 (sparse compressed head block) - restoring those
# reproduces the old code path.  `bench_ops.py ab --before-ref <merge>` remains
# the independent check.
PRE_PORT_CONSTS = {
    "_PROJ_BN": 128,         # compressor: projection N tile
    "_PROJ_BK": 128,         # compressor: projection K tile
    "_BLOCK_M_NARROW": 128,  # gqa: narrow tile, i.e. no 64x64 gate
    "_BLOCK_N_NARROW": 128,
    "_HEAD_BLOCK_CMP": 32,   # sparse_attn_sharedkv: no 16-head cmp block
}


def _const_string(consts):
    return ",".join(f"{k}={v}" for k, v in consts.items())


SPECS = {
    "causal_conv1d_fn": CC1D,
    "compressor": COMPRESSOR,
    "gqa": GQA,
    "kv_rmsnorm_rope_cache": KVRMS,
    "lightning_indexer": LI,
    "sparse_attn_sharedkv": SAS,
}


def _dt(name):
    import torch

    return getattr(torch, name)


def _log(msg):
    print(msg, flush=True)


def _json_load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _json_dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _sync(device):
    import torch

    if str(device).startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================================================================== #
# Input builders  (deterministic: same seed -> same tensors on every run)
# =========================================================================== #


def build_causal_conv1d_fn(case_id, device):
    import numpy as np
    import torch

    batch, dim, total_len, with_pad, dtype_s, has_bias, silu = CC1D[case_id]
    dtype = _dt(dtype_s)
    width, pad = 4, (3 if with_pad else 0)
    padded = batch + pad

    import random

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    eos = torch.randperm(total_len - 1)[: padded - 1].sort().values
    seqlens = torch.diff(
        torch.cat([torch.tensor([-1]), eos, torch.tensor([total_len - 1])])
    ).tolist()
    cumsum = torch.cumsum(torch.tensor(seqlens), dim=0).to(torch.int32)
    qsl = torch.cat([torch.tensor([0], dtype=torch.int32), cumsum]).to(device)

    pre = 64
    x = torch.randn(1, total_len, pre + dim, device=device, dtype=dtype)
    x = x.transpose(1, 2)[:, pre : pre + dim, :]          # non-contiguous on purpose
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    entries = batch * 10
    conv_states = torch.randn(entries, width - 1, dim, device=device, dtype=dtype).transpose(1, 2)
    has_init = torch.randint(0, 2, (padded,), dtype=torch.bool, device=device)
    idx = torch.randperm(entries, dtype=torch.int32, device=device)[:batch]
    cache_indices = torch.cat(
        [idx, torch.full((pad,), -1, dtype=torch.int32, device=device)]
    )

    kw = dict(
        x=x.squeeze(0),
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=qsl,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        activation="silu" if silu else None,
        pad_slot_id=-1,
    )
    return kw, dict(slots=[int(v) for v in idx.tolist()])


def _compressor_layout(case_id):
    from itertools import accumulate

    batch, hidden, seq, head_dim, block, rope, ratio, coff, start = COMPRESSOR[case_id]

    def per(v):
        return (v,) * batch if isinstance(v, int) else tuple(v)

    seq_lens, starts = per(seq), per(start)
    return dict(
        batch=batch, hidden=hidden, seq_lens=seq_lens, head_dim=head_dim,
        block_size=block, rope_dim=rope, ratio=ratio, coff=coff, starts=starts,
        rank3=case_id.endswith("_rank3") or case_id in COMPRESSOR_RANK3,
        cumulative=list(accumulate(seq_lens)),
    )


def build_compressor(case_id, device):
    import torch

    lay = _compressor_layout(case_id)
    batch, hidden, ratio = lay["batch"], lay["hidden"], lay["ratio"]
    head_dim, coff, rope = lay["head_dim"], lay["coff"], lay["rope_dim"]
    block, seq_lens, starts = lay["block_size"], lay["seq_lens"], lay["starts"]
    total = sum(seq_lens)
    proj = coff * head_dim
    dtype = torch.bfloat16
    torch.manual_seed(42)

    def uniform(shape, lo, hi, dt):
        return (torch.rand(shape, device=device, dtype=torch.float32) * (hi - lo) + lo).to(dt)

    max_pos = max(s + n for s, n in zip(starts, seq_lens))
    max_blocks = (max_pos + block - 1) // block
    table = torch.arange(1, batch * max_blocks + 1, dtype=torch.int32).reshape(batch, max_blocks)
    state_blocks = batch * max_blocks + 1
    out_per_batch = (seq_lens[0] + ratio - 1) // ratio
    rope_rows = (
        batch * out_per_batch if lay["rank3"] else min(total, total // ratio + batch)
    )

    x = uniform((total, hidden), -10, 10, dtype)
    sin = uniform((rope_rows, rope), -1, 1, dtype)
    cos = uniform((rope_rows, rope), -1, 1, dtype)
    if lay["rank3"]:
        x = x.reshape(batch, seq_lens[0], hidden)
        sin = sin.reshape(batch, out_per_batch, rope)
        cos = cos.reshape(batch, out_per_batch, rope)

    kw = dict(
        x=x,
        wkv=uniform((proj, hidden), -10, 10, dtype),
        wgate=uniform((proj, hidden), -10, 10, dtype),
        state_cache=uniform((state_blocks, block, 2 * proj), -10, 10, torch.float32),
        ape=uniform((ratio, proj), -10, 10, torch.float32),
        norm_weight=uniform((head_dim,), -10, 10, dtype),
        rope_sin=sin,
        rope_cos=cos,
        state_block_table=table.to(device=device),
        cu_seqlens=None
        if lay["rank3"]
        else torch.tensor([0, *lay["cumulative"][:-1], total], dtype=torch.int32, device=device),
        seqused=None,
        start_pos=torch.tensor(starts, dtype=torch.int32, device=device),
        rope_head_dim=rope,
        cmp_ratio=ratio,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )
    return kw, {}


def build_gqa(case_id, device):
    import torch

    batch, sq, skv, hq, hkv, hd, causal, dtype_s = GQA[case_id]
    dtype = _dt(dtype_s)
    torch.manual_seed(7)

    def r(*shape):
        return (torch.rand(*shape, device=device, dtype=torch.float32) * 2 - 1).to(dtype)

    return dict(
        query=r(batch, sq, hq, hd),
        key=r(batch, skv, hkv, hd),
        value=r(batch, skv, hkv, hd),
        scaleValue=-1.0,
        is_causal=causal,
    ), {}


def build_kv_rmsnorm_rope_cache(case_id, device):
    import torch

    batch, seq_len, dtype_s = KVRMS[case_id]
    dtype = _dt(dtype_s)
    rms, rope, page = 512, 64, 128
    tokens = batch * seq_len
    pages = math.ceil(tokens / page) + 8
    torch.manual_seed(0)
    kw = dict(
        kv=torch.randn(batch, 1, seq_len, rms + rope, dtype=dtype, device=device),
        gamma=torch.ones(rms, dtype=dtype, device=device),
        cos=torch.randn(batch, 1, seq_len, rope, dtype=dtype, device=device),
        sin=torch.randn(batch, 1, seq_len, rope, dtype=dtype, device=device),
        index=torch.arange(tokens, dtype=torch.int64, device=device),
        k_cache=torch.zeros(pages, page, 1, rope, dtype=dtype, device=device),
        ckv_cache=torch.zeros(pages, page, 1, rms, dtype=dtype, device=device),
        k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
        epsilon=1e-5, cache_mode="PA_BNSD", is_output_kv=True,
    )
    return kw, dict(rms=rms, rope=rope)


LI_HEADS, LI_DIM, LI_BLOCK, LI_TOPK = 64, 128, 128, 512


def build_lightning_indexer(case_id, device):
    import torch

    q_t, local_q, key_lens = LI[case_id]
    torch.manual_seed(3)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(3)
        except Exception:
            pass
    blocks = sum(math.ceil(k / LI_BLOCK) for k in key_lens)
    nblocks = blocks + 8
    width = max(2, max(math.ceil(k / LI_BLOCK) for k in key_lens))

    query = torch.empty((q_t, LI_HEADS, LI_DIM), dtype=torch.bfloat16, device=device).uniform_(-8, 8)
    key = torch.empty((nblocks, LI_BLOCK, 1, LI_DIM), dtype=torch.bfloat16, device=device).uniform_(-8, 8)
    weights = torch.empty((q_t, LI_HEADS), dtype=torch.bfloat16, device=device).uniform_(-1, 1)

    table = torch.zeros((len(key_lens), width), dtype=torch.int32, device=device)
    nxt = 0
    for r, klen in enumerate(key_lens):
        nb = math.ceil(klen / LI_BLOCK)
        if nb:
            table[r, :nb] = torch.arange(nxt, nxt + nb, dtype=torch.int32, device=device)
        nxt += nb
    ends, total = [], 0
    for n in local_q:
        total += int(n)
        ends.append(total)

    kw = dict(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=torch.tensor(ends, dtype=torch.int32, device=device),
        actual_seq_lengths_key=torch.tensor(list(key_lens), dtype=torch.int32, device=device),
        block_table=table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=LI_TOPK,
        sparse_mode=3,
        pre_tokens=9223372036854775807,
        next_tokens=9223372036854775807,
        return_value=False,
    )
    return kw, {}


SAS_QH, SAS_KVH, SAS_D, SAS_PAGE, SAS_SCALE = 64, 1, 512, 128, 0.04419417


def _sas_paged(tokens, gen, device):
    import torch

    pages = math.ceil(tokens / SAS_PAGE)
    table = torch.randperm(pages, generator=gen, dtype=torch.int32, device="cpu").view(1, pages)
    kv = torch.rand(
        (pages, SAS_PAGE, SAS_KVH, SAS_D), generator=gen, dtype=torch.float32, device="cpu"
    )
    return (kv * 15 - 5).to(torch.bfloat16).to(device), table.to(device)


def build_sparse_attn_sharedkv(case_id, device):
    import torch

    mode, q_len, kv_len, ratio, topk = SAS[case_id]
    has_cmp = mode != "SWA"
    has_sparse = mode == "SCFA"
    gen = torch.Generator(device="cpu")
    gen.manual_seed(43)

    q = torch.rand((q_len, SAS_QH, SAS_D), generator=gen, dtype=torch.float32, device="cpu")
    q = (q * 20 - 10).to(torch.bfloat16).to(device)
    ori_kv, ori_bt = _sas_paged(kv_len, gen, device)

    cmp_kv = cmp_bt = sparse_idx = None
    if has_cmp:
        cmp_kv, cmp_bt = _sas_paged(kv_len // ratio, gen, device)
        if has_sparse:
            idx = torch.full((q_len, SAS_KVH, topk), -1, dtype=torch.int32, device="cpu")
            for p in range(q_len):
                visible = kv_len // ratio if q_len == 1 else (p + 1) // ratio
                count = min(visible, topk)
                if count:
                    idx[p, 0, :count] = torch.randperm(
                        visible, generator=gen, dtype=torch.int32, device="cpu"
                    )[:count]
            sparse_idx = idx.to(device)

    sinks = torch.rand((SAS_QH,), generator=gen, dtype=torch.float32, device="cpu").to(device)
    return dict(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=sparse_idx,
        ori_block_table=ori_bt,
        cmp_block_table=cmp_bt,
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32, device=device),
        seqused_kv=torch.tensor([kv_len], dtype=torch.int32, device=device),
        sinks=sinks,
        softmax_scale=SAS_SCALE,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    ), {}


BUILDERS = {
    "causal_conv1d_fn": build_causal_conv1d_fn,
    "compressor": build_compressor,
    "gqa": build_gqa,
    "kv_rmsnorm_rope_cache": build_kv_rmsnorm_rope_cache,
    "lightning_indexer": build_lightning_indexer,
    "sparse_attn_sharedkv": build_sparse_attn_sharedkv,
}


# =========================================================================== #
# Outputs that get digested (the return value plus everything mutated in place)
# =========================================================================== #


def collect_causal_conv1d_fn(kw, meta, result):
    import torch

    slots = torch.tensor(meta["slots"], dtype=torch.long, device=kw["conv_states"].device)
    return {
        "out": result,
        "conv_states": kw["conv_states"].index_select(0, slots),
    }


def collect_compressor(kw, meta, result):
    return {"out": result, "state_cache": kw["state_cache"]}


def collect_gqa(kw, meta, result):
    return {"out": result}


def collect_kv_rmsnorm_rope_cache(kw, meta, result):
    import torch

    slot = kw["index"].to(torch.long).reshape(-1)
    return {
        "k_rope": result[2],
        "c_kv": result[3],
        "k_cache": kw["k_cache"].view(-1, meta["rope"]).index_select(0, slot),
        "ckv_cache": kw["ckv_cache"].view(-1, meta["rms"]).index_select(0, slot),
    }


def collect_lightning_indexer(kw, meta, result):
    return {"indices": result[0]}


def collect_sparse_attn_sharedkv(kw, meta, result):
    return {"out": result[0]}


COLLECTORS = {
    "causal_conv1d_fn": collect_causal_conv1d_fn,
    "compressor": collect_compressor,
    "gqa": collect_gqa,
    "kv_rmsnorm_rope_cache": collect_kv_rmsnorm_rope_cache,
    "lightning_indexer": collect_lightning_indexer,
    "sparse_attn_sharedkv": collect_sparse_attn_sharedkv,
}


def digest(tensors):
    """Order-insensitive fp64 fingerprint of the outputs.

    A kernel change that perturbs the fp32 reduction order moves the last bits
    of these sums, so they are reported as 'identical' or as a relative delta --
    never as pass/fail.
    """
    import torch

    out = {}
    for name, t in tensors.items():
        v = t.detach().float().reshape(-1)
        if v.numel() == 0:
            out[name] = dict(numel=0, sum=0.0, sumsq=0.0, absmax=0.0, nonfinite=0)
            continue
        nonfinite = int((~torch.isfinite(v)).sum())
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).double()
        out[name] = dict(
            numel=int(v.numel()),
            sum=float(v.sum()),
            sumsq=float((v * v).sum()),
            absmax=float(v.abs().max()),
            nonfinite=nonfinite,
        )
    return out


# =========================================================================== #
# Measurement
# =========================================================================== #


def _measure_loop(fn, device, warmup, iters, rounds):
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(rounds):
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    samples.sort()
    n = len(samples)
    return dict(
        mode="loop", iters=iters, rounds=rounds,
        median_ms=samples[n // 2], min_ms=samples[0],
        mean_ms=sum(samples) / n, samples_ms=samples,
    )


def _jit_classes():
    out = []
    for modname in (
        "triton.runtime",
        "triton.runtime.jit",
        "triton.spec.ascend.runtime.jit",
        "triton.backends.ascend.runtime.jit",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, "JITFunction", None)
        if cls is not None and cls not in out:
            out.append(cls)
    return out


def _measure_host_split(fn, device, warmup, iters, rounds):
    total = _measure_loop(fn, device, warmup, iters, rounds)
    classes = _jit_classes()
    saved = [(c, c.run) for c in classes]

    def stub_run(self, *a, **k):
        return None

    for c, _ in saved:
        c.run = stub_run
    try:
        host = _measure_loop(fn, device, warmup, iters, rounds)
    finally:
        for c, orig in saved:
            c.run = orig
    return dict(
        mode="host-split", iters=iters, rounds=rounds,
        median_ms=total["median_ms"], min_ms=total["min_ms"],
        mean_ms=total["mean_ms"], host_ms=host["median_ms"],
        device_ms=max(total["median_ms"] - host["median_ms"], 0.0),
        samples_ms=total["samples_ms"],
    )


def _measure_graph(fn, device, warmup, iters, rounds):
    import torch

    mod = torch.npu if str(device).startswith("npu") else torch.cuda
    graph_cls = getattr(mod, "NPUGraph", None) or getattr(mod, "CUDAGraph", None)
    if graph_cls is None or not hasattr(mod, "graph"):
        raise RuntimeError(f"{device} backend exposes no capture API")
    side = mod.Stream()
    side.wait_stream(mod.current_stream())
    with mod.stream(side):
        for _ in range(warmup):
            fn()
    mod.current_stream().wait_stream(side)
    mod.synchronize()
    graph = graph_cls()
    with mod.graph(graph):
        fn()
    mod.synchronize()
    samples = []
    for _ in range(rounds):
        mod.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        mod.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    samples.sort()
    n = len(samples)
    return dict(
        mode="graph", iters=iters, rounds=rounds,
        median_ms=samples[n // 2], min_ms=samples[0],
        mean_ms=sum(samples) / n, samples_ms=samples,
    )


# =========================================================================== #
# Worker: one (op, case) in its own process
# =========================================================================== #


def _load_op(op, op_dir):
    path = os.path.join(op_dir, op + ".py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"operator not found: {path} (use --op-dir)")
    name = f"_bench_op_{op}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{path} has no top-level run()")
    return mod


def _parse_consts(spec):
    out = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--const expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        v = v.strip()
        for conv in (int, float):
            try:
                out[k.strip()] = conv(v)
                break
            except ValueError:
                continue
        else:
            out[k.strip()] = v
    return out


def worker_main(args):
    import torch

    op, case_id, device = args.op, args.case, args.device
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401

            if args.device_index is not None:
                torch.npu.set_device(args.device_index)
        except Exception as exc:
            _json_dump(
                _raw(args.outdir, op, case_id, args.tag),
                dict(op=op, case=case_id, status="error",
                     error=f"torch_npu unavailable: {exc}"),
            )
            return 1

    rec = dict(op=op, case=case_id, tag=args.tag, device=device, status="ok",
               consts={}, consts_ignored=[])
    try:
        kw, meta = BUILDERS[op](case_id, device)
        mod = _load_op(op, args.op_dir)
        # A const list is shared by every operator in a run, so names an
        # operator does not have are skipped - but recorded, and fatal under
        # --strict-const, so a typo cannot pass silently.
        for name, value in _parse_consts(args.const).items():
            if hasattr(mod, name):
                setattr(mod, name, value)
                rec["consts"][name] = value
            else:
                rec["consts_ignored"].append(name)
        if rec["consts_ignored"] and args.strict_const:
            raise AttributeError(
                f"{op}.py has no module constant(s) {rec['consts_ignored']} "
                f"(has: {[a for a in dir(mod) if a.isupper()]})"
            )

        def invoke():
            return mod.run(**kw)

        with torch.inference_mode():
            _sync(device)
            result = invoke()
            _sync(device)
            rec["digest"] = digest(COLLECTORS[op](kw, meta, result))
        del result
        if device.startswith("npu") and hasattr(torch, "npu"):
            torch.npu.empty_cache()

        if args.no_bench:
            rec["bench"] = dict(mode="skipped")
        else:
            with torch.inference_mode():
                try:
                    if args.measure_mode == "npu-graph":
                        bench = _measure_graph(invoke, device, args.warmup, args.iters, args.rounds)
                    elif args.measure_mode == "host-split":
                        bench = _measure_host_split(invoke, device, args.warmup, args.iters, args.rounds)
                    else:
                        bench = _measure_loop(invoke, device, args.warmup, args.iters, args.rounds)
                except Exception as exc:
                    bench = _measure_loop(invoke, device, args.warmup, args.iters, args.rounds)
                    bench["graph_capture_failed"] = f"{type(exc).__name__}: {exc}"[:300]
            if device.startswith("npu") and hasattr(torch, "npu"):
                try:
                    bench["peak_alloc_mb"] = round(torch.npu.max_memory_allocated() / 2**20, 2)
                except Exception:
                    pass
            rec["bench"] = bench
        rec["env"] = {"torch": torch.__version__, "host": platform.node()}
        try:
            import triton

            rec["env"]["triton"] = triton.__version__
        except Exception:
            pass
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        _log(rec["traceback"])
    _json_dump(_raw(args.outdir, op, case_id, args.tag), rec)
    return 0 if rec["status"] == "ok" else 1


def _raw(outdir, op, case_id, tag):
    stem = f"{op}__{case_id}" + (f"@{tag}" if tag else "")
    return os.path.join(outdir, "raw", stem + ".json")


# =========================================================================== #
# Driver
# =========================================================================== #


def _cases_for(op, spec):
    if spec in CASES[op]:
        return list(CASES[op][spec])
    wanted = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in CASES[op]:
            wanted.extend(CASES[op][tok])
        elif tok:
            wanted.append(tok)
    unknown = [c for c in wanted if c not in SPECS[op]]
    if unknown:
        raise SystemExit(f"unknown case(s) for {op}: {unknown}\navailable: {sorted(SPECS[op])}")
    return wanted


def cmd_run(args):
    ops = list(OP_ORDER) if args.ops == "all" else [o.strip() for o in args.ops.split(",") if o.strip()]
    for o in ops:
        if o not in OP_ORDER:
            raise SystemExit(f"unknown op {o}; available: {list(OP_ORDER)}")
    if getattr(args, "before_consts", False):
        merged = dict(PRE_PORT_CONSTS)
        merged.update(_parse_consts(args.const))
        args.const = _const_string(merged)
    os.makedirs(args.outdir, exist_ok=True)
    _json_dump(
        os.path.join(args.outdir, "meta.json"),
        dict(label=args.label, ops=ops, cases=args.cases, device=args.device,
             measure_mode=args.measure_mode, const=args.const,
             warmup=args.warmup, iters=args.iters, rounds=args.rounds,
             before_consts=getattr(args, "before_consts", False),
             op_dir=os.path.abspath(args.op_dir),
             env={"python": sys.version.split()[0], "platform": platform.platform()}),
    )

    _log("=" * 76)
    _log(f"BENCH  label={args.label!r}  device={args.device}  mode={args.measure_mode}")
    if args.const:
        _log(f"       const overrides: {args.const}")
    _log("=" * 76)
    rows = []
    for op in ops:
        for case_id in _cases_for(op, args.cases):
            reps = []
            for rep in range(max(1, args.repeat)):
                tag = args.tag if rep == 0 else f"{args.tag}#{rep + 1}"
                reps.append(_run_one(args, op, case_id, tag))
            rows.append(_aggregate(op, case_id, reps))
    _json_dump(os.path.join(args.outdir, "results.json"), rows)
    _write_summary(args.outdir, args.label, rows)
    return 0


def _run_one(args, op, case_id, tag):
    cmd = [
        sys.executable, SELF, "worker",
        "--outdir", args.outdir, "--op", op, "--case", case_id,
        "--op-dir", args.op_dir, "--device", args.device,
        "--measure-mode", args.measure_mode,
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--rounds", str(args.rounds), "--tag", tag,
    ]
    if args.const:
        cmd += ["--const", args.const]
    if getattr(args, "strict_const", False):
        cmd.append("--strict-const")
    if args.no_bench:
        cmd.append("--no-bench")
    if args.device_index is not None:
        cmd += ["--device-index", str(args.device_index)]
    log = os.path.join(args.outdir, "logs", f"{op}__{case_id}@{tag}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as lf:
        try:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            pass
    rec = _json_load(_raw(args.outdir, op, case_id, tag), {}) or {}
    bench = rec.get("bench") or {}
    status = rec.get("status", "crash")
    ms = bench.get("median_ms")
    human = f"{ms:8.3f} ms" if isinstance(ms, (int, float)) else "     n/a "
    _log(f"  {op:<22}{case_id:<30}{tag:<10}{status:<7}{human:>12}  "
         f"{bench.get('mode') or ''}")
    return dict(
        op=op, case=case_id, status=status,
        mode=bench.get("mode"), ms=ms, min_ms=bench.get("min_ms"),
        host_ms=bench.get("host_ms"), device_ms=bench.get("device_ms"),
        peak_alloc_mb=bench.get("peak_alloc_mb"),
        capture_failed=bench.get("graph_capture_failed"),
        wall_s=round(time.time() - t0, 1), digest=rec.get("digest"),
        consts=rec.get("consts"),
        consts_ignored=rec.get("consts_ignored") or None,
        error=rec.get("error"),
    )


def _aggregate(op, case_id, reps):
    """Median across repeated runs, plus the spread that a single run hides.

    Within-run rounds are tight; the variation that matters for a before/after
    comparison is between process invocations, so `--repeat` re-runs the case in
    fresh processes and this records min/max/spread over their medians.
    """
    good = [r for r in reps if r["status"] == "ok" and isinstance(r["ms"], (int, float))]
    first = reps[0]
    if not good:
        return first
    vals = sorted(r["ms"] for r in good)
    med = vals[len(vals) // 2]
    out = dict(first)
    out["ms"] = med
    out["repeat_ms"] = [r["ms"] for r in reps]
    out["repeat_min"] = vals[0]
    out["repeat_max"] = vals[-1]
    out["spread_pct"] = (
        round(100.0 * (vals[-1] - vals[0]) / med, 2) if med and len(vals) > 1 else 0.0
    )
    out["digest_identical_across_repeats"] = all(r["digest"] == first["digest"] for r in reps)
    return out


def _write_summary(outdir, label, rows):
    md = [f"# bench_ops — {label}", ""]
    md.append("| op | case | status | mode | ms | spread % | host ms | device ms | peak MB |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    def f(v, sp="{:.3f}"):
        return sp.format(v) if isinstance(v, (int, float)) else "-"

    for r in rows:
        md.append("| {op} | {case} | {st} | {m} | {ms} | {sp} | {h} | {d} | {p} |".format(
            op=r["op"], case=r["case"], st=r["status"], m=r["mode"] or "-",
            ms=f(r["ms"]), sp=f(r.get("spread_pct"), "{:.1f}"),
            h=f(r["host_ms"]), d=f(r["device_ms"]),
            p=f(r["peak_alloc_mb"], "{:.0f}")))
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        md += ["", "## Failed", "", "| op | case | error |", "|---|---|---|"]
        for r in bad:
            md.append(f"| {r['op']} | {r['case']} | {(r.get('error') or '')[:160]} |")
    path = os.path.join(outdir, "summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    _log(f"  -> {path}")
    _log(f"  -> {os.path.join(outdir, 'results.json')}")


def _git(repo, *argv, check=True):
    r = subprocess.run(["git", "-C", repo, *argv], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(argv)} failed:\n{r.stderr.strip()}")
    return r


def _op_py_files(op_dir):
    """The files a state switch may touch: every op, never the tool itself."""
    return sorted(
        f for f in os.listdir(op_dir)
        if f.endswith(".py") and f != os.path.basename(SELF)
    )


def cmd_ab(args):
    """One command: before -> run -> after -> run -> compare.

    The 'before' state is either a git ref (operator files checked out, then
    restored from an in-memory snapshot) or, with --before-consts, the same tree
    with PRE_PORT_CONSTS applied - which needs no git at all.
    """
    if bool(args.before_ref) == bool(args.before_consts):
        raise SystemExit(
            "give exactly one of --before-ref <ref> or --before-consts"
        )
    op_dir = os.path.abspath(args.op_dir)
    repo = _git(op_dir, "rev-parse", "--show-toplevel").stdout.strip()
    files = _op_py_files(op_dir)
    rel = [os.path.relpath(os.path.join(op_dir, f), repo) for f in files]

    if args.before_consts:
        return _ab_by_consts(args, op_dir, repo)

    refs = {}
    for ref in (args.before_ref, args.after_ref):
        sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
        subject = _git(repo, "log", "-1", "--format=%s", sha).stdout.strip()
        refs[ref] = (sha, subject)
    _log("=" * 76)
    _log("A/B  " + op_dir)
    _log(f"  before {args.before_ref} -> {refs[args.before_ref][0][:10]} "
         f"{refs[args.before_ref][1][:52]}")
    _log(f"  after  {args.after_ref} -> {refs[args.after_ref][0][:10]} "
         f"{refs[args.after_ref][1][:52]}")
    _log(f"  files swapped: {len(files)}  (bench_ops.py is never swapped)")
    _log("=" * 76)

    if refs[args.before_ref][0] == refs[args.after_ref][0]:
        raise SystemExit("before-ref and after-ref resolve to the same commit")

    root = os.path.abspath(
        args.root or (args.outdir if getattr(args, "outdir", None) else
                      os.path.join(HERE, "out", "ab"))
    )
    os.makedirs(root, exist_ok=True)

    # snapshot the working tree so the restore is exact even if it is dirty
    snap = {f: open(os.path.join(op_dir, f), "rb").read() for f in files}
    dirty = _git(repo, "status", "--porcelain", "--", *rel).stdout.strip()
    if dirty:
        _log(f"  note: {len(dirty.splitlines())} operator file(s) differ from HEAD "
             f"before starting; they are restored afterwards")

    made = {}
    try:
        for ref, tag in ((args.before_ref, "before"), (args.after_ref, "after")):
            _git(repo, "checkout", ref, "--", *rel)
            diff = _git(repo, "status", "--porcelain", "--", *rel).stdout
            made[tag] = len(diff.splitlines())
            _log(f"\n[{tag}] checked out {ref}: {made[tag]} file(s) differ from HEAD")
            args.outdir = os.path.join(root, tag)
            args.label = tag
            args.tag = tag
            cmd_run(args)
    finally:
        for f, blob in snap.items():
            with open(os.path.join(op_dir, f), "wb") as fh:
                fh.write(blob)
        msg = _git(repo, "status", "--porcelain", "--", *rel).stdout.strip()
        _log("\n[restore] working tree rewritten from the snapshot"
             + (f" ({len(msg.splitlines())} file(s) still differ from HEAD, "
                "as before the run)" if msg else " (matches HEAD)"))

    _json_dump(os.path.join(root, "ab_meta.json"), dict(
        before=dict(ref=args.before_ref, sha=refs[args.before_ref][0],
                    subject=refs[args.before_ref][1], files_changed=made.get("before")),
        after=dict(ref=args.after_ref, sha=refs[args.after_ref][0],
                   subject=refs[args.after_ref][1], files_changed=made.get("after")),
        files=files, cases=args.cases, measure_mode=args.measure_mode,
        repeat=args.repeat, const=args.const,
    ))

    args.before = os.path.join(root, "before")
    args.after = os.path.join(root, "after")
    return cmd_compare(args)


def _ab_by_consts(args, op_dir, repo):
    """before = current tree + PRE_PORT_CONSTS, after = current tree as-is."""
    root = os.path.abspath(
        args.root or (getattr(args, "outdir", None) or os.path.join(HERE, "out", "ab"))
    )
    os.makedirs(root, exist_ok=True)
    user = args.const
    before = dict(PRE_PORT_CONSTS)
    before.update(_parse_consts(user))
    _log("=" * 76)
    _log("A/B  (constant emulation, no git)"
         f"  {op_dir}")
    _log(f"  before : current tree + PRE_PORT_CONSTS {before}")
    _log(f"  after  : current tree, const={user or '<none>'}")
    for name in before:
        _log(f"    {name} -> {before[name]}")
    _log("=" * 76)
    saved_const = args.const
    saved_flag = getattr(args, "before_consts", False)
    try:
        args.before_consts = True
        args.const = _const_string(before)
        args.label = args.tag = "before"
        args.outdir = os.path.join(root, "before")
        cmd_run(args)

        args.before_consts = False          # <-- the 'after' side is the tree as-is
        args.const = user
        args.label = args.tag = "after"
        args.outdir = os.path.join(root, "after")
        cmd_run(args)
    finally:
        args.const = saved_const
        args.before_consts = saved_flag

    # 2) guard: refuse to report a comparison of two identical measurements.
    #    This is the const-mode equivalent of the "same SHA" check in git mode,
    #    and it would have caught the bug above.
    mb = _json_load(os.path.join(root, "before", "meta.json"), {}) or {}
    ma = _json_load(os.path.join(root, "after", "meta.json"), {}) or {}
    if (mb.get("const"), bool(mb.get("before_consts"))) == (
        ma.get("const"), bool(ma.get("before_consts"))
    ):
        raise SystemExit(
            "before and after were measured with the same constants "
            f"(const={mb.get('const')!r}, before_consts={mb.get('before_consts')!r}); "
            "nothing differs, refusing to report a comparison"
        )

    _json_dump(os.path.join(root, "ab_meta.json"), dict(
        mode="const-emulation",
        before=dict(source="PRE_PORT_CONSTS", consts=before),
        after=dict(source="working tree", consts=user),
        cases=args.cases, measure_mode=args.measure_mode,
        repeat=args.repeat,
    ))
    args.before = os.path.join(root, "before")
    args.after = os.path.join(root, "after")
    return cmd_compare(args)


def cmd_compare(args):
    before = _json_load(os.path.join(args.before, "results.json"))
    after = _json_load(os.path.join(args.after, "results.json"))
    if before is None or after is None:
        raise SystemExit("both folders need a results.json (run `run --label ...` first)")
    bmeta = _json_load(os.path.join(args.before, "meta.json"), {}) or {}
    ameta = _json_load(os.path.join(args.after, "meta.json"), {}) or {}
    bmap = {(r["op"], r["case"]): r for r in before}
    amap = {(r["op"], r["case"]): r for r in after}

    rows = []
    for op in OP_ORDER:
        for case_id in sorted({c for (o, c) in list(bmap) + list(amap) if o == op}):
            b, a = bmap.get((op, case_id)), amap.get((op, case_id))
            b_ms = (b or {}).get("ms")
            a_ms = (a or {}).get("ms")
            ratio = (b_ms / a_ms) if (b_ms and a_ms) else None
            same_mode = (b or {}).get("mode") == (a or {}).get("mode")
            digest_rel = None
            if (b or {}).get("digest") and (a or {}).get("digest"):
                worst = 0.0
                for key, bd in b["digest"].items():
                    ad = a["digest"].get(key)
                    if not ad:
                        continue
                    for field in ("sum", "sumsq", "absmax"):
                        x, y = bd.get(field, 0.0), ad.get(field, 0.0)
                        denom = max(abs(x), abs(y), 1e-30)
                        worst = max(worst, abs(x - y) / denom)
                digest_rel = worst
            rows.append(dict(
                op=op, case=case_id,
                before_status=(b or {}).get("status", "missing"),
                after_status=(a or {}).get("status", "missing"),
                before_ms=b_ms, after_ms=a_ms, speedup=ratio,
                same_mode=same_mode,
                digest_rel=digest_rel,
                before_spread=(b or {}).get("spread_pct"),
                after_spread=(a or {}).get("spread_pct"),
                digest_identical=(digest_rel is not None and digest_rel == 0.0),
                before_mode=(b or {}).get("mode"), after_mode=(a or {}).get("mode"),
            ))

    ran_both = [r for r in rows
                if r["before_status"] != "missing" and r["after_status"] != "missing"]
    skipped = [r for r in rows
               if r["before_status"] == "missing" or r["after_status"] == "missing"]
    rows = ran_both

    md = [f"# bench_ops — before/after", ""]
    md.append(f"- before: `{bmeta.get('label')}` ({args.before})")
    md.append(f"- after:  `{ameta.get('label')}` ({args.after})")
    md.append(f"- mode: `{bmeta.get('measure_mode')}` / `{ameta.get('measure_mode')}`")
    md.append("")
    md.append("`after/before` is the latency ratio: **>1x means the `after` state is"
              " faster**, <1x means slower (`0.95x` = 5% slower).")
    md.append("")
    md.append("`status` rows are pairs that could not be compared (one side did not"
              " run); `*` marks a ratio whose two sides were measured in different"
              " timing modes and is therefore not meaningful.")
    md += ["", "| op | case | before ms | after ms | after/before | digest | spread |",
           "|---|---|---|---|---|---|---|"]

    def f(v, sp="{:.3f}"):
        return sp.format(v) if isinstance(v, (int, float)) else "-"

    for r in rows:
        if r["before_status"] != "ok" or r["after_status"] != "ok":
            summary = f"{r['before_status']} -> {r['after_status']}"
            md.append(f"| {r['op']} | {r['case']} | {f(r['before_ms'])} | {f(r['after_ms'])} | {summary} | - |")
            continue
        if r["speedup"]:
            sp = f"{r['speedup']:.2f}x"
            if not r["same_mode"]:
                sp += " *"
        else:
            sp = "-"
        dig = "identical" if r["digest_identical"] else (
            f"Δ {r['digest_rel']:.2e}" if r["digest_rel"] is not None else "-")
        spd = "-"
        if r.get("before_spread") is not None or r.get("after_spread") is not None:
            spd = f"{f(r.get('before_spread'), '{:.1f}')}/{f(r.get('after_spread'), '{:.1f}')} %"
        md.append(f"| {r['op']} | {r['case']} | {f(r['before_ms'])} | {f(r['after_ms'])}"
                  f" | {sp} | {dig} | {spd} |")

    if skipped:
        md += ["", f"_{len(skipped)} (op, case) pair(s) present in only one of the two"
                   f" runs and are not compared: "
                   + ", ".join(f"{r['op']}/{r['case']}"
                               for r in skipped[:8])
                   + (" ..." if len(skipped) > 8 else "") + "_"]

    # per-op geomean of the after/before speedup
    md += ["", "## Per-op geomean of after/before (>1x = faster)",
           "", "| op | geomean | cases |", "|---|---|---|"]
    for op in OP_ORDER:
        vals = [r["speedup"] for r in rows
                if r["op"] == op and r["speedup"] and r["same_mode"]]
        if vals:
            g = math.exp(sum(math.log(v) for v in vals) / len(vals))
            md.append(f"| {op} | {g:.2f}x | {len(vals)} |")

    out = os.path.join(os.path.dirname(os.path.abspath(args.after)), "compare.md")
    with open(out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    _json_dump(os.path.join(os.path.dirname(os.path.abspath(args.after)), "compare.json"), rows)
    _log("\n".join(md))
    _log(f"\n  -> {out}")
    return 0


def cmd_list(args):
    for op in OP_ORDER:
        path = os.path.join(args.op_dir, op + ".py")
        print(f"\n{op}   ({'found' if os.path.isfile(path) else 'MISSING'}: {path})")
        for spec in ("quick", "default", "full"):
            print(f"  {spec:<8}: {', '.join(CASES[op][spec])}")
        print(f"  all ids : {', '.join(sorted(SPECS[op]))}")
    return 0


def cmd_selfcheck(args):
    """Build every case on CPU: catches a broken builder without needing a device."""
    import torch  # noqa: F401

    bad = 0
    for op in OP_ORDER:
        for case_id in sorted(SPECS[op]):
            try:
                kw, _ = BUILDERS[op](case_id, "cpu")
                shapes = sum(
                    1 for v in kw.values() if hasattr(v, "shape")
                )
                print(f"  ok   {op:<22}{case_id:<30}{shapes} tensors")
            except Exception as exc:
                bad += 1
                print(f"  FAIL {op:<22}{case_id:<30}{type(exc).__name__}: {exc}")
    print("\nselfcheck", "FAILED" if bad else "OK")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bench_ops", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, with_cases=True):
        p.add_argument("--op-dir", default=HERE, help="directory holding <op>.py")
        p.add_argument("--ops", default="all")
        if with_cases:
            p.add_argument("--cases", default="default",
                           help="quick|default|full or a comma separated list")

    def run_options(p, outdir_required=True):
        if outdir_required:
            p.add_argument("--outdir", required=True)
        p.add_argument("--label", default="run", help="name for this state, e.g. before/after")
        p.add_argument("--tag", default="", help="suffix for raw/, logs/ (default: label)")
        p.add_argument("--device", default=None)
        p.add_argument("--device-index", type=int, default=None)
        p.add_argument("--measure-mode", choices=("loop", "npu-graph", "host-split"),
                       default="npu-graph")
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--iters", type=int, default=20)
        p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--timeout", type=int, default=3600)
        p.add_argument("--no-bench", action="store_true")
        p.add_argument(
            "--repeat", type=int, default=1,
            help="run each case N times in fresh processes; results.json then "
            "holds the median across repeats plus spread_pct (the "
            "between-process noise that a single run cannot show)",
        )
        p.add_argument("--const", default=None,
                       metavar="NAME=VALUE[,NAME=VALUE]",
                       help="override module constants before calling run(); "
                       "names an operator does not have are skipped and listed "
                       "under consts_ignored in results.json")
        p.add_argument("--strict-const", action="store_true",
                       help="fail instead of skipping unknown constant names")

    p_run = sub.add_parser("run", help="measure every (op, case) in one source state")
    common(p_run)
    run_options(p_run)
    p_run.add_argument("--before-consts", action="store_true",
                       help="also apply PRE_PORT_CONSTS, i.e. measure the "
                       "pre-port configuration of the current tree (no git "
                       "checkout needed)")

    p_ab = sub.add_parser(
        "ab",
        help="measure <before-ref> and <after-ref> and compare them, in one command",
        description="Checks the operator files out of --before-ref, runs the whole "
        "benchmark, checks them out of --after-ref, runs it again, compares, and "
        "always restores the working tree.  bench_ops.py itself is never swapped, "
        "so both sides are measured with the same tool.",
    )
    common(p_ab)
    run_options(p_ab, outdir_required=False)
    p_ab.add_argument("--before-ref", default=None, metavar="REF",
                      help="git ref for the 'before' state, e.g. the merge commit")
    p_ab.add_argument("--before-consts", action="store_true",
                      help="take the 'before' state from PRE_PORT_CONSTS "
                      "instead of a git ref: same tree, no checkout, no git "
                      "mutation at all")
    p_ab.add_argument("--after-ref", default="HEAD", metavar="REF")
    p_ab.add_argument("--root", default=None,
                      help="output root; before/ and after/ are created inside it "
                      "(default: out/ab next to bench_ops.py)")

    p_cmp = sub.add_parser("compare", help="diff two run folders")
    p_cmp.add_argument("--before", required=True)
    p_cmp.add_argument("--after", required=True)

    p_w = sub.add_parser("worker", help="single (op, case); used by run")
    common(p_w)
    p_w.add_argument("--outdir", required=True)
    p_w.add_argument("--op", required=True)
    p_w.add_argument("--case", required=True)
    p_w.add_argument("--tag", default="")
    p_w.add_argument("--device", default=None)
    p_w.add_argument("--device-index", type=int, default=None)
    p_w.add_argument("--measure-mode", choices=("loop", "npu-graph", "host-split"),
                     default="npu-graph")
    p_w.add_argument("--warmup", type=int, default=5)
    p_w.add_argument("--iters", type=int, default=20)
    p_w.add_argument("--rounds", type=int, default=5)
    p_w.add_argument("--no-bench", action="store_true")
    p_w.add_argument("--const", default=None)
    p_w.add_argument("--strict-const", action="store_true")

    p_list = sub.add_parser("list", help="show the case registry")
    common(p_list)
    p_chk = sub.add_parser("selfcheck", help="build every case on CPU")
    common(p_chk)

    args = ap.parse_args(argv)
    if getattr(args, "tag", None) == "" and getattr(args, "label", None):
        args.tag = args.label
    if args.cmd != "compare" and getattr(args, "device", None) is None:
        args.device = _detect_device()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "ab":
        return cmd_ab(args)
    if args.cmd == "worker":
        return worker_main(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "selfcheck":
        return cmd_selfcheck(args)
    raise SystemExit(f"unknown command {args.cmd}")


def _detect_device():
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


if __name__ == "__main__":
    sys.exit(main())
