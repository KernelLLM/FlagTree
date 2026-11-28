# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host geometry validation and compressed-cache decode strategy selection.

TLA port of cannbot-dsl attention_geometry.py. These are host-side-only
functions (no device code), so they remain plain Python without TLA decorators.
The policy uses public tensor geometry; it never copies device index or length
values to the host.
"""

from numbers import Integral


SUPPORTED_QUERY_HEADS = (8, 16, 64)
HEAD_DIM = 512
ATTENTION_K_ALIGNMENT = 128


def validate_aic_grid(grid, physical_cores=None):
    """Accept positive AIC counts, bounded by the input device when available."""
    if isinstance(grid, bool) or not isinstance(grid, Integral) or grid <= 0:
        raise ValueError("grid must be a positive integer")
    if physical_cores is not None and grid > physical_cores:
        raise ValueError(
            f"grid={grid} exceeds the input device's physical AIC count {physical_cores}"
        )
    return int(grid)


def validate_cache_layout(name, cache, record_bytes, *, page_size=None):
    """Accept tightly packed records with an independent, nonoverlapping page stride."""
    if (
        cache.ndim != 4
        or cache.shape[0] <= 0
        or cache.shape[1] <= 0
        or tuple(cache.shape[2:]) != (1, record_bytes)
    ):
        raise ValueError(f"{name} must have shape [N,page_size,1,{record_bytes}]")
    rows = int(cache.shape[1])
    if page_size is not None and rows != page_size:
        raise ValueError(f"{name} page_size must equal {page_size}")
    strides = tuple(cache.stride())
    if (
        strides[1:] != (record_bytes, record_bytes, 1)
        or strides[0] < rows * record_bytes
    ):
        raise ValueError(
            f"{name} requires tight records and stride (S,{record_bytes},{record_bytes},1), "
            "with S >= page_size * record_bytes"
        )
    return rows


def validate_attention_geometry(
    *, k=None, page_size=128, win_len=128, grid=32, tokens=None
):
    """Validate scalar geometry before constructing channels or GM workspace.

    QK uses N64 tiles, while PV reduces K128 tiles. Window boundaries must
    remain identical in the page-based attention core and the index buffers.
    """
    for name, value in (
        ("k", k),
        ("page_size", page_size),
        ("win_len", win_len),
        ("tokens", tokens),
    ):
        if value is None and name in ("k", "tokens"):
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    validate_aic_grid(grid)
    if win_len % 64 or win_len % page_size:
        raise ValueError("win_len must be a multiple of 64 and page_size")
    if k is not None and (k <= win_len or k % ATTENTION_K_ALIGNMENT):
        raise ValueError("k must exceed win_len and be a multiple of 128")


def validate_query_heads(heads):
    """Reject implicit rounding / bools before allocating or compiling anything."""
    if (
        isinstance(heads, bool)
        or not isinstance(heads, Integral)
        or heads not in SUPPORTED_QUERY_HEADS
    ):
        raise ValueError("heads must be one of 8, 16, 64")
    return int(heads)


def select_small_k_attention_head_tile(heads, tokens, grid, k, *, win_len=128):
    """Select H64 CV once at least half the token owners can be active.

    Keep the balanced two-head-group layout for T16/grid32. Other eligible
    shapes use one M64 task per token and reuse its KV across all heads.

    H8/H16 use logical-M GM QK/PV; splitting them into M4/M8 AIV pieces
    would violate the existing CV layout, so never enter that pipeline.
    """
    heads = validate_query_heads(heads)
    if heads != 64 or k > 640 or k % 128 or win_len % 128:
        return 0
    if tokens == 16 and grid == 32:
        return 32
    return 64 if 2 * tokens >= grid else 0


COMPRESSED_PREDECODE_MIN_ROWS = 4096
COMPRESSED_DECODE_STRATEGIES = ("auto", "predecode", "indexed")

# Conservative switch points from matched full-mega msprof sweeps on
# Ascend 950DT. Key: (T, H, Kc, grid, P, Kw).
# Values are the minimum cache rows selecting indexed, inclusive.
COMPRESSED_PROFILED_INDEXED_FROM_ROWS = {
    (1, 16, 512, 28, 128, 128): 24 * 128,
    (8, 16, 512, 28, 128, 128): 32 * 128,
    (8, 16, 2048, 28, 128, 128): 96 * 128,
    (16, 16, 512, 28, 128, 128): 32 * 128,
    (64, 64, 512, 28, 128, 128): 5 * 128,
}


def choose_compressed_cache_decode_strategy(
    tokens,
    cache_rows,
    topk,
    strategy="auto",
    *,
    heads=None,
    grid=None,
    page_size=None,
    win_len=None,
    mode="attn",
    has_lengths=False,
):
    """Choose from public geometry without copying device lengths to the host."""
    if strategy not in COMPRESSED_DECODE_STRATEGIES:
        raise ValueError("cmp_decode_strategy must be auto, predecode or indexed")
    for name, value in (("tokens", tokens), ("cache_rows", cache_rows), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if strategy != "auto":
        return strategy
    key = (int(tokens), heads, int(topk), grid, page_size, win_len)
    if mode == "attn" and not has_lengths:
        indexed_from = COMPRESSED_PROFILED_INDEXED_FROM_ROWS.get(key)
        if indexed_from is not None:
            return "indexed" if int(cache_rows) >= indexed_from else "predecode"
    limit = max(COMPRESSED_PREDECODE_MIN_ROWS, int(tokens) * int(topk))
    return "indexed" if int(cache_rows) > limit else "predecode"
