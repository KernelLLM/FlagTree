# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared host/device layout for the private, single-allocation GM workspace.

TLA port of cannbot-dsl attention_workspace.py. This is host-side-only
layout logic, identical to the original. The device binding creates TLA
views from workspace pointers instead of cannbot-dsl Tensor slices.
"""

from math import prod


class AttentionWorkspaceLayout:
    """Disjoint typed regions; offsets and sizes are expressed in bytes.

    Every region starts at a 512-byte boundary. Regions retain their original
    lifetimes and never overlap, including the immutable zero-fill record.
    The caller owns the GM allocation; device binding only creates alias views.
    Typed host aliases share that allocation and cross the private device ABI
    separately. This avoids dtype-changing GM views in the compiler's
    FIXPIPE atomic and ND2NZ lowering; no data is copied or converted.
    """

    ALIGNMENT = 512

    def __init__(
        self,
        tokens,
        heads,
        k,
        compressed_pages,
        page_size,
        strategy,
        *,
        filter_indices=False,
        win_len=128,
        compressed_page_size=None,
    ):
        decoded = (
            (1, 1, 512)
            if strategy == "indexed"
            else (
                compressed_pages,
                page_size if compressed_page_size is None else compressed_page_size,
                512,
            )
        )
        specs = (
            ("qr", (tokens, 1280), "float32", 4),
            ("kv", (tokens, 512), "float32", 4),
            ("qfull", (tokens, heads * 512), "bfloat16", 2),
            ("x8", (tokens, 5120), "float8_e4m3fn", 1),
            ("sx", (tokens, 80, 2), "float8_e8m0", 1),
            ("qn8", (tokens, 1280), "float8_e4m3fn", 1),
            ("sqn", (tokens, 20, 2), "float8_e8m0", 1),
            ("kvn", (tokens, 512), "bfloat16", 2),
            ("qb", (tokens, heads, 512), "bfloat16", 2),
            ("kg", (tokens, k, 512), "bfloat16", 2),
            ("s", (tokens, heads, k), "float32", 4),
            ("p", (tokens, heads, k), "bfloat16", 2),
            ("cmp_dec", decoded, "bfloat16", 2),
            ("zero", (1, 512), "bfloat16", 2),
        )
        if filter_indices:
            # Cross-core publication: producer AIVs filter/map each index once;
            # projection, gather and softmax owners consume these GM views.
            specs += (
                ("seq_win_idx", (tokens, win_len), "int32", 4),
                ("seq_cmp_idx", (tokens, k - win_len), "int32", 4),
                ("seq_slot", (tokens,), "int32", 4),
            )
        regions = []
        cursor = 0
        for name, shape, dtype, width in specs:
            cursor = self._align(cursor)
            size = prod(shape) * width
            regions.append((name, shape, dtype, cursor, size))
            cursor += size
        self.regions = tuple(regions)
        self.nbytes = self._align(cursor)

    @classmethod
    def _align(cls, value):
        return (value + cls.ALIGNMENT - 1) // cls.ALIGNMENT * cls.ALIGNMENT

    def allocate(self, device):
        """Allocate one FP32 tensor and return byte-offset diagnostic views."""
        import torch

        storage = torch.empty(self.nbytes // 4, dtype=torch.float32, device=device)
        byte_storage = storage.view(torch.uint8)
        views = {}
        for name, shape, dtype, offset, size in self.regions:
            torch_dtype = (
                torch.float8_e8m0fnu
                if dtype == "float8_e8m0"
                else getattr(torch, dtype)
            )
            views[name] = (
                byte_storage[offset : offset + size].view(torch_dtype).view(shape)
            )
        return storage, views

    def arguments(self, storage):
        """Return five typed aliases of the same host allocation."""
        import torch

        return (
            storage,
            storage.view(torch.bfloat16),
            storage.view(torch.float8_e4m3fn),
            storage.view(torch.float8_e8m0fnu),
            storage.view(torch.int32),
        )

    # ---- TLA device-side binding ----
    # In TLA kernels, workspace views are created with tla.view from
    # the five typed pointer arguments. The dtype_to_tl mapping and
    # the offset arithmetic match the host layout exactly.
    #
    # Usage in a @tla.jit kernel:
    #   regions = layout.tla_view_specs()
    #   for name, shape, tl_dtype, elem_offset in regions:
    #       buf = tla.view(workspace_ptr_for_dtype, shape=shape,
    #                       offsets=[elem_offset], space="gm")
    #
    # The five workspace pointer arguments correspond to:
    #   workspace_gm      (float32)
    #   workspace_bf16_gm (bfloat16)
    #   workspace_fp8_gm  (float8_e4m3fn)
    #   workspace_e8m0_gm (float8_e8m0)
    #   workspace_int32_gm(int32)

    DTYPE_WIDTH = {
        "float32": 4,
        "bfloat16": 2,
        "float8_e4m3fn": 1,
        "float8_e8m0": 1,
        "int32": 4,
    }

    # Map workspace dtype strings to TLA dtype strings for tla.view
    DTYPE_TO_TL = {
        "float32": "tl.float32",
        "bfloat16": "tl.bfloat16",
        "float8_e4m3fn": "tl.float8e4nv",
        "float8_e8m0": "tl.uint8",  # E8M0 stored as uint8 in TLA
        "int32": "tl.int32",
    }

    # Map workspace dtype strings to the workspace pointer argument they alias
    DTYPE_TO_PTR_ARG = {
        "float32": "workspace_gm",
        "bfloat16": "workspace_bf16_gm",
        "float8_e4m3fn": "workspace_fp8_gm",
        "float8_e8m0": "workspace_e8m0_gm",
        "int32": "workspace_int32_gm",
    }

    def tla_view_specs(self):
        """Return (name, shape, dtype_str, ptr_arg, elem_offset) for each region.

        Each element offset is the byte offset divided by the element width,
        giving the linear element index into the typed pointer alias.
        """
        specs = []
        for name, shape, dtype, byte_offset, _size in self.regions:
            width = self.DTYPE_WIDTH[dtype]
            elem_offset = byte_offset // width
            ptr_arg = self.DTYPE_TO_PTR_ARG[dtype]
            tl_dtype = self.DTYPE_TO_TL[dtype]
            specs.append((name, shape, tl_dtype, ptr_arg, elem_offset))
        return specs

    def bind(self, fp32, bf16, fp8, e8m0, int32):
        """Create device Tensor views without allocating or copying data.

        This is the cannbot-dsl compatible path. TLA kernels use
        tla_view_specs() with tla.view instead.
        """
        roots = {
            "float32": (fp32, 4),
            "bfloat16": (bf16, 2),
            "float8_e4m3fn": (fp8, 1),
            "float8_e8m0": (e8m0, 1),
            "int32": (int32, 4),
        }
        return {
            name: roots[dtype][0][
                offset // roots[dtype][1] : (offset + size) // roots[dtype][1],
            ].view(shape)
            for name, shape, dtype, offset, size in self.regions
        }
