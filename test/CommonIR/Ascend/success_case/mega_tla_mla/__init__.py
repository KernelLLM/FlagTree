# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TLA port of the mega_mla fused attention kernel.

This package rewrites the cannbot-dsl mega_mla kernel using the TLA
(triton-ascend low-level API) programming model. The algorithmic structure
is preserved; only the API surface changes.

Modules:
    mega_mla_kernel   -- main kernel entry points and projection pipeline
    qkv_projection    -- MXFP8 projection helpers (quantize, RMS norm, matmul tiles)
    attention_ops      -- QK/softmax/PV matmul and gather operations
    rotary_embedding   -- RoPE (rotary position embedding) helpers
    cache_ops          -- cache decode, quantize, dequantize, gather
    attention_geometry -- host-side geometry validation and constants
"""
