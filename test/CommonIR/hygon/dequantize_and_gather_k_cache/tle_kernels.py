# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Exact original JIT function; full source retained under tle/."""
import triton
import triton.language as tl


@triton.jit
def _dequantize_and_gather_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    scale_slots: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    cache_block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    num_workers: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    HAVE_GATHER_LENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    if HAVE_GATHER_LENS:
        gather_len = tl.load(gather_lens_ptr + req_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for local_i in range(worker_idx, gather_len, num_workers):
        pos = start_pos + local_i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos - block_in_seq * cache_block_size
        physical_block = tl.load(
            block_table_ptr + req_idx * max_blocks_per_seq + block_in_seq
        )
        cache_block = k_cache_ptr + physical_block.to(tl.int64) * cache_block_stride
        token_data = cache_block + pos_in_block * token_data_size
        scale_base = (
            cache_block
            + cache_block_size * token_data_size
            + pos_in_block * scale_slots
        )
        out_row = out_ptr + req_idx * out_stride0 + (offset + local_i) * out_stride1

        if nope_dim % quant_block == 0:
            for qblock in tl.static_range(0, n_quant_blocks):
                qoffs = qblock * quant_block + tl.arange(0, quant_block)
                x_u8 = tl.load(token_data + qoffs)
                x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                encoded = tl.load(scale_base + qblock)
                scale = tl.exp2(encoded.to(tl.float32) - 127.0)
                x = x_fp8 * scale
                tl.store(out_row + qoffs, x.to(tl.bfloat16))
        else:
            for qblock in tl.static_range(0, n_quant_blocks - 1):
                qoffs = qblock * quant_block + tl.arange(0, quant_block)
                x_u8 = tl.load(token_data + qoffs)
                x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                encoded = tl.load(scale_base + qblock)
                scale = tl.exp2(encoded.to(tl.float32) - 127.0)
                x = x_fp8 * scale
                tl.store(out_row + qoffs, x.to(tl.bfloat16))

            qblock = n_quant_blocks - 1
            qoffs = qblock * quant_block + tl.arange(0, quant_block)
            qmask = qoffs < nope_dim
            x_u8 = tl.load(token_data + qoffs, mask=qmask, other=0)
            x_fp8 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            encoded = tl.load(scale_base + qblock)
            scale = tl.exp2(encoded.to(tl.float32) - 127.0)
            x = x_fp8 * scale
            tl.store(out_row + qoffs, x.to(tl.bfloat16), mask=qmask)

        bf16_ptr = (token_data + nope_dim).to(tl.pointer_type(tl.bfloat16))
        for rblock in tl.static_range(0, rope_dim, 64):
            roffs = rblock + tl.arange(0, 64)
            rmask = roffs < rope_dim
            vals = tl.load(bf16_ptr + roffs, mask=rmask, other=0.0)
            tl.store(out_row + nope_dim + roffs, vals, mask=rmask)

