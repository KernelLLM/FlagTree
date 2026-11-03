"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""
import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice
@triton.jit
def moe_sum_kernel(input_ptr, output_ptr, num_tokens, topk, hidden_size, input_stride_token, input_stride_topk, input_stride_hidden, output_stride_token, output_stride_hidden, BLOCK_SIZE: tl.constexpr):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token
    for expert_idx in range(topk):
        expert_ptr = input_base + expert_idx * input_stride_topk
        expert_data = tl.load(expert_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
        acc += expert_data
    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets
    tl.store(output_ptr_pos, acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc, mask=hidden_mask)
