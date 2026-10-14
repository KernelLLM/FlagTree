# Native implementation template provenance is recorded in provenance.json.
# Hard20 revalidates all lowering and numeric claims independently.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

@g.jit
def moe_sum_kernel(input_ptr, output_ptr, num_tokens, topk, hidden_size, input_stride_token, input_stride_topk, input_stride_hidden, output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr, LAYOUT: gl.constexpr):
    token = gl.program_id(0)
    offsets = gl.program_id(1) * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=LAYOUT)
    mask = offsets < hidden_size
    if token >= num_tokens:
        return
    accumulator = gl.full((BLOCK_SIZE,), 0, gl.float32, layout=LAYOUT)
    for expert in range(topk):
        values = cdna3.buffer_load(input_ptr, (token * input_stride_token + expert * input_stride_topk + offsets).to(gl.int32), mask=mask, other=0.0)
        accumulator = accumulator + values
    if input_ptr.dtype.element_ty == gl.float16:
        accumulator = accumulator.to(gl.float16)
    cdna3.buffer_store(accumulator.to(output_ptr.dtype.element_ty), output_ptr, (token * output_stride_token + offsets).to(gl.int32), mask=mask)
