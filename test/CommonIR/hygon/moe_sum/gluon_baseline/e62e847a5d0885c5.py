# Gluon implementation exported from the pinned source.
# Original: /public/home/scnethpc2653/lhd/env/projects/hard81_group2_20260911/src/FlagGems-vllm/src/flaggems_vllm/ops/moe_sum.py
# Provenance: results/gluon/gluon_baseline/e62e847a5d0885c5.json
# Original library copyright and license: see ../../licenses/ and the project README.
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hcu_ops as __h81

from importlib import import_module as _import_module

@g.jit
def moe_sum_kernel_443b0376d4(input_ptr, output_ptr, num_tokens, topk, hidden_size, input_stride_token, input_stride_topk, input_stride_hidden, output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    token_idx = __h81.freeze(gl.program_id(0))
    block_idx = __h81.freeze(gl.program_id(1))
    hidden_start = __h81.freeze(__h81.to_auto(block_idx) * __h81.to_auto(BLOCK_SIZE))
    hidden_offsets = __h81.freeze(__h81.to_auto(hidden_start) + __h81.arange(0, __h81.to_auto(BLOCK_SIZE), layout=gl.BlockedLayout([1], [64], [2], [0])), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    hidden_mask = __h81.freeze(__h81.compare(__h81.to_auto(hidden_offsets), __h81.to_auto(hidden_size), 'lt'), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    if __h81.compare(__h81.to_auto(token_idx), __h81.to_auto(num_tokens), 'ge'):
        return
    acc = __h81.freeze(__h81.zeros((__h81.to_auto(BLOCK_SIZE),), dtype=gl.float32), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    input_base = __h81.freeze(__h81.to_auto(input_ptr) + __h81.to_auto(token_idx) * __h81.to_auto(input_stride_token))
    for expert_idx in range(__h81.to_auto(topk)):
        expert_ptr = __h81.freeze(__h81.to_auto(input_base) + __h81.to_auto(expert_idx) * __h81.to_auto(input_stride_topk))
        expert_data = __h81.freeze(__h81.load(__h81.to_auto(expert_ptr) + __h81.to_auto(hidden_offsets), mask=__h81.to_auto(hidden_mask), other=0.0, _layout=gl.BlockedLayout([1], [64], [2], [0])), _layout=gl.BlockedLayout([1], [64], [2], [0]))
        acc = __h81.freeze(__h81.to_auto(acc) + __h81.to_auto(expert_data), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    output_ptr_pos = __h81.freeze(__h81.to_auto(output_ptr) + __h81.to_auto(token_idx) * __h81.to_auto(output_stride_token) + __h81.to_auto(hidden_offsets), _layout=gl.BlockedLayout([1], [64], [2], [0]))
    __h81.store(__h81.to_auto(output_ptr_pos), __h81.cast(acc, gl.float16) if __h81.compare(__h81.pointer_element_type(__h81.to_auto(input_ptr), True), gl.float16, 'eq') else __h81.to_auto(acc), mask=__h81.to_auto(hidden_mask), _layout=gl.BlockedLayout([1], [64], [2], [0]))

@g.jit
def zeros_1ed385672d(shape, dtype):
    """HCU helper SHA256: 64e253bc2ea2d0226c3aa239e6a8d880a9a88457caf7a1c38287b1279c838b63"""
    '\n    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.\n\n    :param shape: Shape of the new array, e.g., (8, 16) or (8, )\n    :type shape: tuple of ints\n    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`\n    :type dtype: DType\n    '
    return __h81.full(__h81.to_auto(shape), 0, __h81.to_auto(dtype))

KERNEL = moe_sum_kernel_443b0376d4
