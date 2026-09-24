"""Evidence-selected Gluon implementation for the stage-two MoE sum matrix."""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def moe_sum_topk2_u(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + gl.arange(
        0, BLOCK_SIZE, layout=gl.BlockedLayout([1], [64], [2], [0])
    )
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc = expert0.to(gl.float32) + expert1.to(gl.float32)
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk2_p(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + gl.arange(
        0, BLOCK_SIZE, layout=gl.BlockedLayout([2], [64], [2], [0])
    )
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc = expert0.to(gl.float32) + expert1.to(gl.float32)
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk2_wave1_p(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + gl.arange(
        0, BLOCK_SIZE, layout=gl.BlockedLayout([2], [64], [1], [0])
    )
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc = expert0.to(gl.float32) + expert1.to(gl.float32)
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk6_u(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
    offsets = block_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    acc = gl.zeros((BLOCK_SIZE,), dtype=gl.float32, layout=layout)
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    acc += expert0
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert1
    expert2 = gl.load(base + 2 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert2
    expert3 = gl.load(base + 3 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert3
    expert4 = gl.load(base + 4 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert4
    expert5 = gl.load(base + 5 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert5
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk6_p(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [2], [0])
    offsets = block_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    acc = gl.zeros((BLOCK_SIZE,), dtype=gl.float32, layout=layout)
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    acc += expert0
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert1
    expert2 = gl.load(base + 2 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert2
    expert3 = gl.load(base + 3 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert3
    expert4 = gl.load(base + 4 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert4
    expert5 = gl.load(base + 5 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert5
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk6_wave1_u(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    acc = gl.zeros((BLOCK_SIZE,), dtype=gl.float32, layout=layout)
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    acc += expert0
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert1
    expert2 = gl.load(base + 2 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert2
    expert3 = gl.load(base + 3 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert3
    expert4 = gl.load(base + 4 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert4
    expert5 = gl.load(base + 5 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert5
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk6_wave1_p(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    offsets = block_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    acc = gl.zeros((BLOCK_SIZE,), dtype=gl.float32, layout=layout)
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    acc += expert0
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert1
    expert2 = gl.load(base + 2 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert2
    expert3 = gl.load(base + 3 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert3
    expert4 = gl.load(base + 4 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert4
    expert5 = gl.load(base + 5 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert5
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


@g.jit
def moe_sum_topk6_quad(
    input_ptr, output_ptr, num_tokens, topk, hidden_size,
    input_stride_token, input_stride_topk, input_stride_hidden,
    output_stride_token, output_stride_hidden, BLOCK_SIZE: gl.constexpr,
):
    token_idx = gl.program_id(0)
    block_idx = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    offsets = block_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < hidden_size
    if token_idx >= num_tokens:
        return
    base = input_ptr + token_idx * input_stride_token
    acc = gl.zeros((BLOCK_SIZE,), dtype=gl.float32, layout=layout)
    expert0 = gl.load(base + offsets, mask=mask, other=0.0)
    acc += expert0
    expert1 = gl.load(base + input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert1
    expert2 = gl.load(base + 2 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert2
    expert3 = gl.load(base + 3 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert3
    expert4 = gl.load(base + 4 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert4
    expert5 = gl.load(base + 5 * input_stride_topk + offsets, mask=mask, other=0.0)
    acc += expert5
    gl.store(output_ptr + token_idx * output_stride_token + offsets, acc, mask=mask)


def _selection(dtype, topk, hidden_size):
    is_half = dtype in (torch.float16, torch.bfloat16)
    if topk == 2:
        if dtype == torch.float16 and hidden_size == 128:
            return moe_sum_topk2_wave1_p, 128, 1, "topk2_wave1_p"
        if is_half and hidden_size == 1024:
            return moe_sum_topk2_p, 256, 2, "topk2_p"
        return moe_sum_topk2_u, 256, 2, "topk2_u"
    if topk == 6:
        if hidden_size == 128:
            if is_half:
                return moe_sum_topk6_wave1_p, 128, 1, "topk6_wave1_p"
            return moe_sum_topk6_wave1_u, 128, 1, "topk6_wave1_u"
        if hidden_size == 1024:
            if is_half:
                return moe_sum_topk6_p, 256, 2, "topk6_p"
            return moe_sum_topk6_quad, 256, 1, "topk6_quad"
        return moe_sum_topk6_u, 256, 2, "topk6_u"
    raise ValueError(f"optimized moe_sum supports topk=2 or 6, got {topk}")


def launch_spec(input, output):
    num_tokens, topk, hidden_size = input.shape
    kernel, block_size, num_warps, variant = _selection(input.dtype, topk, hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    args = (
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        block_size,
    )
    launch = {
        "grid": grid,
        "num_warps": num_warps,
        "num_stages": 3,
        "waves_per_eu": 1,
    }
    metadata = {"variant": variant, "block_size": block_size, "num_warps": num_warps}
    return kernel, args, launch, metadata


def moe_sum(input, output):
    kernel, args, launch, _ = launch_spec(input, output)
    grid = launch.pop("grid")
    try:
        kernel[grid](*args, **launch)
    finally:
        launch["grid"] = grid

