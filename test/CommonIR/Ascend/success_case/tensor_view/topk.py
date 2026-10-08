"""Single-stage row-wise TopK implemented with TensorView for Ascend.

The bitonic value/index sort follows the FlagGems TopK operator.  After the
sort, the data-dependent value lookup is represented by a gather/scatter view
instead of pointer arithmetic; input and output tiles use partition views.
"""

import argparse
import math

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, zeros_like


@triton.jit
def _compare_and_swap(x, indices, flip, stage: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**stage, 2, 2 ** (n_dims - stage - 1)]
    values = core.reshape(x, shape)
    ids = core.reshape(indices, shape)
    side = core.arange(0, 2)[None, :, None]

    left = core.where(side == 0, values, 0.0)
    right = core.where(side == 1, values, 0.0)
    left = core.reshape(
        core.broadcast_to(tl.sum(left, 1)[:, None, :], shape), x.shape
    ).to(x.dtype)
    right = core.reshape(
        core.broadcast_to(tl.sum(right, 1)[:, None, :], shape), x.shape
    ).to(x.dtype)

    left_id = core.where(side == 0, ids, 0)
    right_id = core.where(side == 1, ids, 0)
    left_id = core.reshape(
        core.broadcast_to(tl.sum(left_id, 1)[:, None, :], shape), indices.shape
    ).to(indices.dtype)
    right_id = core.reshape(
        core.broadcast_to(tl.sum(right_id, 1)[:, None, :], shape), indices.shape
    ).to(indices.dtype)

    if core.constexpr(x.dtype.primitive_bitwidth) == 16:
        value_int_type = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        value_int_type = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        value_int_type = core.int64
    else:
        raise ValueError("unsupported TopK value dtype")

    left_bits = left.to(value_int_type, bitcast=True)
    right_bits = right.to(value_int_type, bitcast=True)
    value_bits = x.to(value_int_type, bitcast=True)
    swap = (left > right) ^ flip
    result_bits = value_bits ^ core.where(
        swap, left_bits ^ right_bits, zeros_like(value_bits)
    )

    left_id_bits = left_id.to(core.int32, bitcast=True)
    right_id_bits = right_id.to(core.int32, bitcast=True)
    id_bits = indices.to(core.int32, bitcast=True)
    result_ids = id_bits ^ core.where(
        swap, left_id_bits ^ right_id_bits, zeros_like(id_bits)
    )
    return result_bits.to(x.dtype, bitcast=True), result_ids.to(
        indices.dtype, bitcast=True
    )


@triton.jit
def _bitonic_merge(
    x,
    indices,
    stage: core.constexpr,
    order: core.constexpr,
    n_dims: core.constexpr,
):
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    if order == 2:
        shape: core.constexpr = [
            n_outer * 2 ** (n_dims - 1 - stage),
            2,
            2**stage,
        ]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        ) != 0
    else:
        flip = order != 0
    for merge_stage in core.static_range(stage):
        x, indices = _compare_and_swap(
            x, indices, flip, merge_stage + (n_dims - stage), n_dims
        )
    return x, indices


@triton.jit
def _argsort(x, indices, descending: core.constexpr):
    n_dims: core.constexpr = _log2(x.shape[0])
    for stage in core.static_range(1, n_dims + 1):
        x, indices = _bitonic_merge(
            x, indices, stage, 2 if stage < n_dims else descending, n_dims
        )
    return x, indices


@triton.jit
def topk_tensor_view_kernel(
    x_ptr,
    values_ptr,
    indices_ptr,
    rows,
    cols: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < cols
    pad_value = float("-inf") if DESCENDING else float("inf")

    # No pointer is offset before it becomes a view.  Row selection is carried
    # entirely by the TensorView index tuple.
    input_tiles = tl.make_partition_view(
        x_ptr,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[1, BLOCK_SIZE],
    )
    input_gather = tl.make_gather_scatter_view(
        x_ptr,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[1, BLOCK_SIZE],
        sparse_dim=[1],
    )
    output_elements = rows * k
    value_scatter = tl.make_gather_scatter_view(
        values_ptr,
        shape=[output_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )
    index_scatter = tl.make_gather_scatter_view(
        indices_ptr,
        shape=[output_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )

    values = tl.reshape(
        tl.load(input_tiles, index=(row, 0)), (BLOCK_SIZE,)
    )
    values = tl.where(valid, values, pad_value).to(tl.float32)
    source_indices = offsets.to(tl.int32)
    _, sorted_indices = _argsort(values, source_indices, DESCENDING)

    # Only the first k ranks are observable.  Give every other gather lane a
    # valid coordinate so the backend never forms an extreme or negative
    # address.  Their scatter coordinates are one-past-the-end and TensorView
    # boundary handling drops those stores.
    output_valid = offsets < k
    gather_indices = tl.where(output_valid, sorted_indices, 0).to(tl.int32)
    selected = tl.reshape(
        tl.load(input_gather, index=(row, gather_indices)), (BLOCK_SIZE,)
    )
    output_indices = tl.where(
        output_valid, row * k + offsets, output_elements
    ).to(tl.int32)
    tl.store(value_scatter, selected, index=(output_indices,))
    tl.store(
        index_scatter,
        sorted_indices.to(tl.int64),
        index=(output_indices,),
    )


def topk(x, k, largest=True):
    """Return sorted TopK values and indices along the last dimension."""
    if x.device.type != "npu":
        raise ValueError("TensorView TopK requires an NPU tensor")
    if not x.is_contiguous():
        raise ValueError("TensorView TopK expects a contiguous input")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("TensorView TopK supports fp16, bf16, and fp32 inputs")
    if not 0 < k <= x.shape[-1]:
        raise ValueError("k must satisfy 0 < k <= input.shape[-1]")

    cols = x.shape[-1]
    if cols >= 4096:
        raise ValueError("single-stage TensorView TopK requires cols < 4096")
    rows = math.prod(x.shape[:-1])
    output_shape = x.shape[:-1] + (k,)
    values = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=x.device)
    if rows == 0:
        return values, indices
    block_size = triton.next_power_of_2(cols)

    topk_tensor_view_kernel[(rows,)](
        x,
        values,
        indices,
        rows,
        cols,
        k,
        BLOCK_SIZE=block_size,
        DESCENDING=largest,
    )
    return values, indices


def main():
    parser = argparse.ArgumentParser(description="Ascend TensorView TopK")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=255)
    parser.add_argument("--k", type=int, default=17)
    parser.add_argument("--smallest", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), dtype=torch.float32, device="npu")
    largest = not args.smallest
    actual_values, actual_indices = topk(x, args.k, largest=largest)
    expected_values, expected_indices = torch.topk(
        x, args.k, dim=-1, largest=largest, sorted=True
    )
    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_indices, expected_indices)
    print("TensorView TopK: Test Passed!")


if __name__ == "__main__":
    main()
