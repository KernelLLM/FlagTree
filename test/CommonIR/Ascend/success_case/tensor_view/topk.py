"""Single-stage row-wise TopK implemented with TensorView for Ascend.

The iterative selection follows the FlagGems TopK stage kernels.  The
data-dependent value lookup and output stores are represented by
gather/scatter views instead of pointer arithmetic.
"""

import argparse
import math

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def topk_tensor_view_kernel(
    x_ptr,
    values_ptr,
    indices_ptr,
    rows,
    cols: tl.constexpr,
    k,
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

    candidates = tl.reshape(
        tl.load(input_tiles, index=(row, 0)), (BLOCK_SIZE,)
    )
    candidates = tl.where(valid, candidates, pad_value).to(tl.float32)
    available = valid
    selected_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # This is the selection strategy used by the FlagGems TopK stage kernels.
    # It avoids the reshape/broadcast/bitcast sequence of bitonic argsort,
    # which can produce unaligned Ascend UB vector accesses.
    for rank in range(k):
        if DESCENDING:
            selected_value = tl.max(candidates)
        else:
            selected_value = tl.min(candidates)
        is_candidate = available & (candidates == selected_value)
        candidate_indices = tl.where(is_candidate, offsets, BLOCK_SIZE)
        selected_index = tl.argmin(candidate_indices, axis=0)
        selected_indices = tl.where(
            offsets == rank, selected_index, selected_indices
        )
        available = available & (offsets != selected_index)
        candidates = tl.where(
            offsets == selected_index, pad_value, candidates
        )

    # Only the first k ranks are observable.  Give every other gather lane a
    # valid coordinate so the backend never forms an extreme or negative
    # address.  Their scatter coordinates are one-past-the-end and TensorView
    # boundary handling drops those stores.
    output_valid = offsets < k
    gather_indices = tl.where(output_valid, selected_indices, 0).to(tl.int32)
    selected = tl.reshape(
        tl.load(input_gather, index=(row, gather_indices)), (BLOCK_SIZE,)
    )
    output_indices = tl.where(
        output_valid, row * k + offsets, output_elements
    ).to(tl.int32)
    tl.store(value_scatter, selected, index=(output_indices,))
    tl.store(
        index_scatter,
        selected_indices.to(tl.int64),
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
