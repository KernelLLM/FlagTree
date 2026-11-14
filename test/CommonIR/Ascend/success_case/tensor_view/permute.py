"""TensorView implementation of a materializing permute for Ascend.

The index calculation follows FlagGems' experimental permute operator, while
all input/output data movement is expressed through TensorView.  Metadata is
kept in separate tensors and does not alias either TensorView base pointer.
"""

import argparse

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def permute_tensor_view_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    input_span,
    ndim,
    in_strides_perm_ptr,
    out_shape_ptr,
    out_postfix_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_i64 = offsets.to(tl.int64)

    input_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for axis in range(MAX_DIMS):
        active = axis < ndim
        postfix = tl.load(out_postfix_ptr + axis, mask=active, other=1).to(
            tl.int64
        )
        axis_size = tl.load(out_shape_ptr + axis, mask=active, other=1).to(
            tl.int64
        )
        input_stride = tl.load(
            in_strides_perm_ptr + axis, mask=active, other=0
        ).to(tl.int64)
        coordinate = (offsets_i64 // postfix) % axis_size
        input_indices += coordinate * input_stride

    # The base of each view is the original kernel pointer argument.  The
    # gather index tensor describes the non-contiguous source addresses, while
    # the output is one contiguous partition per program.
    input_view = tl.make_gather_scatter_view(
        x_ptr,
        shape=[input_span],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )
    output_view = tl.make_partition_view(
        y_ptr,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    values = tl.load(input_view, index=(input_indices,))
    tl.store(output_view, values, index=(pid,))


def permute(x, dims, block_size=256):
    """Materialize ``x.permute(dims)`` through the TensorView kernel."""
    if x.device.type != "npu":
        raise ValueError("TensorView permute requires an NPU tensor")

    dims = tuple(int(dim) for dim in dims)
    ndim = x.ndim
    if ndim > 16:
        raise ValueError("TensorView permute supports at most 16 dimensions")
    if len(dims) != ndim:
        raise ValueError(f"expected {ndim} dimensions, got {len(dims)}")
    dims = tuple(dim % ndim for dim in dims)
    if len(set(dims)) != ndim:
        raise ValueError("dims must be a permutation without repeated axes")

    input_shape = tuple(x.shape)
    input_strides = tuple(x.stride())
    output_shape = tuple(input_shape[dim] for dim in dims)
    permuted_strides = tuple(input_strides[dim] for dim in dims)

    output_postfix = []
    product = 1
    for size in reversed(output_shape):
        output_postfix.append(product)
        product *= int(size)
    output_postfix.reverse()

    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    # Valid physical offsets can exceed numel for a positive-stride,
    # non-contiguous tensor.  Describe the full span rooted at x.data_ptr().
    input_span = 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(input_shape, input_strides)
    )
    strides_tensor = torch.tensor(
        permuted_strides, dtype=torch.int64, device=x.device
    )
    shape_tensor = torch.tensor(output_shape, dtype=torch.int64, device=x.device)
    postfix_tensor = torch.tensor(
        output_postfix, dtype=torch.int64, device=x.device
    )

    grid = (triton.cdiv(n_elements, block_size),)
    permute_tensor_view_kernel[grid](
        x,
        output,
        n_elements,
        input_span,
        ndim,
        strides_tensor,
        shape_tensor,
        postfix_tensor,
        BLOCK_SIZE=block_size,
        MAX_DIMS=16,
    )
    return output


def _parse_int_tuple(value):
    return tuple(int(item) for item in value.split(",") if item)


def main():
    parser = argparse.ArgumentParser(description="Ascend TensorView permute")
    parser.add_argument("--shape", default="2,3,5")
    parser.add_argument("--dims", default="2,0,1")
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    shape = _parse_int_tuple(args.shape)
    dims = _parse_int_tuple(args.dims)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="npu")
    actual = permute(x, dims, block_size=args.block_size)
    expected = torch.permute(x, dims).contiguous()
    torch.testing.assert_close(actual, expected)
    print("TensorView permute: Test Passed!")


if __name__ == "__main__":
    main()
