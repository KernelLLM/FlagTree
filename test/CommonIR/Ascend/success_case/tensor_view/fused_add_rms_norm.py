"""fused_add_rms_norm: add the residual into the input, then normalise it in place.

Both tensors are read through PartitionViews and written back through GatherScatterViews
addressed by row index. The operator writes both of its results in place, so the input and
residual pointers are only ever used through views: once a pointer backs a view it cannot also
take part in pointer arithmetic in the same kernel. Rows past the end of the tensor fall
outside the view and are dropped, which covers the tail block without a mask.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle
from triton.experimental.tle.language.dsa import tile_load, tile_store

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_ROWS = 4096
ROW_WIDTH = 512
BLOCK_M = 8
EPS = 1e-5


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def fused_add_rms_norm_kernel(x, residual, weight, num_rows, eps, WIDTH: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_WIDTH: tl.constexpr):
    row_block = tl.program_id(0)
    rows = (row_block * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)

    x_view = tl.make_partition_view(x, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                    tile=[BLOCK_M, BLOCK_WIDTH])
    residual_view = tl.make_partition_view(residual, shape=[num_rows, WIDTH],
                                           strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    weight_view = tl.make_partition_view(weight, shape=[WIDTH], strides=[1],
                                         tile=[BLOCK_WIDTH])
    tile = tile_load(x_view, index=(row_block, 0),
                     dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    res = tile_load(residual_view, index=(row_block, 0),
                    dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    scale = tile_load(weight_view, index=(0,),
                      dst_space=tle.language.dsa.ascend.UB).to(tl.float32)

    summed = tile + res
    rrms = 1.0 / tl.sqrt(tl.sum(summed * summed, axis=1) / WIDTH + eps)
    normalized = summed * rrms[:, None] * scale[None, :]

    x_out = tl.make_gather_scatter_view(x, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                        tile=[BLOCK_M, BLOCK_WIDTH], sparse_dim=[0])
    residual_out = tl.make_gather_scatter_view(residual, shape=[num_rows, WIDTH],
                                               strides=[WIDTH, 1],
                                               tile=[BLOCK_M, BLOCK_WIDTH], sparse_dim=[0])
    tile_store(residual_out, value=summed.to(residual.dtype.element_ty), index=(rows, 0),
               src_space=tle.language.dsa.ascend.UB)
    tile_store(x_out, value=normalized.to(x.dtype.element_ty), index=(rows, 0),
               src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=EPS, block_m=BLOCK_M):
    """Modifies x and residual in place: residual receives the sum, x the normalised sum."""
    width = normalized_shape[-1]
    assert x.shape[-1] == width and weight.numel() == width
    rows = x.numel() // width
    grid = (triton.cdiv(rows, block_m),)
    fused_add_rms_norm_kernel[grid](x, residual, weight, rows, eps, WIDTH=width,
                                   BLOCK_M=block_m,
                                   BLOCK_WIDTH=triton.next_power_of_2(width))
    return x, residual


def run_case(num_rows, width, device):
    x = torch.randn((num_rows, width), dtype=torch.float16, device=device)
    residual = torch.randn((num_rows, width), dtype=torch.float16, device=device)
    weight = torch.randn((width,), dtype=torch.float16, device=device)

    summed = x.float() + residual.float()
    rrms = 1.0 / torch.sqrt((summed * summed).mean(-1, keepdim=True) + EPS)
    expected_x = (summed * rrms * weight.float()[None, :]).to(torch.float16)
    expected_residual = summed.to(torch.float16)

    fused_add_rms_norm(x, residual, (width,), weight)

    torch.testing.assert_close(x, expected_x, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual, expected_residual)
    print(f"  rows={num_rows} width={width}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fused_add_rms_norm through TensorView")
    parser.add_argument("--rows", type=int, default=NUM_ROWS)
    parser.add_argument("--width", type=int, default=ROW_WIDTH)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.rows, args.width, device)
    # A row count that is not a multiple of the tile height leaves a partial tail block;
    # a width that is not a power of two makes the dense tile wider than the view.
    run_case(args.rows - 3, args.width, device)
    run_case(1024, 768, device)
    print("Test Passed!")
