"""
Test cases for tile.concat operation in Python/TLE DSL

These tests demonstrate the usage of tle.tile_concat for sinkhorn optimization,
showing how to combine small tensors into larger ones for better SIMD utilization.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.dsa as tle


@triton.jit
def concat_kernel_1d(
    x_ptr,
    y_ptr,
    z_ptr,
    N: tl.constexpr,
):
    """
    Test 1D tensor concat: <128> + <128> -> <256>

    This demonstrates the basic 1D concat pattern that will lower to
    tensor.insert_slice in Linalg IR.
    """
    # Load two 128-element vectors
    x = tl.load(x_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    y = tl.load(y_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)

    # Concatenate along dim=0 using TLE primitive
    z = tle.tile_concat(x, y, dim=0)  # tensor<256xf32>

    # Store result
    tl.store(z_ptr + tl.arange(0, 2*N), z, mask=tl.arange(0, 2*N) < 2*N)


@triton.jit
def concat_kernel_2d(
    x_ptr,
    y_ptr,
    z_ptr,
    N: tl.constexpr,
):
    """
    Test 2D tensor concat: <1x128> + <1x128> -> <2x128>

    This demonstrates the 2D concat pattern that will preserve tensor.concat
    in Linalg IR.
    """
    # Load two 1x128 tensors
    x = tl.load(x_ptr + tl.arange(0, N)[None, :], mask=tl.arange(0, N)[None, :] < N, other=0.0)
    y = tl.load(y_ptr + tl.arange(0, N)[None, :], mask=tl.arange(0, N)[None, :] < N, other=0.0)

    # Concatenate along dim=0 (row dimension) using TLE primitive
    z = tle.tile_concat(x, y, dim=0)  # tensor<2x128xf32>

    # Store result
    offs = tl.arange(0, 2)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(z_ptr + offs, z, mask=offs < 2*N)


@triton.jit
def sinkhorn_concat_pattern(
    row_ptr,
    out_ptr,
    rcp_sum_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sinkhorn optimization pattern: combine small 1x4 tensors into 2x4
    for better SIMD utilization.

    Before optimization:
        for i in range(8):
            row = load(ptr + i * 4)      # tensor<4xf32>
            row = row * rcp_sum          # 4-wide mul
            store(out + i * 4, row)

    After optimization with concat:
        row_0 = load(ptr + 0)            # tensor<1x4xf32>
        row_1 = load(ptr + 4)            # tensor<1x4xf32>
        row_01 = concat(row_0, row_1, dim=0)  # tensor<2x4xf32>
        row_01 = row_01 * rcp_sum        # 8-wide mul
        store(out, row_01)
    """
    # Load reciprocal sum (broadcast scalar)
    rcp_sum = tl.load(rcp_sum_ptr)

    # Load first row (1x4)
    row_0 = tl.load(row_ptr + tl.arange(0, BLOCK_SIZE)[None, :],
                    mask=tl.arange(0, BLOCK_SIZE)[None, :] < BLOCK_SIZE,
                    other=0.0)

    # Load second row (1x4)
    row_1 = tl.load(row_ptr + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :],
                    mask=tl.arange(0, BLOCK_SIZE)[None, :] < BLOCK_SIZE,
                    other=0.0)

    # Concat to get 2x4 tensor for wider SIMD using TLE primitive
    row_01 = tle.tile_concat(row_0, row_1, dim=0)  # tensor<2x4xf32>

    # Element-wise multiply with broadcast (8-wide operation)
    result = row_01 * rcp_sum

    # Store result
    offs = tl.arange(0, 2)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    tl.store(out_ptr + offs, result, mask=offs < 2*BLOCK_SIZE)


@triton.jit
def concat_chain_pattern(
    a_ptr, b_ptr, c_ptr, d_ptr,
    out_ptr,
    N: tl.constexpr,
):
    """
    Test chaining multiple concat operations:
    (a + b) + (c + d) -> final tensor<256xf32>
    """
    # Load four 64-element vectors
    a = tl.load(a_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    b = tl.load(b_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    c = tl.load(c_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    d = tl.load(d_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)

    # First level concat: 64 + 64 -> 128 using TLE primitive
    ab = tle.tile_concat(a, b, dim=0)
    cd = tle.tile_concat(c, d, dim=0)

    # Second level concat: 128 + 128 -> 256 using TLE primitive
    result = tle.tile_concat(ab, cd, dim=0)

    # Store result
    tl.store(out_ptr + tl.arange(0, 4*N), result, mask=tl.arange(0, 4*N) < 4*N)


@triton.jit
def concat_with_compute(
    x_ptr, y_ptr, scale_ptr,
    out_ptr,
    N: tl.constexpr,
):
    """
    Test concat followed by element-wise operations.
    Demonstrates that concat output can feed into standard Triton compute ops.
    """
    # Load inputs
    x = tl.load(x_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    y = tl.load(y_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    scale = tl.load(scale_ptr + tl.arange(0, 2*N), mask=tl.arange(0, 2*N) < 2*N, other=1.0)

    # Concat using TLE primitive
    z = tle.tile_concat(x, y, dim=0)  # tensor<256xf32>

    # Element-wise operations on concatenated tensor
    result = z * scale  # 256-wide multiply
    result = tl.exp(result)  # 256-wide exp

    # Store result
    tl.store(out_ptr + tl.arange(0, 2*N), result, mask=tl.arange(0, 2*N) < 2*N)


# ---------------------------------------------------------------------------
# 精度测试: 运行上面定义的 kernel, 与 torch 参考结果对比
# 运行: python test_tile_concat.py
# ---------------------------------------------------------------------------
DEVICE = "npu"


def _check(name, actual, ref, rtol=1e-3, atol=1e-3):
    """对比 kernel 输出与参考结果, 通过打印 ✅, 失败打印 ❌ 但不中断。"""
    try:
        torch.testing.assert_close(actual.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name}: {type(e).__name__}: {e}")
        return False


def run_precision_tests():
    torch.manual_seed(0)
    grid = (1,)
    results = []

    # 1D concat: <N> + <N> -> <2N>
    N = 128
    x = torch.randn(N, dtype=torch.float32, device=DEVICE)
    y = torch.randn(N, dtype=torch.float32, device=DEVICE)
    z = torch.zeros(2 * N, dtype=torch.float32, device=DEVICE)
    concat_kernel_1d[grid](x, y, z, N=N)
    results.append(_check("concat_kernel_1d", z, torch.cat([x, y], dim=0)))

    # 2D concat dim=0: <1xN> + <1xN> -> <2xN>
    x2 = torch.randn(1, N, dtype=torch.float32, device=DEVICE)
    y2 = torch.randn(1, N, dtype=torch.float32, device=DEVICE)
    z2 = torch.zeros(2, N, dtype=torch.float32, device=DEVICE)
    concat_kernel_2d[grid](x2, y2, z2, N=N)
    results.append(_check("concat_kernel_2d", z2, torch.cat([x2, y2], dim=0)))

    # sinkhorn: cat([row_0, row_1], 0) * rcp_sum
    BLK = 4
    row = torch.randn(2 * BLK, dtype=torch.float32, device=DEVICE)
    rcp = torch.randn(1, dtype=torch.float32, device=DEVICE)
    out_s = torch.zeros(2, BLK, dtype=torch.float32, device=DEVICE)
    sinkhorn_concat_pattern[grid](row, out_s, rcp, BLOCK_SIZE=BLK)
    ref_s = torch.cat([row[:BLK].reshape(1, BLK), row[BLK:].reshape(1, BLK)], dim=0) * rcp
    results.append(_check("sinkhorn_concat_pattern", out_s, ref_s))

    # chain: cat([a, b, c, d], 0)
    Nc = 64
    a = torch.randn(Nc, dtype=torch.float32, device=DEVICE)
    b = torch.randn(Nc, dtype=torch.float32, device=DEVICE)
    c = torch.randn(Nc, dtype=torch.float32, device=DEVICE)
    d = torch.randn(Nc, dtype=torch.float32, device=DEVICE)
    out_c = torch.zeros(4 * Nc, dtype=torch.float32, device=DEVICE)
    concat_chain_pattern[grid](a, b, c, d, out_c, N=Nc)
    results.append(_check("concat_chain_pattern", out_c, torch.cat([a, b, c, d], dim=0)))

    # compute: exp(cat([x, y], 0) * scale)
    xw = torch.randn(N, dtype=torch.float32, device=DEVICE)
    yw = torch.randn(N, dtype=torch.float32, device=DEVICE)
    scale = torch.randn(2 * N, dtype=torch.float32, device=DEVICE)
    out_w = torch.zeros(2 * N, dtype=torch.float32, device=DEVICE)
    concat_with_compute[grid](xw, yw, scale, out_w, N=N)
    ref_w = torch.exp(torch.cat([xw, yw], dim=0) * scale)
    results.append(_check("concat_with_compute", out_w, ref_w))

    # f16: cat([x, y], 0), 半精度放宽容差
    xf = torch.randn(N, dtype=torch.float32, device=DEVICE)
    yf = torch.randn(N, dtype=torch.float32, device=DEVICE)
    out_f = torch.zeros(2 * N, dtype=torch.float16, device=DEVICE)
    concat_f16_pattern[grid](xf, yf, out_f, N=N)
    ref_f = torch.cat([xf.half(), yf.half()], dim=0)
    results.append(_check("concat_f16_pattern", out_f, ref_f, rtol=1e-2, atol=1e-2))

    # dim=1: <MxN> + <MxN> -> <Mx2N>
    M, Nd = 32, 64
    xd = torch.randn(M, Nd, dtype=torch.float32, device=DEVICE)
    yd = torch.randn(M, Nd, dtype=torch.float32, device=DEVICE)
    out_d = torch.zeros(M, 2 * Nd, dtype=torch.float32, device=DEVICE)
    concat_dim1_pattern[grid](xd, yd, out_d, M=M, N=Nd)
    results.append(_check("concat_dim1_pattern", out_d, torch.cat([xd, yd], dim=1)))

    passed = sum(results)
    print("-" * 60)
    print(f"精度测试: {passed}/{len(results)} 通过")
    return passed == len(results)


@triton.jit
def concat_f16_pattern(
    x_ptr, y_ptr,
    out_ptr,
    N: tl.constexpr,
):
    """
    Test concat with fp16 tensors.
    """
    # Load fp16 inputs
    x = tl.load(x_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0).to(tl.float16)
    y = tl.load(y_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0).to(tl.float16)

    # Concat fp16 tensors using TLE primitive
    z = tle.tile_concat(x, y, dim=0)  # tensor<256xf16>

    # Store result
    tl.store(out_ptr + tl.arange(0, 2*N), z, mask=tl.arange(0, 2*N) < 2*N)


@triton.jit
def concat_dim1_pattern(
    x_ptr, y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    """
    Test concat along dim=1 (column dimension).
    <128x64> + <128x64> -> <128x128>
    """
    # Load two 128x64 tensors
    x_offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    y_offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]

    x = tl.load(x_ptr + x_offs, mask=x_offs < M*N, other=0.0)
    y = tl.load(y_ptr + y_offs, mask=y_offs < M*N, other=0.0)

    # Concat along dim=1 using TLE primitive
    z = tle.tile_concat(x, y, dim=1)  # tensor<128x128xf32>

    # Store result
    out_offs = tl.arange(0, M)[:, None] * (2*N) + tl.arange(0, 2*N)[None, :]
    tl.store(out_ptr + out_offs, z, mask=out_offs < M*2*N)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_precision_tests() else 1)
