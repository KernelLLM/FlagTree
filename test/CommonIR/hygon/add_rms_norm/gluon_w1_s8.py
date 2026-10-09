import sys as _sys
if _sys.flags.no_site:
    import sysconfig as _sysconfig
    from pathlib import Path as _Path
    _python_version = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
    for _package_path in dict.fromkeys((
        _sysconfig.get_path("purelib"),
        _sysconfig.get_path("platlib"),
        f"/usr/local/lib/{_python_version}/dist-packages",
        f"/usr/lib/{_python_version}/dist-packages",
        "/usr/lib/python3/dist-packages",
    )):
        if _package_path and _Path(_package_path).is_dir() and _package_path not in _sys.path:
            _sys.path.append(_package_path)

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

VARIANT = "w1_s8"
LAYOUT = gl.BlockedLayout([8], [64], [1], [0])
NUM_WARPS = 1
NUM_STAGES = 2
WAVES_PER_EU = 1

@g.jit(do_not_specialize=['eps'])
def small(out_ptr, in_ptr1, in_ptr2, w_ptr, y_stride_r, y_stride_c, x1_stride_r, x1_stride_c, x2_stride_r, x2_stride_c, N, eps, BLOCK_SIZE: gl.constexpr):
    pid = gl.program_id(0)
    cols = gl.arange(0, BLOCK_SIZE, layout=LAYOUT)
    mask = cols < N
    x1 = cdna3.buffer_load(in_ptr1, pid * x1_stride_r + cols * x1_stride_c, mask=mask, other=0.0).to(gl.float32)
    x2 = cdna3.buffer_load(in_ptr2, pid * x2_stride_r + cols * x2_stride_c, mask=mask, other=0.0).to(gl.float32)
    x = x1 + x2
    var = gl.sum(x * x, axis=0) / N
    rrms = 1 / gl.sqrt(var + eps)
    if BLOCK_SIZE == 1:
        w = gl.load(w_ptr + cols, mask=mask, other=0.0)
    else:
        w = cdna3.buffer_load(w_ptr, cols, mask=mask, other=0.0)
    y = (x * rrms * w).to(out_ptr.dtype.element_ty)
    cdna3.buffer_store(y, out_ptr, pid * y_stride_r + cols * y_stride_c, mask=mask)

@g.jit(do_not_specialize=['eps'])
def loop(out_ptr, in_ptr1, in_ptr2, w_ptr, N, eps, TILE_N: gl.constexpr):
    pid = gl.program_id(0).to(gl.int64)
    acc = gl.full((TILE_N,), 0, gl.float32, layout=LAYOUT)
    num_steps = gl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + gl.arange(0, TILE_N, layout=LAYOUT)
        x1 = cdna3.buffer_load(in_ptr1, (pid * N + n_offsets).to(gl.int32)).to(gl.float32)
        x2 = cdna3.buffer_load(in_ptr2, (pid * N + n_offsets).to(gl.int32)).to(gl.float32)
        x = x1 + x2
        acc += x * x
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + gl.arange(0, TILE_N, layout=LAYOUT)
    mask = n_offsets < N
    x1 = cdna3.buffer_load(in_ptr1, (pid * N + n_offsets).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    x2 = cdna3.buffer_load(in_ptr2, (pid * N + n_offsets).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
    x = x1 + x2
    acc += x * x
    var = gl.sum(acc, axis=0) / N
    rrms = 1 / gl.sqrt(var + eps)
    prev_multiple = gl.cdiv(N, TILE_N) * TILE_N - TILE_N
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = prev_multiple - start_n + gl.arange(0, TILE_N, layout=LAYOUT)
        mask = n_offsets < N
        x1 = cdna3.buffer_load(in_ptr1, (pid * N + n_offsets).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
        x2 = cdna3.buffer_load(in_ptr2, (pid * N + n_offsets).to(gl.int32), mask=mask, other=0.0).to(gl.float32)
        x = x1 + x2
        w = cdna3.buffer_load(w_ptr, n_offsets, mask=mask, other=0.0)
        y = (x * rrms * w).to(out_ptr.dtype.element_ty)
        out_offsets = n_offsets
        cdna3.buffer_store(y, out_ptr, (pid * N + out_offsets).to(gl.int32), mask=out_offsets < N)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = prev_multiple - start_n + gl.arange(0, TILE_N, layout=LAYOUT)
        x1 = cdna3.buffer_load(in_ptr1, (pid * N + n_offsets).to(gl.int32)).to(gl.float32)
        x2 = cdna3.buffer_load(in_ptr2, (pid * N + n_offsets).to(gl.int32)).to(gl.float32)
        x = x1 + x2
        w = cdna3.buffer_load(w_ptr, n_offsets)
        y = (x * rrms * w).to(out_ptr.dtype.element_ty)
        out_offsets = n_offsets
        cdna3.buffer_store(y, out_ptr, (pid * N + out_offsets).to(gl.int32))

def add_rms_norm(x1, x2, weight, eps=1e-5):
    """Return RMSNorm(x1 + x2) * weight, using this file's fixed layout.

    Inputs: contiguous GPU x1/x2 [M,N] and weight [N], with a common
    FP16/BF16/FP32 dtype/device. No layout argument, autotuning, hidden copies,
    external configuration, or autograd support. Performance is archived for
    the 45 configurations in README; other shapes are not performance claims.
    """
    import torch
    import triton

    if x1.ndim != 2 or x2.shape != x1.shape or weight.ndim != 1 or weight.numel() != x1.shape[1]:
        raise ValueError("expected x1/x2 [M,N] and weight [N]")
    if x1.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes: FP16, BF16, FP32")
    if any(t.dtype != x1.dtype or t.device != x1.device or not t.is_cuda or not t.is_contiguous()
           for t in (x1, x2, weight)):
        raise ValueError("inputs must be contiguous GPU tensors with matching dtype/device")
    m, n = x1.shape
    if n <= 0:
        raise ValueError("N must be positive")
    # The delivered raw-buffer kernels use 32-bit byte-addressable offsets.
    if x1.numel() * x1.element_size() >= (1 << 31):
        raise ValueError("input exceeds this raw-buffer implementation's safe address range")
    out = torch.empty_like(x1)
    if m == 0:
        return out
    launch = dict(num_warps=NUM_WARPS, num_stages=NUM_STAGES, waves_per_eu=WAVES_PER_EU)
    if n <= 4096:
        small[(m,)](out, x1, x2, weight, n, 1, n, 1, n, 1, n, eps,
                    triton.next_power_of_2(n), **launch)
    else:
        tile = 8192 if x1.dtype == torch.bfloat16 else 4096
        loop[(m,)](out, x1, x2, weight, n, eps, tile, **launch)
    return out


def _self_test():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("a compatible HIP/DCU environment is required")
    torch.manual_seed(20261009)
    for m, n, dtype, tolerance in (
        (64, 64, torch.float16, 2e-3),
        (32, 16384, torch.bfloat16, 3e-2),
    ):
        x1 = torch.randn((m, n), device="cuda", dtype=dtype)
        x2 = torch.randn_like(x1)
        weight = torch.randn((n,), device="cuda", dtype=dtype)
        summed = x1.float() + x2.float()
        reference = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-5)
                     * weight.float()).to(dtype)
        actual = add_rms_norm(x1, x2, weight)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
        print(f"PASS add_rms_norm {VARIANT}: {m}x{n} {str(dtype).removeprefix('torch.')}", flush=True)


if __name__ == "__main__":
    _self_test()
