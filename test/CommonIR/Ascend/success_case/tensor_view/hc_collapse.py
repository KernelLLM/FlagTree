"""Persistent HC=4 weighted collapse using Ascend TensorView.

Compute Y[t, h] = sum_c X[t, c, h] * P[t, c], c in [0, 4).
The public entry point preserves the measured BF16 configuration: B=1024,
up to 40 persistent programs, vector P loads, and FP32 reduction.
Requires the TensorView-enabled FlagTree build used for validation.
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
from triton.experimental.tle.language.dsa import tile_load


# =============================================================================
# Persistent TensorView kernel (the measured kernel AST is preserved)
# =============================================================================
@triton.jit
def _hc_vector4(X, P, Y, T: tl.constexpr, H: tl.constexpr,
                B: tl.constexpr, NPROGRAMS: tl.constexpr,
                PERSISTENT: tl.constexpr, USE_VIEW: tl.constexpr):
    # Flatten the output into T * ceil(H / B) independent hidden tiles.
    NH: tl.constexpr = triton.cdiv(H, B)
    if USE_VIEW:
        # Contiguous X[T, 4, H] is viewed as [T * 4, H]. A [4, B]
        # partition groups all four channels of one token for this hidden tile.
        xv = tle.make_partition_view(X, shape=[T * 4, H],
                                    strides=[H, 1], tile=[4, B])
    start = tl.program_id(0)
    if PERSISTENT:
        end = T * NH
    else:
        end = start + 1
    c = tl.arange(0, 4)
    bh = tl.arange(0, B)
    # A persistent program handles jobs pid, pid + NPROGRAMS, ... .
    # Distinct jobs write disjoint output tiles; no atomic accumulation is needed.
    for job in range(start, end, NPROGRAMS):
        t = job // NH
        ht = job % NH
        h = ht * B + bh
        # The four FP32 weights are shared by all B hidden positions.
        # This is one vector load in the source, not a claim about DMA count.
        w = tl.load(P + t * 4 + c).to(tl.float32)  # four FP32 values, 16 B
        if USE_VIEW:
            # index uses tile coordinates: row tile t covers channels 0..3.
            # The partition view supplies boundary handling for the X tile.
            x = tile_load(xv, index=(t, ht),
                          dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
        else:
            x = tl.load(X + (t * 4 + c[:, None]) * H + h[None, :],
                        mask=h[None, :] < H, other=0.).to(tl.float32)
        # BF16 input values are widened before multiplication and reduction.
        # tl.sum may use a different FP32 association than a left-fold sum;
        # tolerance agreement does not imply bitwise agreement with AscendC.
        weighted = x * w[:, None]
        y = tl.sum(weighted, axis=0)
        # Round once to the output dtype and suppress stores beyond H.
        tl.store(Y + t * H + h, y.to(Y.dtype.element_ty), mask=h < H)


# =============================================================================
# Host-side configuration and launch
# =============================================================================
_DEFAULT_T = 128
_DEFAULT_H = 4096
_BLOCK_H = 1024
_MAX_PROGRAMS = 40


def compile_contract():
    """Require the tested TensorView API instead of silently changing paths."""
    from triton.backends.ascend import compiler
    if not hasattr(tle, 'make_partition_view'):
        raise RuntimeError('Need the TensorView-enabled FlagTree build; no fallback.')
    if not {'index', 'dst_space'} <= set(inspect.signature(tile_load).parameters):
        raise RuntimeError('Incompatible tile_load API')
    if 'use_commonir_tensor_view' not in getattr(compiler.NPUOptions, '__dataclass_fields__', {}):
        raise RuntimeError('Missing use_commonir_tensor_view compiler option')
    if getattr(compiler, '_USE_CUSTOM_COMPILE_OPT', None):
        raise RuntimeError('Remove USE_CUSTOM_COMPILE_OPT in a fresh process')
    return dict(use_commonir_tensor_view=True, enable_fp_fusion=False, debug=True)


def prepare_hc_collapse(x, p, out):
    """Validate arguments and return an allocation-free launch callable.

    Args:
        x: Contiguous BF16 NPU tensor with shape [T, 4, H].
        p: Contiguous FP32 tensor with shape [T, 4], on the same NPU.
        out: Contiguous BF16 tensor with shape [T, H], on the same NPU.

    T and H must be positive. The output must not overlap either input.
    This entry point is inference-only and deliberately exposes only the
    measured configuration, rather than unvalidated tuning options/dtypes.
    The caller must keep input/output buffers alive through device completion.
    """
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("Expected X[T, 4, H]")
    t, _, h = x.shape
    if t < 1 or h < 1 or tuple(p.shape) != (t, 4) or tuple(out.shape) != (t, h):
        raise ValueError("Expected positive T/H, P[T, 4], and output[T, H]")
    if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or p.dtype != torch.float32:
        raise TypeError("Expected BF16 input/output and FP32 weights")
    if x.device.type != "npu" or p.device != x.device or out.device != x.device:
        raise ValueError("All tensors must be on the same NPU")
    if not all(v.is_contiguous() for v in (x, p, out)):
        raise ValueError("All tensors must be contiguous")
    if any(v.requires_grad for v in (x, p, out)):
        raise ValueError("HC collapse is inference-only")
    for v in (x, p):
        if (out.data_ptr() < v.data_ptr() + v.numel() * v.element_size()
                and v.data_ptr() < out.data_ptr() + out.numel() * out.element_size()):
            raise ValueError("Output must not overlap inputs")

    # B=1024 and a cap of 40 programs match the measured Ascend910B4-1 setup.
    # The cap is a tested launch choice, not a portable hardware core query.
    block_h = _BLOCK_H
    programs = min(_MAX_PROGRAMS, t * triton.cdiv(h, block_h))
    options = compile_contract()

    def run():
        _hc_vector4[(programs,)](
            x, p, out, T=t, H=h, B=block_h, NPROGRAMS=programs,
            PERSISTENT=True, USE_VIEW=True, **options,
        )
        return out

    return run


def hc_collapse(x, p, out=None):
    """Launch HC collapse and return output; completion follows NPU stream order.

    Supply ``out`` or use ``prepare_hc_collapse`` when allocation must be
    excluded from timing. No backward implementation is provided.
    """
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("Expected X[T, 4, H]")
    if out is None:
        out = torch.empty((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
    return prepare_hc_collapse(x, p, out)()


# =============================================================================
# Correctness checks: CPU FP64 oracle and output guard regions
# =============================================================================
_PATTERNS = (
    "random", "positive", "zero", "onehot0", "onehot1", "onehot2", "onehot3",
    "cancel", "near_cancel",
)
_VERIFY_SHAPES = ((1, 16), (3, 1040), (41, 64), (2, 4096), (17, 5120), (65, 7168))


def check_case(t, h, device="npu:0", pattern="random", seed=0):
    """Run one case; bound CPU reference memory by processing 64 tokens at a time."""
    if t < 1 or h < 1 or pattern not in _PATTERNS:
        raise ValueError("Invalid shape or input pattern")
    generator = torch.Generator().manual_seed(20261029 + t + h + seed)
    x = torch.empty((t, 4, h), dtype=torch.bfloat16, device=device)
    p = torch.empty((t, 4), dtype=torch.float32, device=device)
    for lo in range(0, t, 64):
        n = min(64, t - lo)
        cx = torch.randn(n, 4, h, generator=generator).bfloat16()
        cp = torch.randn(n, 4, generator=generator)
        if pattern == "positive":
            cx = (2 * torch.rand(n, 4, h, generator=generator)).bfloat16()
            cp = 2 * torch.rand(n, 4, generator=generator)
        elif pattern == "zero":
            cp.zero_()
        elif pattern.startswith("onehot"):
            cp.zero_()
            cp[:, int(pattern[-1])] = 1
        elif pattern in ("cancel", "near_cancel"):
            cx[:, 1] = cx[:, 0]
            cx[:, 3] = cx[:, 2]
            cp[:] = torch.tensor([1., -1., .5, -.5])
            if pattern == "near_cancel":
                cp[:, 1] += 2. ** -20
        x[lo:lo + n].copy_(cx)
        p[lo:lo + n].copy_(cp)

    guard = torch.full((t * h + 32,), 123., dtype=torch.bfloat16, device=device)
    out = guard[16:-16].view(t, h)
    out.fill_(float("nan"))
    hc_collapse(x, p, out)
    torch.npu.synchronize()
    result = dict(T=t, H=h, pattern=pattern, passed=True, bitwise_equal=True,
                  max_abs=0., max_error_over_bound=0., left_max_abs=0.)
    for lo in range(0, t, 64):
        cx, cp, actual = x[lo:lo + 64].cpu(), p[lo:lo + 64].cpu(), out[lo:lo + 64].cpu()
        products64 = cx.double() * cp.double()[:, :, None]
        reference = products64.sum(1)
        # BF16 output rounding plus FP32 multiplication/reduction error.
        # sumabs handles cancellation, where relative error alone is misleading.
        u = 2. ** -8
        bound = u * reference.abs() + (1 + u) * 8 * (2. ** -24) * products64.abs().sum(1) + 1e-30
        error = (actual.double() - reference).abs()
        if not bool(torch.isfinite(actual).all()) or not bool((error <= bound).all()):
            raise AssertionError(f"Numerical check failed: T={t}, H={h}, pattern={pattern}, token={lo}")
        products32 = cx.float() * cp[:, :, None]
        left = (((products32[:, 0] + products32[:, 1]) + products32[:, 2]) + products32[:, 3]).bfloat16()
        result["bitwise_equal"] &= bool(torch.equal(actual.contiguous().view(torch.uint8),
                                                    left.contiguous().view(torch.uint8)))
        result["max_abs"] = max(result["max_abs"], float(error.max()))
        result["max_error_over_bound"] = max(result["max_error_over_bound"], float((error / bound).max()))
        result["left_max_abs"] = max(result["left_max_abs"], float((actual.float() - left.float()).abs().max()))
    if not bool((guard[:16].cpu() == 123).all()) or not bool((guard[-16:].cpu() == 123).all()):
        raise AssertionError("Output guard corruption")
    return result


def check_rejected_inputs(device):
    """Exercise host checks without launching an invalid device operation."""
    x = torch.empty((2, 4, 32), dtype=torch.bfloat16, device=device)
    p = torch.empty((2, 4), dtype=torch.float32, device=device)
    out = torch.empty((2, 16), dtype=torch.bfloat16, device=device)
    cases = ((x[:, :, ::2], p, out, "contiguous"),
             (x, p, x.view(-1)[:64].view(2, 32), "overlap"))
    for xx, pp, yy, expected in cases:
        try:
            prepare_hc_collapse(xx, pp, yy)
        except ValueError as exc:
            if expected not in str(exc):
                raise
        else:
            raise AssertionError(f"Expected rejection: {expected}")


# =============================================================================
# Intermediate TTIR dump (compiler required; no device launch)
# =============================================================================
def dump_ttir(path=None, T=_DEFAULT_T, H=_DEFAULT_H, return_module=False):
    """Compile the final BF16 specialization to TTIR, without running on NPU.

    This uses FlagTree internal frontend APIs, following native_matmul.py.
    It does not run the complete backend pipeline or validate device correctness.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir, ascend
    from triton._C.libtriton import tle as tle_ir
    from triton.backends.ascend.compiler import NPUOptions

    if T < 1 or H < 1:
        raise ValueError("T and H must be positive")
    # Use the installed backend's option dataclass, including its version-specific
    # defaults, rather than duplicating an incomplete options class here.
    options = NPUOptions(**compile_contract())
    constants = dict(T=T, H=H, B=_BLOCK_H,
                     NPROGRAMS=min(_MAX_PROGRAMS, T * triton.cdiv(H, _BLOCK_H)),
                     PERSISTENT=True, USE_VIEW=True)
    src = ASTSource(_hc_vector4, {"X": "*bf16", "P": "*fp32", "Y": "*bf16"}, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.dsa_ir.load_tile_dialects(context)
    ascend_ir = getattr(ascend, "ir", None)
    if ascend_ir is not None and hasattr(ascend_ir, "load_dialects"):
        ascend_ir.load_dialects(context)
    codegen_fns = {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}
    module = ast_to_ttir(_hc_vector4, src, context, options, codegen_fns, {})
    if not module.verify():
        raise RuntimeError("HC TTIR verification failed")
    text = str(module)
    target = Path(path) if path else Path.cwd() / "hc_collapse.ttir.mlir"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"[TTIR] {target.resolve()}")
    return module if return_module else text


# =============================================================================
# CLI entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Persistent HC=4 collapse with Ascend TensorView")
    parser.add_argument("--T", type=int, default=_DEFAULT_T)
    parser.add_argument("--H", type=int, default=_DEFAULT_H)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pattern", choices=_PATTERNS, default="random")
    parser.add_argument("--verify-all", action="store_true", help="54 numerical cases plus one input-contract check")
    parser.add_argument("--dump-ttir", nargs="?", const="hc_collapse.ttir.mlir", default=None,
                        help="Write TTIR and exit without a device launch")
    args = parser.parse_args()
    if args.T < 1 or args.H < 1:
        parser.error("T and H must be positive")
    if args.dump_ttir is not None:
        if args.verify_all:
            parser.error("--dump-ttir and --verify-all are separate modes")
        dump_ttir(args.dump_ttir, T=args.T, H=args.H)
        return

    # Register the NPU backend explicitly; never silently fall back to CUDA.
    import torch_npu  # noqa: F401
    if not args.device.startswith("npu") or not torch.npu.is_available():
        raise RuntimeError("An available Ascend NPU is required")
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    print(f"Triton: {triton.__version__} ({triton.__file__})")
    print(f"Device: {torch.npu.get_device_name(torch.npu.current_device())}")
    with torch.inference_mode():
        if args.verify_all:
            count = 0
            for t, h in _VERIFY_SHAPES:
                for pattern in _PATTERNS:
                    print(json.dumps(check_case(t, h, args.device, pattern, args.seed)), flush=True)
                    count += 1
            check_rejected_inputs(args.device)
            print(f"Test Passed! {count} numerical cases + 1 input-contract check")
        else:
            print(json.dumps(check_case(args.T, args.H, args.device, args.pattern, args.seed)))
            print("Test Passed!")


if __name__ == "__main__":
    main()

