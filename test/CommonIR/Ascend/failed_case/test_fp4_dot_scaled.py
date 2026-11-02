#!/usr/bin/env python3
"""Test fp4 (e2m1) dot_scaled compilation through the triton-ascend pipeline.

E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
Two fp4 values are packed into one uint8 byte (low nibble = first element).
Scale factors are uint8 (e8m0 format), one per 32-element group.

This test constructs a minimal dot_scaled kernel with e2m1 inputs and
attempts to compile it through ast_to_ttir + the full ascend lowering pipeline.
"""

import os
import sys
import tempfile

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import triton
import triton.language as tl
from triton._C.libtriton import ir, passes as ir_passes
from triton._C.libtriton.ascend import passes as ascend_passes

try:
    from triton._C.libtriton.ascend import ir as ascend_ir
    _has_ascend_ir = True
except ImportError:
    _has_ascend_ir = False

try:
    from triton._C.libtriton import tle as tle_ir
    _has_tle = True
except ImportError:
    tle_ir = None
    _has_tle = False


# =====================================================================
# Kernel: fp4 e2m1 dot_scaled matmul
# =====================================================================
# For e2m1: K=64 means 64 fp4 elements → packed into 32 uint8 bytes
# Scale: one uint8 per 32 elements → K//32 scale entries per row
#
# Shapes (e2m1, lhs_k_pack=True):
#   A:       (M, K//2) uint8       — K fp4 values packed into K//2 bytes
#   A_scale: (M, K//32) uint8      — e8m0 scale per 32-element group
#   B:       (K//2, N) uint8       — same packing for B
#   B_scale: (N, K//32) uint8      — scale for B (note: indexed by N)
#   C:       (M, N)    float32     — output accumulator

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 64  # number of fp4 elements in the K dimension


@triton.jit
def fp4_dot_scaled_kernel(
    a_ptr,       # *uint8, shape (M, K//2)
    a_scale_ptr, # *uint8, shape (M, K//32)
    b_ptr,       # *uint8, shape (K//2, N)
    b_scale_ptr, # *uint8, shape (N, K//32)
    c_ptr,       # *float32, shape (M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Minimal fp4 e2m1 dot_scaled kernel for compilation testing."""
    # A: (BLOCK_M, BLOCK_K // 2) uint8 — K fp4 values packed into K/2 bytes
    PACKED_K: tl.constexpr = BLOCK_K // 2  # 32 bytes for 64 fp4 values
    SCALE_K: tl.constexpr = BLOCK_K // 32  # 2 scale entries per row

    # Load A block
    a_offs = tl.arange(0, BLOCK_M)[:, None] * PACKED_K + tl.arange(0, PACKED_K)[None, :]
    a = tl.load(a_ptr + a_offs)  # (BLOCK_M, PACKED_K) uint8

    # Load A scale
    a_scale_offs = tl.arange(0, BLOCK_M)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :]
    a_scale = tl.load(a_scale_ptr + a_scale_offs)  # (BLOCK_M, SCALE_K) uint8

    # Load B block: (PACKED_K, BLOCK_N) — same K packing
    b_offs = tl.arange(0, PACKED_K)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    b = tl.load(b_ptr + b_offs)  # (PACKED_K, BLOCK_N) uint8

    # Load B scale
    b_scale_offs = tl.arange(0, BLOCK_N)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :]
    b_scale = tl.load(b_scale_ptr + b_scale_offs)  # (BLOCK_N, SCALE_K) uint8

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # dot_scaled with e2m1 format for both lhs and rhs
    acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc)

    # Store output
    c_offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(c_ptr + c_offs, acc)


# =====================================================================
# Also test: e2m1 lhs × e2m1 rhs, rhs_scale=None
# =====================================================================
@triton.jit
def fp4_dot_scaled_no_rhs_scale_kernel(
    a_ptr, a_scale_ptr, b_ptr, c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP4 dot_scaled with rhs_scale=None (only lhs has microscaling)."""
    PACKED_K: tl.constexpr = BLOCK_K // 2
    SCALE_K: tl.constexpr = BLOCK_K // 32

    a_offs = tl.arange(0, BLOCK_M)[:, None] * PACKED_K + tl.arange(0, PACKED_K)[None, :]
    a = tl.load(a_ptr + a_offs)

    a_scale_offs = tl.arange(0, BLOCK_M)[:, None] * SCALE_K + tl.arange(0, SCALE_K)[None, :]
    a_scale = tl.load(a_scale_ptr + a_scale_offs)

    b_offs = tl.arange(0, PACKED_K)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    b = tl.load(b_ptr + b_offs)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot_scaled(a, a_scale, "e2m1", b, None, "e2m1", acc=acc)

    c_offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(c_ptr + c_offs, acc)


# =====================================================================
# Compilation infrastructure
# =====================================================================
class _Opts:
    num_warps = 4
    num_stages = 1
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = True
    allowed_dot_input_precisions = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default = 0
    default_dot_input_precision = "ieee"


def make_context():
    ctx = ir.context()
    ir.load_dialects(ctx)
    if _has_tle:
        tle_ir.load_dialects(ctx)
        tle_ir.load_tile_dialects(ctx)
    if _has_ascend_ir:
        ascend_ir.load_dialects(ctx)
    return ctx


def compile_to_ttir(kernel_fn, signature, constants, name):
    """Compile kernel to TTIR (CommonIR) and return (module, context, mlir_str)."""
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir

    src = ASTSource(kernel_fn, signature, constants)
    context = make_context()
    codegen_fns = {"min_dot_size": lambda a, b: (1, 1, 1)}
    try:
        module = ast_to_ttir(kernel_fn, src, context, _Opts(), codegen_fns, {})
        ok = module.verify()
        mlir = module.str()
        print(f"  [ast_to_ttir] OK  verify={ok}  {len(mlir)} chars")
        return module, context, mlir
    except Exception as e:
        print(f"  [ast_to_ttir] FAILED: {e}")
        return None, None, None


def run_lowering_pipeline(module, context, name):
    """Run the full ascend lowering pipeline on a module, reporting each pass."""
    passes_list = [
        ("inliner",       lambda pm: ir_passes.common.add_inliner(pm)),
        ("structure",     lambda pm: ascend_passes.ttir.add_triton_to_structure(pm, False, False)),
        ("discrete_mask", lambda pm: ascend_passes.ttir.add_discrete_mask_access_conversion(pm, False, False, False)),
        ("unstructure",   lambda pm: ascend_passes.ttir.add_triton_to_unstructure(pm, False, False)),
        ("hivm",          lambda pm: ascend_passes.ttir.add_triton_to_hivm(pm)),
        ("hfusion",       lambda pm: ascend_passes.ttir.add_triton_to_hfusion(pm)),
        ("llvm",          lambda pm: ascend_passes.ttir.add_triton_to_llvm(pm)),
        ("bubble_up",     lambda pm: ascend_passes.ttir.add_bubble_up_operation(pm)),
        ("structure_r2",  lambda pm: ascend_passes.ttir.add_triton_to_structure(pm, False, False)),
        ("linalg",        lambda pm: ascend_passes.ttir.add_triton_to_linalg(pm, False, True, False, False, False)),
        ("annotation",    lambda pm: ascend_passes.ttir.add_triton_to_annotation(pm)),
        ("dag_scope",     lambda pm: ascend_passes.ttir.add_dag_scope(pm)),
        ("dag_ssbuffer",  lambda pm: ascend_passes.ttir.add_dag_ssbuffer(pm)),
        ("dag_sync",      lambda pm: ascend_passes.ttir.add_dag_sync(pm)),
    ]

    out_dir = "/dev/shm/yuansheng/IR_triton_ascend_diff"
    all_ok = True
    for pname, add_fn in passes_list:
        pm = ir.pass_manager(context)
        pm.enable_debug()
        add_fn(pm)
        try:
            pm.run(module)
            v = module.verify()
            print(f"  [{pname:20s}] OK  verify={v}")
        except RuntimeError as e:
            err = str(e)[:120]
            print(f"  [{pname:20s}] FAILED: {err}")
            all_ok = False
            break

    # Dump final IR
    final_mlir = module.str()
    final_path = os.path.join(out_dir, f"fp4_{name}_final.mlir")
    with open(final_path, "w") as f:
        f.write(final_mlir)
    print(f"  → Final IR: {final_path} ({len(final_mlir)} chars)")
    return all_ok


def test_kernel(kernel_fn, name, signature, constants):
    """Full test: compile + lower."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")

    # Phase 1: ast_to_ttir
    print("Phase 1: AST → TTIR")
    module, context, mlir = compile_to_ttir(kernel_fn, signature, constants, name)
    if module is None:
        return False

    # Save CommonIR
    out_dir = "/dev/shm/yuansheng/IR_triton_ascend_diff"
    ttir_path = os.path.join(out_dir, f"fp4_{name}_ttir.mlir")
    with open(ttir_path, "w") as f:
        f.write(mlir)
    print(f"  → TTIR: {ttir_path}")

    # Phase 2: lowering
    print("Phase 2: Lowering pipeline")
    ok = run_lowering_pipeline(module, context, name)
    return ok


# =====================================================================
# Precision test infrastructure
# =====================================================================
import torch
import numpy as np


# ── E2M1 LUT: nibble value (0-15) → float ────────────────────────────
# bits: [S][E1 E0][M0]
# E==0 subnormal: val = (-1)^S * 0.M0       → 0 or 0.5
# E!=0 normal:    val = (-1)^S * 2^(E-1) * 1.M0
_E2M1_LUT = np.array([
    0.0,   0.5,  1.0,  1.5,   2.0,  3.0,   4.0,  6.0,    # positive: 0b0_XX_X
   -0.0,  -0.5, -1.0, -1.5,  -2.0, -3.0,  -4.0, -6.0,    # negative: 0b1_XX_X
], dtype=np.float32)


def decode_e2m1_packed(packed: np.ndarray) -> np.ndarray:
    """Decode uint8-packed e2m1 → float32.  Each byte → 2 floats (lo nibble first)."""
    lo = (packed & 0x0F).astype(np.int32)
    hi = ((packed >> 4) & 0x0F).astype(np.int32)
    flo = _E2M1_LUT[lo]
    fhi = _E2M1_LUT[hi]
    # interleave: (*, K//2) → (*, K)
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float32)
    out[..., 0::2] = flo
    out[..., 1::2] = fhi
    return out


def decode_e8m0_scale(scale: np.ndarray) -> np.ndarray:
    """Decode e8m0 scale (uint8) → float32.  val = 2^(scale - 127)."""
    return np.ldexp(np.ones_like(scale, dtype=np.float32), scale.astype(np.int32) - 127)


def apply_mx_scale(decoded_fp: np.ndarray, scale: np.ndarray, group_size: int = 32) -> np.ndarray:
    """Apply MX block scaling: one scale per `group_size` elements along last axis."""
    # decoded_fp: (..., K)   scale: (..., K // group_size)
    scale_fp = decode_e8m0_scale(scale)                    # (..., K//group_size)
    scale_expanded = np.repeat(scale_fp, group_size, axis=-1)  # (..., K)
    return decoded_fp * scale_expanded


def reference_mx_matmul(a_packed, a_scale, b_packed, b_scale, group_size=32):
    """CPU reference: (A_e2m1 * A_scale) @ (B_e2m1 * B_scale)^T-style matmul.

    a_packed: (M, K//2) uint8
    a_scale:  (M, K//32) uint8
    b_packed: (K//2, N) uint8  — note: K dim is rows
    b_scale:  (N, K//32) uint8 — note: indexed by N (one scale row per output col)
    """
    a_packed_np = a_packed.cpu().numpy()
    b_packed_np = b_packed.cpu().numpy()
    a_scale_np = a_scale.cpu().numpy()

    # Decode A: (M, K//2) → (M, K)
    a_fp = decode_e2m1_packed(a_packed_np)
    a_fp = apply_mx_scale(a_fp, a_scale_np, group_size)

    # Decode B: (K//2, N) → first transpose to (N, K//2), decode, then get (N, K)
    # But B is stored (K//2, N), so decode along axis 0 is tricky.
    # Transpose → (N, K//2), decode → (N, K), scale, transpose back → (K, N)
    b_T_packed = b_packed_np.T               # (N, K//2)
    b_T_fp = decode_e2m1_packed(b_T_packed)  # (N, K)
    if b_scale is not None:
        b_scale_np = b_scale.cpu().numpy()   # (N, K//32)
        b_T_fp = apply_mx_scale(b_T_fp, b_scale_np, group_size)
    b_fp = b_T_fp.T                          # (K, N)

    # Matmul in fp32
    c = a_fp.astype(np.float64) @ b_fp.astype(np.float64)
    return c.astype(np.float32)


# ── Triton kernel launcher (on NPU) ──────────────────────────────────
def run_fp4_dot_scaled_on_npu(a_packed, a_scale, b_packed, b_scale):
    """Launch fp4_dot_scaled_kernel on NPU and return the result tensor."""
    M = a_packed.shape[0]
    N = b_packed.shape[1]
    c = torch.empty((M, N), dtype=torch.float32, device=a_packed.device)
    grid = (1,)
    fp4_dot_scaled_kernel[grid](
        a_packed, a_scale, b_packed, b_scale, c,
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=a_packed.shape[1] * 2,
    )
    return c


def run_fp4_dot_scaled_no_rhs_scale_on_npu(a_packed, a_scale, b_packed):
    """Launch fp4_dot_scaled_no_rhs_scale_kernel on NPU."""
    M = a_packed.shape[0]
    N = b_packed.shape[1]
    c = torch.empty((M, N), dtype=torch.float32, device=a_packed.device)
    grid = (1,)
    fp4_dot_scaled_no_rhs_scale_kernel[grid](
        a_packed, a_scale, b_packed, c,
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=a_packed.shape[1] * 2,
    )
    return c


# ── Precision test runner ─────────────────────────────────────────────
def run_precision_test(M, N, K, with_rhs_scale, seed=42):
    """Run a single precision test case."""
    device = "npu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    PACKED_K = K // 2
    SCALE_K = K // 32

    # Generate random packed fp4 data (uint8)
    # Clamp to valid e2m1 nibbles (0-15 are all valid, but avoid large values
    # to keep the matmul result in a reasonable range for comparison)
    a_packed = torch.randint(0, 256, (M, PACKED_K), dtype=torch.uint8, device=device)
    b_packed = torch.randint(0, 256, (PACKED_K, N), dtype=torch.uint8, device=device)

    # Generate scales: use moderate exponents to avoid overflow
    # e8m0: scale = 2^(val - 127), so val=127 → scale=1.0
    # Use range [120, 134] → scale in [2^-7, 2^7] = [0.0078, 128]
    a_scale = torch.randint(120, 135, (M, SCALE_K), dtype=torch.uint8, device=device)

    if with_rhs_scale:
        b_scale = torch.randint(120, 135, (N, SCALE_K), dtype=torch.uint8, device=device)
    else:
        b_scale = None

    # CPU reference
    ref = reference_mx_matmul(a_packed, a_scale, b_packed, b_scale, group_size=32)
    ref_tensor = torch.from_numpy(ref)

    # NPU result
    try:
        if with_rhs_scale:
            npu_result = run_fp4_dot_scaled_on_npu(a_packed, a_scale, b_packed, b_scale)
        else:
            npu_result = run_fp4_dot_scaled_no_rhs_scale_on_npu(a_packed, a_scale, b_packed)
        npu_cpu = npu_result.cpu()
    except Exception as e:
        return False, f"NPU kernel launch failed: {e}", None, None

    # Compare
    abs_diff = (npu_cpu.float() - ref_tensor.float()).abs()
    ref_abs = ref_tensor.float().abs().clamp(min=1e-6)
    rel_diff = abs_diff / ref_abs

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()
    mean_rel = rel_diff.mean().item()

    # Tolerance: fp4 has very low precision (only 16 representable values),
    # and the matmul accumulates K products, so we allow generous tolerance.
    atol = 1e-2
    rtol = 5e-2
    passed = torch.allclose(npu_cpu.float(), ref_tensor.float(), atol=atol, rtol=rtol)

    stats = {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "mean_rel_diff": mean_rel,
        "atol": atol,
        "rtol": rtol,
    }
    return passed, stats, npu_cpu, ref_tensor


def main():
    os.makedirs("/dev/shm/yuansheng/IR_triton_ascend_diff", exist_ok=True)

    # =====================================================================
    # Part 1: Compilation tests (pure front-end, no device needed)
    # =====================================================================
    print("=" * 70)
    print("Part 1: Compilation tests")
    print("=" * 70)
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")

    sig1 = {
        "a_ptr": "*u8", "a_scale_ptr": "*u8",
        "b_ptr": "*u8", "b_scale_ptr": "*u8",
        "c_ptr": "*fp32",
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
    }
    const1 = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
    ok1 = test_kernel(fp4_dot_scaled_kernel, "e2m1_both_scale", sig1, const1)

    sig2 = {
        "a_ptr": "*u8", "a_scale_ptr": "*u8",
        "b_ptr": "*u8",
        "c_ptr": "*fp32",
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
    }
    const2 = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
    ok2 = test_kernel(fp4_dot_scaled_no_rhs_scale_kernel, "e2m1_no_rhs_scale", sig2, const2)

    print(f"\nCompilation results:")
    print(f"  e2m1 × e2m1 (both scales):  {'PASS' if ok1 else 'FAIL'}")
    print(f"  e2m1 × e2m1 (no rhs scale): {'PASS' if ok2 else 'FAIL'}")

    # =====================================================================
    # Part 2: Precision tests (requires NPU)
    # =====================================================================
    print()
    print("=" * 70)
    print("Part 2: Precision tests (NPU)")
    print("=" * 70)

    has_npu = hasattr(torch, "npu") and torch.npu.is_available()
    if not has_npu:
        print("  ⚠ No NPU available, skipping precision tests.")
        return

    import torch_npu  # noqa: F401
    print(f"  NPU: {torch.npu.get_device_name(0)}")
    print()

    test_cases = [
        # (M,  N,  K,   with_rhs_scale, description)
        (32,  32,  64,  True,  "32×32×64,  both scales"),
        (32,  32,  64,  False, "32×32×64,  no rhs scale"),
        (32,  32,  128, True,  "32×32×128, both scales"),
        (32,  32,  128, False, "32×32×128, no rhs scale"),
        (64,  64,  64,  True,  "64×64×64,  both scales"),
        (64,  64,  128, True,  "64×64×128, both scales"),
    ]

    all_passed = True
    print(f"  {'Test':<35s} {'Status':>8s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'max_rel':>10s}  {'mean_rel':>10s}")
    print(f"  {'-'*35} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for M, N, K, with_rhs_scale, desc in test_cases:
        passed, stats, npu_out, ref_out = run_precision_test(M, N, K, with_rhs_scale)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        if isinstance(stats, dict):
            print(f"  {desc:<35s} {status:>8s}  {stats['max_abs_diff']:>10.4e}  "
                  f"{stats['mean_abs_diff']:>10.4e}  {stats['max_rel_diff']:>10.4e}  "
                  f"{stats['mean_rel_diff']:>10.4e}")
        else:
            # stats is an error message string
            print(f"  {desc:<35s} {'ERROR':>8s}  {stats}")

    # ── Final summary ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Final: Compilation {'PASS' if (ok1 and ok2) else 'FAIL'} | "
          f"Precision {'ALL PASS' if all_passed else 'SOME FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
