import argparse
import os

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle


_DEFAULT_M = 128
_DEFAULT_N = 128
_DEFAULT_K = 128

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32


@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    a_block_ptr = tl.make_block_ptr(
        mat_a, (M, K), (K, 1),
        (m_start, 0),
        (BLOCK_M, BLOCK_K), (1, 0))
    b_block_ptr = tl.make_block_ptr(
        mat_b, (K, N), (N, 1),
        (0, n_start),
        (BLOCK_K, BLOCK_N), (1, 0))

    mat_a_shared = tle.gpu.alloc(
        [BLOCK_M, BLOCK_K], dtype=mat_a.dtype.element_ty, mem_addr_space=tle.gpu.shared)
    mat_b_shared = tle.gpu.alloc(
        [BLOCK_K, BLOCK_N], dtype=mat_b.dtype.element_ty, mem_addr_space=tle.gpu.shared)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        tle.gpu.copy(a_block_ptr, mat_a_shared, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_block_ptr, mat_b_shared, [BLOCK_K, BLOCK_N])
        acc = tl.dot(
            tle.gpu.to_tensor(mat_a_shared, writable=False),
            tle.gpu.to_tensor(mat_b_shared, writable=False),
            acc,
            out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_K])
        b_block_ptr = tl.advance(b_block_ptr, [BLOCK_K, 0])

    tl.store(
        tl.make_block_ptr(
            mat_c, (M, N), (N, 1),
            (m_start, n_start),
            (BLOCK_M, BLOCK_N), (1, 0)),
        acc.to(mat_c.dtype.element_ty))


def _grid(M, N):
    return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)


def call(mat_a, mat_b):
    M = mat_a.shape[0]
    K = mat_a.shape[1]
    N = mat_b.shape[1]
    mat_c = torch.empty((M, N), dtype=mat_a.dtype, device=mat_a.device)
    matmul_kernel[_grid(M, N)](
        mat_a, mat_b, mat_c, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3)
    return mat_c


class _DumpOptions:
    num_warps = 4
    num_stages = 1
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    allowed_dot_input_precisions = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default = 0
    default_dot_input_precision = "ieee"
    sanitize_overflow = False


def _signature():
    return {
        "mat_a": "*fp16",
        "mat_b": "*fp16",
        "mat_c": "*fp16",
    }


def _compile_frontend_module(M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K):
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    constants = {
        "M": M,
        "N": N,
        "K": K,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
    }
    src = ASTSource(matmul_kernel.fn, _signature(), constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (16, 16, 16)}
    module = ast_to_ttir(matmul_kernel, src, context, _DumpOptions(), codegen_fns, {})
    if not module.verify():
        raise RuntimeError("_compile_frontend_module: module.verify() failed")
    return module


def dump_ttir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native_matmul_gpu_ttir.mlir")
    module = _compile_frontend_module(M=M, N=N, K=K)
    mlir = str(module)
    with open(path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"[dump_ttir] module.verify() = True; wrote TTIR ({len(mlir)} chars) to {path}")
    return mlir


def dump_tileir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native_matmul_gpu_tileir.mlir")
    module = _compile_frontend_module(M=M, N=N, K=K)
    mlir = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor"):
        if needle not in mlir:
            raise RuntimeError(f"dump_tileir: expected {needle} in frontend IR")
    with open(path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"[dump_tileir] module.verify() = True; wrote TileIR ({len(mlir)} chars) to {path}")
    return mlir


def dump_ttgir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native_matmul_gpu_ttgir.mlir")
    device = "cuda"
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)
    mat_c = torch.empty((M, N), dtype=torch.float16, device=device)
    compiled = matmul_kernel.warmup(
        mat_a, mat_b, mat_c, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3, grid=_grid(M, N))
    mlir = compiled.asm.get("ttgir")
    if not mlir:
        raise RuntimeError("dump_ttgir: compiled kernel does not contain ttgir asm")
    with open(path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"[dump_ttgir] wrote TTGIR ({len(mlir)} chars) to {path}")
    print(f"[dump_ttgir] local_alloc={mlir.count('triton_gpu.local_alloc')} "
          f"async_copy={mlir.count('triton_gpu.async_copy_global_to_local')} "
          f"local_load={mlir.count('triton_gpu.local_load')}")
    return mlir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matmul kernel using tle.gpu alloc/copy/to_tensor")
    parser.add_argument("--M", type=int, default=_DEFAULT_M)
    parser.add_argument("--N", type=int, default=_DEFAULT_N)
    parser.add_argument("--K", type=int, default=_DEFAULT_K)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--dump-ttir", nargs="?", const="", default=None)
    parser.add_argument("--dump-tileir", nargs="?", const="", default=None)
    parser.add_argument("--dump-ttgir", nargs="?", const="", default=None)
    args = parser.parse_args()

    if args.dump_ttir is not None:
        dump_ttir(path=(args.dump_ttir or None), M=args.M, N=args.N, K=args.K)
        raise SystemExit(0)
    if args.dump_tileir is not None:
        dump_tileir(path=(args.dump_tileir or None), M=args.M, N=args.N, K=args.K)
        raise SystemExit(0)
    if args.dump_ttgir is not None:
        dump_ttgir(path=(args.dump_ttgir or None), M=args.M, N=args.N, K=args.K)
        raise SystemExit(0)

    torch.manual_seed(0)
    mat_a = torch.randn((args.M, args.K), dtype=torch.float16, device="cuda")
    mat_b = torch.randn((args.K, args.N), dtype=torch.float16, device="cuda")
    mat_c = call(mat_a, mat_b)

    if not args.no_check:
        ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
        torch.testing.assert_close(ref, mat_c, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
