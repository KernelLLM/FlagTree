import argparse
import os

import triton
import triton.language as tl

import triton.experimental.tle as tle


BLOCK = 32


@triton.jit
def gpu_primitives_kernel(src, M: tl.constexpr, BLOCK: tl.constexpr):
    whole = tle.gpu.alloc([2, BLOCK], dtype=src.dtype.element_ty, mem_addr_space=tle.gpu.shared)
    slot0 = tle.gpu.subview(whole, [0, 0], [1, BLOCK], [1, 1])
    tmp = tle.gpu.alloc([1, BLOCK], dtype=src.dtype.element_ty, mem_addr_space=tle.gpu.local)

    src_tile = tl.make_block_ptr(
        src, (1, M), (M, 1),
        (0, 0),
        (1, BLOCK), (1, 0))
    tle.gpu.copy(src_tile, slot0, [1, BLOCK])
    value = tle.gpu.to_tensor(slot0, writable=False)
    tle.gpu.store_tensor(value, tmp)


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


def dump_tileir(path=None, M=BLOCK):
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir
    from triton._C.libtriton import tle as tle_ir

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tle_gpu_primitives_tileir.mlir")

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    src = ASTSource(
        gpu_primitives_kernel.fn,
        {"src": "*fp16"},
        {"M": M, "BLOCK": BLOCK},
    )
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    module = ast_to_ttir(
        gpu_primitives_kernel,
        src,
        context,
        _DumpOptions(),
        {"min_dot_size": lambda lhsType, rhsType: (16, 16, 16)},
        {},
    )
    if not module.verify():
        raise RuntimeError("dump_tileir: module.verify() failed")
    mlir = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.subview", "tile.to_tensor", "tile.store_tensor"):
        if needle not in mlir:
            raise RuntimeError(f"dump_tileir: expected {needle} in frontend IR")
    with open(path, "w", encoding="utf-8") as f:
        f.write(mlir)
    print(f"[dump_tileir] module.verify() = True; wrote TileIR ({len(mlir)} chars) to {path}")
    return mlir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontend dump for tle.gpu primitive coverage")
    parser.add_argument("--M", type=int, default=BLOCK)
    parser.add_argument("--dump-tileir", nargs="?", const="", default=None)
    args = parser.parse_args()

    dump_tileir(path=(args.dump_tileir or None), M=args.M)
