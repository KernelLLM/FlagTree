import argparse
from pathlib import Path

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _gpu_primitives_kernel(src, out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    storage = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=2, scope="cta", name="p", payload=storage)
    writer = pipe.writer()
    reader = pipe.reader()

    write_slot = writer.acquire(0).payload
    tle.gpu.copy(src + idx, write_slot, [BLOCK])
    writer.commit(0)
    wait_result = reader.wait(0)
    read_slot = wait_result.slot.payload
    value = tl.load(tle.gpu.local_ptr(read_slot))
    reader.release(0)
    writer.close(0)
    tl.store(out + idx, value)


def _compile_options():
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options({
        "num_warps": 4,
        "num_stages": 3,
    })
    return target, backend, options


def _frontend_module():
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    target, backend, options = _compile_options()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(_gpu_primitives_kernel, {
        "src": "*fp32",
        "out": "*fp32",
    }, {"BLOCK": 16})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    if not module.verify():
        raise RuntimeError("frontend module.verify() failed")
    return module


def dump_tileir(path):
    module = _frontend_module()
    text = str(module)
    for needle in (
            "tile.alloc",
            "tile.copy",
            "tile.subview",
            "builtin.unrealized_conversion_cast",
            "tle.local_pointers",
            "tle.pipe.create",
            "tle.pipe.writer_acquire",
            "tle.pipe.reader_wait",
    ):
        if needle not in text:
            raise RuntimeError(f"expected {needle} in primitive TileIR dump")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-tileir] wrote {path}")


def dump_ttir(path):
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes
    from triton._C.libtriton import nvidia

    module = _frontend_module()
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    pm.run(module, "tle_gpu_primitives.inliner")
    pm = ir.pass_manager(module.context)
    nvidia.passes.commonir.add_to_ttgir(pm)
    pm.run(module, "tle_gpu_primitives.tileir_to_ttgir")
    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.subview", "builtin.unrealized_conversion_cast"):
        if needle in text:
            raise RuntimeError(f"primitive converted TTIR still contains {needle}")
    for needle in ("ttg.local_alloc", "ttg.memdesc_index", "tle.local_pointers", "tle.pipe.create"):
        if needle not in text:
            raise RuntimeError(f"primitive converted TTIR expected {needle}")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-ttir] wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-tileir", nargs="?", const="tle_gpu_primitives_tileir.mlir")
    parser.add_argument("--dump-ttir", nargs="?", const="tle_gpu_primitives_ttir.mlir")
    args = parser.parse_args()
    if args.dump_tileir:
        dump_tileir(args.dump_tileir)
    if args.dump_ttir:
        dump_ttir(args.dump_ttir)
    if not args.dump_tileir and not args.dump_ttir:
        dump_tileir("tle_gpu_primitives_tileir.mlir")


if __name__ == "__main__":
    main()
