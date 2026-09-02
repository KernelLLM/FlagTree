import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _flash_gpu_kernel(q_ptr, k_ptr, v_ptr, o_ptr, stride_qm, stride_qd, stride_km, stride_kd, stride_vm, stride_vd,
                      stride_om, stride_od, N_CTX: tl.constexpr, D: tl.constexpr, sm_scale: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_buf = tle.gpu.alloc([BLOCK_M, D], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    k_buf = tle.gpu.alloc([BLOCK_N, D], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    v_buf = tle.gpu.alloc([BLOCK_N, D], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)

    q_tile = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    tle.gpu.copy(q_tile, q_buf, [BLOCK_M, D])
    q = tle.gpu.to_tensor(q_buf, writable=False)

    m_i = tl.full((BLOCK_M, ), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M, ), tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k_tile = k_ptr + (start_n + offs_n)[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_tile = v_ptr + (start_n + offs_n)[:, None] * stride_vm + offs_d[None, :] * stride_vd
        tle.gpu.copy(k_tile, k_buf, [BLOCK_N, D])
        tle.gpu.copy(v_tile, v_buf, [BLOCK_N, D])
        k = tle.gpu.to_tensor(k_buf, writable=False)
        v = tle.gpu.to_tensor(v_buf, writable=False)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    out_buf = tle.gpu.alloc([BLOCK_M, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tle.gpu.store_tensor(out, out_buf)
    out_vals = tle.gpu.to_tensor(out_buf, writable=False)
    o_tile = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_tile, out_vals)


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


def _frontend_module(kernel, signature, constants):
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    target, backend, options = _compile_options()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(kernel, signature, constants)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    if not module.verify():
        raise RuntimeError("frontend module.verify() failed")
    return module


def dump_tileir(path):
    constants = {
        "N_CTX": 32,
        "D": 32,
        "sm_scale": 0.1767766952966369,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
    }
    signature = {
        "q_ptr": "*fp16",
        "k_ptr": "*fp16",
        "v_ptr": "*fp16",
        "o_ptr": "*fp32",
        "stride_qm": "i32",
        "stride_qd": "i32",
        "stride_km": "i32",
        "stride_kd": "i32",
        "stride_vm": "i32",
        "stride_vd": "i32",
        "stride_om": "i32",
        "stride_od": "i32",
    }
    module = _frontend_module(_flash_gpu_kernel, signature, constants)
    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor", "tile.store_tensor"):
        if needle not in text:
            raise RuntimeError(f"expected {needle} in TileIR dump")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-tileir] wrote {path}")


def dump_ttir(path):
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes
    from triton._C.libtriton import nvidia

    constants = {
        "N_CTX": 32,
        "D": 32,
        "sm_scale": 0.1767766952966369,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
    }
    signature = {
        "q_ptr": "*fp16",
        "k_ptr": "*fp16",
        "v_ptr": "*fp16",
        "o_ptr": "*fp32",
        "stride_qm": "i32",
        "stride_qd": "i32",
        "stride_km": "i32",
        "stride_kd": "i32",
        "stride_vm": "i32",
        "stride_vd": "i32",
        "stride_om": "i32",
        "stride_od": "i32",
    }
    module = _frontend_module(_flash_gpu_kernel, signature, constants)
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    pm.run(module, "native_flash_gpu.inliner")
    pm = ir.pass_manager(module.context)
    nvidia.passes.commonir.add_to_ttgir(pm)
    pm.run(module, "native_flash_gpu.tileir_to_ttgir")
    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor", "tile.store_tensor"):
        if needle in text:
            raise RuntimeError(f"converted TTIR dump still contains {needle}")
    for needle in ("ttg.local_alloc", "ttg.local_load", "ttg.local_store"):
        if needle not in text:
            raise RuntimeError(f"converted TTIR dump expected {needle}")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-ttir] wrote {path}")


def run_check(n_ctx=32, d=32):
    torch.manual_seed(0)
    q = torch.randn((n_ctx, d), device="cuda", dtype=torch.float16)
    k = torch.randn((n_ctx, d), device="cuda", dtype=torch.float16)
    v = torch.randn((n_ctx, d), device="cuda", dtype=torch.float16)
    out = torch.empty((n_ctx, d), device="cuda", dtype=torch.float32)
    sm_scale = d**-0.5
    grid = (triton.cdiv(n_ctx, 16), )
    _flash_gpu_kernel[grid](q, k, v, out, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                            out.stride(0), out.stride(1), N_CTX=n_ctx, D=d, sm_scale=sm_scale, BLOCK_M=16, BLOCK_N=16)
    ref = torch.softmax(torch.matmul(q.float(), k.float().T) * sm_scale, dim=-1).matmul(v.float())
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)
    print("[check] native_flash_gpu precision PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-tileir", nargs="?", const="native_flash_gpu_tileir.mlir")
    parser.add_argument("--dump-ttir", nargs="?", const="native_flash_gpu_ttir.mlir")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--D", type=int, default=32)
    args = parser.parse_args()
    if args.dump_tileir:
        dump_tileir(args.dump_tileir)
    if args.dump_ttir:
        dump_ttir(args.dump_ttir)
    if not args.no_check:
        run_check(args.N, args.D)


if __name__ == "__main__":
    main()
