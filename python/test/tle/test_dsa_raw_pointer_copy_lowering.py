"""Regression tests for TLE DSA copies using raw Triton pointer tensors.

Expressions such as ``base + tl.arange(...)`` have the type
``tensor<...x!tt.ptr<T>>``.  The public DSA examples use this form for both
GM-to-buffer and buffer-to-GM copies, so the tests intentionally continue
beyond frontend TileIR generation through the Ascend Linalg pipeline.
"""

import pytest
import triton
import triton.experimental.tle as tle
import triton.language as tl

from triton._C.libtriton import ascend, ir, passes
from triton._C.libtriton import tle as tle_ir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource


class Options:
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


@triton.jit
def raw_pointer_roundtrip(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    buf = tle.dsa.alloc(
        [BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    tle.dsa.copy(x + offsets, buf, [BLOCK])
    values = tle.dsa.to_tensor(buf, writable=False) + 1.0
    out_buf = tle.dsa.to_buffer(values, tle.dsa.ascend.UB)
    tle.dsa.copy(out_buf, out + offsets, [BLOCK])


@triton.jit
def raw_pointer_loop(x, out, BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    buf = tle.dsa.alloc(
        [BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    acc = tl.zeros([], dtype=tl.float32)
    for i in range(NUM_BLOCKS):
        tle.dsa.copy(x + i * BLOCK + lane, buf, [BLOCK])
        acc += tl.sum(tle.dsa.to_tensor(buf, writable=False), axis=0)
    tl.store(out, acc)


@triton.jit
def raw_pointer_pipeline(x, out, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    buf = tle.dsa.alloc(
        [BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    acc = tl.zeros([], dtype=tl.float32)
    for i in tle.dsa.pipeline(0, N, BLOCK, num_stages=2):
        tle.dsa.copy(x + i + lane, buf, [BLOCK])
        acc += tl.sum(tle.dsa.to_tensor(buf, writable=False), axis=0)
    tl.store(out, acc)


@triton.jit
def mismatched_raw_pointer_copy(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    buf = tle.dsa.alloc(
        [BLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    # x is bf16 while buf is f32.  DSA copy is not a numeric cast.
    tle.dsa.copy(x + offsets, buf, [BLOCK])
    tl.store(out, tl.sum(tle.dsa.to_tensor(buf, writable=False), axis=0))


CASES = (
    (
        raw_pointer_roundtrip,
        {"x": "*fp32", "out": "*fp32"},
        {"BLOCK": 128},
    ),
    (
        raw_pointer_loop,
        {"x": "*fp32", "out": "*fp32"},
        {"BLOCK": 128, "NUM_BLOCKS": 4},
    ),
    (
        raw_pointer_pipeline,
        {"x": "*fp32", "out": "*fp32"},
        {"N": 512, "BLOCK": 128},
    ),
)


def compile_kernel(kernel, signature, constants):
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    from triton._C.libtriton.ascend import ir as ascend_ir

    ascend_ir.load_dialects(context)
    source = ASTSource(kernel.fn, signature, constants)
    module = ast_to_ttir(
        kernel,
        source,
        context,
        Options(),
        {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)},
        {},
    )
    assert module.verify()
    return module


def run_pass(module, configure):
    manager = ir.pass_manager(module.context)
    configure(manager)
    manager.run(module)
    assert module.verify()


def lower_to_linalg(module):
    run_pass(
        module,
        lambda pm: (
            passes.common.add_inliner(pm),
            ascend.passes.ttir.add_tileir_to_hivm(pm),
        ),
    )
    run_pass(
        module,
        lambda pm: (
            ascend.passes.ttir.add_erase_linalg_casts(pm),
            passes.common.add_canonicalizer(pm),
        ),
    )
    run_pass(
        module,
        lambda pm: (
            ascend.passes.ttir.add_triton_to_structure_incubated(
                pm, False, False, False
            ),
            ascend.passes.ttir.add_discrete_mask_access_conversion(
                pm, False, False
            ),
        ),
    )
    run_pass(
        module,
        lambda pm: (
            ascend.passes.ttir.add_triton_to_unstructure_incubated(
                pm, False, False
            ),
            ascend.passes.ttir.add_triton_to_hivm(pm),
            ascend.passes.ttir.add_triton_to_hfusion(pm),
            ascend.passes.ttir.add_triton_to_llvm(pm),
        ),
    )
    run_pass(
        module,
        lambda pm: (
            ascend.passes.ttir.add_bubble_up_operation(pm),
            ascend.passes.ttir.add_triton_to_structure_incubated(
                pm, False, False, False
            ),
        ),
    )
    run_pass(
        module,
        lambda pm: (
            passes.common.add_inliner(pm),
            passes.common.add_canonicalizer(pm),
        ),
    )
    run_pass(
        module,
        lambda pm: ascend.passes.ttir.add_triton_to_linalg_incubated(
            pm, False, True, False, False, False
        ),
    )
    run_pass(
        module, lambda pm: ascend.passes.ttir.add_fold_staging_copy(pm)
    )
    run_pass(
        module, lambda pm: ascend.passes.ttir.add_erase_linalg_casts(pm)
    )
    run_pass(
        module,
        lambda pm: (
            passes.common.add_canonicalizer(pm),
            passes.common.add_cse(pm),
            passes.common.add_symbol_dce(pm),
        ),
    )


@pytest.mark.parametrize("kernel,signature,constants", CASES)
def test_raw_pointer_copy_full_lowering(kernel, signature, constants):
    module = compile_kernel(kernel, signature, constants)
    frontend_ir = str(module)
    assert "tile.copy" in frontend_ir
    assert "!tt.ptr<f32>" in frontend_ir

    lower_to_linalg(module)
    lowered_ir = str(module)
    assert "tile.copy" not in lowered_ir
    assert "builtin.unrealized_conversion_cast" not in lowered_ir


def test_raw_pointer_copy_becomes_load_and_store():
    kernel, signature, constants = CASES[0]
    module = compile_kernel(kernel, signature, constants)
    run_pass(
        module,
        lambda pm: (
            passes.common.add_inliner(pm),
            ascend.passes.ttir.add_tileir_to_hivm(pm),
        ),
    )
    lowered_ir = str(module)
    assert "tile.copy" not in lowered_ir
    assert "tt.load" in lowered_ir
    assert "tt.store" in lowered_ir


def test_mismatched_copy_element_types_are_rejected():
    module = compile_kernel(
        mismatched_raw_pointer_copy,
        {"x": "*bf16", "out": "*fp32"},
        {"BLOCK": 128},
    )
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        run_pass(
            module,
            lambda pm: (
                passes.common.add_inliner(pm),
                ascend.passes.ttir.add_tileir_to_hivm(pm),
            ),
        )
