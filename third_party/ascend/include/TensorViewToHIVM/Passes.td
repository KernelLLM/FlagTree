//===-- Passes.td - TensorViewToHIVM pass definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_TO_HIVM_CONVERSION_PASSES
#define TENSORVIEW_TO_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TensorViewToHIVM : Pass<"tensor-view-to-hivm", "mlir::ModuleOp"> {
  let summary = "Lower TensorView access ops to memref + HIVM (Route A)";
  let description = [{
    Lowers the TensorView access ops to memref + HIVM DMA:
      - `!tv.ptr<T>` function inputs -> `memref<?xT, #hivm.address_space<gm>>`
      - `tv.view_load`  -> reinterpret_cast(GM tile) + alloc(UB) + hivm.hir.load
                           + bufferization.to_tensor (boundary bridge)
      - `tv.view_store` -> bufferization.to_buffer(UB) + reinterpret_cast(GM) + hivm.hir.store
    The compute (arith on tensor) and the boundary to_tensor/to_buffer bridges
    are left for the downstream linalg bufferization to fold.
  }];
  let constructor = "triton::createTensorViewToHIVMPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "mlir::triton::tv::TensorViewDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::triton::TritonDialect"];
}

#endif // TENSORVIEW_TO_HIVM_CONVERSION_PASSES
