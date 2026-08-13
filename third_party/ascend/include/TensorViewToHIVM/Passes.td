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
  let summary = "Lower TensorView access ops to generic community memref ops";
  let description = [{
    GENERIC lowering variant: lowers the TensorView access ops to *community*
    memref/bufferization/scf/tensor ops only (no HIVM dialect), mirroring the
    native tt->linalg access form so the backend does placement + DMA lowering:
      - `!tv.ptr<T>` function inputs -> plain `memref<?xT>`
      - `tv.view_load`  -> reinterpret_cast + alloc + memref.copy + to_tensor
      - `tv.view_store` -> to_buffer + reinterpret_cast + memref.copy
      - gather load  -> reinterpret window + memref.copy + scalar scf.for
                        (tensor.extract/insert)
      - scatter store -> scalar scf.for + materialize_in_destination
      - ptr_load/ptr_store -> scalar scf.for (memref.load / materialize)
  }];
  let constructor = "triton::createTensorViewToHIVMPass()";
  let dependentDialects = [
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
