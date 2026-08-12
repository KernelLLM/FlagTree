//===-- Passes.td - TritonToTensorView pass definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_TENSORVIEW_CONVERSION_PASSES
#define TRITON_TO_TENSORVIEW_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToTensorView : Pass<"triton-to-tensor-view", "mlir::ModuleOp"> {
  let summary = "Convert Triton pointer accesses to the TensorView (tv) dialect";
  let description = [{
    Replaces `!tt.ptr<T>` function inputs with `!tv.ptr<T>` and lowers the
    supported `tt.load`/`tt.store` access clusters (regular tiled or the 2-D
    single-sparse-dimension form) into `tv.make_tensor_view` plus an encoded
    view and `tv.view_load`/`tv.view_store`.
    The compute side (arith / tt.dot) is left untouched and keeps flowing as
    tensors.
  }];
  let constructor = "triton::createTritonToTensorViewPass()";
  let dependentDialects = [
    "mlir::triton::tv::TensorViewDialect",
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect"];
}

#endif // TRITON_TO_TENSORVIEW_CONVERSION_PASSES
