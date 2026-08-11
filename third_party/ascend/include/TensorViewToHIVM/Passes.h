//===- Passes.h - TensorViewToHIVM passes -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_TENSORVIEW_TO_HIVM_CONVERSION_PASSES_H
#define TRITON_ADAPTER_TENSORVIEW_TO_HIVM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass to lower TensorView access ops to memref + HIVM.
std::unique_ptr<OperationPass<ModuleOp>> createTensorViewToHIVMPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/TensorViewToHIVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_TENSORVIEW_TO_HIVM_CONVERSION_PASSES_H
