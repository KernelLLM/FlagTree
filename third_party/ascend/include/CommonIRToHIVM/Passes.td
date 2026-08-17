/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TILE_IR_TO_HIVM_CONVERSION_PASSES
#define TILE_IR_TO_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CommonIRToHIVM : Pass<"commonir-to-hivm", "mlir::ModuleOp"> {
  let summary = "Convert CommonIR dialect to HIVM dialect";
  let constructor = "triton::createCommonIRToHIVMPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "triton::tile::CommonIRDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::triton::TritonDialect",
  ];
}

#endif // TILE_IR_TO_HIVM_CONVERSION_PASSES
