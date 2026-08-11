//===- TensorViewDialect.cpp - TensorView dialect registration ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::tv;

void TensorViewDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ascend/include/Dialect/TensorView/IR/TensorViewTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ascend/include/Dialect/TensorView/IR/TensorViewAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "ascend/include/Dialect/TensorView/IR/TensorViewOps.cpp.inc"
      >();
}

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.cpp.inc"
