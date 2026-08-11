//===- TritonToTensorView.cpp - tt -> tv conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass A of the TensorView flow.  It:
//   1. Replaces `!tt.ptr<T>` function inputs with `!tv.ptr<T>`.
//   2. Lowers contiguous `tt.load` / `tt.store` clusters
//        %s  = tt.splat  %base            : !tt.ptr -> tensor<Nx!tt.ptr>
//        %p  = tt.addptr %s, %offset       (offset = splat(pid*BLOCK) + arange)
//        %v  = tt.load   %p [, %mask, %other]
//      into
//        %tv  = tv.make_tensor_view %base, sizes=[%N], strides=[1]
//        %pv  = tv.make_partition_view %tv  (#tv.partition_view<tile=[BLOCK]>)
//        %v   = tv.view_load %pv[%pid]
//   The compute side (arith / tt.dot) is untouched.
//
// Scope (Stage 2): 1-D contiguous partition accesses whose base pointer is a
// function argument.  Strided / gather / mask lowering come later.
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonToTensorView/Passes.h"

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOTENSORVIEW
#include "ascend/include/TritonToTensorView/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// A recognized 1-D contiguous partition access.
struct ContigAccess {
  Value basePtr;                 // scalar !tv.ptr<T>
  Type elementType;              // T
  int64_t tileSize = 0;          // BLOCK
  Value tileIndex;               // i32 tile index (e.g. program_id)
  Value fullSize;                // i32 full extent (from the mask compare)
  triton::AddPtrOp addptr;       // op to erase afterwards
  triton::SplatOp baseSplat;     // op to erase afterwards
};

static std::optional<int64_t> getConstIntValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// Match `tt.load`/`tt.store` pointer operand of the 1-D contiguous form.
static bool matchContiguous1D(Value ptrTensor, Value maskVal, ContigAccess &out) {
  auto addptr = ptrTensor.getDefiningOp<triton::AddPtrOp>();
  if (!addptr)
    return false;
  auto splat = addptr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return false;
  // The base pointer must already have been rewritten to !tv.ptr (i.e. it is a
  // function argument handled by rewriteFuncPtrArgs).
  auto tvPtr = dyn_cast<tv::PtrType>(splat.getSrc().getType());
  if (!tvPtr)
    return false;
  out.basePtr = splat.getSrc();
  out.elementType = tvPtr.getPointeeType();
  out.addptr = addptr;
  out.baseSplat = splat;

  // offset = addi(splat(pid*BLOCK), arange) in either operand order.
  auto addi = addptr.getOffset().getDefiningOp<arith::AddIOp>();
  if (!addi)
    return false;
  triton::MakeRangeOp range;
  triton::SplatOp startSplat;
  for (Value operand : {addi.getLhs(), addi.getRhs()}) {
    if (auto m = operand.getDefiningOp<triton::MakeRangeOp>())
      range = m;
    else if (auto s = operand.getDefiningOp<triton::SplatOp>())
      startSplat = s;
  }
  if (!range || !startSplat || range.getStart() != 0)
    return false;
  out.tileSize = static_cast<int64_t>(range.getEnd()) -
                 static_cast<int64_t>(range.getStart());

  // block_start = muli(tileIndex, BLOCK)
  auto muli = startSplat.getSrc().getDefiningOp<arith::MulIOp>();
  if (!muli)
    return false;
  if (auto c = getConstIntValue(muli.getLhs()); c && *c == out.tileSize)
    out.tileIndex = muli.getRhs();
  else if (auto c = getConstIntValue(muli.getRhs()); c && *c == out.tileSize)
    out.tileIndex = muli.getLhs();
  else
    return false;

  // full extent from the mask compare: cmpi(offset, splat(N)).
  if (maskVal) {
    if (auto cmp = maskVal.getDefiningOp<arith::CmpIOp>()) {
      for (Value operand : {cmp.getLhs(), cmp.getRhs()})
        if (auto s = operand.getDefiningOp<triton::SplatOp>())
          out.fullSize = s.getSrc();
    }
  }
  // A full extent is required to build the base view size operand.
  return static_cast<bool>(out.fullSize);
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Build make_tensor_view + make_partition_view; returns the partition view.
static Value buildPartitionView(OpBuilder &b, Location loc,
                                const ContigAccess &a) {
  MLIRContext *ctx = b.getContext();
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nIdx =
      b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.fullSize);

  SmallVector<int64_t> dynShape{ShapedType::kDynamic};
  SmallVector<int64_t> unitStride{1};
  auto baseTy = tv::TensorViewType::get(dynShape, a.elementType, unitStride,
                                        /*encoding=*/Attribute());
  auto baseView = b.create<tv::MakeTensorViewOp>(
      loc, baseTy, a.basePtr, ValueRange{nIdx}, ValueRange{c1});

  auto enc = tv::PartitionViewAttr::get(ctx, ArrayRef<int64_t>{a.tileSize});
  auto partTy =
      tv::TensorViewType::get(dynShape, a.elementType, unitStride, enc);
  auto partView =
      b.create<tv::MakePartitionViewOp>(loc, partTy, baseView.getResult());
  return partView.getResult();
}

/// Erase the (now dead) addptr + splat feeding a converted access.  Both ops
/// are always set when a match succeeded; guard only on remaining uses.
/// Taken by value (non-const op wrappers) so getOperation()/erase() are usable.
static void eraseDeadPtrChain(triton::AddPtrOp addptr, triton::SplatOp baseSplat) {
  if (addptr->use_empty()) {
    addptr->erase();
    if (baseSplat->use_empty())
      baseSplat->erase();
  }
}

static LogicalResult rewriteLoad(triton::LoadOp load) {
  ContigAccess a;
  if (!matchContiguous1D(load.getPtr(), load.getMask(), a))
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  Value partView = buildPartitionView(b, loc, a);
  Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.tileIndex);
  auto viewLoad = b.create<tv::ViewLoadOp>(loc, load.getResult().getType(),
                                           partView, ValueRange{idx},
                                           /*mask=*/Value());
  load.getResult().replaceAllUsesWith(viewLoad.getResult());
  load.erase();
  eraseDeadPtrChain(a.addptr, a.baseSplat);
  return success();
}

static LogicalResult rewriteStore(triton::StoreOp store) {
  ContigAccess a;
  if (!matchContiguous1D(store.getPtr(), store.getMask(), a))
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();
  Value partView = buildPartitionView(b, loc, a);
  Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.tileIndex);
  b.create<tv::ViewStoreOp>(loc, partView, store.getValue(), ValueRange{idx},
                            /*mask=*/Value());
  store.erase();
  eraseDeadPtrChain(a.addptr, a.baseSplat);
  return success();
}

//===----------------------------------------------------------------------===//
// Function-argument rewrite: !tt.ptr<T> -> !tv.ptr<T>
//===----------------------------------------------------------------------===//

static void rewriteFuncPtrArgs(triton::FuncOp func) {
  auto funcTy = func.getFunctionType();
  bool changed = false;
  SmallVector<Type> inputs;
  for (Type t : funcTy.getInputs()) {
    if (auto p = dyn_cast<triton::PointerType>(t)) {
      inputs.push_back(tv::PtrType::get(p.getPointeeType()));
      changed = true;
    } else {
      inputs.push_back(t);
    }
  }
  if (!changed)
    return;

  func.setFunctionType(
      FunctionType::get(func.getContext(), inputs, funcTy.getResults()));
  if (!func.empty()) {
    for (BlockArgument arg : func.front().getArguments())
      if (auto p = dyn_cast<triton::PointerType>(arg.getType()))
        arg.setType(tv::PtrType::get(p.getPointeeType()));
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TritonToTensorViewPass
    : public mlir::triton::impl::TritonToTensorViewBase<
          TritonToTensorViewPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 1. Rewrite pointer function arguments to !tv.ptr.
    module.walk([](triton::FuncOp func) { rewriteFuncPtrArgs(func); });

    // 2. Collect then rewrite accesses (avoid mutating during the walk).
    SmallVector<triton::LoadOp> loads;
    SmallVector<triton::StoreOp> stores;
    module.walk([&](Operation *op) {
      if (auto l = dyn_cast<triton::LoadOp>(op))
        loads.push_back(l);
      else if (auto s = dyn_cast<triton::StoreOp>(op))
        stores.push_back(s);
    });

    for (triton::LoadOp l : loads) {
      if (failed(rewriteLoad(l)))
        l.emitError("TritonToTensorView: unsupported tt.load access pattern");
    }
    for (triton::StoreOp s : stores) {
      if (failed(rewriteStore(s)))
        s.emitError("TritonToTensorView: unsupported tt.store access pattern");
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToTensorViewPass() {
  return std::make_unique<TritonToTensorViewPass>();
}
