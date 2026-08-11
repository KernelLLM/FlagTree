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

#include <limits>
#include <optional>

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

/// A recognized 1-D tiled access: contiguous partition (traversalStride == tile)
/// or overlapping/gapped strided view (traversalStride != tile).
struct ContigAccess {
  Value basePtr;                 // scalar !tv.ptr<T>
  Type elementType;              // T
  int64_t tileSize = 0;          // WINDOW (tile size)
  int64_t traversalStride = 0;   // STEP between tile origins (== tile for partition)
  Value tileIndex;               // i32 tile index (e.g. program_id)
  Value fullSize;                // i32 full extent (from the mask compare; may be null)
  triton::AddPtrOp addptr;       // outer (tensor) addptr, to erase afterwards
  triton::SplatOp baseSplat;     // splat feeding it, to erase afterwards
  triton::AddPtrOp scalarAddptr; // hoisted scalar addptr on the base (may be null)
};

static std::optional<int64_t> getConstIntValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// Match `tt.load`/`tt.store` pointer of a 1-D tiled access.  The per-element
/// pointer is `base + tile_origin + arange`, where the tile-origin scalar
/// (`pid * STEP`) may appear in two frontend forms:
///   (non-hoisted) addptr(splat(base),        addi(splat(pid*STEP), arange))
///   (hoisted)     addptr(splat(addptr(base, pid*STEP)), arange)
/// The hoisted form appears when the offset is not shared with a mask.
static bool matchContiguous1D(Value ptrTensor, Value maskVal, ContigAccess &out) {
  auto addptr = ptrTensor.getDefiningOp<triton::AddPtrOp>();
  if (!addptr)
    return false;
  auto splat = addptr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return false;
  out.addptr = addptr;
  out.baseSplat = splat;

  // The splatted scalar pointer is either the base or a hoisted
  // addptr(base, pid*STEP).
  Value scalarPtr = splat.getSrc();
  Value originScalar; // i32 = pid*STEP from the hoisted scalar addptr (or null)
  if (auto scalarAp = scalarPtr.getDefiningOp<triton::AddPtrOp>()) {
    out.scalarAddptr = scalarAp;
    originScalar = scalarAp.getOffset();
    scalarPtr = scalarAp.getPtr();
  }
  // The base pointer must already have been rewritten to !tv.ptr.
  auto tvPtr = dyn_cast<tv::PtrType>(scalarPtr.getType());
  if (!tvPtr)
    return false;
  out.basePtr = scalarPtr;
  out.elementType = tvPtr.getPointeeType();

  // Tensor offset: make_range, or addi(splat(pid*STEP), make_range).
  Value tensorOff = addptr.getOffset();
  triton::MakeRangeOp range;
  Value originFromOffset;
  if (auto mr = tensorOff.getDefiningOp<triton::MakeRangeOp>()) {
    range = mr;
  } else if (auto addi = tensorOff.getDefiningOp<arith::AddIOp>()) {
    for (Value operand : {addi.getLhs(), addi.getRhs()}) {
      if (auto m = operand.getDefiningOp<triton::MakeRangeOp>())
        range = m;
      else if (auto s = operand.getDefiningOp<triton::SplatOp>())
        originFromOffset = s.getSrc();
    }
  }
  if (!range || range.getStart() != 0)
    return false;
  out.tileSize = static_cast<int64_t>(range.getEnd()) -
                 static_cast<int64_t>(range.getStart());

  // tile_origin = muli(tileIndex, STEP): from the hoisted scalar addptr or the
  // splat inside the tensor offset.  STEP == tile => partition, else strided.
  Value origin = originScalar ? originScalar : originFromOffset;
  if (!origin)
    return false;
  auto muli = origin.getDefiningOp<arith::MulIOp>();
  if (!muli)
    return false;
  if (auto c = getConstIntValue(muli.getLhs())) {
    out.traversalStride = *c;
    out.tileIndex = muli.getRhs();
  } else if (auto c = getConstIntValue(muli.getRhs())) {
    out.traversalStride = *c;
    out.tileIndex = muli.getLhs();
  } else {
    return false;
  }

  // full extent from the mask compare: cmpi(offset, splat(N)).  Optional: an
  // unmasked access has no explicit bound (handled with a sentinel in buildView).
  if (maskVal) {
    if (auto cmp = maskVal.getDefiningOp<arith::CmpIOp>()) {
      for (Value operand : {cmp.getLhs(), cmp.getRhs()})
        if (auto s = operand.getDefiningOp<triton::SplatOp>())
          out.fullSize = s.getSrc();
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Build make_tensor_view + make_partition_view / make_strided_view; returns the
/// encoded view.  Partition when traversalStride == tile, strided otherwise.
static Value buildView(OpBuilder &b, Location loc, const ContigAccess &a) {
  MLIRContext *ctx = b.getContext();
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nIdx;
  if (a.fullSize) {
    nIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.fullSize);
  } else {
    // Unmasked access: no explicit bound.  Use a large sentinel so Pass B's tail
    // clamp folds to a full tile (the kernel guarantees in-bounds access).
    nIdx = b.create<arith::ConstantIndexOp>(
        loc, std::numeric_limits<int64_t>::max());
  }

  SmallVector<int64_t> dynShape{ShapedType::kDynamic};
  SmallVector<int64_t> unitStride{1};
  auto baseTy = tv::TensorViewType::get(dynShape, a.elementType, unitStride,
                                        /*encoding=*/Attribute());
  auto baseView = b.create<tv::MakeTensorViewOp>(
      loc, baseTy, a.basePtr, ValueRange{nIdx}, ValueRange{c1});

  bool isPartition = (a.traversalStride == a.tileSize);
  Attribute enc;
  if (isPartition)
    enc = tv::PartitionViewAttr::get(ctx, ArrayRef<int64_t>{a.tileSize});
  else
    enc = tv::StridedViewAttr::get(ctx, ArrayRef<int64_t>{a.tileSize},
                                   ArrayRef<int64_t>{a.traversalStride});
  auto viewTy = tv::TensorViewType::get(dynShape, a.elementType, unitStride, enc);
  if (isPartition)
    return b.create<tv::MakePartitionViewOp>(loc, viewTy, baseView.getResult())
        .getResult();
  return b.create<tv::MakeStridedViewOp>(loc, viewTy, baseView.getResult())
      .getResult();
}

/// Erase the (now dead) tensor addptr + splat (+ hoisted scalar addptr) feeding a
/// converted access.  Taken by value (non-const op wrappers) so
/// getOperation()/erase() are usable.  scalarAddptr may be null (non-hoisted).
static void eraseDeadPtrChain(triton::AddPtrOp addptr, triton::SplatOp baseSplat,
                              triton::AddPtrOp scalarAddptr) {
  if (addptr->use_empty()) {
    addptr->erase();
    if (baseSplat->use_empty()) {
      baseSplat->erase();
      if (scalarAddptr && scalarAddptr->use_empty())
        scalarAddptr->erase();
    }
  }
}

static LogicalResult rewriteLoad(triton::LoadOp load) {
  ContigAccess a;
  if (!matchContiguous1D(load.getPtr(), load.getMask(), a))
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  Value partView = buildView(b, loc, a);
  Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.tileIndex);
  auto viewLoad = b.create<tv::ViewLoadOp>(loc, load.getResult().getType(),
                                           partView, ValueRange{idx},
                                           /*mask=*/Value());
  load.getResult().replaceAllUsesWith(viewLoad.getResult());
  load.erase();
  eraseDeadPtrChain(a.addptr, a.baseSplat, a.scalarAddptr);
  return success();
}

static LogicalResult rewriteStore(triton::StoreOp store) {
  ContigAccess a;
  if (!matchContiguous1D(store.getPtr(), store.getMask(), a))
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();
  Value partView = buildView(b, loc, a);
  Value idx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.tileIndex);
  b.create<tv::ViewStoreOp>(loc, partView, store.getValue(), ValueRange{idx},
                            /*mask=*/Value());
  store.erase();
  eraseDeadPtrChain(a.addptr, a.baseSplat, a.scalarAddptr);
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
