//===- TensorViewToHIVM.cpp - tv access ops -> memref + HIVM ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass B of the TensorView flow (Route A, late placement).  Lowers the tv
// access ops to memref + HIVM DMA, bridging to/from tensor at the boundary and
// leaving the compute (arith on tensor) + bridges for the downstream linalg
// bufferization to fold.
//
//   !tv.ptr<T> func input        -> memref<?xT, #hivm.address_space<gm>>
//   tv.view_load %pv[%i]         -> %off = %i * TILE
//                                   %gm  = reinterpret_cast %base off:[%off] sizes:[TILE] strides:[S]
//                                   %ub  = alloc() : memref<TILExT, #ub>
//                                   hivm.hir.load ins(%gm) outs(%ub)
//                                   %t   = bufferization.to_tensor %ub    (bridge)
//   tv.view_store %pv[%i], %v    -> %vm  = bufferization.to_buffer %v : memref<TILExT,#ub>  (bridge)
//                                   %gm  = reinterpret_cast %base_out ...
//                                   hivm.hir.store ins(%vm) outs(%gm)
//
// Scope (Stage 3): 1-D partition accesses (matching Pass A's output).
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TensorViewToHIVM/Passes.h"

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TENSORVIEWTOHIVM
#include "ascend/include/TensorViewToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Info recovered from a partition view chain: base memref + tile + stride.
struct ViewInfo {
  Value base;        // memref<?xT, #gm> (the rewritten function argument)
  Type elementType;  // T
  int64_t tile = 0;  // TILE (1-D)
  int64_t stride = 0;
  tv::MakePartitionViewOp partitionOp;
  tv::MakeTensorViewOp baseOp;
  bool ok = false;
};

/// Trace `viewVal` (a partition view SSA) back to its base memref + params.
static ViewInfo traceView(Value viewVal) {
  ViewInfo vi;
  auto pv = viewVal.getDefiningOp<tv::MakePartitionViewOp>();
  if (!pv)
    return vi;
  auto partTy = dyn_cast<tv::TensorViewType>(pv.getResult().getType());
  if (!partTy)
    return vi;
  auto enc = dyn_cast_or_null<tv::PartitionViewAttr>(partTy.getEncoding());
  if (!enc || enc.getTile().size() != 1)
    return vi;
  auto mtv = pv.getSource().getDefiningOp<tv::MakeTensorViewOp>();
  if (!mtv)
    return vi;
  auto baseTy = dyn_cast<tv::TensorViewType>(mtv.getResult().getType());
  if (!baseTy || baseTy.getStrides().size() != 1)
    return vi;
  // NOTE: read the source operand untyped. rewriteFuncPtrArgs has already
  // changed the underlying function argument to a memref, so the typed
  // accessor mtv.getSource() (which casts to TypedValue<tv::PtrType>) would
  // assert. getOperand(0) is the source operand.
  Value base = mtv->getOperand(0);
  if (!isa<MemRefType>(base.getType()))
    return vi; // base must already be a memref (function argument rewritten)

  vi.base = base;
  vi.elementType = baseTy.getElementType();
  vi.tile = enc.getTile()[0];
  vi.stride = baseTy.getStrides()[0];
  vi.partitionOp = pv;
  vi.baseOp = mtv;
  vi.ok = true;
  return vi;
}

/// Emit the GM tile view: reinterpret_cast %base to offset:[%i*TILE] sizes:[TILE] strides:[S].
static Value emitGmTile(OpBuilder &b, Location loc, const ViewInfo &vi,
                        Value index, hivm::AddressSpaceAttr gmSpace) {
  MLIRContext *ctx = b.getContext();
  Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile);
  Value off = b.create<arith::MulIOp>(loc, index, cTile);
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                       /*strides=*/{vi.stride});
  auto gmTileTy = MemRefType::get({vi.tile}, vi.elementType, layout, gmSpace);
  return b.create<memref::ReinterpretCastOp>(
      loc, gmTileTy, vi.base, /*offset=*/OpFoldResult(off),
      /*sizes=*/ArrayRef<OpFoldResult>{b.getIndexAttr(vi.tile)},
      /*strides=*/ArrayRef<OpFoldResult>{b.getIndexAttr(vi.stride)});
}

//===----------------------------------------------------------------------===//
// Lowering
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load,
                                   hivm::AddressSpaceAttr gmSpace,
                                   hivm::AddressSpaceAttr ubSpace) {
  ViewInfo vi = traceView(load.getView());
  if (!vi.ok || load.getIndices().size() != 1)
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  Value gm = emitGmTile(b, loc, vi, load.getIndices()[0], gmSpace);
  auto ubTy = MemRefType::get({vi.tile}, vi.elementType,
                              MemRefLayoutAttrInterface{}, ubSpace);
  Value ub = b.create<memref::AllocOp>(loc, ubTy);
  b.create<hivm::LoadOp>(loc, TypeRange{}, gm, ub);

  auto tensorTy = cast<RankedTensorType>(load.getResult().getType());
  Value t = b.create<bufferization::ToTensorOp>(loc, tensorTy, ub,
                                                /*restrict=*/true,
                                                /*writable=*/false);
  load.getResult().replaceAllUsesWith(t);
  load.erase();
  return success();
}

static LogicalResult lowerViewStore(tv::ViewStoreOp store,
                                    hivm::AddressSpaceAttr gmSpace,
                                    hivm::AddressSpaceAttr ubSpace) {
  ViewInfo vi = traceView(store.getView());
  if (!vi.ok || store.getIndices().size() != 1)
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();
  auto ubTy = MemRefType::get({vi.tile}, vi.elementType,
                              MemRefLayoutAttrInterface{}, ubSpace);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value vm = b.create<bufferization::ToMemrefOp>(loc, ubTy, store.getValue());
#else
  Value vm = b.create<bufferization::ToBufferOp>(loc, ubTy, store.getValue());
#endif
  Value gm = emitGmTile(b, loc, vi, store.getIndices()[0], gmSpace);
  b.create<hivm::StoreOp>(loc, TypeRange{}, vm, gm);
  store.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Function-argument rewrite: !tv.ptr<T> -> memref<?xT, #gm>
//===----------------------------------------------------------------------===//

static void rewriteFuncPtrArgs(triton::FuncOp func,
                               hivm::AddressSpaceAttr gmSpace) {
  auto ptrToMemref = [&](Type t) -> Type {
    if (auto p = dyn_cast<tv::PtrType>(t))
      return MemRefType::get({ShapedType::kDynamic}, p.getPointeeType(),
                             MemRefLayoutAttrInterface{}, gmSpace);
    return t;
  };

  auto funcTy = func.getFunctionType();
  bool changed = false;
  SmallVector<Type> inputs;
  for (Type t : funcTy.getInputs()) {
    Type nt = ptrToMemref(t);
    changed |= (nt != t);
    inputs.push_back(nt);
  }
  if (!changed)
    return;

  func.setFunctionType(
      FunctionType::get(func.getContext(), inputs, funcTy.getResults()));
  if (!func.empty())
    for (BlockArgument arg : func.front().getArguments())
      if (isa<tv::PtrType>(arg.getType()))
        arg.setType(ptrToMemref(arg.getType()));
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TensorViewToHIVMPass
    : public mlir::triton::impl::TensorViewToHIVMBase<TensorViewToHIVMPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto gmSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);
    auto ubSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);

    // 1. Rewrite !tv.ptr function inputs to GM memrefs.
    module.walk([&](triton::FuncOp func) { rewriteFuncPtrArgs(func, gmSpace); });

    // 2. Lower access ops (collect first to avoid mutation during walk).
    SmallVector<tv::ViewLoadOp> loads;
    SmallVector<tv::ViewStoreOp> stores;
    module.walk([&](Operation *op) {
      if (auto l = dyn_cast<tv::ViewLoadOp>(op))
        loads.push_back(l);
      else if (auto s = dyn_cast<tv::ViewStoreOp>(op))
        stores.push_back(s);
    });
    for (tv::ViewLoadOp l : loads)
      if (failed(lowerViewLoad(l, gmSpace, ubSpace)))
        l.emitError("TensorViewToHIVM: unsupported view_load");
    for (tv::ViewStoreOp s : stores)
      if (failed(lowerViewStore(s, gmSpace, ubSpace)))
        s.emitError("TensorViewToHIVM: unsupported view_store");

    // 3. Erase the now-dead make_partition_view / make_tensor_view ops.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      module.walk([&](Operation *op) {
        if ((isa<tv::MakePartitionViewOp>(op) ||
             isa<tv::MakeStridedViewOp>(op) ||
             isa<tv::MakeGatherScatterViewOp>(op) ||
             isa<tv::MakeTensorViewOp>(op)) &&
            op->use_empty())
          dead.push_back(op);
      });
      for (Operation *op : dead) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTensorViewToHIVMPass() {
  return std::make_unique<TensorViewToHIVMPass>();
}
