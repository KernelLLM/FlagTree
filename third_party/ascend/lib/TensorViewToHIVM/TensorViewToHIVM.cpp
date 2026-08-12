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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

/// Info recovered from a partition/strided view chain (rank-generic).
struct ViewInfo {
  Value base;                        // flat memref<?xT, #gm>
  Type elementType;
  unsigned rank = 0;
  SmallVector<int64_t> tile;         // per-dim tile size
  SmallVector<int64_t> traversal;    // per-dim traversal stride (== tile for partition)
  SmallVector<int64_t> strideStatic; // per-dim element stride, or kDynamic
  SmallVector<Value> strideVal;      // per-dim element stride SSA (index)
  SmallVector<Value> n;              // per-dim extent (index) from make_tensor_view
  bool ok = false;
};

static ViewInfo traceView(Value viewVal) {
  ViewInfo vi;
  auto encTy = dyn_cast<tv::TensorViewType>(viewVal.getType());
  if (!encTy)
    return vi;
  // Tile + traversal from the view encoding (partition: traversal == tile).
  ArrayRef<int64_t> tile, traversal;
  Attribute encoding = encTy.getEncoding();
  if (auto p = dyn_cast_or_null<tv::PartitionViewAttr>(encoding)) {
    tile = p.getTile();
    traversal = p.getTile();
  } else if (auto s = dyn_cast_or_null<tv::StridedViewAttr>(encoding)) {
    if (s.getTile().size() != s.getTraversalStrides().size())
      return vi;
    tile = s.getTile();
    traversal = s.getTraversalStrides();
  } else {
    return vi; // gather_scatter is not handled by this (contiguous-tile) path
  }

  // make_partition_view / make_strided_view -> its source is the base view.
  Operation *viewOp = viewVal.getDefiningOp();
  Value baseView;
  if (auto p = dyn_cast_or_null<tv::MakePartitionViewOp>(viewOp))
    baseView = p.getSource();
  else if (auto s = dyn_cast_or_null<tv::MakeStridedViewOp>(viewOp))
    baseView = s.getSource();
  else
    return vi;

  auto mtv = baseView.getDefiningOp<tv::MakeTensorViewOp>();
  if (!mtv)
    return vi;
  auto baseTy = dyn_cast<tv::TensorViewType>(mtv.getResult().getType());
  if (!baseTy)
    return vi;
  // NOTE: read the source operand untyped -- rewriteFuncPtrArgs has changed the
  // underlying function argument to a memref, so mtv.getSource() would assert.
  Value base = mtv->getOperand(0);
  if (!isa<MemRefType>(base.getType()))
    return vi;

  unsigned rank = tile.size();
  if (baseTy.getStrides().size() != rank || mtv.getSizes().size() != rank ||
      mtv.getStrides().size() != rank)
    return vi;

  vi.base = base;
  vi.elementType = baseTy.getElementType();
  vi.rank = rank;
  vi.tile.assign(tile.begin(), tile.end());
  vi.traversal.assign(traversal.begin(), traversal.end());
  vi.strideStatic.assign(baseTy.getStrides().begin(), baseTy.getStrides().end());
  vi.strideVal.assign(mtv.getStrides().begin(), mtv.getStrides().end());
  vi.n.assign(mtv.getSizes().begin(), mtv.getSizes().end());
  vi.ok = true;
  return vi;
}

/// reinterpret_cast %base to offset:[off] sizes:[tile...] strides:[stride...],
/// yielding the (possibly strided, N-D) GM tile view.  No memory is accessed.
static Value emitGmTile(OpBuilder &b, Location loc, const ViewInfo &vi,
                        Value off, hivm::AddressSpaceAttr gmSpace) {
  MLIRContext *ctx = b.getContext();
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                       vi.strideStatic);
  auto gmTileTy = MemRefType::get(vi.tile, vi.elementType, layout, gmSpace);
  SmallVector<OpFoldResult> sizes, strides;
  for (unsigned d = 0; d < vi.rank; ++d) {
    sizes.push_back(b.getIndexAttr(vi.tile[d]));
    if (vi.strideStatic[d] == ShapedType::kDynamic)
      strides.push_back(vi.strideVal[d]);
    else
      strides.push_back(b.getIndexAttr(vi.strideStatic[d]));
  }
  return b.create<memref::ReinterpretCastOp>(loc, gmTileTy, vi.base,
                                             OpFoldResult(off), sizes, strides);
}

/// Physical tile-origin offset = sum_d idx_d * traversal_d * element_stride_d.
static Value computeOffset(OpBuilder &b, Location loc, const ViewInfo &vi,
                           ValueRange indices) {
  Value off;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
    Value logical = b.create<arith::MulIOp>(loc, indices[d], cTrav);
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    off = d == 0 ? phys : b.create<arith::AddIOp>(loc, off, phys);
  }
  return off;
}

/// `[0 : len]` subview of a 1-D memref (the in-bounds prefix).
static Value emitPrefixSubview(OpBuilder &b, Location loc, Value src,
                               Value len) {
  return b.create<memref::SubViewOp>(
      loc, src, ArrayRef<OpFoldResult>{b.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{OpFoldResult(len)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

//===----------------------------------------------------------------------===//
// Lowering.  1-D uses contiguous tail handling (pad + partial DMA); rank > 1 is
// currently block-aligned (full tile).
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load,
                                   hivm::AddressSpaceAttr gmSpace,
                                   hivm::AddressSpaceAttr ubSpace) {
  ViewInfo vi = traceView(load.getView());
  if (!vi.ok || load.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  auto ubTy = MemRefType::get(vi.tile, vi.elementType,
                              MemRefLayoutAttrInterface{}, ubSpace);
  Value ub = b.create<memref::AllocOp>(loc, ubTy);
  auto tensorTy = cast<RankedTensorType>(load.getResult().getType());

  if (vi.rank == 1) {
    Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[0]);
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[0]);
    Value offLogical =
        b.create<arith::MulIOp>(loc, load.getIndices()[0], cTrav);
    Value nMinusOff = b.create<arith::SubIOp>(loc, vi.n[0], offLogical);
    Value len = b.create<arith::MinSIOp>(loc, nMinusOff, cTile);
    Value off = b.create<arith::MulIOp>(loc, offLogical, vi.strideVal[0]);
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    // Zero-pad the whole tile (padding = zero), then DMA the valid prefix.
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(vi.elementType));
    b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{ub});
    Value gmSub = emitPrefixSubview(b, loc, gm, len);
    Value ubSub = emitPrefixSubview(b, loc, ub, len);
    b.create<hivm::LoadOp>(loc, TypeRange{}, gmSub, ubSub);
  } else {
    Value off = computeOffset(b, loc, vi, load.getIndices());
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    b.create<hivm::LoadOp>(loc, TypeRange{}, gm, ub);
  }

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
  if (!vi.ok || store.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();
  auto ubTy = MemRefType::get(vi.tile, vi.elementType,
                              MemRefLayoutAttrInterface{}, ubSpace);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value vm = b.create<bufferization::ToMemrefOp>(loc, ubTy, store.getValue());
#else
  Value vm = b.create<bufferization::ToBufferOp>(loc, ubTy, store.getValue());
#endif

  if (vi.rank == 1) {
    Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[0]);
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[0]);
    Value offLogical =
        b.create<arith::MulIOp>(loc, store.getIndices()[0], cTrav);
    Value nMinusOff = b.create<arith::SubIOp>(loc, vi.n[0], offLogical);
    Value len = b.create<arith::MinSIOp>(loc, nMinusOff, cTile);
    Value off = b.create<arith::MulIOp>(loc, offLogical, vi.strideVal[0]);
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    Value gmSub = emitPrefixSubview(b, loc, gm, len);
    Value vmSub = emitPrefixSubview(b, loc, vm, len);
    b.create<hivm::StoreOp>(loc, TypeRange{}, vmSub, gmSub);
  } else {
    Value off = computeOffset(b, loc, vi, store.getIndices());
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    b.create<hivm::StoreOp>(loc, TypeRange{}, vm, gm);
  }

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
