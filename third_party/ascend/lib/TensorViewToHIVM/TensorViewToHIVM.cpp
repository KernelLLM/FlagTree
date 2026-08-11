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

/// Info recovered from a partition/strided view chain.
struct ViewInfo {
  Value base;           // memref<?xT, #gm> (the rewritten function argument)
  Type elementType;     // T
  int64_t tile = 0;     // WINDOW / tile size (1-D)
  int64_t traversal = 0;// STEP between tile origins (== tile for partition)
  int64_t stride = 0;   // element stride (tensor_view.strides)
  Value n;              // full extent (index), from make_tensor_view sizes[0]
  bool ok = false;
};

/// Trace `viewVal` (a partition/strided view SSA) back to its base memref + params.
static ViewInfo traceView(Value viewVal) {
  ViewInfo vi;
  auto encTy = dyn_cast<tv::TensorViewType>(viewVal.getType());
  if (!encTy)
    return vi;
  // Tile + traversal from the view encoding (partition: traversal == tile).
  int64_t tile, traversal;
  Attribute encoding = encTy.getEncoding();
  if (auto p = dyn_cast_or_null<tv::PartitionViewAttr>(encoding)) {
    if (p.getTile().size() != 1)
      return vi;
    tile = p.getTile()[0];
    traversal = tile;
  } else if (auto s = dyn_cast_or_null<tv::StridedViewAttr>(encoding)) {
    if (s.getTile().size() != 1 || s.getTraversalStrides().size() != 1)
      return vi;
    tile = s.getTile()[0];
    traversal = s.getTraversalStrides()[0];
  } else {
    return vi; // gather_scatter is not handled by this (contiguous-tile) path
  }

  // The op producing the encoded view (make_partition_view / make_strided_view);
  // its source is the base tensor_view (make_tensor_view result).
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
  if (!baseTy || baseTy.getStrides().size() != 1)
    return vi;
  // NOTE: read the source operand untyped. rewriteFuncPtrArgs has already
  // changed the underlying function argument to a memref, so the typed
  // accessor mtv.getSource() (which casts to TypedValue<tv::PtrType>) would
  // assert. getOperand(0) is the source operand.
  Value base = mtv->getOperand(0);
  if (!isa<MemRefType>(base.getType()))
    return vi; // base must already be a memref (function argument rewritten)
  if (mtv.getSizes().size() != 1)
    return vi;

  vi.base = base;
  vi.elementType = baseTy.getElementType();
  vi.tile = tile;
  vi.traversal = traversal;
  vi.stride = baseTy.getStrides()[0];
  vi.n = mtv.getSizes()[0];
  vi.ok = true;
  return vi;
}

/// Emit the full GM tile view at element offset `off`:
///   reinterpret_cast %base to offset:[off] sizes:[TILE] strides:[S].
/// This only builds a view descriptor; no memory is accessed until a subview is
/// DMAed, so a tile that spills past the extent is fine here.
static Value emitGmTile(OpBuilder &b, Location loc, const ViewInfo &vi,
                        Value off, hivm::AddressSpaceAttr gmSpace) {
  MLIRContext *ctx = b.getContext();
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                       /*strides=*/{vi.stride});
  auto gmTileTy = MemRefType::get({vi.tile}, vi.elementType, layout, gmSpace);
  return b.create<memref::ReinterpretCastOp>(
      loc, gmTileTy, vi.base, /*offset=*/OpFoldResult(off),
      /*sizes=*/ArrayRef<OpFoldResult>{b.getIndexAttr(vi.tile)},
      /*strides=*/ArrayRef<OpFoldResult>{b.getIndexAttr(vi.stride)});
}

/// `[0 : len]` subview of a 1-D memref (the in-bounds prefix).
static Value emitPrefixSubview(OpBuilder &b, Location loc, Value src,
                               Value len) {
  return b.create<memref::SubViewOp>(
      loc, src, ArrayRef<OpFoldResult>{b.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{OpFoldResult(len)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

/// len = min(n - off, TILE): the in-bounds prefix length of this tile.
static Value emitValidLen(OpBuilder &b, Location loc, const ViewInfo &vi,
                          Value off, Value cTile) {
  Value nMinusOff = b.create<arith::SubIOp>(loc, vi.n, off);
  return b.create<arith::MinSIOp>(loc, nMinusOff, cTile);
}

//===----------------------------------------------------------------------===//
// Lowering (with contiguous tail handling: pad the tile to zero, DMA the valid
// prefix only, so out-of-bounds elements are never accessed).
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load,
                                   hivm::AddressSpaceAttr gmSpace,
                                   hivm::AddressSpaceAttr ubSpace) {
  ViewInfo vi = traceView(load.getView());
  if (!vi.ok || load.getIndices().size() != 1)
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile);
  Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal);
  // Tile origin steps by the traversal stride (== tile for a partition).
  Value off = b.create<arith::MulIOp>(loc, load.getIndices()[0], cTrav);
  Value len = emitValidLen(b, loc, vi, off, cTile);

  Value gm = emitGmTile(b, loc, vi, off, gmSpace);
  auto ubTy = MemRefType::get({vi.tile}, vi.elementType,
                              MemRefLayoutAttrInterface{}, ubSpace);
  Value ub = b.create<memref::AllocOp>(loc, ubTy);
  // Zero-pad the whole tile (encoding padding = zero), then DMA the valid prefix.
  Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(vi.elementType));
  b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{ub});
  Value gmSub = emitPrefixSubview(b, loc, gm, len);
  Value ubSub = emitPrefixSubview(b, loc, ub, len);
  b.create<hivm::LoadOp>(loc, TypeRange{}, gmSub, ubSub);

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
  Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile);
  Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal);
  Value off = b.create<arith::MulIOp>(loc, store.getIndices()[0], cTrav);
  Value len = emitValidLen(b, loc, vi, off, cTile);

  auto ubTy = MemRefType::get({vi.tile}, vi.elementType,
                              MemRefLayoutAttrInterface{}, ubSpace);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value vm = b.create<bufferization::ToMemrefOp>(loc, ubTy, store.getValue());
#else
  Value vm = b.create<bufferization::ToBufferOp>(loc, ubTy, store.getValue());
#endif
  Value gm = emitGmTile(b, loc, vi, off, gmSpace);
  // Only DMA the valid prefix back to GM; the out-of-bounds tail is not written.
  Value gmSub = emitPrefixSubview(b, loc, gm, len);
  Value vmSub = emitPrefixSubview(b, loc, vm, len);
  b.create<hivm::StoreOp>(loc, TypeRange{}, vmSub, gmSub);
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
