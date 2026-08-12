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
//                                   %buf = alloc() : memref<TILExT>       (untagged)
//                                   hivm.hir.load ins(%gm) outs(%buf)
//                                   %t   = bufferization.to_tensor %buf   (bridge)
//   tv.view_store %pv[%i], %v    -> %vm  = bufferization.to_buffer %v : memref<TILExT>  (untagged)
//                                   %gm  = reinterpret_cast %base_out ...
//                                   hivm.hir.store ins(%vm) outs(%gm)
//
// Address-space policy: only the GM source/dest (function pointer args) is
// tagged #gm -- the anchor hivm.hir.load/store need (src==gm for load, dst==gm
// for store).  The on-chip staging buffer is left UNTAGGED: the HIVM DMA memory-
// space verifier is skipped when either side is unspaced, so a single
// hir.load ins(#gm) outs(<untagged>) is valid, and bishengir's mem-planning
// assigns the buffer's real space by consumer (ub for vector, L0A/L0B for the
// cube).  Pass B therefore no longer pre-commits ub vs L0 -- placement stays
// with the backend, matching the native lowering path.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

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

/// `[0 : len_d]` rank-N subview of a memref (the in-bounds tile prefix).
static Value emitPrefixSubview(OpBuilder &b, Location loc, Value src,
                               ValueRange lens) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (Value len : lens) {
    offsets.push_back(b.getIndexAttr(0));
    sizes.push_back(OpFoldResult(len));
    strides.push_back(b.getIndexAttr(1));
  }
  return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
}

/// True if `v` is the "unmasked" extent sentinel (INT64_MAX constant) emitted by
/// Pass A for dims without a mask bound.
static bool isSentinelExtent(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue() == std::numeric_limits<int64_t>::max();
  return false;
}

/// Per-dim in-bounds length `len_d = min(n_d - idx_d*trav_d, tile_d)` and the
/// physical tile-origin offset `Σ (idx_d*trav_d) * stride_d`.
static Value emitTailGeometry(OpBuilder &b, Location loc, const ViewInfo &vi,
                              ValueRange indices, SmallVectorImpl<Value> &lens) {
  Value off;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value cTrav = b.create<arith::ConstantIndexOp>(loc, vi.traversal[d]);
    Value offLog = b.create<arith::MulIOp>(loc, indices[d], cTrav);
    Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    Value nMinus = b.create<arith::SubIOp>(loc, vi.n[d], offLog);
    lens.push_back(b.create<arith::MinSIOp>(loc, nMinus, cTile));
    Value phys = b.create<arith::MulIOp>(loc, offLog, vi.strideVal[d]);
    off = d == 0 ? phys : b.create<arith::AddIOp>(loc, off, phys);
  }
  return off;
}

//===----------------------------------------------------------------------===//
// Lowering.  Unmasked tiles DMA the full block (this clean form is what the
// native cube path recognizes for tt.dot operands).  Masked tiles (any dim with
// a real extent) zero-pad the buffer and DMA only the in-bounds rank-N prefix.
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load,
                                   hivm::AddressSpaceAttr gmSpace) {
  ViewInfo vi = traceView(load.getView());
  if (!vi.ok || load.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  auto tensorTy = cast<RankedTensorType>(load.getResult().getType());

  // On-chip staging buffer left UNTAGGED (see file header): a single
  // hir.load ins(#gm) outs(<untagged>) is valid and bishengir assigns the
  // buffer's real space (ub / L0A / L0B) by consumer.
  auto localTy = MemRefType::get(vi.tile, vi.elementType);
  Value buf = b.create<memref::AllocOp>(loc, localTy);

  bool masked = false;
  for (Value n : vi.n)
    if (!isSentinelExtent(n)) {
      masked = true;
      break;
    }

  if (!masked) {
    Value off = computeOffset(b, loc, vi, load.getIndices());
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    b.create<hivm::LoadOp>(loc, TypeRange{}, gm, buf);
  } else {
    SmallVector<Value> lens;
    Value off = emitTailGeometry(b, loc, vi, load.getIndices(), lens);
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    // Zero-pad the whole tile (padding = zero), then DMA the in-bounds prefix.
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(vi.elementType));
    b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buf});
    Value gmSub = emitPrefixSubview(b, loc, gm, lens);
    Value bufSub = emitPrefixSubview(b, loc, buf, lens);
    b.create<hivm::LoadOp>(loc, TypeRange{}, gmSub, bufSub);
  }

  Value t = b.create<bufferization::ToTensorOp>(loc, tensorTy, buf,
                                                /*restrict=*/true,
                                                /*writable=*/false);
  load.getResult().replaceAllUsesWith(t);
  load.erase();
  return success();
}

static LogicalResult lowerViewStore(tv::ViewStoreOp store,
                                    hivm::AddressSpaceAttr gmSpace) {
  ViewInfo vi = traceView(store.getView());
  if (!vi.ok || store.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();
  // Untagged on-chip source buffer (see file header): hir.store dst==gm is the
  // only anchor; bishengir assigns the source buffer's space by producer.
  auto localTy = MemRefType::get(vi.tile, vi.elementType);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value vm = b.create<bufferization::ToMemrefOp>(loc, localTy, store.getValue());
#else
  Value vm = b.create<bufferization::ToBufferOp>(loc, localTy, store.getValue());
#endif

  bool masked = false;
  for (Value n : vi.n)
    if (!isSentinelExtent(n)) {
      masked = true;
      break;
    }

  if (!masked) {
    Value off = computeOffset(b, loc, vi, store.getIndices());
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    b.create<hivm::StoreOp>(loc, TypeRange{}, vm, gm);
  } else {
    SmallVector<Value> lens;
    Value off = emitTailGeometry(b, loc, vi, store.getIndices(), lens);
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    Value gmSub = emitPrefixSubview(b, loc, gm, lens);
    Value vmSub = emitPrefixSubview(b, loc, vm, lens);
    b.create<hivm::StoreOp>(loc, TypeRange{}, vmSub, gmSub);
  }

  store.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Discrete gather/scatter: ptr_load / ptr_store -> scalar scf.for loop
// (reinterpret_cast base offset:[idx[k]] sizes:[1] + memref.load/store), the
// same form the native discrete-access lowering produces.  The base is a scalar
// !tv.ptr (func arg, already rewritten to memref); the gather index comes from
// the op's index operand.
//===----------------------------------------------------------------------===//

/// reinterpret_cast %base to offset:[%ik] sizes:[1] strides:[1] : ... #gm .
static Value emitScalarGmElem(OpBuilder &b, Location loc, Value base, Value ik,
                              Type elemTy, hivm::AddressSpaceAttr gmSpace) {
  auto layout = StridedLayoutAttr::get(b.getContext(),
                                       /*offset=*/ShapedType::kDynamic, {1});
  auto ty = MemRefType::get({1}, elemTy, layout, gmSpace);
  return b.create<memref::ReinterpretCastOp>(
      loc, ty, base, OpFoldResult(ik),
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

static LogicalResult lowerPtrLoad(tv::PtrLoadOp op,
                                  hivm::AddressSpaceAttr gmSpace) {
  // NOTE: read the base untyped -- rewriteFuncPtrArgs has changed the underlying
  // function argument to a memref, so op.getBase() (TypedValue<PtrType>) asserts.
  Value base = op->getOperand(0);
  if (!isa<MemRefType>(base.getType()) || op.getIndices().empty())
    return failure();
  Value idx = op.getIndices()[0]; // tensor<Nxindex>
  Value count = op.getCount();    // optional valid-lane count (null if absent)
  auto resTy = cast<RankedTensorType>(op.getResult().getType());
  int64_t n = resTy.getShape()[0];
  if (ShapedType::isDynamic(n))
    return failure();
  Type elemTy = resTy.getElementType();

  OpBuilder b(op);
  Location loc = op.getLoc();
  auto bufTy = MemRefType::get({n}, elemTy);
  Value buf = b.create<memref::AllocOp>(loc, bufTy);
  Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(elemTy));
  b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buf});

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  // Loop only over the in-bounds lanes (native form); the pad (0) covers the
  // rest.  A per-lane scf.if is NOT used -- the backend mis-lowers it.
  Value ub = count ? count : b.create<arith::ConstantIndexOp>(loc, n);
  auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    auto ext = b.create<tensor::ExtractOp>(loc, idx, ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value rc = emitScalarGmElem(b, loc, base, ext.getResult(), elemTy, gmSpace);
    Value v = b.create<memref::LoadOp>(loc, rc, ValueRange{c0});
    b.create<memref::StoreOp>(loc, v, buf, ValueRange{k});
  }

  Value t = b.create<bufferization::ToTensorOp>(loc, resTy, buf,
                                                /*restrict=*/true,
                                                /*writable=*/false);
  op.getResult().replaceAllUsesWith(t);
  op.erase();
  return success();
}

static LogicalResult lowerPtrStore(tv::PtrStoreOp op,
                                   hivm::AddressSpaceAttr gmSpace) {
  // Base read untyped (see lowerPtrLoad): the arg is now a memref.
  Value base = op->getOperand(0);
  if (!isa<MemRefType>(base.getType()) || op.getIndices().empty())
    return failure();
  Value idx = op.getIndices()[0];
  Value count = op.getCount(); // optional valid-lane count (null if absent)
  Value value = op.getValue();
  auto valTy = cast<RankedTensorType>(value.getType());
  int64_t n = valTy.getShape()[0];
  if (ShapedType::isDynamic(n))
    return failure();
  Type elemTy = valTy.getElementType();

  OpBuilder b(op);
  Location loc = op.getLoc();
  auto bufTy = MemRefType::get({n}, elemTy);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value vm = b.create<bufferization::ToMemrefOp>(loc, bufTy, value);
#else
  Value vm = b.create<bufferization::ToBufferOp>(loc, bufTy, value);
#endif

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  // Loop only over the in-bounds lanes (crucial: the padded tail must NOT
  // scatter, or it would clobber base[idx_pad]).  No per-lane scf.if.
  Value ub = count ? count : b.create<arith::ConstantIndexOp>(loc, n);
  auto loop = b.create<scf::ForOp>(loc, c0, ub, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    Value v = b.create<memref::LoadOp>(loc, vm, ValueRange{k});
    auto ext = b.create<tensor::ExtractOp>(loc, idx, ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value rc = emitScalarGmElem(b, loc, base, ext.getResult(), elemTy, gmSpace);
    b.create<memref::StoreOp>(loc, v, rc, ValueRange{c0});
  }

  op.erase();
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

    // 1. Rewrite !tv.ptr function inputs to GM memrefs.
    module.walk([&](triton::FuncOp func) { rewriteFuncPtrArgs(func, gmSpace); });

    // 2. Lower access ops (collect first to avoid mutation during walk).
    SmallVector<tv::ViewLoadOp> loads;
    SmallVector<tv::ViewStoreOp> stores;
    SmallVector<tv::PtrLoadOp> ptrLoads;
    SmallVector<tv::PtrStoreOp> ptrStores;
    module.walk([&](Operation *op) {
      if (auto l = dyn_cast<tv::ViewLoadOp>(op))
        loads.push_back(l);
      else if (auto s = dyn_cast<tv::ViewStoreOp>(op))
        stores.push_back(s);
      else if (auto pl = dyn_cast<tv::PtrLoadOp>(op))
        ptrLoads.push_back(pl);
      else if (auto ps = dyn_cast<tv::PtrStoreOp>(op))
        ptrStores.push_back(ps);
    });
    for (tv::ViewLoadOp l : loads)
      if (failed(lowerViewLoad(l, gmSpace)))
        l.emitError("TensorViewToHIVM: unsupported view_load");
    for (tv::ViewStoreOp s : stores)
      if (failed(lowerViewStore(s, gmSpace)))
        s.emitError("TensorViewToHIVM: unsupported view_store");
    for (tv::PtrLoadOp pl : ptrLoads)
      if (failed(lowerPtrLoad(pl, gmSpace)))
        pl.emitError("TensorViewToHIVM: unsupported ptr_load");
    for (tv::PtrStoreOp ps : ptrStores)
      if (failed(lowerPtrStore(ps, gmSpace)))
        ps.emitError("TensorViewToHIVM: unsupported ptr_store");

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
