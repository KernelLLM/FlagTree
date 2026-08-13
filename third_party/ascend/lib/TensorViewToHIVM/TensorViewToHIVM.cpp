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
  SmallVector<int64_t> sparseDims;   // gather/scatter dimensions (currently one)
  bool ok = false;

  bool isGatherScatter() const { return !sparseDims.empty(); }
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
  } else if (auto g = dyn_cast_or_null<tv::GatherScatterViewAttr>(encoding)) {
    tile = g.getTile();
    traversal = g.getTile();
    vi.sparseDims.assign(g.getSparseDim().begin(), g.getSparseDim().end());
    if (vi.sparseDims.size() != 1)
      return vi;
  } else {
    return vi;
  }

  // make_partition_view / make_strided_view -> its source is the base view.
  Operation *viewOp = viewVal.getDefiningOp();
  Value baseView;
  if (auto p = dyn_cast_or_null<tv::MakePartitionViewOp>(viewOp))
    baseView = p.getSource();
  else if (auto s = dyn_cast_or_null<tv::MakeStridedViewOp>(viewOp))
    baseView = s.getSource();
  else if (auto g = dyn_cast_or_null<tv::MakeGatherScatterViewOp>(viewOp))
    baseView = g.getSource();
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

/// Return true for the sentinel used by Pass A when a raw pointer has no
/// recoverable source extent.
static bool isUnknownExtent(Value extent) {
  auto constant = extent.getDefiningOp<arith::ConstantIndexOp>();
  return constant && constant.value() == std::numeric_limits<int64_t>::max();
}

static Value asIndex(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
}

/// With no source extent available from a raw Triton pointer, materialize only
/// the prefix needed by the sparse indices instead of allocating Pass A's
/// INT64_MAX sentinel.  Negative/out-of-range indices retain the frontend's UB
/// semantics.
static Value computeSparseSpan(OpBuilder &b, Location loc, Value sparseIndex,
                               int64_t length) {
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = b.create<arith::ConstantIndexOp>(loc, length);
  auto loop = b.create<scf::ForOp>(
      loc, c0, upper, c1, ValueRange{c0},
      [&](OpBuilder &nested, Location nestedLoc, Value iv, ValueRange args) {
        Value idx =
            nested.create<tensor::ExtractOp>(nestedLoc, sparseIndex, iv);
        idx = asIndex(nested, nestedLoc, idx);
        Value max = nested.create<arith::MaxSIOp>(nestedLoc, args[0], idx);
        nested.create<scf::YieldOp>(nestedLoc, max);
      });
  return b.create<arith::AddIOp>(loc, loop.getResult(0), c1);
}

/// GM/local source window used before vgather or scalar gather decomposition.
/// Regular dimensions contain one tile; the sparse dimension contains the
/// addressable prefix [0, sparseSpan).
struct GatherWindow {
  Value local;
  MemRefType localType;
};

static GatherWindow emitGatherWindow(OpBuilder &b, Location loc,
                                     const ViewInfo &vi, ValueRange indices,
                                     unsigned sparseDim, Value sparseSpan,
                                     hivm::AddressSpaceAttr gmSpace) {
  SmallVector<int64_t> shape(vi.tile.begin(), vi.tile.end());
  shape[sparseDim] = ShapedType::kDynamic;
  auto layout = StridedLayoutAttr::get(
      b.getContext(), /*offset=*/ShapedType::kDynamic, vi.strideStatic);
  auto gmTy = MemRefType::get(shape, vi.elementType, layout, gmSpace);

  Value off;
  for (unsigned d = 0; d < vi.rank; ++d) {
    if (d == sparseDim)
      continue;
    Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    Value logical = b.create<arith::MulIOp>(loc, indices[d], cTile);
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    off = off ? b.create<arith::AddIOp>(loc, off, phys) : phys;
  }
  if (!off)
    off = b.create<arith::ConstantIndexOp>(loc, 0);

  SmallVector<OpFoldResult> sizes, strides;
  for (unsigned d = 0; d < vi.rank; ++d) {
    sizes.push_back(d == sparseDim ? OpFoldResult(sparseSpan)
                                   : OpFoldResult(b.getIndexAttr(vi.tile[d])));
    strides.push_back(vi.strideStatic[d] == ShapedType::kDynamic
                          ? OpFoldResult(vi.strideVal[d])
                          : OpFoldResult(b.getIndexAttr(vi.strideStatic[d])));
  }
  Value gm = b.create<memref::ReinterpretCastOp>(
      loc, gmTy, vi.base, OpFoldResult(off), sizes, strides);
  auto localTy = MemRefType::get(shape, vi.elementType);
  Value local = b.create<memref::AllocOp>(loc, localTy, ValueRange{sparseSpan});
  b.create<hivm::LoadOp>(loc, TypeRange{}, gm, local);
  return {local, localTy};
}

static Value castSparseTensorToI32(OpBuilder &b, Location loc, Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getElementType().isInteger(32))
    return value;
  auto i32Type = RankedTensorType::get(type.getShape(), b.getI32Type());
  if (type.getElementType().isIndex())
    return b.create<arith::IndexCastOp>(loc, i32Type, value);
  return b.create<arith::TruncIOp>(loc, i32Type, value);
}

static bool supportsVGather(Type elementType) {
  if (elementType.isF16() || elementType.isBF16() || elementType.isF32())
    return true;
  auto integerType = dyn_cast<IntegerType>(elementType);
  return integerType &&
         (integerType.getWidth() == 16 || integerType.getWidth() == 32);
}

/// Expand a rank-1 sparse index to rank N and broadcast it across all regular
/// dimensions, producing the index shape required by hivm.hir.vgather.
static Value broadcastSparseIndices(OpBuilder &b, Location loc,
                                    const ViewInfo &vi, Value sparseIndex,
                                    unsigned sparseDim) {
  sparseIndex = castSparseTensorToI32(b, loc, sparseIndex);
  SmallVector<int64_t> expandedShape(vi.rank, 1);
  expandedShape[sparseDim] = vi.tile[sparseDim];
  auto expandedType = RankedTensorType::get(expandedShape, b.getI32Type());
  SmallVector<ReassociationIndices> reassociation(1);
  for (unsigned d = 0; d < vi.rank; ++d)
    reassociation[0].push_back(d);
  Value expanded = b.create<tensor::ExpandShapeOp>(loc, expandedType,
                                                   sparseIndex, reassociation);

  auto expandedMemrefType =
      MemRefType::get(expandedShape, b.getI32Type());
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value expandedBuffer = b.create<bufferization::ToMemrefOp>(
      loc, expandedMemrefType, expanded);
#else
  Value expandedBuffer = b.create<bufferization::ToBufferOp>(
      loc, expandedMemrefType, expanded);
#endif

  auto fullType = MemRefType::get(vi.tile, b.getI32Type());
  Value full = b.create<memref::AllocOp>(loc, fullType);
  SmallVector<int64_t> broadcastDims;
  for (unsigned d = 0; d < vi.rank; ++d)
    if (d != sparseDim)
      broadcastDims.push_back(d);
  b.create<hivm::VBrcOp>(loc, TypeRange{}, expandedBuffer, full,
                         b.getDenseI64ArrayAttr(broadcastDims));
  return full;
}

/// Native AscendNPU-IR fallback for a gather axis that is not the last one:
/// nested scalar tensor.extract/tensor.insert loops.
static Value emitScalarGather(OpBuilder &b, Location loc, const ViewInfo &vi,
                              Value source, Value sparseIndex,
                              unsigned sparseDim, RankedTensorType resultType) {
  Value init = b.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                         resultType.getElementType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> coords;

  for (unsigned d = 0; d < vi.rank; ++d) {
    Value iterArg = loops.empty() ? init : loops.back().getRegionIterArg(0);
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, upper, c1, iterArg);
    if (!loops.empty())
      b.create<scf::YieldOp>(loc, loop.getResult(0));
    loops.push_back(loop);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  Value sparse =
      b.create<tensor::ExtractOp>(loc, sparseIndex, coords[sparseDim]);
  sparse = asIndex(b, loc, sparse);
  SmallVector<Value> sourceCoords(coords);
  sourceCoords[sparseDim] = sparse;
  Value element = b.create<tensor::ExtractOp>(loc, source, sourceCoords);
  Value target = loops.back().getRegionIterArg(0);
  Value result = b.create<tensor::InsertOp>(loc, element, target, coords);
  b.create<scf::YieldOp>(loc, result);
  b.setInsertionPointAfter(loops.front());
  return loops.front().getResult(0);
}

/// Construct per-element physical offsets for hivm.hir.scatter_store.
static Value emitScatterOffsets(OpBuilder &b, Location loc, const ViewInfo &vi,
                                ValueRange indices, unsigned sparseDim) {
  auto offsetType = RankedTensorType::get(vi.tile, b.getI64Type());
  Value init = b.create<tensor::EmptyOp>(loc, offsetType.getShape(),
                                         offsetType.getElementType());
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> coords;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value iterArg = loops.empty() ? init : loops.back().getRegionIterArg(0);
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, upper, c1, iterArg);
    if (!loops.empty())
      b.create<scf::YieldOp>(loc, loop.getResult(0));
    loops.push_back(loop);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  Value offset;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value logical;
    if (d == sparseDim) {
      logical = b.create<tensor::ExtractOp>(loc, indices[d], coords[d]);
      logical = asIndex(b, loc, logical);
    } else {
      Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
      Value origin = b.create<arith::MulIOp>(loc, indices[d], cTile);
      logical = b.create<arith::AddIOp>(loc, origin, coords[d]);
    }
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    offset = offset ? b.create<arith::AddIOp>(loc, offset, phys) : phys;
  }
  Value offset64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offset);
  Value target = loops.back().getRegionIterArg(0);
  Value result = b.create<tensor::InsertOp>(loc, offset64, target, coords);
  b.create<scf::YieldOp>(loc, result);
  b.setInsertionPointAfter(loops.front());
  return loops.front().getResult(0);
}

//===----------------------------------------------------------------------===//
// Lowering.  1-D uses contiguous tail handling (pad + partial DMA); rank > 1 is
// currently block-aligned (full tile).
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load,
                                   hivm::AddressSpaceAttr gmSpace) {
  ViewInfo vi = traceView(load.getView());
  if (!vi.ok || load.getIndices().size() != vi.rank)
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  auto tensorTy = cast<RankedTensorType>(load.getResult().getType());

  if (vi.isGatherScatter()) {
    unsigned sparseDim = vi.sparseDims.front();
    if (sparseDim >= vi.rank ||
        !isa<RankedTensorType>(load.getIndices()[sparseDim].getType()))
      return failure();
    for (unsigned d = 0; d < vi.rank; ++d)
      if (d != sparseDim && !load.getIndices()[d].getType().isIndex())
        return failure();

    Value sparseIndex = load.getIndices()[sparseDim];
    Value sparseSpan = vi.n[sparseDim];
    if (isUnknownExtent(sparseSpan))
      sparseSpan = computeSparseSpan(b, loc, sparseIndex, vi.tile[sparseDim]);
    GatherWindow window = emitGatherWindow(b, loc, vi, load.getIndices(),
                                           sparseDim, sparseSpan, gmSpace);

    Value result;
    if (sparseDim == vi.rank - 1 && supportsVGather(vi.elementType)) {
      Value fullIndices =
          broadcastSparseIndices(b, loc, vi, sparseIndex, sparseDim);
      auto resultMemrefType = MemRefType::get(vi.tile, vi.elementType);
      Value resultBuffer = b.create<memref::AllocOp>(loc, resultMemrefType);
      b.create<hivm::VGatherOp>(loc, TypeRange{}, window.local, fullIndices,
                                resultBuffer);
      result = b.create<bufferization::ToTensorOp>(
          loc, tensorTy, resultBuffer, /*restrict=*/true, /*writable=*/false);
    } else {
      auto sourceTensorType =
          RankedTensorType::get(window.localType.getShape(), vi.elementType);
      Value source = b.create<bufferization::ToTensorOp>(
          loc, sourceTensorType, window.local, /*restrict=*/true,
          /*writable=*/false);
      result = emitScalarGather(b, loc, vi, source, sparseIndex, sparseDim,
                                tensorTy);
    }
    load.getResult().replaceAllUsesWith(result);
    load.erase();
    return success();
  }

  // On-chip staging buffer left UNTAGGED (see file header): a single
  // hir.load ins(#gm) outs(<untagged>) is valid and bishengir assigns the
  // buffer's real space (ub / L0A / L0B) by consumer.
  auto localTy = MemRefType::get(vi.tile, vi.elementType);
  Value buf = b.create<memref::AllocOp>(loc, localTy);

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
    b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buf});
    Value gmSub = emitPrefixSubview(b, loc, gm, len);
    Value bufSub = emitPrefixSubview(b, loc, buf, len);
    b.create<hivm::LoadOp>(loc, TypeRange{}, gmSub, bufSub);
  } else {
    Value off = computeOffset(b, loc, vi, load.getIndices());
    Value gm = emitGmTile(b, loc, vi, off, gmSpace);
    b.create<hivm::LoadOp>(loc, TypeRange{}, gm, buf);
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

  if (vi.isGatherScatter()) {
    unsigned sparseDim = vi.sparseDims.front();
    if (sparseDim >= vi.rank ||
        !isa<RankedTensorType>(store.getIndices()[sparseDim].getType()))
      return failure();
    for (unsigned d = 0; d < vi.rank; ++d)
      if (d != sparseDim && !store.getIndices()[d].getType().isIndex())
        return failure();

    Value offsets =
        emitScatterOffsets(b, loc, vi, store.getIndices(), sparseDim);
    int64_t burstLength = 1;
    int64_t expectedStride = 1;
    for (int64_t d = static_cast<int64_t>(vi.rank) - 1;
         d > static_cast<int64_t>(sparseDim); --d) {
      if (vi.strideStatic[d] != expectedStride)
        break;
      burstLength *= vi.tile[d];
      expectedStride *= vi.tile[d];
    }
    Value burst = b.create<arith::ConstantIntOp>(loc, burstLength, 64);
    b.create<hivm::ScatterStoreOp>(
        loc, TypeRange{}, offsets, store.getValue(), burst, Value(), vi.base,
        /*cache=*/nullptr, /*evict=*/nullptr);
    store.erase();
    return success();
  }

  // Untagged on-chip source buffer (see file header): hir.store dst==gm is the
  // only anchor; bishengir assigns the source buffer's space by producer.
  auto localTy = MemRefType::get(vi.tile, vi.elementType);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value vm = b.create<bufferization::ToMemrefOp>(loc, localTy, store.getValue());
#else
  Value vm = b.create<bufferization::ToBufferOp>(loc, localTy, store.getValue());
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
      if (failed(lowerViewLoad(l, gmSpace)))
        l.emitError("TensorViewToHIVM: unsupported view_load");
    for (tv::ViewStoreOp s : stores)
      if (failed(lowerViewStore(s, gmSpace)))
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
