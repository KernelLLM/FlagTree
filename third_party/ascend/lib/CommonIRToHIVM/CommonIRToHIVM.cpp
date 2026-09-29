/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

//===----------------------------------------------------------------------===//
// CommonIRToHIVM — Lowers CommonIR dialect ops to HIVM (Ascend NPU) dialect
// ops.
//
// Strategy: greedy rewrite runs patterns iteratively to fixed point.
//   Step 1: tile.alloc → memref.alloc (produces memref with
//   #hivm.address_space) Step 2: tile.to_tensor → RAUW: replace all uses of
//   result with src operand,
//           then erase. The src is now memref (from Step 1), consumers get
//           memref instead of tensor, bridged by UnrealizedConversionCast.
//   Step 3: tile.copy → hivm.copy (tile.* src, on-chip DMA)
//                    or memref.copy (!tt.ptr src, GM↔local DMA; the ptr is
//                    bridged to memref via UnrealizedConversionCast)
//           tile.store_tensor → hivm.copy.
//   Step 4: tile.load/store → hivm.load/store
//   Step 5: sync ops → hivm sync ops
//===----------------------------------------------------------------------===//

#include "ascend/include/CommonIRToHIVM/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
namespace tile = mlir::triton::tile;
namespace tv = mlir::triton::tv;
namespace hivm = mlir::hivm;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_COMMONIRTVPTRLEGALIZE
#define GEN_PASS_DEF_COMMONIRTOHIVM
#include "ascend/include/CommonIRToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

// =============================================================================
// Common Helpers
// =============================================================================

static bool isOneToOne(UnrealizedConversionCastOp op) {
  return op.getInputs().size() == 1 && op->getNumResults() == 1;
}

/// Return true if \p op is an annotation.mark operation.
static bool isAnnotationMark(Operation *op) {
  return op->getName().getStringRef() == "annotation.mark";
}

static Value castDefaultMemrefToGM(Value value, Location loc,
                                   PatternRewriter &rewriter) {
  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (!memrefTy || memrefTy.getMemorySpace())
    return value;

  auto gmSpace = hivm::AddressSpaceAttr::get(rewriter.getContext(),
                                             hivm::AddressSpace::GM);
  auto gmTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                              memrefTy.getLayout(), gmSpace);
  return rewriter.create<memref::MemorySpaceCastOp>(loc, gmTy, value);
}

static hivm::AddressSpace mapMemSpaceToHIVM(tile::MemorySpace tileSpace) {
  switch (tileSpace) {
  case tile::MemorySpace::GM:
    return hivm::AddressSpace::GM;
  case tile::MemorySpace::L1:
    return hivm::AddressSpace::L1;
  case tile::MemorySpace::L0A:
    return hivm::AddressSpace::L0A;
  case tile::MemorySpace::L0B:
    return hivm::AddressSpace::L0B;
  case tile::MemorySpace::L0C:
    return hivm::AddressSpace::L0C;
  case tile::MemorySpace::UB:
    return hivm::AddressSpace::UB;
  // Generic GPU memory spaces (Global/Shared/Local/Register) have no HIVM
  // counterpart and should never reach this lowering pass.
  case tile::MemorySpace::Global:
  case tile::MemorySpace::Shared:
  case tile::MemorySpace::Local:
  case tile::MemorySpace::Register:
    llvm_unreachable(
        "unsupported generic memory space in CommonIR-to-HIVM lowering");
  }
  llvm_unreachable("unknown CommonIR memory space");
}

static MemRefType convertBufToMemRef(tile::BufType bufTy) {
  auto shape = bufTy.getShape();
  auto elemTy = bufTy.getElementType();
  auto space = bufTy.getMemorySpace();
  auto *ctx = elemTy.getContext();
  return MemRefType::get(
      llvm::SmallVector<int64_t>(shape), elemTy, MemRefLayoutAttrInterface{},
      hivm::AddressSpaceAttr::get(ctx, mapMemSpaceToHIVM(space)));
}

static MemRefType convertTensorToMemRef(tile::TensorType tensorTy) {
  auto shape = tensorTy.getShape();
  auto elemTy = tensorTy.getElementType();
  auto space = tensorTy.getMemorySpace();
  auto *ctx = elemTy.getContext();
  return MemRefType::get(
      llvm::SmallVector<int64_t>(shape), elemTy, MemRefLayoutAttrInterface{},
      hivm::AddressSpaceAttr::get(ctx, mapMemSpaceToHIVM(space)));
}

/// Get the underlying memref value, converting tile/builtin tensor types to
/// memref.
static Value getAsMemRef(Value val, PatternRewriter &rewriter) {
  if (isa<MemRefType>(val.getType()))
    return val;
  // Look through UnrealizedConversionCast
  if (auto cast = val.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumResults() == 1 && cast->getNumOperands() == 1) {
      auto inner = cast->getOperand(0);
      if (isa<MemRefType>(inner.getType()))
        return inner;
    }
  }
  // Convert tile/builtin types to memref
  Type targetTy;
  if (auto bufTy = dyn_cast<tile::BufType>(val.getType()))
    targetTy = convertBufToMemRef(bufTy);
  else if (auto tensorTy = dyn_cast<tile::TensorType>(val.getType()))
    targetTy = convertTensorToMemRef(tensorTy);
  else if (auto rankedTy = dyn_cast<RankedTensorType>(val.getType())) {
    auto elemTy = rankedTy.getElementType();
    if (auto ptrElemTy = dyn_cast<triton::PointerType>(elemTy)) {
      auto gmSpace = hivm::AddressSpaceAttr::get(val.getType().getContext(),
                                                 hivm::AddressSpace::GM);
      targetTy =
          MemRefType::get(rankedTy.getShape(), ptrElemTy.getPointeeType(),
                          MemRefLayoutAttrInterface{}, gmSpace);
    } else {
      targetTy = MemRefType::get(rankedTy.getShape(), elemTy);
    }
  } else if (auto ptrTy = dyn_cast<triton::PointerType>(val.getType())) {
    // tt.ptr<tensor<...>> → memref<...>
    auto pointeeTy = ptrTy.getPointeeType();
    auto gmSpace = hivm::AddressSpaceAttr::get(val.getType().getContext(),
                                               hivm::AddressSpace::GM);
    if (auto rankedTy = dyn_cast<RankedTensorType>(pointeeTy))
      targetTy = MemRefType::get(rankedTy.getShape(), rankedTy.getElementType(),
                                 MemRefLayoutAttrInterface{}, gmSpace);
    else
      targetTy = MemRefType::get({ShapedType::kDynamic}, pointeeTy,
                                 MemRefLayoutAttrInterface{}, gmSpace);
  } else
    return val;
  return rewriter
      .create<UnrealizedConversionCastOp>(val.getLoc(), targetTy, val)
      ->getResult(0);
}

static bool isTensorOfPointer(Type ty) {
  auto rankedTy = dyn_cast<RankedTensorType>(ty);
  return rankedTy && isa<triton::PointerType>(rankedTy.getElementType());
}

// ============================================================================
// TensorView tile.load / tile.store direct lowering.
//
// A `tile.load`/`tile.store` whose operand is a `tv.tensor_view` is lowered
// here: a partition or strided encoding becomes a contiguous GM<->UB DMA, a
// gather_scatter encoding becomes an element-wise discrete gather/scatter.
// Both carry the explicit dst_space/src_space memory space.
// ============================================================================

struct ViewTileInfo {
  Value base;
  Type elementType;
  SmallVector<int64_t> tile;
  SmallVector<int64_t> traversal;
  SmallVector<int64_t> dimMap;
  SmallVector<int64_t> strideStatic;
  SmallVector<int64_t> sparseDims;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  tv::PaddingValue padding = tv::PaddingValue::ZERO;
  bool ok = false;
};

static Value asIndexValue(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value);
}

// Trace a `tile.load`/`tile.store` operand back to the `tv.make_*_view` that
// produced it and gather the tile geometry. Handles both partition and strided
// encodings; for a partition view the traversal stride equals the tile size.
static ViewTileInfo traceView(Value view, PatternRewriter &rewriter) {
  ViewTileInfo info;
  auto viewTy = dyn_cast<tv::TensorViewType>(view.getType());
  if (!viewTy)
    return info;
  Attribute enc = viewTy.getEncoding();

  ArrayRef<int64_t> tile;
  ArrayRef<int64_t> dimMap;
  SmallVector<int64_t> dimMapStorage;
  SmallVector<int64_t> traversal;
  SmallVector<int64_t> sparseDims;
  tv::PaddingValue padding = tv::PaddingValue::ZERO;
  Value source;

  if (auto pv = dyn_cast_or_null<tv::PartitionViewAttr>(enc)) {
    auto make = view.getDefiningOp<tv::MakePartitionViewOp>();
    if (!make)
      return info;
    tile = pv.getTile();
    dimMap = pv.getDimMap();
    padding = pv.getPaddingValue();
    traversal.assign(tile.begin(), tile.end()); // step == size
    source = make.getSource();
  } else if (auto sv = dyn_cast_or_null<tv::StridedViewAttr>(enc)) {
    auto make = view.getDefiningOp<tv::MakeStridedViewOp>();
    if (!make)
      return info;
    tile = sv.getTile();
    dimMap = sv.getDimMap();
    padding = sv.getPaddingValue();
    ArrayRef<int64_t> ts = sv.getTraversalStrides();
    traversal.assign(ts.begin(), ts.end());
    source = make.getSource();
  } else if (auto gs = dyn_cast_or_null<tv::GatherScatterViewAttr>(enc)) {
    auto make = view.getDefiningOp<tv::MakeGatherScatterViewOp>();
    if (!make)
      return info;
    tile = gs.getTile();
    padding = gs.getPaddingValue();
    traversal.assign(tile.begin(), tile.end()); // contiguous step per dim
    sparseDims.assign(gs.getSparseDim().begin(), gs.getSparseDim().end());
    // A gather_scatter view keeps the base dimension order, so the tile
    // dimensions map onto the base dimensions identically.
    dimMapStorage.resize(tile.size());
    for (unsigned i = 0; i < tile.size(); ++i)
      dimMapStorage[i] = i;
    dimMap = dimMapStorage;
    source = make.getSource();
  } else {
    return info;
  }

  auto makeView = source.getDefiningOp<tv::MakeTensorViewOp>();
  auto baseTy = dyn_cast<tv::TensorViewType>(source.getType());
  if (!makeView || !baseTy)
    return info;

  if (tile.empty() || tile.size() != dimMap.size() ||
      tile.size() != traversal.size() ||
      makeView.getSizes().size() != makeView.getStrides().size() ||
      makeView.getSizes().size() != baseTy.getStrides().size())
    return info;

  Value base = getAsMemRef(makeView.getSource(), rewriter);
  if (!isa<MemRefType>(base.getType()))
    return info;
  base = castDefaultMemrefToGM(base, makeView.getLoc(), rewriter);

  for (int64_t tileSize : tile)
    if (tileSize <= 0)
      return ViewTileInfo();
  for (int64_t step : traversal)
    if (step <= 0)
      return ViewTileInfo();
  for (unsigned i = 0; i < dimMap.size(); ++i) {
    int64_t d = dimMap[i];
    if (d < 0 || static_cast<size_t>(d) >= makeView.getSizes().size())
      return ViewTileInfo();
    for (unsigned j = 0; j < i; ++j)
      if (dimMap[j] == d)
        return ViewTileInfo();
  }

  info.base = base;
  info.elementType = viewTy.getElementType();
  info.tile.assign(tile.begin(), tile.end());
  info.traversal = std::move(traversal);
  info.dimMap.assign(dimMap.begin(), dimMap.end());
  info.strideStatic.assign(baseTy.getStrides().begin(),
                           baseTy.getStrides().end());
  info.shape.assign(makeView.getSizes().begin(), makeView.getSizes().end());
  info.strides.assign(makeView.getStrides().begin(),
                      makeView.getStrides().end());
  info.padding = padding;
  info.sparseDims = std::move(sparseDims);
  info.ok = true;
  return info;
}

// Compute the flat element offset of the tile selected by `indices`, and
// optionally the per-dimension valid (in-bounds) lengths used for tail padding.
// The tile *starts* at `index * traversal` along each view dimension; its
// extent is `tile`. For a partition view traversal == tile, so this reduces to
// the familiar `index * tile` start. Valid length always clamps against the
// tile extent, independent of the traversal stride.
static Value emitViewGeometry(OpBuilder &builder, Location loc,
                              const ViewTileInfo &info, ValueRange indices,
                              SmallVectorImpl<Value> *validLengths) {
  Value offset;
  for (unsigned viewDim = 0; viewDim < info.tile.size(); ++viewDim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[viewDim]);
    Value index = asIndexValue(builder, loc, indices[viewDim]);
    Value step =
        builder.create<arith::ConstantIndexOp>(loc, info.traversal[viewDim]);
    Value logical = builder.create<arith::MulIOp>(loc, index, step);

    if (validLengths) {
      Value tileSize =
          builder.create<arith::ConstantIndexOp>(loc, info.tile[viewDim]);
      Value extent = asIndexValue(builder, loc, info.shape[baseDim]);
      Value remaining = builder.create<arith::SubIOp>(loc, extent, logical);
      Value length =
          builder.create<arith::MinSIOp>(loc, remaining, tileSize);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      validLengths->push_back(
          builder.create<arith::MaxSIOp>(loc, length, zero));
    }

    Value stride = asIndexValue(builder, loc, info.strides[baseDim]);
    Value physical = builder.create<arith::MulIOp>(loc, logical, stride);
    offset = offset ? builder.create<arith::AddIOp>(loc, offset, physical)
                    : physical;
  }
  return offset;
}

// Build the GM-side memref for one tile via reinterpret_cast. The tile shape is
// `tile` (the traversal stride only moved the start offset, not the extent),
// so this helper is identical for partition and strided views.
static Value emitViewGmTile(OpBuilder &builder, Location loc,
                            const ViewTileInfo &info, Value offset) {
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> staticStrides;
  for (unsigned viewDim = 0; viewDim < info.tile.size(); ++viewDim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[viewDim]);
    sizes.push_back(builder.getIndexAttr(info.tile[viewDim]));
    int64_t stride = info.strideStatic[baseDim];
    staticStrides.push_back(stride);
    if (stride == ShapedType::kDynamic)
      strides.push_back(info.strides[baseDim]);
    else
      strides.push_back(builder.getIndexAttr(stride));
  }

  auto layout = StridedLayoutAttr::get(builder.getContext(),
                                       ShapedType::kDynamic, staticStrides);
  auto gmSpace = hivm::AddressSpaceAttr::get(builder.getContext(),
                                             hivm::AddressSpace::GM);
  auto gmTy =
      MemRefType::get(info.tile, info.elementType, layout, gmSpace);
  return builder.create<memref::ReinterpretCastOp>(
      loc, gmTy, info.base, OpFoldResult(offset), sizes, strides);
}

// Restrict a full tile memref to its in-bounds prefix along every dimension so
// only valid lanes are moved; the padding fill covers the rest.
static Value emitPrefixSubview(OpBuilder &builder, Location loc, Value source,
                               ValueRange validLengths) {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  for (Value length : validLengths) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(OpFoldResult(length));
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
}

static MemRefType makeSpaceMemref(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                  Type elementType, hivm::AddressSpace space) {
  auto spaceAttr = hivm::AddressSpaceAttr::get(ctx, space);
  return MemRefType::get(shape, elementType, MemRefLayoutAttrInterface{},
                         spaceAttr);
}

// Resolve the on-chip memory space carried by a tile.load/tile.store into a
// HIVM address space. A null/absent attribute means the DSL did not specify a
// space, so we default to UB (the Vector working buffer). Generic GPU spaces
// are rejected: they never occur on the Ascend lowering path.
static hivm::AddressSpace
resolveTileSpace(std::optional<tile::MemorySpace> space) {
  if (!space)
    return hivm::AddressSpace::UB;
  return mapMemSpaceToHIVM(*space);
}

// Build the constant used to fill a tile's out-of-bounds lanes according to the
// view's padding value: the tail lanes of a dense boundary tile or the
// out-of-range lanes of a gather_scatter gather. Float element types support the
// IEEE special pads (nan / +inf / -inf); integers only support zero.
static Value emitPadConstant(OpBuilder &builder, Location loc, Type elementType,
                             tv::PaddingValue padding) {
  if (auto floatTy = dyn_cast<FloatType>(elementType)) {
    const llvm::fltSemantics &sem = floatTy.getFloatSemantics();
    llvm::APFloat value = llvm::APFloat::getZero(sem);
    switch (padding) {
    case tv::PaddingValue::NAN_VALUE:
      value = llvm::APFloat::getNaN(sem);
      break;
    case tv::PaddingValue::POS_INF:
      value = llvm::APFloat::getInf(sem, /*Negative=*/false);
      break;
    case tv::PaddingValue::NEG_INF:
      value = llvm::APFloat::getInf(sem, /*Negative=*/true);
      break;
    default:
      break;
    }
    return builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(elementType, value));
  }
  return builder.create<arith::ConstantOp>(loc,
                                           builder.getZeroAttr(elementType));
}

// A tile.load/tile.store lowers through the TensorView path when its operand
// carries a partition, strided, or gather_scatter encoding and the index
// count matches the tile rank. The dense encodings share the contiguous DMA
// lowering; a gather_scatter view adds the discrete access path.
static bool isViewLowerable(tv::TensorViewType viewTy, size_t numIndices) {
  Attribute enc = viewTy.getEncoding();
  if (auto pv = dyn_cast_or_null<tv::PartitionViewAttr>(enc))
    return numIndices == pv.getTile().size();
  if (auto sv = dyn_cast_or_null<tv::StridedViewAttr>(enc))
    return numIndices == sv.getTile().size();
  if (auto gs = dyn_cast_or_null<tv::GatherScatterViewAttr>(enc))
    return numIndices == gs.getTile().size();
  return false;
}

// Build a single-element GM view (memref<1xT>) at a dynamic flat offset so an
// individual scalar element can be loaded from global memory.
static Value emitGmElementView(OpBuilder &b, Location loc, Value base,
                               Value offset, Type elementType) {
  auto layout = StridedLayoutAttr::get(b.getContext(),
                                       /*offset=*/ShapedType::kDynamic, {1});
  auto gmSpace =
      hivm::AddressSpaceAttr::get(b.getContext(), hivm::AddressSpace::GM);
  auto ty = MemRefType::get({1}, elementType, layout, gmSpace);
  return b.create<memref::ReinterpretCastOp>(
      loc, ty, base, OpFoldResult(offset),
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

// Physical (flat) element offset and in-bounds predicate of one tile element.
struct ElementAccess {
  Value physOffset;
  Value inBounds;
};

// Resolve one tile element to its base coordinate along every dimension. A
// gather_scatter sparse dimension reads its logical coordinate from the
// per-lane index tensor via tensor.extract; every other dimension follows the
// dense rule index * traversal + coord. The element is in bounds when all of
// its logical coordinates fall inside the base view; physOffset is the flat
// element offset (sum of logical * stride) into the base memref.
static ElementAccess emitElementAccess(OpBuilder &b, Location loc,
                                       const ViewTileInfo &info,
                                       ValueRange indices, ValueRange coords) {
  ElementAccess access;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  access.inBounds = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);
  for (unsigned viewDim = 0; viewDim < info.tile.size(); ++viewDim) {
    unsigned baseDim = static_cast<unsigned>(info.dimMap[viewDim]);
    Value logical;
    if (llvm::is_contained(info.sparseDims, static_cast<int64_t>(viewDim))) {
      Value extracted =
          b.create<tensor::ExtractOp>(loc, indices[viewDim], coords[viewDim]);
      logical = asIndexValue(b, loc, extracted);
    } else {
      Value index = asIndexValue(b, loc, indices[viewDim]);
      Value step =
          b.create<arith::ConstantIndexOp>(loc, info.traversal[viewDim]);
      Value origin = b.create<arith::MulIOp>(loc, index, step);
      logical = b.create<arith::AddIOp>(loc, origin, coords[viewDim]);
    }

    Value extent = asIndexValue(b, loc, info.shape[baseDim]);
    Value nonNeg =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, logical, zero);
    Value below = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                          logical, extent);
    Value dimIn = b.create<arith::AndIOp>(loc, nonNeg, below);
    access.inBounds = b.create<arith::AndIOp>(loc, access.inBounds, dimIn);

    Value stride = asIndexValue(b, loc, info.strides[baseDim]);
    Value physical = b.create<arith::MulIOp>(loc, logical, stride);
    access.physOffset =
        access.physOffset
            ? b.create<arith::AddIOp>(loc, access.physOffset, physical)
            : physical;
  }
  return access;
}

// Element-wise padded gather into a caller-provided UB buffer, valid for any
// tile rank and for both a dense boundary tile and a gather_scatter view. One
// scf.for is nested per tile dimension; for each element emitElementAccess
// resolves its base coordinate (a gather_scatter sparse dimension reads it
// from the index tensor), and an scf.if loads from GM when the element is in
// bounds or stores the pad constant otherwise. Writing element-by-element
// into `ub` (rather than assembling a tensor) lets a dense caller share the
// same destination buffer with its whole-tile DMA fast path.
static void emitElementwiseFill(OpBuilder &b, Location loc,
                                const ViewTileInfo &info, ValueRange indices,
                                Value ub, Value pad) {
  unsigned rank = info.tile.size();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> coords;
  OpBuilder::InsertionGuard guard(b);
  for (unsigned d = 0; d < rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, zero, upper, one);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  ElementAccess access = emitElementAccess(b, loc, info, indices, coords);
  auto ifOp = b.create<scf::IfOp>(
      loc, access.inBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value elemView = emitGmElementView(nested, nestedLoc, info.base,
                                           access.physOffset, info.elementType);
        Value loaded =
            nested.create<memref::LoadOp>(nestedLoc, elemView, ValueRange{zero});
        nested.create<scf::YieldOp>(nestedLoc, loaded);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        nested.create<scf::YieldOp>(nestedLoc, pad);
      });
  b.create<memref::StoreOp>(loc, ifOp.getResult(0), ub, coords);
}

// A dense tile is fully in bounds when every dimension's valid length
// (computed by emitViewGeometry) covers the whole tile extent. The base
// view's shape is a runtime value, so this can only be decided at runtime,
// not from the static tile size alone.
static Value emitAllInBounds(OpBuilder &b, Location loc,
                             const ViewTileInfo &info,
                             ValueRange validLengths) {
  Value allIn = b.create<arith::ConstantIntOp>(loc, 1, /*width=*/1);
  for (unsigned d = 0; d < info.tile.size(); ++d) {
    Value tileSize = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    Value full = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         validLengths[d], tileSize);
    allIn = b.create<arith::AndIOp>(loc, allIn, full);
  }
  return allIn;
}

// Load a dense (partition/strided) tile, choosing at runtime between a
// whole-tile DMA and the element-wise padded gather. Whether the tile stays
// in bounds is a runtime property of the base view's shape, so scf.if picks
// the branch once per tile invocation rather than at compile time. Both
// branches fill the same UB buffer so the result has a single definition
// regardless of which one ran: a tensor-typed scf.if result would instead
// require both branches to bufferize to the same memory space, which a
// whole-tile DMA and an element-wise loop do not do identically. A boundary
// tile takes the gather branch even though the view is dense, because the
// in-bounds region of a multi-dim tile is generally L-shaped and cannot be
// expressed as a single rectangular subview of the whole-tile DMA.
static Value emitDenseTileLoad(OpBuilder &b, Location loc,
                               const ViewTileInfo &info, ValueRange indices,
                               Value offset, ValueRange validLengths,
                               hivm::AddressSpace dstSpace,
                               RankedTensorType resultTy, Value pad) {
  Value ub = b.create<memref::AllocOp>(
      loc, makeSpaceMemref(b.getContext(), info.tile, info.elementType,
                          dstSpace));
  Value allInBounds = emitAllInBounds(b, loc, info, validLengths);
  b.create<scf::IfOp>(
      loc, allInBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value gm = emitViewGmTile(nested, nestedLoc, info, offset);
        nested.create<hivm::LoadOp>(nestedLoc, TypeRange{}, gm, ub);
        nested.create<scf::YieldOp>(nestedLoc);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        emitElementwiseFill(nested, nestedLoc, info, indices, ub, pad);
        nested.create<scf::YieldOp>(nestedLoc);
      });
  return b.create<bufferization::ToTensorOp>(loc, resultTy, ub, true, false);
}

// Element-wise padded gather for a gather_scatter view, valid for any tile
// rank. One scf.for is nested per tile dimension; for each element
// emitElementAccess resolves its base coordinate from the sparse dimension's
// index tensor, and an scf.if loads from GM when the element is in bounds or
// yields the pad constant otherwise. Because the padded lane's value is
// produced as the element's sole definition (rather than a whole-tile fill
// that a later DMA partially overwrites), it survives backend dead-store
// elimination.
static Value emitPaddedGatherLoad(OpBuilder &b, Location loc,
                                  const ViewTileInfo &info, ValueRange indices,
                                  RankedTensorType resultTy, Value pad) {
  unsigned rank = info.tile.size();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value init = b.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                         resultTy.getElementType());

  SmallVector<scf::ForOp> loops;
  SmallVector<Value> coords;
  OpBuilder::InsertionGuard guard(b);
  for (unsigned d = 0; d < rank; ++d) {
    Value iterArg = loops.empty() ? init : loops.back().getRegionIterArg(0);
    Value upper = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, zero, upper, one, iterArg);
    if (!loops.empty())
      b.create<scf::YieldOp>(loc, loop.getResult(0));
    loops.push_back(loop);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  Value target = loops.back().getRegionIterArg(0);
  ElementAccess access = emitElementAccess(b, loc, info, indices, coords);

  auto ifOp = b.create<scf::IfOp>(
      loc, access.inBounds,
      [&](OpBuilder &nested, Location nestedLoc) {
        Value elemView = emitGmElementView(nested, nestedLoc, info.base,
                                           access.physOffset, info.elementType);
        Value loaded =
            nested.create<memref::LoadOp>(nestedLoc, elemView, ValueRange{zero});
        nested.create<scf::YieldOp>(nestedLoc, loaded);
      },
      [&](OpBuilder &nested, Location nestedLoc) {
        nested.create<scf::YieldOp>(nestedLoc, pad);
      });

  Value inserted =
      b.create<tensor::InsertOp>(loc, ifOp.getResult(0), target, coords);
  b.create<scf::YieldOp>(loc, inserted);
  return loops.front().getResult(0);
}

// A gather_scatter view pairs each sparse dimension with a tensor-valued index
// and every other dimension with a scalar index; a dense view (empty
// sparseDims) therefore expects all-scalar indices. Reject a mismatched operand
// so the element-access emitters can assume the pairing holds.
static bool viewIndicesWellTyped(const ViewTileInfo &info, ValueRange indices) {
  for (unsigned d = 0; d < indices.size(); ++d) {
    bool isSparse = llvm::is_contained(info.sparseDims, static_cast<int64_t>(d));
    bool isTensor = isa<RankedTensorType>(indices[d].getType());
    if (isSparse != isTensor)
      return false;
  }
  return true;
}

// Element-wise discrete scatter, symmetric to emitPaddedGatherLoad. One scf.for
// is nested per tile dimension; emitElementAccess resolves each element's base
// coordinate and in-bounds predicate, and an scf.if stores the source tensor's
// element to GM only when it is in bounds. Out-of-bounds lanes are dropped,
// matching the view's scatter semantics.
static void emitDiscreteScatter(OpBuilder &b, Location loc,
                                const ViewTileInfo &info, ValueRange indices,
                                Value source) {
  unsigned rank = info.tile.size();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> coords;
  OpBuilder::InsertionGuard guard(b);
  for (unsigned d = 0; d < rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, info.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, zero, upper, one);
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  ElementAccess access = emitElementAccess(b, loc, info, indices, coords);
  b.create<scf::IfOp>(
      loc, access.inBounds, [&](OpBuilder &nested, Location nestedLoc) {
        Value element =
            nested.create<tensor::ExtractOp>(nestedLoc, source, coords);
        Value elemView = emitGmElementView(nested, nestedLoc, info.base,
                                           access.physOffset, info.elementType);
        Value empty = nested.create<tensor::EmptyOp>(
            nestedLoc, ArrayRef<int64_t>{1}, info.elementType);
        Value inserted = nested.create<tensor::InsertOp>(nestedLoc, element,
                                                         empty, ValueRange{zero});
        auto materialize =
            nested.create<bufferization::MaterializeInDestinationOp>(
                nestedLoc, inserted, elemView);
        materialize->setAttr("writable", nested.getUnitAttr());
        nested.create<scf::YieldOp>(nestedLoc);
      });
}

// Lower a `tile.load` off a TensorView. A dense (partition/strided) view maps
// to a contiguous GM -> UB DMA (hivm.load, MTE2) when the tile is fully in
// bounds, falling back to an element-wise gather at any boundary; a
// gather_scatter view always takes the element-wise gather, since its sparse
// dimension has no contiguous physical layout to DMA. The tile geometry
// comes from `traceView`.
static LogicalResult lowerViewLoad(tile::LoadOp op, PatternRewriter &rewriter) {
  auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultTy)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

  ViewTileInfo info = traceView(op.getSrc(), rewriter);
  if (!info.ok || op.getIndices().size() != info.tile.size() ||
      resultTy.getRank() != static_cast<int64_t>(info.tile.size()) ||
      resultTy.getElementType() != info.elementType ||
      resultTy.getShape() != ArrayRef<int64_t>(info.tile))
    return rewriter.notifyMatchFailure(op, "unsupported TensorView load shape");
  if (!viewIndicesWellTyped(info, op.getIndices()))
    return rewriter.notifyMatchFailure(
        op, "TensorView load: sparse dims need tensor indices, dense dims scalar");

  Location loc = op.getLoc();
  // Only the GM -> UB DMA is lowered here; other on-chip spaces
  // (L1/L0A/L0B/L0C) are unsupported.
  hivm::AddressSpace dstSpace = resolveTileSpace(op.getDstSpace());
  if (dstSpace != hivm::AddressSpace::UB)
    return rewriter.notifyMatchFailure(
        op, "TensorView load: dst_space other than UB is unsupported");

  Value pad = emitPadConstant(rewriter, loc, info.elementType, info.padding);

  if (info.sparseDims.empty()) {
    SmallVector<Value> validLengths;
    Value offset =
        emitViewGeometry(rewriter, loc, info, op.getIndices(), &validLengths);
    Value tensor = emitDenseTileLoad(rewriter, loc, info, op.getIndices(),
                                     offset, validLengths, dstSpace, resultTy,
                                     pad);
    rewriter.replaceOp(op, tensor);
    return success();
  }

  // Element-wise gather: a gather_scatter view reads scattered rows, so each
  // lane loads independently. In-bounds lanes load from GM and out-of-bounds
  // lanes take the view's padding value.
  Value tensor =
      emitPaddedGatherLoad(rewriter, loc, info, op.getIndices(), resultTy, pad);
  rewriter.replaceOp(op, tensor);
  return success();
}

// Lower a `tile.store` into a TensorView, symmetric to lowerViewLoad. A dense
// view becomes a contiguous UB -> GM DMA (hivm.store, MTE3); a gather_scatter
// view becomes an element-wise discrete scatter that writes each lane to its
// sparse coordinate and drops out-of-bounds lanes.
static LogicalResult lowerViewStore(tile::StoreOp op,
                                    PatternRewriter &rewriter) {
  auto valueTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
  if (!valueTy)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor source");

  ViewTileInfo info = traceView(op.getDst(), rewriter);
  if (!info.ok || op.getIndices().size() != info.tile.size() ||
      valueTy.getRank() != static_cast<int64_t>(info.tile.size()) ||
      valueTy.getElementType() != info.elementType ||
      valueTy.getShape() != ArrayRef<int64_t>(info.tile))
    return rewriter.notifyMatchFailure(op,
                                       "unsupported TensorView store shape");
  if (!viewIndicesWellTyped(info, op.getIndices()))
    return rewriter.notifyMatchFailure(
        op,
        "TensorView store: sparse dims need tensor indices, dense dims scalar");

  Location loc = op.getLoc();
  // Only the UB -> GM DMA is lowered here; other on-chip spaces
  // (L1/L0A/L0B/L0C) are unsupported.
  hivm::AddressSpace srcSpace = resolveTileSpace(op.getSrcSpace());
  if (srcSpace != hivm::AddressSpace::UB)
    return rewriter.notifyMatchFailure(
        op, "TensorView store: src_space other than UB is unsupported");

  // A gather_scatter view writes each lane to its own sparse coordinate;
  // out-of-bounds lanes are dropped.
  if (!info.sparseDims.empty()) {
    emitDiscreteScatter(rewriter, loc, info, op.getIndices(), op.getSrc());
    rewriter.eraseOp(op);
    return success();
  }

  SmallVector<Value> validLengths;
  Value offset =
      emitViewGeometry(rewriter, loc, info, op.getIndices(), &validLengths);
  Value gm = emitViewGmTile(rewriter, loc, info, offset);
  Value ub = rewriter.create<memref::AllocOp>(
      loc, makeSpaceMemref(rewriter.getContext(), info.tile, info.elementType,
                           srcSpace));
  Value gmSub = emitPrefixSubview(rewriter, loc, gm, validLengths);
  Value ubSub = emitPrefixSubview(rewriter, loc, ub, validLengths);

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  for (Value length : validLengths) {
    offsets.push_back(rewriter.getIndexAttr(0));
    sizes.push_back(OpFoldResult(length));
    strides.push_back(rewriter.getIndexAttr(1));
  }
  Value slice = rewriter.create<tensor::ExtractSliceOp>(
      loc, op.getSrc(), offsets, sizes, strides);
  auto materialize = rewriter.create<bufferization::MaterializeInDestinationOp>(
      loc, slice, ubSub);
  materialize->setAttr("writable", rewriter.getUnitAttr());
  rewriter.create<hivm::StoreOp>(loc, TypeRange{}, ubSub, gmSub);
  rewriter.eraseOp(op);
  return success();
}

static void lowerScatteredLoad(tile::CopyOp op, Value ptr, Value buf,
                               PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value loaded = rewriter.create<triton::LoadOp>(
      loc, ptr, Value(), Value(), ArrayRef<int32_t>{},
      std::optional<triton::PaddingOption>(), triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, false);

  auto ty = cast<RankedTensorType>(loaded.getType());
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  Value mem = rewriter.create<bufferization::ToMemrefOp>(
      loc, MemRefType::get(ty.getShape(), ty.getElementType()), loaded);
#else
  Value mem = rewriter.create<bufferization::ToBufferOp>(
      loc, MemRefType::get(ty.getShape(), ty.getElementType()), loaded);
#endif

  rewriter.create<hivm::CopyOp>(loc, TypeRange{}, mem,
                                getAsMemRef(buf, rewriter));
  rewriter.eraseOp(op);
}

static void lowerScatteredStore(tile::CopyOp op, Value buf, Value ptr,
                                PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value mem = getAsMemRef(buf, rewriter);
  auto m = cast<MemRefType>(mem.getType());

  Value t = rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(m.getShape(), m.getElementType()), mem, true,
      false);

  rewriter.replaceOpWithNewOp<triton::StoreOp>(
      op, ptr, t, Value(), ArrayRef<int32_t>{}, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);
}

static bool isTritonPointerLike(Type ty) {
  if (isa<triton::PointerType>(ty))
    return true;
  if (auto rankedTy = dyn_cast<RankedTensorType>(ty))
    return isa<triton::PointerType>(rankedTy.getElementType());
  return false;
}

static hivm::PIPE mapPipe(int64_t tilePipe) {
  switch (static_cast<tile::Pipe>(tilePipe)) {
  case tile::Pipe::PIPE_M:
    return hivm::PIPE::PIPE_M;
  case tile::Pipe::PIPE_V:
    return hivm::PIPE::PIPE_V;
  case tile::Pipe::PIPE_MTE1:
    return hivm::PIPE::PIPE_MTE1;
  case tile::Pipe::PIPE_MTE2:
    return hivm::PIPE::PIPE_MTE2;
  case tile::Pipe::PIPE_MTE3:
    return hivm::PIPE::PIPE_MTE3;
  case tile::Pipe::PIPE_FIX:
    return hivm::PIPE::PIPE_FIX;
  case tile::Pipe::PIPE_S:
    return hivm::PIPE::PIPE_S;
  }
  llvm_unreachable("unknown CommonIR pipe");
}

static hivm::EVENT mapEvent(int64_t tileEvent) {
  return static_cast<hivm::EVENT>(tileEvent);
}

// =============================================================================
// Step 1: tile.alloc → memref.alloc + #hivm.address_space
// =============================================================================
struct TileAllocToMemRef : OpRewritePattern<tile::AllocOp> {
  using OpRewritePattern<tile::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::AllocOp op,
                                PatternRewriter &rewriter) const final {
    auto bufTy = op.getResult().getType();
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, convertBufToMemRef(bufTy));
    return success();
  }
};

// =============================================================================
// step 1.5
// =============================================================================

struct TileSubviewToMemrefSubview : public OpRewritePattern<tile::SubViewOp> {
  using OpRewritePattern<tile::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Convert the source buffer and deduce the result type
    Value sourceMemRef = getAsMemRef(op.getOperand(0), rewriter);
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    auto sourceMemRefType = sourceMemRef.getType().cast<MemRefType>();
#else
    auto sourceMemRefType = mlir::cast<MemRefType>(sourceMemRef.getType());
#endif

    // 2. Map dynamic offsets directly to OpFoldResult
    SmallVector<OpFoldResult> mixedOffsets = llvm::to_vector<4>(llvm::map_range(
        op.getOffsets(), [](Value v) { return OpFoldResult(v); }));

    // 3. Unpack static sizes and strides into OpFoldResult arrays
    SmallVector<OpFoldResult> mixedSizes = getAsOpFoldResult(op.getSizesAttr());
    SmallVector<OpFoldResult> mixedStrides =
        getAsOpFoldResult(op.getStridesAttr());

    auto targetShape = op.getType().getShape();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    auto inferredType = memref::SubViewOp::inferRankReducedResultType(
                            targetShape, sourceMemRefType, mixedOffsets,
                            mixedSizes, mixedStrides)
                            .cast<MemRefType>();
#else
    auto inferredType =
        mlir::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            targetShape, sourceMemRefType, mixedOffsets, mixedSizes,
            mixedStrides));
#endif

    // 4. Replace the old op with the standard memref.subview
    auto memrefSubview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), inferredType, sourceMemRef, mixedOffsets, mixedSizes,
        mixedStrides);

    rewriter.replaceOp(op, memrefSubview.getResult());

    return success();
  }
};

// =============================================================================
// Step 2: tile.to_tensor → UnrealizedConversionCast (memref → tensor)
// =============================================================================
struct TileToTensorEliminate : OpRewritePattern<tile::ToTensorOp> {
  using OpRewritePattern<tile::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::ToTensorOp op,
                                PatternRewriter &rewriter) const final {
    // The src is now memref (from Step 1).  The result type is tensor<>.
    // Bridge the gap with UnrealizedConversionCast: memref → tensor.
    Value src = op.getOperand();
    auto resultTy = op.getResult().getType();
    if (op->getAttr("writable")) {
      // The caller (tle.dsa.to_tensor defaults to writable=True) expects an
      // in-place view: custom ops use such a tensor as a DPS out and write
      // the results back into the source buffer. Lower it to a writable
      // bufferization.to_tensor; a plain cast would make One-Shot
      // Bufferization copy the out into a fresh UB allocation that
      // hivm-plan-memory cannot plan, which ends up as a `call @malloc` in
      // the device code (AICore link failure) and a broken dataflow.
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
          op, cast<RankedTensorType>(resultTy), src,
          /*restrict=*/true, /*writable=*/true);
      return success();
    }
    if (src.getType() != resultTy) {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(),
                                                              resultTy, src);
      rewriter.replaceOp(op, cast->getResult(0));
    } else {
      rewriter.replaceOp(op, src);
    }
    return success();
  }
};

// =============================================================================
// Step 3: tile.copy → hivm.copy / memref.copy
//   - tile.* source  → hivm.copy (on-chip buffer DMA)
//   - !tt.ptr source → memref.copy (GM→local DMA; later passes lower it)
// =============================================================================
struct TileCopyToHIVM : OpRewritePattern<tile::CopyOp> {
  using OpRewritePattern<tile::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value src = op.getOperand(0);
    Value dst = op.getOperand(1);

    if (isTensorOfPointer(src.getType())) {
      lowerScatteredLoad(op, src, dst, rewriter);
      return success();
    }
    if (isTensorOfPointer(dst.getType())) {
      lowerScatteredStore(op, src, dst, rewriter);
      return success();
    }

    Value srcMem = getAsMemRef(src, rewriter);
    Value dstMem = getAsMemRef(dst, rewriter);
    if (isTritonPointerLike(src.getType())) {
      rewriter.replaceOpWithNewOp<memref::CopyOp>(op, srcMem, dstMem);
      return success();
    }
    rewriter.create<hivm::CopyOp>(op.getLoc(), TypeRange{}, srcMem, dstMem);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TileStoreTensorToHIVM : OpRewritePattern<tile::StoreTensorOp> {
  using OpRewritePattern<tile::StoreTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::StoreTensorOp op,
                                PatternRewriter &rewriter) const final {
    Value dstMem = getAsMemRef(op.getOperand(1), rewriter);
    Value srcMem = op.getOperand(0);
    if (!isa<MemRefType>(srcMem.getType())) {
      auto tensorTy = dyn_cast<RankedTensorType>(srcMem.getType());
      if (!tensorTy)
        return failure();
      auto memrefTy =
          MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
      srcMem = rewriter.create<bufferization::ToMemrefOp>(op.getLoc(), memrefTy,
                                                          srcMem);
#else
      srcMem = rewriter.create<bufferization::ToBufferOp>(op.getLoc(), memrefTy,
                                                          srcMem);
#endif
    }

    rewriter.create<hivm::CopyOp>(op.getLoc(), TypeRange{}, srcMem, dstMem);
    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// Step 4: tile.load → hivm.load
// =============================================================================
struct TileLoadToHIVM : OpRewritePattern<tile::LoadOp> {
  using OpRewritePattern<tile::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::LoadOp op,
                                PatternRewriter &rewriter) const final {
    // TensorView operand -> direct lowering (dense DMA or gather_scatter gather).
    if (auto viewTy = dyn_cast<tv::TensorViewType>(op.getSrc().getType())) {
      if (isViewLowerable(viewTy, op.getIndices().size()))
        return lowerViewLoad(op, rewriter);
      return failure();
    }
    auto resultTy = op.getResult().getType();
    if (auto t = dyn_cast<tile::TensorType>(resultTy)) {
      auto memrefTy = convertTensorToMemRef(t);
      rewriter.replaceOpWithNewOp<hivm::LoadOp>(op, memrefTy, op.getSrc());
      return success();
    }
    return failure();
  }
};

// =============================================================================
// Step 4: tile.store → hivm.store
// =============================================================================
struct TileStoreToHIVM : OpRewritePattern<tile::StoreOp> {
  using OpRewritePattern<tile::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::StoreOp op,
                                PatternRewriter &rewriter) const final {
    // TensorView destination -> direct lowering (dense DMA or gather_scatter scatter).
    if (auto viewTy = dyn_cast<tv::TensorViewType>(op.getDst().getType())) {
      if (isViewLowerable(viewTy, op.getIndices().size()))
        return lowerViewStore(op, rewriter);
      return failure();
    }
    if (!op.getIndices().empty())
      return failure();
    Value src = getAsMemRef(op.getSrc(), rewriter);
    rewriter.replaceOpWithNewOp<hivm::StoreOp>(op, Type(), src, op.getDst());
    return success();
  }
};

// =============================================================================
// Step 5: Sync ops
// =============================================================================
struct TileSetFlagToHIVM : OpRewritePattern<tile::SetFlagOp> {
  using OpRewritePattern<tile::SetFlagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::SetFlagOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::SetFlagOp>(
        op,
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getProducer()))),
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getConsumer()))),
        hivm::EventAttr::get(ctx,
                             mapEvent(static_cast<int64_t>(op.getEvent()))),
        Value());
    return success();
  }
};

struct TileWaitFlagToHIVM : OpRewritePattern<tile::WaitFlagOp> {
  using OpRewritePattern<tile::WaitFlagOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::WaitFlagOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::WaitFlagOp>(
        op,
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getProducer()))),
        hivm::PipeAttr::get(ctx,
                            mapPipe(static_cast<int64_t>(op.getConsumer()))),
        hivm::EventAttr::get(ctx,
                             mapEvent(static_cast<int64_t>(op.getEvent()))),
        Value());
    return success();
  }
};

struct TilePipeBarrierToHIVM : OpRewritePattern<tile::PipeBarrierOp> {
  using OpRewritePattern<tile::PipeBarrierOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::PipeBarrierOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::PipeBarrierOp>(
        op,
        hivm::PipeAttr::get(ctx, mapPipe(static_cast<int64_t>(op.getPipe()))));
    return success();
  }
};

struct TileCubeWaitToHIVM : OpRewritePattern<tile::CubeWaitOp> {
  using OpRewritePattern<tile::CubeWaitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::CubeWaitOp op,
                                PatternRewriter &rewriter) const final {
    auto *ctx = op->getContext();
    rewriter.replaceOpWithNewOp<hivm::SyncBlockWaitOp>(
        op, hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR),
        hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX),
        hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3),
        OpFoldResult(rewriter.getIndexAttr(0)));
    return success();
  }
};

struct TileGmOffsetToHIVM : OpRewritePattern<tile::GmOffsetOp> {
  using OpRewritePattern<tile::GmOffsetOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tile::GmOffsetOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value result = op.getBase();
    auto indices = op.getIndices();
    auto strides = op.getStrides();
    for (size_t i = 0; i < indices.size(); ++i) {
      Value off = rewriter.create<arith::MulIOp>(loc, indices[i], strides[i]);
      result = rewriter.create<arith::AddIOp>(loc, result, off);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// =============================================================================
// tile.concat
//     - 1D: tensor.empty + a sequence of tensor.insert_slice
//     - 2D+: tensor.concat
// =============================================================================
struct TileConcatToHIVM : OpRewritePattern<tile::ConcatOp> {
  using OpRewritePattern<tile::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::ConcatOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    int64_t dim = op.getDim();

    auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!resType || !lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");

    int64_t rank = resType.getRank();
    if (dim < 0 || dim >= rank)
      return rewriter.notifyMatchFailure(op, "concat dim out of range");

    // 2D+ tensors: lower directly to tensor.concat, which supports
    // concatenation along an arbitrary static dimension for any rank.
    if (rank >= 2) {
      rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, resType, dim,
                                                    ValueRange{lhs, rhs});
      return success();
    }

    // 1D tensors: build the result via tensor.empty + a sequence of
    // tensor.insert_slice, placing lhs then rhs back-to-back along `dim`.
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                    resType.getElementType());

    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    SmallVector<OpFoldResult> lhsSizes =
        llvm::map_to_vector(lhsType.getShape(), [&](int64_t t) {
          return OpFoldResult(rewriter.getIndexAttr(t));
        });
    SmallVector<OpFoldResult> rhsSizes =
        llvm::map_to_vector(rhsType.getShape(), [&](int64_t t) {
          return OpFoldResult(rewriter.getIndexAttr(t));
        });

    auto insertLhs = rewriter.create<tensor::InsertSliceOp>(
        loc, lhs, emptyOp, offsets, lhsSizes, strides);

    offsets[dim] = rewriter.getIndexAttr(lhsType.getShape()[dim]);
    auto insertRhs = rewriter.create<tensor::InsertSliceOp>(
        loc, rhs, insertLhs, offsets, rhsSizes, strides);

    rewriter.replaceOp(op, insertRhs.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern A: !tt.ptr<tensor<>> -> memref<>
//===----------------------------------------------------------------------===//
struct LowerPtrTensorCastToMemref
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOne(op))
      return failure();
    Value in = op.getInputs().front();
    auto ptrTy = dyn_cast<triton::PointerType>(in.getType());
    if (!ptrTy)
      return failure();
    auto tensorPointee = dyn_cast<RankedTensorType>(ptrTy.getPointeeType());
    if (!tensorPointee)
      return failure();
    auto outMemref = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (!outMemref)
      return failure();
    if (outMemref.getShape() != tensorPointee.getShape() ||
        outMemref.getElementType() != tensorPointee.getElementType())
      return failure();

    Location loc = op.getLoc();
    // %t = tt.load %ptr : !tt.ptr<tensor<...>> (tensor-pointer overload)
    Value loaded = rewriter.create<triton::LoadOp>(
        loc, in, /*boundaryCheck=*/ArrayRef<int32_t>{},
        /*padding=*/std::optional<triton::PaddingOption>(),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);
    // %m = bufferization.to_memref %t
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    Value asMemref =
        rewriter.create<bufferization::ToMemrefOp>(loc, outMemref, loaded);
#else
    Value asMemref =
        rewriter.create<bufferization::ToBufferOp>(loc, outMemref, loaded);
#endif
    rewriter.replaceOp(op, asMemref);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern B: memref<..., #space> -> tensor<...>
//===----------------------------------------------------------------------===//
struct LowerMemrefCastToTensor
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOne(op))
      return failure();
    Value in = op.getInputs().front();
    auto memrefTy = dyn_cast<MemRefType>(in.getType());
    if (!memrefTy)
      return failure();
    auto tensorOut = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorOut)
      return failure();
    if (memrefTy.getShape() != tensorOut.getShape() ||
        memrefTy.getElementType() != tensorOut.getElementType())
      return failure();

    Location loc = op.getLoc();
    // %t = bufferization.to_tensor %m restrict
    Value asTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorOut, in, /*restrict=*/true, /*writable=*/false);
    rewriter.replaceOp(op, asTensor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern C (utility): simplify chains of one-to-one unrealized casts whose
// outer source type equals the outer result type.  This catches cases like
//   %a = unrealized_conversion_cast %x : memref -> tensor
//   %b = unrealized_conversion_cast %a : tensor -> memref
// which arise after the linalg pass partially converts surrounding ops.
//===----------------------------------------------------------------------===//
struct FoldRoundtripCast : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isOneToOne(op))
      return failure();
    Value in = op.getInputs().front();
    auto prev = in.getDefiningOp<UnrealizedConversionCastOp>();
    if (!prev || !isOneToOne(prev))
      return failure();
    Value origin = prev.getInputs().front();
    if (origin.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, origin);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Rewrite pattern: fold staging alloc + copy1 + to_tensor + copy2
//===----------------------------------------------------------------------===//
struct FoldStagingCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value stage = copyOp.getSource();
    auto stageCast = stage.getDefiningOp<memref::MemorySpaceCastOp>();
    if (stageCast)
      stage = stageCast.getSource();
    Value dst = copyOp.getTarget();

    // ① Destination must have an explicit memory space (on-chip buffer).
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType || !dstType.getMemorySpace())
      return failure();

    // ② Source must be a staging memref.alloc in default/GM address space.
    auto stageAlloc = stage.getDefiningOp<memref::AllocOp>();
    if (!stageAlloc)
      return failure();
    auto stageType = stageAlloc.getType();
    if (auto stageSpace = stageType.getMemorySpace()) {
      auto stageAddr = dyn_cast<hivm::AddressSpaceAttr>(stageSpace);
      if (!stageAddr || stageAddr.getAddressSpace() != hivm::AddressSpace::GM)
        return failure();
    }

    // ③ Enumerate the source copy: memref.copy %src, %stage.
    //    Collect all users of %stage to verify the expected pattern.
    memref::CopyOp srcCopy;
    Operation *stageMark = nullptr;
    bufferization::ToTensorOp toTensor;
    Operation *tensorMark = nullptr;
    SmallVector<Operation *> otherUses;

    for (Operation *user : stage.getUsers()) {
      if (auto c = dyn_cast<memref::CopyOp>(user)) {
        if (c.getTarget() == stage) {
          if (srcCopy)
            return failure(); // multiple incoming copies — ambiguous
          srcCopy = c;
        } else if (c == copyOp.getOperation()) {
          continue; // the outgoing copy we are folding
        } else {
          otherUses.push_back(user);
        }
      } else if (isAnnotationMark(user)) {
        if (stageMark)
          otherUses.push_back(user);
        else
          stageMark = user;
      } else if (stageCast && user == stageCast.getOperation()) {
        continue;
      } else if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (toTensor)
          otherUses.push_back(user);
        else
          toTensor = tt;
      } else {
        otherUses.push_back(user);
      }
    }

    // srcCopy is mandatory; toTensor is optional (Flavour A has it, B doesn't).
    if (!srcCopy || !otherUses.empty())
      return failure();

    // Collect annotation.mark users of the tensor value (only if toTensor
    // exists — Flavour A path).
    if (toTensor) {
      Value tensorVal = toTensor.getResult();
      for (Operation *user : tensorVal.getUsers()) {
        if (isAnnotationMark(user) && !tensorMark)
          tensorMark = user;
      }
    }

    Location loc = copyOp.getLoc();
    Value src = castDefaultMemrefToGM(srcCopy.getSource(), loc, rewriter);
    Value dstCbuf = dst;

    // ④ Create the merged copy: memref.copy %src, %dst(#space)
    rewriter.create<memref::CopyOp>(loc, src, dstCbuf);

    // ⑤ If there was an annotation.mark on %stage, move it to %dst.
    if (stageMark) {
      OperationState state(loc, "annotation.mark");
      state.addOperands(dstCbuf);
      for (auto &namedAttr : stageMark->getAttrs())
        state.addAttribute(namedAttr.getName(), namedAttr.getValue());
      rewriter.create(state);
    }

    // ⑥ For Flavour A: drop the address space via memory_space_cast and
    //    re-create the to_tensor pointing at the cbuf alloc.
    //    For Flavour B: nothing to do — the tensor path never existed.
    if (toTensor) {
      auto genericType =
          MemRefType::get(dstType.getShape(), dstType.getElementType());
      Value cast =
          rewriter.create<memref::MemorySpaceCastOp>(loc, genericType, dstCbuf);
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          toTensor.getLoc(), toTensor.getType(), cast, toTensor.getRestrict(),
          toTensor.getWritable());

      // ⑦ If there was an annotation.mark on the tensor value, re-create it.
      if (tensorMark) {
        OperationState state(loc, "annotation.mark");
        state.addOperands(tensor);
        for (auto &namedAttr : tensorMark->getAttrs())
          state.addAttribute(namedAttr.getName(), namedAttr.getValue());
        rewriter.create(state);
      }

      rewriter.replaceOp(toTensor, tensor);
    }

    // ⑧ Erase the dead ops.
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(srcCopy);
    if (stageCast)
      rewriter.eraseOp(stageCast);
    if (stageMark)
      rewriter.eraseOp(stageMark);
    if (tensorMark)
      rewriter.eraseOp(tensorMark);

    return success();
  }
};

// =============================================================================
// tv.ptr signature legalization
// =============================================================================

// Convert a tv.ptr<T> kernel base pointer to memref<?xT, GM> (a ranked pointee
// keeps its shape; otherwise a dynamic 1-D memref). The GM address space is
// preserved so the downstream lowering sees a global-memory buffer.
static MemRefType convertTvPtrToMemRef(tv::PtrType ptrTy) {
  auto pointeeTy = ptrTy.getPointeeType();
  auto *ctx = pointeeTy.getContext();
  auto gmSpace = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);
  if (auto rankedTy = dyn_cast<RankedTensorType>(pointeeTy))
    return MemRefType::get(rankedTy.getShape(), rankedTy.getElementType(),
                           MemRefLayoutAttrInterface{}, gmSpace);
  return MemRefType::get({ShapedType::kDynamic}, pointeeTy,
                         MemRefLayoutAttrInterface{}, gmSpace);
}

// Rewrite every tv.ptr<T> kernel argument of \p func to memref<?xT, GM>. Each
// retyped entry-block argument is bridged back to its original tv.ptr type with
// an unrealized_conversion_cast so that ops still consuming the base pointer
// (make_tensor_view) keep verifying; the CommonIR-to-HIVM lowering recovers the
// memref through that cast. The base pointer materialized by
// convertToTensorViewPtr enters the pipeline as tv.ptr, so legalizing it here
// keeps the tv.ptr type from leaking into the lowering.
static void legalizeTvPtrSignature(triton::FuncOp func) {
  auto funcType = func.getFunctionType();
  bool changed = false;

  SmallVector<Type> newInputs;
  for (auto ty : funcType.getInputs()) {
    if (auto tvPtrTy = dyn_cast<tv::PtrType>(ty)) {
      newInputs.push_back(convertTvPtrToMemRef(tvPtrTy));
      changed = true;
    } else {
      newInputs.push_back(ty);
    }
  }

  if (!changed)
    return;

  func.setFunctionType(FunctionType::get(func.getContext(), newInputs,
                                         funcType.getResults()));

  if (func.empty())
    return;

  Block &entry = func.front();
  OpBuilder builder(&entry, entry.begin());
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    BlockArgument arg = entry.getArgument(i);
    auto tvPtrTy = dyn_cast<tv::PtrType>(arg.getType());
    if (!tvPtrTy)
      continue;
    arg.setType(convertTvPtrToMemRef(tvPtrTy));
    auto cast = builder.create<UnrealizedConversionCastOp>(func.getLoc(),
                                                           tvPtrTy, arg);
    arg.replaceAllUsesExcept(cast.getResult(0), cast);
  }
}

namespace {
struct CommonIRTvPtrLegalizePass
    : public mlir::triton::impl::CommonIRTvPtrLegalizeBase<
          CommonIRTvPtrLegalizePass> {
  void runOnOperation() override {
    getOperation().walk(
        [](triton::FuncOp func) { legalizeTvPtrSignature(func); });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCommonIRTvPtrLegalizePass() {
  return std::make_unique<CommonIRTvPtrLegalizePass>();
}

// =============================================================================
// Pass
// =============================================================================
namespace {
struct CommonIRToHIVMPass
    : public mlir::triton::impl::CommonIRToHIVMBase<CommonIRToHIVMPass> {
  void runOnOperation() override;
};
} // namespace

void CommonIRToHIVMPass::runOnOperation() {
  auto module = getOperation();

  // Use greedy rewrite to iteratively apply patterns to fixed point.
  // This avoids the type-conversion complexity of the dialect conversion
  // framework: tile.alloc → memref.alloc replaces !tile.buf with memref,
  // then tile.copy (now seeing memref operands) → hivm.copy, etc.

// apply pattern separately to ensure their relative order
// template lambda is a C++ 20 feature, so we'll have to use MACRO
#define APPLY_REWRITE_PATTERN(...)                                             \
  {                                                                            \
    RewritePatternSet patterns(&getContext());                                 \
    patterns.add<__VA_ARGS__>(&getContext());                                  \
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {   \
      return signalPassFailure();                                              \
    }                                                                          \
  }

  APPLY_REWRITE_PATTERN(TileAllocToMemRef);
  APPLY_REWRITE_PATTERN(TileSubviewToMemrefSubview);
  APPLY_REWRITE_PATTERN(TileToTensorEliminate);
  APPLY_REWRITE_PATTERN(TileCopyToHIVM);
  APPLY_REWRITE_PATTERN(TileStoreTensorToHIVM);
  APPLY_REWRITE_PATTERN(TileLoadToHIVM);
  APPLY_REWRITE_PATTERN(TileStoreToHIVM);
  APPLY_REWRITE_PATTERN(TileSetFlagToHIVM);
  APPLY_REWRITE_PATTERN(TileWaitFlagToHIVM);
  APPLY_REWRITE_PATTERN(TilePipeBarrierToHIVM);
  APPLY_REWRITE_PATTERN(TileCubeWaitToHIVM);
  APPLY_REWRITE_PATTERN(TileGmOffsetToHIVM);
  APPLY_REWRITE_PATTERN(TileConcatToHIVM);

  // Step N: After all tile ops are lowered, convert !tile.buf types remaining
  // in tt.func signatures and tt.call ops (these arise when tile.alloc results
  // are passed across function boundaries).
  module.walk([](triton::FuncOp funcOp) {
    auto funcType = funcOp.getFunctionType();
    bool changed = false;

    // Convert input types
    SmallVector<Type> newInputs;
    for (auto ty : funcType.getInputs()) {
      if (auto bufTy = dyn_cast<tile::BufType>(ty)) {
        newInputs.push_back(convertBufToMemRef(bufTy));
        changed = true;
      } else {
        newInputs.push_back(ty);
      }
    }

    // Convert result types
    SmallVector<Type> newResults;
    for (auto ty : funcType.getResults()) {
      if (auto bufTy = dyn_cast<tile::BufType>(ty)) {
        newResults.push_back(convertBufToMemRef(bufTy));
        changed = true;
      } else {
        newResults.push_back(ty);
      }
    }

    if (!changed)
      return;

    // Update function type
    auto newFuncType =
        FunctionType::get(funcOp.getContext(), newInputs, newResults);
    funcOp.setFunctionType(newFuncType);

    // Update block argument types
    if (!funcOp.empty()) {
      Block &entry = funcOp.front();
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        if (auto bufTy =
                dyn_cast<tile::BufType>(entry.getArgument(i).getType())) {
          entry.getArgument(i).setType(convertBufToMemRef(bufTy));
        }
      }
    }
  });

  // Update tt.call result types to match new function signatures.
  module.walk([](triton::CallOp callOp) {
    for (unsigned i = 0; i < callOp->getNumResults(); ++i) {
      if (auto bufTy =
              dyn_cast<tile::BufType>(callOp->getResult(i).getType())) {
        callOp->getResult(i).setType(convertBufToMemRef(bufTy));
      }
    }
  });

  APPLY_REWRITE_PATTERN(LowerPtrTensorCastToMemref);
  APPLY_REWRITE_PATTERN(LowerMemrefCastToTensor);
  APPLY_REWRITE_PATTERN(FoldRoundtripCast);

  // Finally, fold any residual A<->B unrealized cast chains.
  SmallVector<UnrealizedConversionCastOp> casts;
  module->walk([&](UnrealizedConversionCastOp c) { casts.push_back(c); });
  reconcileUnrealizedCasts(casts);

  APPLY_REWRITE_PATTERN(FoldStagingCopyPattern);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCommonIRToHIVMPass() {
  return std::make_unique<CommonIRToHIVMPass>();
}
