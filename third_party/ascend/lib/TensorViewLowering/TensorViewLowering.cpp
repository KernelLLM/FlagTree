//===- TensorViewLowering.cpp - tv access ops -> generic memref -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass B of the TensorView flow -- GENERIC lowering variant.  Lowers the tv
// access ops to *community* memref/bufferization/scf/tensor ops only (no HIVM
// dialect dependency), mirroring the native tt->linalg access form.  The
// backend (bishengir) does address-space marking and the memref.copy ->
// hir.load/store lowering, exactly as it does for the native path.
//
//   !tv.ptr<T> func input        -> memref<?xT>                 (plain, no space)
//   tv.view_load %pv[%i]         -> %gm  = reinterpret_cast %base off:[%off] sizes:[TILE] strides:[S]
//                                   %buf = alloc() : memref<TILExT>
//                                   memref.copy %gm, %buf       (bishengir -> DMA)
//                                   %t   = bufferization.to_tensor %buf
//   tv.view_store %pv[%i], %v    -> %gm  = reinterpret_cast %base_out ...
//                                   bufferization.materialize_in_destination %v in %gm
//   gather/scatter view          -> scalar scf.for over the tile: reinterpret
//                                   sizes:[1] + memref.load (gather) /
//                                   materialize_in_destination (scatter)
//   ptr_load/ptr_store           -> scalar scf.for (memref.load / materialize)
//
// No address space is committed and no hivm op is emitted -- placement (ub /
// L0A-L0B / cc) and DMA form are left entirely to the backend.  This decouples
// tv from AscendNPU-IR at the cost of the HIVM vgather/scatter_store fast paths
// (gather/scatter fall back to the portable scalar decomposition).
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TensorViewLowering/Passes.h"

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

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
#define GEN_PASS_DEF_TENSORVIEWLOWERING
#include "ascend/include/TensorViewLowering/Passes.h.inc"
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
  Value base;                        // flat memref<?xT> (plain, no address space)
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
                        Value off) {
  MLIRContext *ctx = b.getContext();
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/ShapedType::kDynamic,
                                       vi.strideStatic);
  auto gmTileTy = MemRefType::get(vi.tile, vi.elementType, layout);
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

static Value asIndex(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
}

/// reinterpret_cast %base to offset:[%ik] sizes:[1] strides:[1] (1-element view;
/// plain memref, address space assigned by the backend).
static Value emitScalarGmElem(OpBuilder &b, Location loc, Value base, Value ik,
                              Type elemTy) {
  auto layout = StridedLayoutAttr::get(b.getContext(),
                                       /*offset=*/ShapedType::kDynamic, {1});
  auto ty = MemRefType::get({1}, elemTy, layout);
  return b.create<memref::ReinterpretCastOp>(
      loc, ty, base, OpFoldResult(ik),
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

/// Per-dim block geometry for the "extracted load/store" form: the tile with the
/// sparse dim collapsed to 1 (the contiguous block DMA'd each iteration), the
/// data-dependent base offset, and the result subview offsets.
static void emitBlockGeometry(OpBuilder &b, Location loc, const ViewInfo &vi,
                              ValueRange indices, unsigned sparseDim, Value sIdx,
                              Value &off, SmallVectorImpl<OpFoldResult> &sizes,
                              SmallVectorImpl<OpFoldResult> &strides,
                              SmallVectorImpl<OpFoldResult> &subOffs) {
  for (unsigned d = 0; d < vi.rank; ++d) {
    int64_t blk = (d == sparseDim) ? 1 : vi.tile[d];
    Value logical =
        d == sparseDim
            ? sIdx
            : b.create<arith::MulIOp>(
                  loc, indices[d], b.create<arith::ConstantIndexOp>(loc, vi.tile[d]));
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    off = off ? b.create<arith::AddIOp>(loc, off, phys) : phys;
    sizes.push_back(b.getIndexAttr(blk));
    strides.push_back(vi.strideStatic[d] == ShapedType::kDynamic
                          ? OpFoldResult(vi.strideVal[d])
                          : OpFoldResult(b.getIndexAttr(vi.strideStatic[d])));
    subOffs.push_back(b.getIndexAttr(0)); // sparse offset overwritten by caller
  }
}

/// Structured gather in the native "extracted load" form: loop over the SPARSE
/// dim only; each iteration reinterprets a contiguous block (tile with the
/// sparse dim = 1) at the data-dependent offset and memref.copy's it into the
/// result buffer.  The {ExtractedLoadOrStore}/{DiscreteMemAccess} markers let
/// the backend recognize the discrete access (vector core + per-block DMA)
/// instead of a per-element scalar fallback.
static Value emitBlockGather(OpBuilder &b, Location loc, const ViewInfo &vi,
                             Value base, ValueRange indices, unsigned sparseDim,
                             RankedTensorType resultType) {
  auto bufTy = MemRefType::get(vi.tile, vi.elementType);
  Value buf = b.create<memref::AllocOp>(loc, bufTy);
  SmallVector<int64_t> blk(vi.tile.begin(), vi.tile.end());
  blk[sparseDim] = 1;
  auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                       vi.strideStatic);
  auto gmBlkTy = MemRefType::get(blk, vi.elementType, layout);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[sparseDim]);
  auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    auto ext = b.create<tensor::ExtractOp>(loc, indices[sparseDim], ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value sIdx = asIndex(b, loc, ext.getResult());

    Value off;
    SmallVector<OpFoldResult> sizes, strides, subOffs;
    emitBlockGeometry(b, loc, vi, indices, sparseDim, sIdx, off, sizes, strides,
                      subOffs);
    subOffs[sparseDim] = OpFoldResult(k);
    SmallVector<OpFoldResult> subStr(vi.rank, b.getIndexAttr(1));
    Value gm = b.create<memref::ReinterpretCastOp>(loc, gmBlkTy, base,
                                                   OpFoldResult(off), sizes, strides);
    Value sub = b.create<memref::SubViewOp>(loc, buf, subOffs, sizes, subStr);
    b.create<memref::CopyOp>(loc, gm, sub);
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
  return b.create<bufferization::ToTensorOp>(loc, resultType, buf,
                                             /*restrict=*/true, /*writable=*/false);
}

/// Structured scatter, symmetric to emitBlockGather: loop over the SPARSE dim;
/// each iteration materialize_in_destination's a contiguous value block into the
/// data-dependent GM block (store direction cannot use memref.copy).
static void emitBlockScatter(OpBuilder &b, Location loc, const ViewInfo &vi,
                             Value base, Value value, ValueRange indices,
                             unsigned sparseDim) {
  SmallVector<int64_t> blk(vi.tile.begin(), vi.tile.end());
  blk[sparseDim] = 1;
  auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                       vi.strideStatic);
  auto gmBlkTy = MemRefType::get(blk, vi.elementType, layout);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[sparseDim]);
  auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    auto ext = b.create<tensor::ExtractOp>(loc, indices[sparseDim], ValueRange{k});
    ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value sIdx = asIndex(b, loc, ext.getResult());

    Value off;
    SmallVector<OpFoldResult> sizes, strides, subOffs;
    emitBlockGeometry(b, loc, vi, indices, sparseDim, sIdx, off, sizes, strides,
                      subOffs);
    subOffs[sparseDim] = OpFoldResult(k);
    SmallVector<OpFoldResult> subStr(vi.rank, b.getIndexAttr(1));
    Value gm = b.create<memref::ReinterpretCastOp>(loc, gmBlkTy, base,
                                                   OpFoldResult(off), sizes, strides);
    Value slice =
        b.create<tensor::ExtractSliceOp>(loc, value, subOffs, sizes, subStr);
    auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, slice, gm);
    mat->setAttr("writable", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
}

/// The block form collapses the sparse dim to 1 and DMAs the contiguous run of
/// the remaining (regular) dims.  It is only valid when some regular dim carries
/// unit stride, i.e. the sparse dim is NOT the innermost contiguous dim.  When
/// the sparse dim itself is the unit-stride dim (a column gather/scatter) there
/// is no contiguous run: a collapsed [.,1] block strided by the row pitch is a
/// data-dependent single-column DMA whose base violates MTE alignment (vector
/// core exception).  Such cases must fall back to the per-element form.
static bool canBlockCopy(const ViewInfo &vi, unsigned sparseDim) {
  for (unsigned d = 0; d < vi.rank; ++d)
    if (d != sparseDim && vi.strideStatic[d] == 1)
      return true;
  return false;
}

/// Per-element gather: nested scf.for over the whole tile, scalar load of each
/// data-dependent element into the result tensor.  Used when the sparse dim is
/// the contiguous dim (no block DMA is expressible).
static Value emitElementwiseGather(OpBuilder &b, Location loc, const ViewInfo &vi,
                                   Value base, ValueRange indices,
                                   unsigned sparseDim,
                                   RankedTensorType resultType) {
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

  Value offset;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value logical;
    if (d == sparseDim) {
      auto ext = b.create<tensor::ExtractOp>(loc, indices[d], coords[d]);
      ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
      logical = asIndex(b, loc, ext.getResult());
    } else {
      Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
      Value origin = b.create<arith::MulIOp>(loc, indices[d], cTile);
      logical = b.create<arith::AddIOp>(loc, origin, coords[d]);
    }
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    offset = offset ? b.create<arith::AddIOp>(loc, offset, phys) : phys;
  }
  Value rc = emitScalarGmElem(b, loc, base, offset, vi.elementType);
  Value v = b.create<memref::LoadOp>(loc, rc, ValueRange{c0});
  Value target = loops.back().getRegionIterArg(0);
  Value result = b.create<tensor::InsertOp>(loc, v, target, coords);
  b.create<scf::YieldOp>(loc, result);
  b.setInsertionPointAfter(loops.front());
  return loops.front().getResult(0);
}

/// Per-element scatter, symmetric to emitElementwiseGather.
static void emitElementwiseScatter(OpBuilder &b, Location loc,
                                   const ViewInfo &vi, Value base, Value value,
                                   ValueRange indices, unsigned sparseDim) {
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  OpBuilder::InsertionGuard guard(b);
  SmallVector<Value> coords;
  scf::ForOp outer;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value upper = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, upper, c1);
    if (d == 0)
      outer = loop;
    b.setInsertionPointToStart(loop.getBody());
    coords.push_back(loop.getInductionVar());
  }

  Value offset;
  for (unsigned d = 0; d < vi.rank; ++d) {
    Value logical;
    if (d == sparseDim) {
      auto ext = b.create<tensor::ExtractOp>(loc, indices[d], coords[d]);
      ext->setAttr("DiscreteMemAccess", b.getUnitAttr());
      logical = asIndex(b, loc, ext.getResult());
    } else {
      Value cTile = b.create<arith::ConstantIndexOp>(loc, vi.tile[d]);
      Value origin = b.create<arith::MulIOp>(loc, indices[d], cTile);
      logical = b.create<arith::AddIOp>(loc, origin, coords[d]);
    }
    Value phys = b.create<arith::MulIOp>(loc, logical, vi.strideVal[d]);
    offset = offset ? b.create<arith::AddIOp>(loc, offset, phys) : phys;
  }
  auto elem = b.create<tensor::ExtractOp>(loc, value, coords);
  elem->setAttr("DiscreteMemAccess", b.getUnitAttr());
  Value rc = emitScalarGmElem(b, loc, base, offset, vi.elementType);
  Value empty =
      b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{1}, vi.elementType);
  Value ins =
      b.create<tensor::InsertOp>(loc, elem.getResult(), empty, ValueRange{c0});
  auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, ins, rc);
  mat->setAttr("writable", b.getUnitAttr());
  outer->setAttr("ExtractedLoadOrStore", b.getUnitAttr());
}

/// Dispatch: block DMA when the sparse dim is not the contiguous dim (native,
/// efficient), else per-element.
static Value emitScalarGather(OpBuilder &b, Location loc, const ViewInfo &vi,
                              Value base, ValueRange indices, unsigned sparseDim,
                              RankedTensorType resultType) {
  if (canBlockCopy(vi, sparseDim))
    return emitBlockGather(b, loc, vi, base, indices, sparseDim, resultType);
  return emitElementwiseGather(b, loc, vi, base, indices, sparseDim, resultType);
}

static void emitScalarScatter(OpBuilder &b, Location loc, const ViewInfo &vi,
                              Value base, Value value, ValueRange indices,
                              unsigned sparseDim) {
  if (canBlockCopy(vi, sparseDim))
    emitBlockScatter(b, loc, vi, base, value, indices, sparseDim);
  else
    emitElementwiseScatter(b, loc, vi, base, value, indices, sparseDim);
}

//===----------------------------------------------------------------------===//
// Lowering.  Unmasked tiles DMA the full block (this clean form is what the
// native cube path recognizes for tt.dot operands).  Masked tiles (any dim with
// a real extent) zero-pad the buffer and DMA only the in-bounds rank-N prefix.
//===----------------------------------------------------------------------===//

static LogicalResult lowerViewLoad(tv::ViewLoadOp load) {
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

    // Scalar-by-scalar gather straight from GM (reinterpret sizes:[1] +
    // memref.load per element).  No pre-loaded window -> no dynamic-shape alloc.
    Value result = emitScalarGather(b, loc, vi, vi.base, load.getIndices(),
                                    sparseDim, tensorTy);
    load.getResult().replaceAllUsesWith(result);
    load.erase();
    return success();
  }

  // On-chip staging buffer left UNTAGGED: memref.copy from the GM tile; the
  // backend assigns the buffer's real space (ub / L0A-L0B) by consumer.
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
    Value gm = emitGmTile(b, loc, vi, off);
    b.create<memref::CopyOp>(loc, gm, buf);
  } else {
    SmallVector<Value> lens;
    Value off = emitTailGeometry(b, loc, vi, load.getIndices(), lens);
    Value gm = emitGmTile(b, loc, vi, off);
    // Zero-pad the whole tile (padding = zero), then copy the in-bounds prefix.
    Value zero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(vi.elementType));
    b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{buf});
    Value gmSub = emitPrefixSubview(b, loc, gm, lens);
    Value bufSub = emitPrefixSubview(b, loc, buf, lens);
    b.create<memref::CopyOp>(loc, gmSub, bufSub);
  }

  Value t = b.create<bufferization::ToTensorOp>(loc, tensorTy, buf,
                                                /*restrict=*/true,
                                                /*writable=*/false);
  load.getResult().replaceAllUsesWith(t);
  load.erase();
  return success();
}

static LogicalResult lowerViewStore(tv::ViewStoreOp store) {
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

    // Scalar scatter (portable form; no hivm.scatter_store).
    emitScalarScatter(b, loc, vi, vi.base, store.getValue(),
                      store.getIndices(), sparseDim);
    store.erase();
    return success();
  }

  // Store the tensor into the GM tile with bufferization.materialize_in_
  // destination (NOT memref.copy: the store-direction memref.copy(local->gm) is
  // mis-lowered by the backend; materialize is the form the native path uses).
  Value value = store.getValue();

  bool masked = false;
  for (Value n : vi.n)
    if (!isSentinelExtent(n)) {
      masked = true;
      break;
    }

  if (!masked) {
    Value off = computeOffset(b, loc, vi, store.getIndices());
    Value gm = emitGmTile(b, loc, vi, off);
    auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, value, gm);
    mat->setAttr("writable", b.getUnitAttr());
  } else {
    SmallVector<Value> lens;
    Value off = emitTailGeometry(b, loc, vi, store.getIndices(), lens);
    Value gm = emitGmTile(b, loc, vi, off);
    Value gmSub = emitPrefixSubview(b, loc, gm, lens);
    // Store only the in-bounds prefix: extract_slice(value)[0:len] -> subview.
    SmallVector<OpFoldResult> offs, szs, strs;
    for (Value len : lens) {
      offs.push_back(b.getIndexAttr(0));
      szs.push_back(OpFoldResult(len));
      strs.push_back(b.getIndexAttr(1));
    }
    Value slice = b.create<tensor::ExtractSliceOp>(loc, value, offs, szs, strs);
    auto mat =
        b.create<bufferization::MaterializeInDestinationOp>(loc, slice, gmSub);
    mat->setAttr("writable", b.getUnitAttr());
  }

  store.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Discrete gather/scatter: ptr_load / ptr_store -> scalar scf.for loop
// (reinterpret_cast base offset:[idx[k]] sizes:[1] + memref.load / materialize),
// the same portable form the native discrete-access lowering produces.  The base
// is a scalar !tv.ptr (func arg, already rewritten to a plain memref); the gather
// index comes from the op's index operand.
//===----------------------------------------------------------------------===//

static LogicalResult lowerPtrLoad(tv::PtrLoadOp op) {
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
    Value rc = emitScalarGmElem(b, loc, base, ext.getResult(), elemTy);
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

static LogicalResult lowerPtrStore(tv::PtrStoreOp op) {
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
    // Discrete GM write must use the native tensor-materialize form (a raw
    // memref.store to a #gm reinterpret is NOT lowered by the backend): extract
    // idx[k]/value[k] from the tensors (DiscreteMemAccess), wrap the scalar in a
    // tensor<1>, and materialize_in_destination into the 1-element GM slot.
    auto extIdx = b.create<tensor::ExtractOp>(loc, idx, ValueRange{k});
    extIdx->setAttr("DiscreteMemAccess", b.getUnitAttr());
    auto extVal = b.create<tensor::ExtractOp>(loc, value, ValueRange{k});
    extVal->setAttr("DiscreteMemAccess", b.getUnitAttr());
    Value rc = emitScalarGmElem(b, loc, base, extIdx.getResult(), elemTy);
    Value empty = b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{1}, elemTy);
    Value ins =
        b.create<tensor::InsertOp>(loc, extVal.getResult(), empty, ValueRange{c0});
    auto mat = b.create<bufferization::MaterializeInDestinationOp>(loc, ins, rc);
    mat->setAttr("writable", b.getUnitAttr());
  }
  loop->setAttr("ExtractedLoadOrStore", b.getUnitAttr());

  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Function-argument rewrite: !tv.ptr<T> -> plain memref<?xT> (backend marks GM)
//===----------------------------------------------------------------------===//

static void rewriteFuncPtrArgs(triton::FuncOp func) {
  auto ptrToMemref = [&](Type t) -> Type {
    if (auto p = dyn_cast<tv::PtrType>(t))
      return MemRefType::get({ShapedType::kDynamic}, p.getPointeeType());
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

struct TensorViewLoweringPass
    : public mlir::triton::impl::TensorViewLoweringBase<TensorViewLoweringPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 1. Rewrite !tv.ptr function inputs to plain memrefs.
    module.walk([&](triton::FuncOp func) { rewriteFuncPtrArgs(func); });

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
      if (failed(lowerViewLoad(l)))
        l.emitError("TensorViewLowering: unsupported view_load");
    for (tv::ViewStoreOp s : stores)
      if (failed(lowerViewStore(s)))
        s.emitError("TensorViewLowering: unsupported view_store");
    for (tv::PtrLoadOp pl : ptrLoads)
      if (failed(lowerPtrLoad(pl)))
        pl.emitError("TensorViewLowering: unsupported ptr_load");
    for (tv::PtrStoreOp ps : ptrStores)
      if (failed(lowerPtrStore(ps)))
        ps.emitError("TensorViewLowering: unsupported ptr_store");

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
mlir::triton::createTensorViewLoweringPass() {
  return std::make_unique<TensorViewLoweringPass>();
}
