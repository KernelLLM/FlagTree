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
// Supported access families include regular partition/strided tiles and the
// 2-D single-sparse-dimension gather/scatter form recognized below.  More
// general pointer tensors and mask lowering remain unsupported.
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonToTensorView/Passes.h"

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
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

/// A recognized tiled access (rank-generic).  Per dim: tile size, traversal
/// stride (== tile for partition, else strided), element stride (static value
/// or ShapedType::kDynamic + an SSA i32), and the tile index.  Dead
/// pointer-chain ops are cleaned by a post-pass sweep rather than tracked here.
struct Access {
  Value basePtr;                 // scalar !tv.ptr<T>
  Type elementType;              // T
  unsigned rank = 0;
  SmallVector<int64_t> tile;
  SmallVector<int64_t> traversal;
  SmallVector<int64_t> stride;    // element stride, or ShapedType::kDynamic
  SmallVector<Value> strideDyn;   // dynamic element stride (i32) if kDynamic, else null
  SmallVector<Value> index;       // scalar tile index, or tensor index for a sparse dim
  SmallVector<int64_t> sparseDims;
  Value fullSize;                 // 1-D masked bound (i32), else null
};

static std::optional<int64_t> getConstIntValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// A uniform integer tensor constant: `tt.splat` of a scalar constant, or an
/// `arith.constant dense<C>` splat.
static std::optional<int64_t> getSplatConstIntValue(Value v) {
  if (auto s = v.getDefiningOp<triton::SplatOp>())
    return getConstIntValue(s.getSrc());
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<DenseIntElementsAttr>(c.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}

/// Recognize a per-dim logical index tensor:
///   addi(splat(muli(pid, STEP)), makeRange(0, TILE))
/// yielding TILE, the traversal stride STEP, and the tile index (pid).  A bare
/// makeRange (no pid origin, e.g. a matmul reduction dim tiled by a single tile)
/// yields a null index, meaning a constant-0 tile position.
static bool matchLogicalIndex(Value idxTensor, int64_t &tile, int64_t &traversal,
                              Value &index) {
  if (auto mr = idxTensor.getDefiningOp<triton::MakeRangeOp>()) {
    if (mr.getStart() != 0)
      return false;
    tile = static_cast<int64_t>(mr.getEnd());
    traversal = tile;
    index = Value(); // origin 0
    return true;
  }
  auto addi = idxTensor.getDefiningOp<arith::AddIOp>();
  if (!addi)
    return false;
  triton::MakeRangeOp range;
  Value originSplatSrc;
  for (Value op : {addi.getLhs(), addi.getRhs()}) {
    if (auto m = op.getDefiningOp<triton::MakeRangeOp>())
      range = m;
    else if (auto s = op.getDefiningOp<triton::SplatOp>())
      originSplatSrc = s.getSrc();
  }
  if (!range || range.getStart() != 0 || !originSplatSrc)
    return false;
  tile = static_cast<int64_t>(range.getEnd());
  auto muli = originSplatSrc.getDefiningOp<arith::MulIOp>();
  if (!muli)
    return false;
  if (auto c = getConstIntValue(muli.getLhs())) {
    traversal = *c;
    index = muli.getRhs();
    return true;
  }
  if (auto c = getConstIntValue(muli.getRhs())) {
    traversal = *c;
    index = muli.getLhs();
    return true;
  }
  return false;
}

/// Match a 1-D tiled access.  The tile-origin scalar (`pid * STEP`) may appear
/// non-hoisted (inside the offset tensor) or hoisted (a scalar addptr on the
/// base); the whole offset may be scaled by a constant element stride.
static bool match1D(Value ptrTensor, Value maskVal, Access &out) {
  auto addptr = ptrTensor.getDefiningOp<triton::AddPtrOp>();
  if (!addptr)
    return false;
  auto splat = addptr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return false;

  Value scalarPtr = splat.getSrc();
  Value originScalar; // i32 = pid*STEP from the hoisted scalar addptr (or null)
  if (auto scalarAp = scalarPtr.getDefiningOp<triton::AddPtrOp>()) {
    originScalar = scalarAp.getOffset();
    scalarPtr = scalarAp.getPtr();
  }
  auto tvPtr = dyn_cast<tv::PtrType>(scalarPtr.getType());
  if (!tvPtr)
    return false;

  int64_t elementStride = 1;
  Value tensorOff = addptr.getOffset();
  if (auto emul = tensorOff.getDefiningOp<arith::MulIOp>()) {
    if (auto c = getSplatConstIntValue(emul.getLhs())) {
      elementStride = *c;
      tensorOff = emul.getRhs();
    } else if (auto c = getSplatConstIntValue(emul.getRhs())) {
      elementStride = *c;
      tensorOff = emul.getLhs();
    }
  }

  triton::MakeRangeOp range;
  Value originFromOffset;
  if (auto mr = tensorOff.getDefiningOp<triton::MakeRangeOp>()) {
    range = mr;
  } else if (auto addi = tensorOff.getDefiningOp<arith::AddIOp>()) {
    for (Value op : {addi.getLhs(), addi.getRhs()}) {
      if (auto m = op.getDefiningOp<triton::MakeRangeOp>())
        range = m;
      else if (auto s = op.getDefiningOp<triton::SplatOp>())
        originFromOffset = s.getSrc();
    }
  }
  if (!range || range.getStart() != 0)
    return false;
  int64_t tile =
      static_cast<int64_t>(range.getEnd()) - static_cast<int64_t>(range.getStart());

  Value origin = originScalar ? originScalar : originFromOffset;
  if (!origin)
    return false;
  auto muli = origin.getDefiningOp<arith::MulIOp>();
  if (!muli)
    return false;
  int64_t traversal;
  Value index;
  if (auto c = getConstIntValue(muli.getLhs())) {
    traversal = *c;
    index = muli.getRhs();
  } else if (auto c = getConstIntValue(muli.getRhs())) {
    traversal = *c;
    index = muli.getLhs();
  } else {
    return false;
  }

  Value fullSize;
  if (maskVal) {
    if (auto cmp = maskVal.getDefiningOp<arith::CmpIOp>()) {
      for (Value op : {cmp.getLhs(), cmp.getRhs()})
        if (auto s = op.getDefiningOp<triton::SplatOp>())
          fullSize = s.getSrc();
    }
  }

  out.basePtr = scalarPtr;
  out.elementType = tvPtr.getPointeeType();
  out.rank = 1;
  out.tile = {tile};
  out.traversal = {traversal};
  out.stride = {elementStride};
  out.strideDyn = {Value()};
  out.index = {index};
  out.fullSize = fullSize;
  return true;
}

/// Match a 2-D block access:
///   addptr(broadcast(addptr(splat(base), muli(expand_dims(rowIdx,1), splat(rowStride)))),
///          broadcast(expand_dims(colIdx, 0)))
/// where rowStride is a runtime value (dynamic element stride) or a constant, and
/// col stride is 1.  (Block-aligned; no mask handled here yet.)
static bool match2D(Value ptrTensor, Access &out) {
  auto outer = ptrTensor.getDefiningOp<triton::AddPtrOp>();
  if (!outer)
    return false;
  auto bcPtr = outer.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto bcOff = outer.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!bcPtr || !bcOff)
    return false;

  // last-dim (col) offset: broadcast(expand_dims(colIdx, axis=0)), stride 1.
  auto colExp = bcOff.getSrc().getDefiningOp<triton::ExpandDimsOp>();
  if (!colExp)
    return false;
  Value colIdxTensor = colExp.getSrc();

  // first-dim (row) ptr: broadcast(addptr(splat(base), rowContrib)).
  auto innerAp = bcPtr.getSrc().getDefiningOp<triton::AddPtrOp>();
  if (!innerAp)
    return false;
  auto splatBase = innerAp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splatBase)
    return false;
  auto tvPtr = dyn_cast<tv::PtrType>(splatBase.getSrc().getType());
  if (!tvPtr)
    return false;

  // rowContrib = muli(expand_dims(rowIdx,1), splat(rowStride))  OR  expand_dims(rowIdx,1).
  Value rowContrib = innerAp.getOffset();
  int64_t rowStride = 1;
  Value rowStrideDyn;
  Value rowExp;
  if (auto mul = rowContrib.getDefiningOp<arith::MulIOp>()) {
    for (Value op : {mul.getLhs(), mul.getRhs()}) {
      if (op.getDefiningOp<triton::ExpandDimsOp>()) {
        rowExp = op;
      } else if (auto c = getSplatConstIntValue(op)) {
        rowStride = *c;
      } else if (auto s = op.getDefiningOp<triton::SplatOp>()) {
        rowStrideDyn = s.getSrc();
        rowStride = ShapedType::kDynamic;
      }
    }
  } else {
    rowExp = rowContrib;
  }
  if (!rowExp)
    return false;
  auto rowExpDims = rowExp.getDefiningOp<triton::ExpandDimsOp>();
  if (!rowExpDims)
    return false;
  Value rowIdxTensor = rowExpDims.getSrc();

  int64_t t0, tr0, t1, tr1;
  Value i0, i1;
  if (!matchLogicalIndex(rowIdxTensor, t0, tr0, i0))
    return false;
  if (!matchLogicalIndex(colIdxTensor, t1, tr1, i1))
    return false;

  out.basePtr = splatBase.getSrc();
  out.elementType = tvPtr.getPointeeType();
  out.rank = 2;
  out.tile = {t0, t1};
  out.traversal = {tr0, tr1};
  out.stride = {rowStride, 1};
  out.strideDyn = {rowStrideDyn, Value()};
  out.index = {i0, i1};
  out.fullSize = Value();
  return true;
}

/// Match a 2-D gather/scatter access with one data-dependent dimension and one
/// regular dimension:
///
///   broadcast(addptr(splat(base), expand_dims(rowIndex, 1) * rowStride))
///     + broadcast(expand_dims(colIndex, 0))
///
/// The data-dependent logical index is retained as the runtime tensor index.
/// This matcher deliberately follows the regular 2-D matcher and therefore
/// cannot steal partition/strided accesses from the existing path.
static bool matchGatherScatter2D(Value ptrTensor, Access &out) {
  auto outer = ptrTensor.getDefiningOp<triton::AddPtrOp>();
  if (!outer)
    return false;
  auto bcPtr = outer.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto bcOff = outer.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!bcPtr || !bcOff)
    return false;

  auto colExp = bcOff.getSrc().getDefiningOp<triton::ExpandDimsOp>();
  if (!colExp)
    return false;

  auto innerAp = bcPtr.getSrc().getDefiningOp<triton::AddPtrOp>();
  if (!innerAp)
    return false;
  auto splatBase = innerAp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splatBase)
    return false;
  auto tvPtr = dyn_cast<tv::PtrType>(splatBase.getSrc().getType());
  if (!tvPtr)
    return false;

  Value rowContrib = innerAp.getOffset();
  int64_t rowStride = 1;
  Value rowStrideDyn;
  Value rowExp;
  if (auto mul = rowContrib.getDefiningOp<arith::MulIOp>()) {
    for (Value operand : {mul.getLhs(), mul.getRhs()}) {
      if (operand.getDefiningOp<triton::ExpandDimsOp>()) {
        rowExp = operand;
      } else if (auto c = getSplatConstIntValue(operand)) {
        rowStride = *c;
      } else if (auto s = operand.getDefiningOp<triton::SplatOp>()) {
        rowStrideDyn = s.getSrc();
        rowStride = ShapedType::kDynamic;
      }
    }
  } else {
    rowExp = rowContrib;
  }
  if (!rowExp)
    return false;
  auto rowExpand = rowExp.getDefiningOp<triton::ExpandDimsOp>();
  if (!rowExpand)
    return false;
  Value rowIndex = rowExpand.getSrc();
  Value colIndexTensor = colExp.getSrc();

  int64_t rowTile, rowTraversal, colTile, colTraversal;
  Value rowRegularIndex, colRegularIndex;
  bool rowIsRegular = matchLogicalIndex(rowIndex, rowTile, rowTraversal,
                                        rowRegularIndex);
  bool colIsRegular = matchLogicalIndex(colIndexTensor, colTile, colTraversal,
                                        colRegularIndex);
  if (rowIsRegular == colIsRegular)
    return false;

  auto getSparseTile = [](Value index) -> std::optional<int64_t> {
    auto ty = dyn_cast<RankedTensorType>(index.getType());
    if (!ty || ty.getRank() != 1 ||
        (!ty.getElementType().isIndex() &&
         !ty.getElementType().isInteger(32) &&
         !ty.getElementType().isInteger(64)) ||
        ShapedType::isDynamic(ty.getShape()[0]))
      return std::nullopt;
    return ty.getShape()[0];
  };

  SmallVector<int64_t> sparseDims;
  Value rowViewIndex, colViewIndex;
  if (rowIsRegular) {
    auto sparseTile = getSparseTile(colIndexTensor);
    if (!sparseTile)
      return false;
    colTile = *sparseTile;
    colTraversal = colTile;
    rowViewIndex = rowRegularIndex;
    colViewIndex = colIndexTensor;
    sparseDims.push_back(1);
  } else {
    auto sparseTile = getSparseTile(rowIndex);
    if (!sparseTile)
      return false;
    rowTile = *sparseTile;
    rowTraversal = rowTile;
    rowViewIndex = rowIndex;
    colViewIndex = colRegularIndex;
    sparseDims.push_back(0);
  }

  out.basePtr = splatBase.getSrc();
  out.elementType = tvPtr.getPointeeType();
  out.rank = 2;
  out.tile = {rowTile, colTile};
  // A sparse dimension does not have a static inter-tile traversal.  Its value
  // is unused for gather/scatter view construction and is set to the tile size.
  out.traversal = {rowTraversal, colTraversal};
  out.stride = {rowStride, 1};
  out.strideDyn = {rowStrideDyn, Value()};
  out.index = {rowViewIndex, colViewIndex};
  out.sparseDims = sparseDims;
  out.fullSize = Value();
  return true;
}

static bool matchAccess(Value ptrTensor, Value maskVal, Access &out) {
  if (match2D(ptrTensor, out))
    return true;
  if (match1D(ptrTensor, maskVal, out))
    return true;
  return matchGatherScatter2D(ptrTensor, out);
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Build make_tensor_view plus the encoded view (rank-N).  A non-empty sparse
/// dimension list selects gather/scatter; otherwise partition when traversal ==
/// tile in every dim, and strided for the remaining regular accesses.
static Value buildView(OpBuilder &b, Location loc, const Access &a) {
  MLIRContext *ctx = b.getContext();
  unsigned rank = a.rank;

  // Per-dim element strides: SSA operand + type stride (static or kDynamic).
  SmallVector<Value> strideOperands;
  SmallVector<int64_t> strideTy;
  for (unsigned d = 0; d < rank; ++d) {
    if (a.stride[d] == ShapedType::kDynamic) {
      strideOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.strideDyn[d]));
      strideTy.push_back(ShapedType::kDynamic);
    } else {
      strideOperands.push_back(
          b.create<arith::ConstantIndexOp>(loc, a.stride[d]));
      strideTy.push_back(a.stride[d]);
    }
  }

  // Per-dim extents: the masked bound (1-D) or a large sentinel (folds to a full
  // tile in Pass B for unmasked accesses).
  SmallVector<Value> sizeOperands;
  for (unsigned d = 0; d < rank; ++d) {
    if (rank == 1 && a.fullSize)
      sizeOperands.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), a.fullSize));
    else
      sizeOperands.push_back(b.create<arith::ConstantIndexOp>(
          loc, std::numeric_limits<int64_t>::max()));
  }

  SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
  auto baseTy = tv::TensorViewType::get(dynShape, a.elementType, strideTy,
                                        /*encoding=*/Attribute());
  auto baseView = b.create<tv::MakeTensorViewOp>(loc, baseTy, a.basePtr,
                                                 sizeOperands, strideOperands);

  if (!a.sparseDims.empty()) {
    Attribute enc = tv::GatherScatterViewAttr::get(ctx, a.tile, a.sparseDims);
    auto viewTy =
        tv::TensorViewType::get(dynShape, a.elementType, strideTy, enc);
    return b
        .create<tv::MakeGatherScatterViewOp>(loc, viewTy, baseView.getResult())
        .getResult();
  }

  bool isPartition = true;
  for (unsigned d = 0; d < rank; ++d)
    if (a.traversal[d] != a.tile[d])
      isPartition = false;

  Attribute enc;
  if (isPartition)
    enc = tv::PartitionViewAttr::get(ctx, a.tile);
  else
    enc = tv::StridedViewAttr::get(ctx, a.tile, a.traversal);
  auto viewTy = tv::TensorViewType::get(dynShape, a.elementType, strideTy, enc);
  if (isPartition)
    return b.create<tv::MakePartitionViewOp>(loc, viewTy, baseView.getResult())
        .getResult();
  return b.create<tv::MakeStridedViewOp>(loc, viewTy, baseView.getResult())
      .getResult();
}

static SmallVector<Value> castIndices(OpBuilder &b, Location loc,
                                      const Access &a) {
  SmallVector<Value> indices;
  for (auto [d, idx] : llvm::enumerate(a.index)) {
    if (llvm::is_contained(a.sparseDims, d)) {
      indices.push_back(idx);
      continue;
    }
    if (idx)
      indices.push_back(
          b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx));
    else
      indices.push_back(b.create<arith::ConstantIndexOp>(loc, 0)); // origin 0
  }
  return indices;
}

static LogicalResult rewriteLoad(triton::LoadOp load) {
  Access a;
  if (!matchAccess(load.getPtr(), load.getMask(), a))
    return failure();

  OpBuilder b(load);
  Location loc = load.getLoc();
  Value view = buildView(b, loc, a);
  auto viewLoad = b.create<tv::ViewLoadOp>(loc, load.getResult().getType(), view,
                                           castIndices(b, loc, a),
                                           /*mask=*/Value());
  load.getResult().replaceAllUsesWith(viewLoad.getResult());
  load.erase();
  return success();
}

static LogicalResult rewriteStore(triton::StoreOp store) {
  Access a;
  if (!matchAccess(store.getPtr(), store.getMask(), a))
    return failure();

  OpBuilder b(store);
  Location loc = store.getLoc();
  Value view = buildView(b, loc, a);
  b.create<tv::ViewStoreOp>(loc, view, store.getValue(), castIndices(b, loc, a),
                            /*mask=*/Value());
  store.erase();
  return success();
}

/// Fixed-point erase of the now-dead pointer-chain ops.  Handles both the 1-D
/// (addptr/splat) and 2-D (broadcast/expand_dims) forms, including the ops left
/// type-inconsistent by the !tv.ptr argument rewrite.
static void eraseDeadPtrOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if ((isa<triton::AddPtrOp>(op) || isa<triton::SplatOp>(op) ||
           isa<triton::BroadcastOp>(op) || isa<triton::ExpandDimsOp>(op)) &&
          op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }
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

    // 3. Clean up the now-dead pointer-chain ops.
    eraseDeadPtrOps(module);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToTensorViewPass() {
  return std::make_unique<TritonToTensorViewPass>();
}
