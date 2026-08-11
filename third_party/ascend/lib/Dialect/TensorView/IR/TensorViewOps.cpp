//===- TensorViewOps.cpp - TensorView operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::tv;

//===----------------------------------------------------------------------===//
// ViewOpInterface
//===----------------------------------------------------------------------===//
#include "ascend/include/Dialect/TensorView/IR/TensorViewInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "ascend/include/Dialect/TensorView/IR/TensorViewOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

// Verify that `res` matches `src` in everything but the encoding, and that
// `src` itself is a base (encoding-free) view.
static LogicalResult verifyEncodingParent(Operation *op, TensorViewType src,
                                          TensorViewType res) {
  if (src.getEncoding())
    return op->emitOpError("source must be a base view (no encoding)");
  if (src.getShape() != res.getShape() || src.getStrides() != res.getStrides() ||
      src.getElementType() != res.getElementType())
    return op->emitOpError("result must match source except for the encoding");
  return success();
}

//===----------------------------------------------------------------------===//
// make_tensor_view
//===----------------------------------------------------------------------===//

LogicalResult MakeTensorViewOp::verify() {
  auto resTy = cast<TensorViewType>(getResult().getType());
  if (resTy.getEncoding())
    return emitOpError("result must be a base view (no encoding)");
  int64_t rank = resTy.getRank();
  if (static_cast<int64_t>(getSizes().size()) != rank)
    return emitOpError("expected ") << rank << " size operands but got "
                                    << getSizes().size();
  if (static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("expected ") << rank << " stride operands but got "
                                    << getStrides().size();
  auto srcTy = cast<PtrType>(getSource().getType());
  if (resTy.getElementType() != srcTy.getPointeeType())
    return emitOpError("result element type must match source pointee type");
  return success();
}

//===----------------------------------------------------------------------===//
// make_partition_view / make_strided_view / make_gather_scatter_view
//===----------------------------------------------------------------------===//

TensorViewType MakePartitionViewOp::getEncodedView() {
  return cast<TensorViewType>(getResult().getType());
}
TensorViewType MakeStridedViewOp::getEncodedView() {
  return cast<TensorViewType>(getResult().getType());
}
TensorViewType MakeGatherScatterViewOp::getEncodedView() {
  return cast<TensorViewType>(getResult().getType());
}

LogicalResult MakePartitionViewOp::verify() {
  auto src = cast<TensorViewType>(getSource().getType());
  auto res = cast<TensorViewType>(getResult().getType());
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<PartitionViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.partition_view encoding");
  int64_t rank = res.getRank();
  for (int64_t d : enc.getDimMap())
    if (d < 0 || d >= rank)
      return emitOpError("dim_map entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}

LogicalResult MakeStridedViewOp::verify() {
  auto src = cast<TensorViewType>(getSource().getType());
  auto res = cast<TensorViewType>(getResult().getType());
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<StridedViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.strided_view encoding");
  int64_t rank = res.getRank();
  for (int64_t d : enc.getDimMap())
    if (d < 0 || d >= rank)
      return emitOpError("dim_map entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}

LogicalResult MakeGatherScatterViewOp::verify() {
  auto src = cast<TensorViewType>(getSource().getType());
  auto res = cast<TensorViewType>(getResult().getType());
  if (failed(verifyEncodingParent(*this, src, res)))
    return failure();
  auto enc = dyn_cast_or_null<GatherScatterViewAttr>(res.getEncoding());
  if (!enc)
    return emitOpError("result must carry a #tv.gather_scatter_view encoding");
  int64_t rank = res.getRank();
  for (int64_t d : enc.getSparseDim())
    if (d < 0 || d >= rank)
      return emitOpError("sparse_dim entry ")
             << d << " out of range [0, " << rank << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// view_load / view_store
//===----------------------------------------------------------------------===//

TensorViewType ViewLoadOp::getEncodedView() {
  return cast<TensorViewType>(getView().getType());
}
TensorViewType ViewStoreOp::getEncodedView() {
  return cast<TensorViewType>(getView().getType());
}

LogicalResult ViewLoadOp::verify() {
  auto view = cast<TensorViewType>(getView().getType());
  Attribute enc = view.getEncoding();
  if (!isViewEncoding(enc))
    return emitOpError("view operand must carry a view encoding");
  if (static_cast<int64_t>(getIndices().size()) != getEncodingIndexSpaceRank(enc))
    return emitOpError("expected ")
           << getEncodingIndexSpaceRank(enc) << " indices but got "
           << getIndices().size();
  auto resTy = cast<RankedTensorType>(getResult().getType());
  SmallVector<int64_t> tile = getEncodingTileShape(enc);
  if (resTy.getShape() != ArrayRef<int64_t>(tile))
    return emitOpError("result shape must equal the view tile shape");
  if (resTy.getElementType() != view.getElementType())
    return emitOpError("result element type must match the view element type");
  if (Value mask = getMask()) {
    auto maskTy = cast<RankedTensorType>(mask.getType());
    if (maskTy.getShape() != resTy.getShape())
      return emitOpError("mask shape must equal result shape");
  }
  return success();
}

LogicalResult ViewStoreOp::verify() {
  auto view = cast<TensorViewType>(getView().getType());
  Attribute enc = view.getEncoding();
  if (!isViewEncoding(enc))
    return emitOpError("view operand must carry a view encoding");
  if (static_cast<int64_t>(getIndices().size()) != getEncodingIndexSpaceRank(enc))
    return emitOpError("expected ")
           << getEncodingIndexSpaceRank(enc) << " indices but got "
           << getIndices().size();
  auto valTy = cast<RankedTensorType>(getValue().getType());
  SmallVector<int64_t> tile = getEncodingTileShape(enc);
  if (valTy.getShape() != ArrayRef<int64_t>(tile))
    return emitOpError("value shape must equal the view tile shape");
  if (valTy.getElementType() != view.getElementType())
    return emitOpError("value element type must match the view element type");
  if (Value mask = getMask()) {
    auto maskTy = cast<RankedTensorType>(mask.getType());
    if (maskTy.getShape() != valTy.getShape())
      return emitOpError("mask shape must equal value shape");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ptr_load / ptr_store
//===----------------------------------------------------------------------===//

// Verify a rank-1, equal-length family of index tensors; returns the common
// length in `len` (or ShapedType::kDynamic when unconstrained).
static LogicalResult verifyIndexTensors(Operation *op, OperandRange indices,
                                        int64_t &len) {
  if (indices.empty())
    return op->emitOpError("expected at least one index tensor");
  len = ShapedType::kDynamic;
  for (Value idx : indices) {
    auto it = cast<RankedTensorType>(idx.getType());
    if (it.getRank() != 1)
      return op->emitOpError("index tensors must be rank-1");
    int64_t l = it.getShape()[0];
    if (ShapedType::isDynamic(len))
      len = l;
    else if (!ShapedType::isDynamic(l) && l != len)
      return op->emitOpError("index tensors must have equal length");
  }
  return success();
}

LogicalResult PtrLoadOp::verify() {
  auto ptrsTy = cast<RankedTensorType>(getPtrs().getType());
  auto ptrElem = dyn_cast<PtrType>(ptrsTy.getElementType());
  if (!ptrElem)
    return emitOpError("ptrs must be a tensor of tv.ptr");
  int64_t len = ShapedType::kDynamic;
  if (failed(verifyIndexTensors(*this, getIndices(), len)))
    return failure();
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (resTy.getRank() != 1)
    return emitOpError("result must be rank-1");
  if (!ShapedType::isDynamic(len) && !ShapedType::isDynamic(resTy.getShape()[0]) &&
      resTy.getShape()[0] != len)
    return emitOpError("result length must match the index length");
  if (resTy.getElementType() != ptrElem.getPointeeType())
    return emitOpError("result element type must match the pointer pointee type");
  return success();
}

LogicalResult PtrStoreOp::verify() {
  auto ptrsTy = cast<RankedTensorType>(getPtrs().getType());
  auto ptrElem = dyn_cast<PtrType>(ptrsTy.getElementType());
  if (!ptrElem)
    return emitOpError("ptrs must be a tensor of tv.ptr");
  int64_t len = ShapedType::kDynamic;
  if (failed(verifyIndexTensors(*this, getIndices(), len)))
    return failure();
  auto valTy = cast<RankedTensorType>(getValue().getType());
  if (valTy.getRank() != 1)
    return emitOpError("value must be rank-1");
  if (!ShapedType::isDynamic(len) && !ShapedType::isDynamic(valTy.getShape()[0]) &&
      valTy.getShape()[0] != len)
    return emitOpError("value length must match the index length");
  if (valTy.getElementType() != ptrElem.getPointeeType())
    return emitOpError("value element type must match the pointer pointee type");
  return success();
}
