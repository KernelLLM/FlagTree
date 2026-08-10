// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#ifdef __FLIR_TILEIR__
#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#endif

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLECONVERTGPUTILETOTTGIR
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

#ifdef __FLIR_TILEIR__
namespace tile = mlir::triton::tile;
namespace ttg = mlir::triton::gpu;

constexpr int kSharedMemoryAddressSpace = 3;

static FailureOr<ttg::MemDescType> getMemDescType(Operation *op,
                                                   tile::BufType bufferType) {
  if (bufferType.getMemorySpace() != tile::MemorySpace::Shared) {
    op->emitError("GPU TileIR conversion currently requires #tile.shared");
    return failure();
  }

  Attribute layout = op->getAttr("tle.gpu_layout");
  if (!layout) {
    op->emitError("is missing the preserved tle.gpu_layout attribute");
    return failure();
  }

  auto memorySpace = ttg::SharedMemorySpaceAttr::get(op->getContext());
  return ttg::MemDescType::get(bufferType.getShape(),
                               bufferType.getElementType(), layout,
                               memorySpace, /*mutableMemory=*/true);
}

static FailureOr<Value> lookupBuffer(Operation *op, Value buffer,
                                     const llvm::DenseMap<Value, Value> &mapped) {
  auto it = mapped.find(buffer);
  if (it == mapped.end()) {
    op->emitError("uses a TileIR buffer without a converted allocation");
    return failure();
  }
  return it->second;
}

static bool isPointerLike(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

static Type getLocalPointerType(Type valueType, Type elementType) {
  Type ptrType = triton::PointerType::get(elementType,
                                          kSharedMemoryAddressSpace);
  if (auto tensorType = dyn_cast<RankedTensorType>(valueType))
    return RankedTensorType::get(tensorType.getShape(), ptrType,
                                 tensorType.getEncoding());
  return ptrType;
}

static SmallVector<Value> createFullIndices(OpBuilder &builder, Location loc,
                                            ArrayRef<int64_t> shape) {
  SmallVector<Value> indices;
  Type i32 = builder.getI32Type();
  for (auto [axis, dim] : llvm::enumerate(shape)) {
    auto rangeType = RankedTensorType::get({dim}, i32);
    Value index = builder.create<triton::MakeRangeOp>(loc, rangeType, 0, dim);
    SmallVector<int64_t> expandedShape{dim};

    for (unsigned i = 0; i < axis; ++i) {
      expandedShape.insert(expandedShape.begin(), 1);
      auto expandedType = RankedTensorType::get(expandedShape, i32);
      index = builder.create<triton::ExpandDimsOp>(loc, expandedType, index, 0);
    }
    for (unsigned i = axis + 1; i < shape.size(); ++i) {
      unsigned expandAxis = expandedShape.size();
      expandedShape.push_back(1);
      auto expandedType = RankedTensorType::get(expandedShape, i32);
      index = builder.create<triton::ExpandDimsOp>(loc, expandedType, index,
                                                    expandAxis);
    }

    auto resultType = RankedTensorType::get(shape, i32);
    indices.push_back(
        builder.create<triton::BroadcastOp>(loc, resultType, index));
  }
  return indices;
}

static Value createFullLocalPointers(OpBuilder &builder, Location loc,
                                     Value memDesc, Type valueType) {
  auto memDescType = cast<ttg::MemDescType>(memDesc.getType());
  Type ptrType =
      getLocalPointerType(valueType, memDescType.getElementType());
  SmallVector<Value> indices =
      createFullIndices(builder, loc, memDescType.getShape());
  return builder.create<tle::LocalPointersOp>(loc, ptrType, memDesc, indices);
}

static Type getPointeeValueType(Type pointerType) {
  if (auto ptr = dyn_cast<triton::PointerType>(pointerType))
    return ptr.getPointeeType();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType);
  if (!tensorType)
    return {};
  auto ptr = dyn_cast<triton::PointerType>(tensorType.getElementType());
  if (!ptr)
    return {};
  return RankedTensorType::get(tensorType.getShape(), ptr.getPointeeType(),
                               tensorType.getEncoding());
}

static FailureOr<Value> castToI32(OpBuilder &builder, Location loc,
                                  Value value) {
  if (value.getType().isInteger(32))
    return value;
  if (value.getType().isIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), value)
        .getResult();
  emitError(loc) << "expected an index or i32 TileIR subview offset";
  return failure();
}

static FailureOr<int32_t> getConstantOffset(Value value) {
  if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>())
    value = indexCast.getIn();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return failure();
  return static_cast<int32_t>(integer.getInt());
}

static LogicalResult convertAllocations(
    ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
    SmallVectorImpl<Operation *> &eraseOps) {
  WalkResult result = module.walk([&](tile::AllocOp op) {
    auto memDescType =
        getMemDescType(op, cast<tile::BufType>(op.getResult().getType()));
    if (failed(memDescType))
      return WalkResult::interrupt();
    OpBuilder builder(op);
    mapped[op.getResult()] =
        builder.create<ttg::LocalAllocOp>(op.getLoc(), *memDescType);
    eraseOps.push_back(op);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static LogicalResult convertSubviews(
    ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
    SmallVectorImpl<Operation *> &eraseOps) {
  WalkResult result = module.walk([&](tile::SubViewOp op) {
    auto source = lookupBuffer(op, op.getSource(), mapped);
    auto resultType =
        getMemDescType(op, cast<tile::BufType>(op.getResult().getType()));
    if (failed(source) || failed(resultType))
      return WalkResult::interrupt();

    OpBuilder builder(op);
    auto sourceType = cast<ttg::MemDescType>((*source).getType());
    Value converted;
    if (sourceType.getRank() == resultType->getRank() + 1) {
      if (op.getOffsets().empty()) {
        op.emitError("rank-reducing tile.subview requires a leading offset");
        return WalkResult::interrupt();
      }
      auto index = castToI32(builder, op.getLoc(), op.getOffsets().front());
      if (failed(index))
        return WalkResult::interrupt();
      converted = builder.create<ttg::MemDescIndexOp>(
          op.getLoc(), *resultType, *source, *index);
    } else if (sourceType.getRank() == resultType->getRank()) {
      SmallVector<int32_t> offsets;
      offsets.reserve(op.getOffsets().size());
      for (Value offset : op.getOffsets()) {
        auto constant = getConstantOffset(offset);
        if (failed(constant)) {
          op.emitError("same-rank tile.subview currently requires static offsets");
          return WalkResult::interrupt();
        }
        offsets.push_back(*constant);
      }
      converted = builder.create<ttg::MemDescSubsliceOp>(
          op.getLoc(), *resultType, *source,
          builder.getDenseI32ArrayAttr(offsets));
    } else {
      op.emitError("unsupported tile.subview rank change");
      return WalkResult::interrupt();
    }

    mapped[op.getResult()] = converted;
    eraseOps.push_back(op);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static LogicalResult convertTileUsers(
    ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
    SmallVectorImpl<Operation *> &eraseOps) {
  bool failedConversion = false;
  OpBuilder builder(module.getContext());

  module.walk([&](Operation *operation) {
    if (failedConversion)
      return;

    if (auto op = dyn_cast<tile::LocalPtrOp>(operation)) {
      auto source = lookupBuffer(op, op.getSource(), mapped);
      if (failed(source)) {
        failedConversion = true;
        return;
      }
      builder.setInsertionPoint(op);
      Value replacement = builder.create<tle::LocalPointersOp>(
          op.getLoc(), op.getResult().getType(), *source, op.getIndices());
      op.getResult().replaceAllUsesWith(replacement);
      eraseOps.push_back(op);
      return;
    }

    if (auto op = dyn_cast<tile::ToTensorOp>(operation)) {
      auto source = lookupBuffer(op, op.getSrc(), mapped);
      if (failed(source)) {
        failedConversion = true;
        return;
      }
      builder.setInsertionPoint(op);
      Value replacement = builder.create<ttg::LocalLoadOp>(
          op.getLoc(), op.getResult().getType(), *source);
      op.getResult().replaceAllUsesWith(replacement);
      eraseOps.push_back(op);
      return;
    }

    if (auto op = dyn_cast<tile::StoreTensorOp>(operation)) {
      auto destination = lookupBuffer(op, op.getDst(), mapped);
      if (failed(destination)) {
        failedConversion = true;
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<ttg::LocalStoreOp>(op.getLoc(), op.getSrc(),
                                        *destination);
      eraseOps.push_back(op);
      return;
    }

    if (auto op = dyn_cast<tile::CopyOp>(operation)) {
      Value src = op.getSrc();
      Value dst = op.getDst();
      if (isa<tile::BufType>(src.getType())) {
        auto converted = lookupBuffer(op, src, mapped);
        if (failed(converted)) {
          failedConversion = true;
          return;
        }
        src = *converted;
      }
      if (isa<tile::BufType>(dst.getType())) {
        auto converted = lookupBuffer(op, dst, mapped);
        if (failed(converted)) {
          failedConversion = true;
          return;
        }
        dst = *converted;
      }

      builder.setInsertionPoint(op);
      if (!op.getIndices().empty()) {
        SmallVector<Value> indices;
        for (Value index : op.getIndices()) {
          auto converted = castToI32(builder, op.getLoc(), index);
          if (failed(converted)) {
            failedConversion = true;
            return;
          }
          indices.push_back(*converted);
        }
        builder.create<ttg::TMACopyOp>(op.getLoc(), src, dst, indices);
      } else if (isa<ttg::MemDescType>(dst.getType())) {
        Value value = src;
        if (isPointerLike(src.getType()))
          value = builder.create<triton::LoadOp>(
              op.getLoc(), src, triton::CacheModifier::NONE,
              triton::EvictionPolicy::NORMAL, false);
        Value ptr = createFullLocalPointers(builder, op.getLoc(), dst,
                                            value.getType());
        builder.create<triton::StoreOp>(op.getLoc(), ptr, value,
                                        triton::CacheModifier::NONE,
                                        triton::EvictionPolicy::NORMAL);
      } else if (isa<ttg::MemDescType>(src.getType()) &&
                 isPointerLike(dst.getType())) {
        Type valueType = getPointeeValueType(dst.getType());
        if (!valueType) {
          op.emitError("cannot determine tile.copy destination value type");
          failedConversion = true;
          return;
        }
        Value ptr = createFullLocalPointers(builder, op.getLoc(), src,
                                            valueType);
        Value value = builder.create<triton::LoadOp>(
            op.getLoc(), ptr, triton::CacheModifier::NONE,
            triton::EvictionPolicy::NORMAL, false);
        builder.create<triton::StoreOp>(op.getLoc(), dst, value,
                                        triton::CacheModifier::NONE,
                                        triton::EvictionPolicy::NORMAL);
      } else {
        op.emitError("unsupported operands for GPU tile.copy conversion");
        failedConversion = true;
        return;
      }
      eraseOps.push_back(op);
      return;
    }

    if (isa<tile::SetFlagOp, tile::WaitFlagOp, tile::PipeBarrierOp>(operation)) {
      builder.setInsertionPoint(operation);
      builder.create<ttg::LocalBarrierOp>(operation->getLoc());
      eraseOps.push_back(operation);
    }
  });

  return failure(failedConversion);
}

static LogicalResult rewritePipeOperands(
    ModuleOp module, const llvm::DenseMap<Value, Value> &mapped) {
  bool failedConversion = false;
  module.walk([&](Operation *op) {
    if (!op->getName().getStringRef().starts_with("tle.pipe."))
      return;
    for (OpOperand &operand : op->getOpOperands()) {
      if (!isa<tile::BufType>(operand.get().getType()))
        continue;
      auto converted = lookupBuffer(op, operand.get(), mapped);
      if (failed(converted)) {
        failedConversion = true;
        return;
      }
      operand.set(*converted);
    }
  });
  return failure(failedConversion);
}
#endif

class ConvertGpuTileToTtgirPass
    : public impl::TritonTleConvertGpuTileToTtgirBase<
          ConvertGpuTileToTtgirPass> {
  void runOnOperation() override {
#ifdef __FLIR_TILEIR__
    ModuleOp module = getOperation();
    llvm::DenseMap<Value, Value> mapped;
    SmallVector<Operation *> eraseOps;

    if (failed(convertAllocations(module, mapped, eraseOps)) ||
        failed(convertSubviews(module, mapped, eraseOps)) ||
        failed(convertTileUsers(module, mapped, eraseOps)) ||
        failed(rewritePipeOperands(module, mapped))) {
      signalPassFailure();
      return;
    }

    for (Operation *op : llvm::reverse(eraseOps)) {
      if (llvm::any_of(op->getResults(),
                       [](Value result) { return !result.use_empty(); })) {
        op->emitError("still has users after GPU TileIR conversion");
        signalPassFailure();
        return;
      }
      op->erase();
    }

    bool hasRemainingGpuTileOps = false;
    module.walk([&](Operation *op) {
      if (isa<tile::AllocOp, tile::SubViewOp, tile::LocalPtrOp,
              tile::CopyOp, tile::ToTensorOp, tile::StoreTensorOp>(op)) {
        op->emitError("was not eliminated by GPU TileIR conversion");
        hasRemainingGpuTileOps = true;
      }
    });
    if (hasRemainingGpuTileOps)
      signalPassFailure();
#endif
  }
};

} // namespace

} // namespace mlir::triton::tle
