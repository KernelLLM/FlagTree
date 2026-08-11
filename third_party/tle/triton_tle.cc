// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "Python.h"
#include "Transforms/Passes.h"
#include "ir.h" // TritonOpBuilder
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#ifdef __FLIR_TILEIR__
#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <cctype>
#include <cstdint>

namespace py = pybind11;
using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace tle = triton::tle;
#ifdef __FLIR_TILEIR__
namespace tile = triton::tile;
#endif

extern std::vector<int64_t>
computeAliasOperandIndices(TritonOpBuilder &self, std::string_view text,
                           const std::vector<Value> &args);

extern tle::DSLRegionOp
createTLERawRegionByLLVMFunc(TritonOpBuilder &self, std::string_view text,
                             const std::vector<Value> &args,
                             const std::vector<int64_t> &aliasOperandIndices);

#ifdef __FLIR_TILEIR__
static std::string attrToLowerString(Attribute attr) {
  if (!attr)
    return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  os.flush();
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return text;
}

static tile::MemorySpace attrToTileMemSpace(Attribute attr) {
  auto text = attrToLowerString(attr);
  if (text.find("register") != std::string::npos)
    return tile::MemorySpace::Register;
  if (text.find("shared") != std::string::npos ||
      text.find("smem") != std::string::npos)
    return tile::MemorySpace::Shared;
  if (text.find("global") != std::string::npos)
    return tile::MemorySpace::Global;
  if (text.find("local") != std::string::npos)
    return tile::MemorySpace::Local;
  return tile::MemorySpace::Shared;
}

static bool isTileBuffer(Value value) {
  return value && isa<tile::BufType>(value.getType());
}

static bool isTTTensorOrPointer(Value value) {
  if (!value)
    return false;
  auto type = value.getType();
  return isa<RankedTensorType, triton::PointerType>(type);
}

static bool isTTPointerLike(Value value) {
  if (!value)
    return false;
  Type type = value.getType();
  if (isa<triton::PointerType>(type))
    return true;
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return isa<triton::PointerType>(tensorType.getElementType());
  return false;
}

static constexpr llvm::StringLiteral kGpuMemDescTypeAttr =
    "tle.gpu_memdesc_type";
static constexpr int kGpuSharedMemoryAddressSpace = 3;

static ttg::MemDescType getGpuMemDescType(Operation *op) {
  if (auto typeAttr = op->getAttrOfType<TypeAttr>(kGpuMemDescTypeAttr))
    return dyn_cast<ttg::MemDescType>(typeAttr.getValue());
  return {};
}

static Value getGpuMemDesc(
    Value value, const llvm::DenseMap<Value, Value> &bufferMemDescs) {
  if (value && isa<ttg::MemDescType>(value.getType()))
    return value;
  return bufferMemDescs.lookup(value);
}

static ttg::MemDescType inferGpuMemDescType(Value value) {
  if (!value)
    return {};
  if (auto type = dyn_cast<ttg::MemDescType>(value.getType()))
    return type;
  if (Operation *def = value.getDefiningOp())
    return getGpuMemDescType(def);
  return {};
}

static RankedTensorType getPointeeTensorType(Value pointer) {
  auto tensorType = dyn_cast<RankedTensorType>(pointer.getType());
  if (!tensorType)
    return {};
  auto pointerType = dyn_cast<triton::PointerType>(tensorType.getElementType());
  if (!pointerType)
    return {};
  return RankedTensorType::get(tensorType.getShape(),
                               pointerType.getPointeeType(),
                               tensorType.getEncoding());
}

static Value materializeTileBuffer(Value value,
                                   llvm::DenseMap<Value, Value> &bufferValues) {
  while (isTileBuffer(value)) {
    auto it = bufferValues.find(value);
    if (it == bufferValues.end())
      return Value();
    value = it->second;
  }
  return value;
}

static void eraseIfUnused(Operation *op) {
  if (!op)
    return;
  if (llvm::all_of(op->getResults(),
                   [](Value result) { return result.use_empty(); }))
    op->erase();
}

static void lowerGpuTileIRToTTIR(ModuleOp module) {
  OpBuilder builder(module.getContext());
  llvm::DenseMap<Value, Value> bufferValues;
  llvm::DenseMap<Value, Value> bufferMemDescs;
  SmallVector<Operation *> eraseOps;

  SmallVector<tile::AllocOp> gpuAllocs;
  module.walk([&](tile::AllocOp allocOp) {
    if (getGpuMemDescType(allocOp))
      gpuAllocs.push_back(allocOp);
  });
  for (tile::AllocOp allocOp : gpuAllocs) {
    builder.setInsertionPoint(allocOp);
    auto localAlloc = builder.create<ttg::LocalAllocOp>(
        allocOp.getLoc(), getGpuMemDescType(allocOp));
    bufferMemDescs[allocOp.getResult()] = localAlloc.getResult();
    allocOp.getResult().replaceAllUsesWith(localAlloc.getResult());
    eraseOps.push_back(allocOp);
  }

  module.walk([&](ttg::WarpSpecializeOp warpSpecialize) {
    auto captures = warpSpecialize.getExplicitCaptures();
    for (Region *region : warpSpecialize.getPartitionRegions()) {
      for (auto [index, capture] : llvm::enumerate(captures))
        region->getArgument(index).setType(capture.getType());
    }
  });

  module.walk([&](triton::CallOp call) {
    auto callee = module.lookupSymbol<triton::FuncOp>(call.getCallee());
    if (!callee)
      return;
    if (isa<UnknownLoc>(call.getLoc())) {
      Location callLoc = callee.getLoc();
      if (isa<UnknownLoc>(callLoc)) {
        if (auto parent = call->getParentOfType<triton::FuncOp>())
          callLoc = parent.getLoc();
      }
      if (isa<UnknownLoc>(callLoc))
        callLoc = FileLineColLoc::get(module.getContext(), "tle-generated", 1,
                                      1);
      call->setLoc(callLoc);
    }
    for (auto [operand, argument] :
         llvm::zip(call.getOperands(), callee.getArguments())) {
      if (auto memDescType = inferGpuMemDescType(operand))
        argument.setType(memDescType);
    }
  });
  for (triton::FuncOp func : module.getOps<triton::FuncOp>()) {
    SmallVector<Type> inputTypes;
    for (BlockArgument argument : func.getArguments())
      inputTypes.push_back(argument.getType());
    SmallVector<Type> resultTypes(func.getResultTypes());
    func.setFunctionType(
        FunctionType::get(module.getContext(), inputTypes, resultTypes));
  }

  module.walk([&](Operation *op) {
    auto mapPipeFields = [&](ValueRange fields) {
      SmallVector<Value> mapped;
      mapped.reserve(fields.size());
      for (Value field : fields) {
        Value memDesc = getGpuMemDesc(field, bufferMemDescs);
        if (!memDesc)
          return SmallVector<Value>{};
        mapped.push_back(memDesc);
      }
      return mapped;
    };

    if (auto pipeOp = dyn_cast<tile::DataflowPipeCreateOp>(op)) {
      auto fields = mapPipeFields(pipeOp.getFields());
      if (fields.size() != pipeOp.getFields().size()) {
        pipeOp.emitError("cannot lower tile.pipe.create without GPU memdescs");
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<tle::PipeCreateOp>(
          op->getLoc(), fields, pipeOp.getCapacityAttr(), pipeOp.getScopeAttr(),
          pipeOp.getPipeNameAttr(), pipeOp.getFieldNamesAttr(),
          pipeOp.getReadersAttr(), pipeOp.getOneShotAttr());
      eraseOps.push_back(op);
      return;
    }

    if (auto pipeOp = dyn_cast<tile::DataflowPipeWriterAcquireOp>(op)) {
      auto fields = mapPipeFields(pipeOp.getFields());
      if (fields.size() != pipeOp.getFields().size()) {
        pipeOp.emitError(
            "cannot lower tile.pipe.writer_acquire without GPU memdescs");
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<tle::PipeWriterAcquireOp>(
          op->getLoc(), fields, pipeOp.getStage(), pipeOp.getPhase(),
          pipeOp.getCapacityAttr(), pipeOp.getScopeAttr(),
          pipeOp.getPipeNameAttr(), pipeOp.getFieldNamesAttr());
      eraseOps.push_back(op);
      return;
    }

    if (auto pipeOp = dyn_cast<tile::DataflowPipeWriterCommitOp>(op)) {
      auto fields = mapPipeFields(pipeOp.getFields());
      if (fields.size() != pipeOp.getFields().size()) {
        pipeOp.emitError(
            "cannot lower tile.pipe.writer_commit without GPU memdescs");
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<tle::PipeWriterCommitOp>(
          op->getLoc(), fields, pipeOp.getStage(), pipeOp.getCapacityAttr(),
          pipeOp.getScopeAttr(), pipeOp.getPipeNameAttr(),
          pipeOp.getFieldNamesAttr());
      eraseOps.push_back(op);
      return;
    }

    if (auto pipeOp = dyn_cast<tile::DataflowPipeWriterCloseOp>(op)) {
      auto fields = mapPipeFields(pipeOp.getFields());
      if (fields.size() != pipeOp.getFields().size()) {
        pipeOp.emitError(
            "cannot lower tile.pipe.writer_close without GPU memdescs");
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<tle::PipeWriterCloseOp>(
          op->getLoc(), fields, pipeOp.getStage(), pipeOp.getPhase(),
          pipeOp.getCapacityAttr(), pipeOp.getScopeAttr(),
          pipeOp.getPipeNameAttr(), pipeOp.getFieldNamesAttr());
      eraseOps.push_back(op);
      return;
    }

    if (auto pipeOp = dyn_cast<tile::DataflowPipeReaderWaitOp>(op)) {
      auto fields = mapPipeFields(pipeOp.getFields());
      if (fields.size() != pipeOp.getFields().size()) {
        pipeOp.emitError(
            "cannot lower tile.pipe.reader_wait without GPU memdescs");
        return;
      }
      builder.setInsertionPoint(op);
      auto lowered = builder.create<tle::PipeReaderWaitOp>(
          op->getLoc(), builder.getI1Type(), fields, pipeOp.getStage(),
          pipeOp.getPhase(), pipeOp.getCapacityAttr(), pipeOp.getScopeAttr(),
          pipeOp.getPipeNameAttr(), pipeOp.getFieldNamesAttr(),
          pipeOp.getReaderNameAttr());
      pipeOp.getIsClosed().replaceAllUsesWith(lowered.getIsClosed());
      eraseOps.push_back(op);
      return;
    }

    if (auto pipeOp = dyn_cast<tile::DataflowPipeReaderReleaseOp>(op)) {
      auto fields = mapPipeFields(pipeOp.getFields());
      if (fields.size() != pipeOp.getFields().size()) {
        pipeOp.emitError(
            "cannot lower tile.pipe.reader_release without GPU memdescs");
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<tle::PipeReaderReleaseOp>(
          op->getLoc(), fields, pipeOp.getStage(), pipeOp.getCapacityAttr(),
          pipeOp.getScopeAttr(), pipeOp.getPipeNameAttr(),
          pipeOp.getFieldNamesAttr(), pipeOp.getReaderNameAttr());
      eraseOps.push_back(op);
      return;
    }

    if (auto copyOp = dyn_cast<tile::CopyOp>(op)) {
      Value copySrc = op->getOperand(0);
      Value copyDst = op->getOperand(1);
      Value srcMemDesc = getGpuMemDesc(copySrc, bufferMemDescs);
      Value dstMemDesc = getGpuMemDesc(copyDst, bufferMemDescs);
      bool srcIsTensorDesc = isa<triton::TensorDescType>(copySrc.getType());
      bool dstIsTensorDesc = isa<triton::TensorDescType>(copyDst.getType());
      if ((srcIsTensorDesc && dstMemDesc) ||
          (srcMemDesc && dstIsTensorDesc)) {
        builder.setInsertionPoint(op);
        Value tmaSrc = srcMemDesc ? srcMemDesc : copySrc;
        Value tmaDst = dstMemDesc ? dstMemDesc : copyDst;
        builder.create<ttg::TMACopyOp>(op->getLoc(), tmaSrc, tmaDst,
                                       copyOp.getIndices());
        eraseOps.push_back(op);
        return;
      }
      Value src = materializeTileBuffer(copySrc, bufferValues);
      if (!src)
        src = copySrc;
      Value dst = copyDst;
      if (dstMemDesc) {
        builder.setInsertionPoint(op);
        Value storeValue = src;
        if (isTTPointerLike(storeValue)) {
          auto load = builder.create<triton::LoadOp>(
              op->getLoc(), storeValue, triton::CacheModifier::NONE,
              triton::EvictionPolicy::NORMAL, false);
          storeValue = load.getResult();
        }
        if (isa<RankedTensorType>(storeValue.getType()))
          builder.create<ttg::LocalStoreOp>(op->getLoc(), storeValue,
                                            dstMemDesc);
        eraseOps.push_back(op);
        return;
      }
      if (isTileBuffer(dst)) {
        bufferValues[dst] = src;
        eraseOps.push_back(op);
        return;
      }
      if (srcMemDesc && isTTPointerLike(dst)) {
        if (auto resultType = getPointeeTensorType(dst)) {
          builder.setInsertionPoint(op);
          src = builder.create<ttg::LocalLoadOp>(op->getLoc(), resultType,
                                                 srcMemDesc);
        }
      }
      if (isTTPointerLike(dst) && isTTTensorOrPointer(src)) {
        builder.setInsertionPoint(op);
        Value storeValue = src;
        if (isTTPointerLike(storeValue)) {
          auto load = builder.create<triton::LoadOp>(
              op->getLoc(), storeValue, triton::CacheModifier::NONE,
              triton::EvictionPolicy::NORMAL, false);
          storeValue = load.getResult();
        }
        builder.create<triton::StoreOp>(
            op->getLoc(), dst, storeValue, triton::CacheModifier::NONE,
            triton::EvictionPolicy::NORMAL);
        eraseOps.push_back(op);
      }
      return;
    }

    if (auto storeTensorOp = dyn_cast<tile::StoreTensorOp>(op)) {
      Value storeSrc = op->getOperand(0);
      Value storeDst = op->getOperand(1);
      bufferValues[storeDst] = storeSrc;
      if (Value dstMemDesc = getGpuMemDesc(storeDst, bufferMemDescs)) {
        builder.setInsertionPoint(op);
        builder.create<ttg::LocalStoreOp>(op->getLoc(), storeSrc, dstMemDesc);
      }
      eraseOps.push_back(op);
      return;
    }

    if (auto subviewOp = dyn_cast<tile::SubViewOp>(op)) {
      Value subviewSource = op->getOperand(0);
      if (Value src = materializeTileBuffer(subviewSource, bufferValues))
        bufferValues[subviewOp.getResult()] = src;

      if (Value srcMemDesc = getGpuMemDesc(subviewSource, bufferMemDescs)) {
        auto resultType = getGpuMemDescType(op);
        if (!resultType) {
          subviewOp.emitError("missing GPU memdesc type for tile.subview");
          return;
        }
        if (subviewOp.getOffsets().empty()) {
          subviewOp.emitError("GPU tile.subview requires a leading slot index");
          return;
        }
        builder.setInsertionPoint(op);
        Value index = subviewOp.getOffsets().front();
        if (!index.getType().isInteger(32))
          index = builder.create<arith::IndexCastOp>(op->getLoc(),
                                                     builder.getI32Type(), index);
        auto memDescIndex = builder.create<ttg::MemDescIndexOp>(
            op->getLoc(), resultType, srcMemDesc, index);
        bufferMemDescs[subviewOp.getResult()] = memDescIndex.getResult();
        subviewOp.getResult().replaceAllUsesWith(memDescIndex.getResult());
      }
      eraseOps.push_back(op);
      return;
    }

    if (auto localPtrOp = dyn_cast<tile::LocalPtrOp>(op)) {
      Value memDesc = getGpuMemDesc(op->getOperand(0), bufferMemDescs);
      if (!memDesc) {
        localPtrOp.emitError(
            "cannot lower tile.local_ptr without a GPU shared-memory memdesc");
        return;
      }
      builder.setInsertionPoint(op);
      auto localPointers = builder.create<tle::LocalPointersOp>(
          op->getLoc(), localPtrOp.getResult().getType(), memDesc,
          localPtrOp.getIndices());
      localPtrOp.getResult().replaceAllUsesWith(localPointers.getResult());
      eraseOps.push_back(op);
      return;
    }

    if (auto gmOffsetOp = dyn_cast<tile::GmOffsetOp>(op)) {
      gmOffsetOp.getResult().replaceAllUsesWith(gmOffsetOp.getBase());
      eraseOps.push_back(op);
      return;
    }

    if (auto toTensorOp = dyn_cast<tile::ToTensorOp>(op)) {
      Value tensorSource = op->getOperand(0);
      Value value;
      if (Value memDesc = getGpuMemDesc(tensorSource, bufferMemDescs)) {
        auto resultType = dyn_cast<RankedTensorType>(
            toTensorOp.getResult().getType());
        if (!resultType) {
          toTensorOp.emitError(
              "GPU tile.to_tensor expects a ranked tensor result");
          return;
        }
        auto pointerType = triton::PointerType::get(
            resultType.getElementType(), kGpuSharedMemoryAddressSpace);
        auto pointerTensorType = RankedTensorType::get(
            resultType.getShape(), pointerType, resultType.getEncoding());
        builder.setInsertionPoint(op);
        auto localPointers = builder.create<tle::LocalPointersOp>(
            op->getLoc(), pointerTensorType, memDesc, ValueRange{});
        value = builder
                    .create<triton::LoadOp>(
                        op->getLoc(), localPointers.getResult(),
                        triton::CacheModifier::NONE,
                        triton::EvictionPolicy::NORMAL, false)
                    .getResult();
      } else {
        value = materializeTileBuffer(tensorSource, bufferValues);
      }
      if (!value) {
        toTensorOp.emitError("cannot lower tile.to_tensor without a "
                             "preceding write or GPU memdesc");
        return;
      }

      builder.setInsertionPoint(op);
      Value replacement = value;
      if (isTTPointerLike(value)) {
        auto load = builder.create<triton::LoadOp>(
            op->getLoc(), value, triton::CacheModifier::NONE,
            triton::EvictionPolicy::NORMAL, false);
        replacement = load.getResult();
      }
      toTensorOp.getResult().replaceAllUsesWith(replacement);
      eraseOps.push_back(op);
    }

    if (isa<tile::SetFlagOp, tile::WaitFlagOp, tile::PipeBarrierOp>(op)) {
      builder.setInsertionPoint(op);
      builder.create<ttg::LocalBarrierOp>(op->getLoc());
      eraseOps.push_back(op);
      return;
    }
  });

  for (Operation *op : llvm::reverse(eraseOps))
    eraseIfUnused(op);

  SmallVector<Operation *> cleanupOps;
  module.walk([&](Operation *op) {
    if (isa<tile::AllocOp, tile::SubViewOp, tile::CopyOp,
            tile::LocalPtrOp, tile::StoreTensorOp, tile::ToTensorOp,
            tile::GmOffsetOp, tile::DataflowPipeCreateOp,
            tile::DataflowPipeWriterAcquireOp,
            tile::DataflowPipeWriterCommitOp,
            tile::DataflowPipeWriterCloseOp,
            tile::DataflowPipeReaderWaitOp,
            tile::DataflowPipeReaderReleaseOp,
            tile::SetFlagOp, tile::WaitFlagOp, tile::PipeBarrierOp>(op))
      cleanupOps.push_back(op);
  });
  for (Operation *op : llvm::reverse(cleanupOps))
    eraseIfUnused(op);

}
#endif

void init_triton_tle_ir(py::module &&m) {

  // Get the existing builder class from the main ir module (TLX style)
  auto &builder_cls = *ir::getBuilderClass();

  // Add TLE extensions to the existing TritonOpBuilder class
  builder_cls
      // TLE-Lite
      .def(
          "create_extract_tile",
          [](TritonOpBuilder &self, Value &input,
             // std::vector<int64_t> &offsets,
             Value &index, std::vector<int64_t> &tileShape) -> Value {
            auto op = self.create<tle::ExtractTileOp>(input, index, tileShape);
            return op.getResult();
          },
          py::arg("input"), py::arg("index"), py::arg("tileShape"),
          "Create extract_tile operation")
      .def(
          "create_insert_tile",
          [](TritonOpBuilder &self, Value &input, Value &tile,
             Value &index) -> Value {
            auto op = self.create<tle::InsertTileOp>(input, tile, index);
            return op.getResult();
          },
          py::arg("input"), py::arg("tile"), py::arg("index"),
          "Create insert_tile operation")
      // TLE-Struct
      .def("make_swizzled_shared_encoding_attr",
           [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
              unsigned maxPhase, std::vector<unsigned> order,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum,
              std::vector<unsigned> CTAOrder) {
             assert(order.size() == CTAsPerCGA.size() && "shape mismatch");
             assert(order.size() == CTASplitNum.size() && "shape mismatch");
             assert(order.size() == CTAOrder.size() && "shape mismatch");
             auto context = self.getBuilder().getContext();
             auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(
                 context, CTAsPerCGA, CTASplitNum, CTAOrder);
             return mlir::cast<Attribute>(ttg::SwizzledSharedEncodingAttr::get(
                 context, vectorSize, perPhase, maxPhase, order, CTALayout));
           })
      .def("make_nv_mma_shared_encoding_attr",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              std::vector<unsigned> order, Type &elemType,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum, std::vector<unsigned> CTAOrder,
              bool fp4Padded, bool swizzled) {
             /* Validation logic for user defined layout encoding begin */
             assert(shape.size() == order.size());
             assert(order.size() == CTAsPerCGA.size());
             assert(CTAsPerCGA.size() == CTASplitNum.size());
             assert(CTASplitNum.size() == CTAOrder.size());
             /* Validation logic for user defined layout encoding end */

             auto context = self.getBuilder().getContext();
             auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(
                 context, CTAsPerCGA, CTASplitNum, CTAOrder);
             if (swizzled) {
               return mlir::cast<Attribute>(ttg::NVMMASharedEncodingAttr::get(
                   context, shape, order, CTALayout, elemType, fp4Padded));
             } else {
               return mlir::cast<Attribute>(ttg::NVMMASharedEncodingAttr::get(
                   context, /*swizzlingByteWidth=*/0,
                   /*transposed=*/order[0] == 0,
                   elemType.getIntOrFloatBitWidth(), fp4Padded, CTALayout));
             }
           })
      .def("make_tensor_memory_encoding_attr",
           [](TritonOpBuilder &self, unsigned blockM, unsigned blockN,
              bool unpacked, unsigned CTASplitM, unsigned CTASplitN) {
             auto context = self.getBuilder().getContext();
             const unsigned colStride = unpacked ? 2 : 1;
             return mlir::cast<Attribute>(ttng::TensorMemoryEncodingAttr::get(
                 context, blockM, blockN, colStride, CTASplitM, CTASplitN,
                 /*twoCTAs=*/false));
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             auto memDesc =
                 ttg::MemDescType::get(shape, elementType, encoding,
                                       memorySpace, /*mutableMemory=*/true);
             return self.create<ttg::LocalAllocOp>(memDesc);
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::LocalAllocOp>(resultTy, value);
           })
      .def("create_tma_copy",
           [](TritonOpBuilder &self, Value src, Value dst,
              std::vector<Value> &indices) {
             self.create<ttg::TMACopyOp>(src, dst, indices);
             return;
           })
      .def("create_local_load",
           [](TritonOpBuilder &self, Type resultTy, Value memDesc) -> Value {
             return self.create<ttg::LocalLoadOp>(resultTy, memDesc);
           })
      .def("create_local_store",
           [](TritonOpBuilder &self, Value &dst, Value &regValues) -> void {
             self.create<ttg::LocalStoreOp>(regValues, dst);
           })
      .def("create_local_pointers",
           [](TritonOpBuilder &self, Type resultTy, Value memDesc,
              py::args args) -> OpState {
             llvm::SmallVector<Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args) {
               indices.push_back(py::cast<Value>(arg));
             }
             return self.create<tle::LocalPointersOp>(resultTy, memDesc,
                                                      indices);
           })
#ifdef __FLIR_TILEIR__
      .def("tile_get_string_attr",
           [](TritonOpBuilder &self, const std::string &name) -> Attribute {
             return self.getBuilder().getStringAttr(name);
           })
      .def("tile_get_buffer_type",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             auto memSpace = attrToTileMemSpace(memorySpace);
             auto *ctx = self.getBuilder().getContext();
             return tile::BufType::get(ctx, shape, elementType, memSpace);
           })
      .def("create_tile_alloc",
           [](TritonOpBuilder &self, Type tileBufType,
              Type gpuMemDescType) -> Value {
             auto bufType = mlir::cast<tile::BufType>(tileBufType);
             auto op = self.create<tile::AllocOp>(
                 tileBufType, bufType.getMemorySpace(),
                 /*shape=*/mlir::ArrayAttr(), /*dtype=*/mlir::TypeAttr(),
                 /*policy=*/tile::PolicyAttr(),
                 /*layout=*/tile::LayoutAttr::get(self.getBuilder().getContext(),
                                                  tile::Layout::ND),
                 /*lifetime=*/tile::LifetimeAttr(),
                 /*comment=*/mlir::StringAttr());
             op->setAttr(kGpuMemDescTypeAttr,
                         mlir::TypeAttr::get(gpuMemDescType));
             return op.getResult();
           })
      .def("create_tile_copy",
           [](TritonOpBuilder &self, Value &src, Value &dst,
              std::vector<Value> &indices, bool interNoAlias) -> void {
             auto op = self.create<tile::CopyOp>(
                 src, dst, indices, /*engine=*/tile::EngineAttr(),
                 /*src_layout=*/tile::LayoutAttr::get(
                     self.getBuilder().getContext(), tile::Layout::ND),
                 /*dst_nz_layout=*/tile::NZLayoutAttr(),
                 /*transpose=*/mlir::UnitAttr(),
                 /*comment=*/mlir::StringAttr());
             if (interNoAlias)
               op->setAttr("inter_no_alias", self.getBuilder().getBoolAttr(true));
           })
      .def("create_tile_subview",
           [](TritonOpBuilder &self, Value source, std::vector<Value> &offsets,
              const std::vector<int64_t> &sizes,
              const std::vector<int64_t> &strides,
              Type gpuMemDescType) -> Value {
             SmallVector<Value> indexOffsets;
             auto &builder = self.getBuilder();
             auto indexType = builder.getIndexType();
             for (Value offset : offsets) {
               if (offset.getType() != indexType)
                 offset = self.create<arith::IndexCastOp>(indexType, offset);
               indexOffsets.push_back(offset);
             }
             auto *ctx = builder.getContext();
             auto srcBuf = mlir::cast<tile::BufType>(source.getType());
             auto resTy = tile::BufType::get(ctx, sizes, srcBuf.getElementType(),
                                             srcBuf.getMemorySpace());
             auto op = self.create<tile::SubViewOp>(
                 resTy, source, indexOffsets, builder.getI64ArrayAttr(sizes),
                 builder.getI64ArrayAttr(strides));
             op->setAttr(kGpuMemDescTypeAttr,
                         mlir::TypeAttr::get(gpuMemDescType));
             return op.getResult();
           })
      .def("create_tile_local_ptr",
           [](TritonOpBuilder &self, Type resultTy, Value source,
              py::args args) -> OpState {
             llvm::SmallVector<Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args)
               indices.push_back(py::cast<Value>(arg));
             return self.create<tile::LocalPtrOp>(resultTy, source, indices);
           })
      .def("create_tile_to_tensor",
           [](TritonOpBuilder &self, Value &src, bool /*writable*/) -> Value {
             auto srcBuf = mlir::cast<tile::BufType>(src.getType());
             auto resTy =
                 RankedTensorType::get(srcBuf.getShape(), srcBuf.getElementType());
             auto op = self.create<tile::ToTensorOp>(resTy, src);
             return op.getResult();
           })
      .def("create_tile_store_tensor",
           [](TritonOpBuilder &self, Value &src, Value &dst) -> void {
             self.create<tile::StoreTensorOp>(src, dst);
           })
      .def("create_tile_pipe_create",
           [](TritonOpBuilder &self, std::vector<Value> fields,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              std::vector<std::string> readerNames, bool oneShot) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             SmallVector<Attribute> readerNameAttrs;
             for (StringRef name : readerNames)
               readerNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             ArrayAttr readerNamesAttr;
             if (!readerNameAttrs.empty())
               readerNamesAttr = builder.getArrayAttr(readerNameAttrs);
             BoolAttr oneShotAttr;
             if (oneShot)
               oneShotAttr = builder.getBoolAttr(true);
             self.create<tile::DataflowPipeCreateOp>(
                 fields, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNamesAttr,
                 oneShotAttr);
           })
      .def("create_tile_pipe_writer_acquire",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tile::DataflowPipeWriterAcquireOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_tile_pipe_writer_commit",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tile::DataflowPipeWriterCommitOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_tile_pipe_writer_close",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tile::DataflowPipeWriterCloseOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_tile_pipe_reader_wait",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName,
              std::vector<std::string>) -> Value {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             return self.create<tile::DataflowPipeReaderWaitOp>(
                 builder.getI1Type(), fields, stage, phase,
                 builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_tile_pipe_reader_release",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName,
              std::vector<std::string>) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             self.create<tile::DataflowPipeReaderReleaseOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_tile_set_flag",
           [](TritonOpBuilder &self, int64_t producer, int64_t consumer,
              int64_t event) -> void {
             self.create<tile::SetFlagOp>(static_cast<tile::Pipe>(producer),
                                          static_cast<tile::Pipe>(consumer),
                                          static_cast<tile::EventID>(event));
           })
      .def("create_tile_wait_flag",
           [](TritonOpBuilder &self, int64_t producer, int64_t consumer,
              int64_t event) -> void {
             self.create<tile::WaitFlagOp>(static_cast<tile::Pipe>(producer),
                                           static_cast<tile::Pipe>(consumer),
                                           static_cast<tile::EventID>(event));
           })
      .def("create_tile_pipe_barrier",
           [](TritonOpBuilder &self, int64_t pipe) -> void {
             self.create<tile::PipeBarrierOp>(static_cast<tile::Pipe>(pipe));
           })
      .def("create_tile_gm_offset",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &indices,
              std::vector<Value> &strides) -> Value {
             SmallVector<Value> indexValues;
             SmallVector<Value> strideValues;
             auto &builder = self.getBuilder();
             auto indexType = builder.getIndexType();
             for (Value index : indices) {
               if (index.getType() != indexType)
                 index = self.create<arith::IndexCastOp>(indexType, index);
               indexValues.push_back(index);
             }
             for (Value stride : strides) {
               if (stride.getType() != indexType)
                 stride = self.create<arith::IndexCastOp>(indexType, stride);
               strideValues.push_back(stride);
             }
             auto op = self.create<tile::GmOffsetOp>(base.getType(), base,
                                                     indexValues, strideValues);
             return op.getResult();
           })
#endif
      .def("create_memdesc_index",
           [](TritonOpBuilder &self, Type resultType, Value src,
              Value index) -> Value {
             return self.create<ttg::MemDescIndexOp>(resultType, src, index);
           })
      .def("create_memdesc_subslice",
           [](TritonOpBuilder &self, Type resultType, Value src,
              std::vector<int32_t> &offsets) -> Value {
             return self.create<ttg::MemDescSubsliceOp>(resultType, src,
                                                        offsets);
           })
      .def("create_warp_return",
           [](TritonOpBuilder &self) -> Operation * {
             return self.create<ttg::WarpReturnOp>();
           })
      .def("create_warp_yield",
           [](TritonOpBuilder &self, std::vector<Value> values) -> Operation * {
             return self.create<ttg::WarpYieldOp>(values);
           })
      .def("create_warp_specialize_partitions",
           [](TritonOpBuilder &self, int numPartitions) -> Operation * {
             return self.create<ttg::WarpSpecializePartitionsOp>(numPartitions);
           })
      .def("create_warp_specialize",
           [](TritonOpBuilder &self, std::vector<Type> resultTypes,
              std::vector<Value> explicitCaptures,
              std::vector<int> partitionNumWarps) {
             return self.create<ttg::WarpSpecializeOp>(
                 resultTypes, explicitCaptures, partitionNumWarps);
           })
      .def("create_pipe_create",
           [](TritonOpBuilder &self, std::vector<Value> fields,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              std::vector<std::string> readerNames, bool oneShot) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             SmallVector<Attribute> readerNameAttrs;
             readerNameAttrs.reserve(readerNames.size());
             for (StringRef name : readerNames)
               readerNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             ArrayAttr readerNamesAttr;
             if (!readerNameAttrs.empty())
               readerNamesAttr = builder.getArrayAttr(readerNameAttrs);
             BoolAttr oneShotAttr;
             if (oneShot)
               oneShotAttr = builder.getBoolAttr(true);
             self.create<tle::PipeCreateOp>(
                 fields, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNamesAttr,
                 oneShotAttr);
           })
      .def("create_pipe_writer_acquire",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterAcquireOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_writer_commit",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterCommitOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_writer_close",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterCloseOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_reader_wait",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName,
              std::vector<std::string>) -> Value {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             return self.create<tle::PipeReaderWaitOp>(
                 builder.getI1Type(), fields, stage, phase,
                 builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_pipe_reader_release",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName, std::vector<std::string>) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             self.create<tle::PipeReaderReleaseOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_exclusive_cumsum",
           [](TritonOpBuilder &self, Type exclusiveTy, Type totalTy, Value src,
              int axis, bool reverse) -> OpState {
             auto &builder = self.getBuilder();
             return self.create<tle::ExclusiveCumsumOp>(
                 TypeRange{exclusiveTy, totalTy}, src,
                 builder.getI32IntegerAttr(axis), builder.getBoolAttr(reverse));
           })
      .def("create_distributed_barrier",
           [](TritonOpBuilder &self) -> void {
             self.create<tle::DistributedBarrierOp>(
                 StringAttr(), IntegerAttr(), DenseI32ArrayAttr(),
                 DenseI32ArrayAttr(), DenseI32ArrayAttr());
           })
      .def(
          "create_distributed_barrier",
          [](TritonOpBuilder &self, const std::string &groupKind,
             const std::vector<int32_t> &groupShape,
             const std::vector<int32_t> &groupAxes,
             const std::vector<int32_t> &groupMask) -> void {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            StringAttr kindAttr;
            IntegerAttr rankAttr;
            DenseI32ArrayAttr shapeAttr;
            DenseI32ArrayAttr axesAttr;
            DenseI32ArrayAttr maskAttr;

            if (!groupKind.empty()) {
              kindAttr = builder.getStringAttr(groupKind);
            }
            // Only materialize subgroup metadata when provided.
            // This allows kind-only barriers (e.g. group_kind="grid").
            if (!groupShape.empty() || !groupAxes.empty() ||
                !groupMask.empty()) {
              rankAttr = builder.getI32IntegerAttr(
                  static_cast<int32_t>(groupShape.size()));
              if (!groupShape.empty()) {
                shapeAttr = DenseI32ArrayAttr::get(ctx, groupShape);
              }
              if (!groupAxes.empty()) {
                axesAttr = DenseI32ArrayAttr::get(ctx, groupAxes);
              }
              if (!groupMask.empty()) {
                maskAttr = DenseI32ArrayAttr::get(ctx, groupMask);
              }
            }

            self.create<tle::DistributedBarrierOp>(
                kindAttr, rankAttr, shapeAttr, axesAttr, maskAttr);
          },
          py::arg("group_kind"), py::arg("group_shape"), py::arg("group_axes"),
          py::arg("group_mask"))
      .def("create_remote_pointers",
           [](TritonOpBuilder &self, Type resultTy, Value src, Value shardId,
              const std::string &space) -> OpState {
             auto &builder = self.getBuilder();
             static const std::unordered_set<std::string> valid = {
                 "cluster", "device", "node"};
             if (valid.find(space) == valid.end()) {
               throw std::invalid_argument(
                   "Invalid space: " + space +
                   ". Expected one of: cluster, device, node.");
             }
             auto space_attr = builder.getStringAttr(space);
             return self.create<tle::RemotePointersOp>(resultTy, src, shardId,
                                                       space_attr);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding,
              std::string storage) -> Type {
             auto context = self.getBuilder().getContext();
             Attribute memorySpace;
             if (storage == "tmem")
               memorySpace = ttng::TensorMemorySpaceAttr::get(context);
             else if (storage == "smem") {
               memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             } else {
               llvm_unreachable("Unknown storage type");
             }
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace, /*mutableMemory=*/true);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding, std::string storage,
              std::vector<int64_t> allocShape) -> Type {
             auto context = self.getBuilder().getContext();
             Attribute memorySpace;
             if (storage == "tmem")
               memorySpace = ttng::TensorMemorySpaceAttr::get(context);
             else if (storage == "smem") {
               memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             } else {
               llvm_unreachable("Unknown storage type");
             }
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace, /*mutableMemory=*/true,
                                          allocShape);
           });
}

void init_triton_tle_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_early_assign_memory_space",
                     tle::createTritonTleEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0("add_select_encodings",
                     tle::createTritonTleSelectEncodings);
  // Backward-compatible alias.
  ADD_PASS_WRAPPER_0("add_assign_local_pointers_encoding",
                     tle::createTritonTleSelectEncodings);
  ADD_PASS_WRAPPER_0("add_insert_local_pointer_barriers",
                     tle::createTritonTleInsertLocalPointerBarriers);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_loads",
                     tle::createTritonTleOptimizeLocalPointerLoads);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_stores",
                     tle::createTritonTleOptimizeLocalPointerStores);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_async_stores",
                     tle::createTritonTleOptimizeLocalPointerAsyncStores);
  ADD_PASS_WRAPPER_0("add_promote_local_store_staging",
                     tle::createTritonTlePromoteLocalStoreStaging);
  ADD_PASS_WRAPPER_0("add_tile_style_pipeline_schedule",
                     tle::createTritonTleTileStylePipelineSchedule);
  ADD_PASS_WRAPPER_0("add_materialize_tile_style_pipeline",
                     tle::createTritonTleMaterializeTileStylePipeline);
  ADD_PASS_WRAPPER_0("add_downgrade_invalid_async_copy",
                     tle::createTritonTleDowngradeInvalidAsyncCopy);
  ADD_PASS_WRAPPER_0("add_optimize_exclusive_cumsum_layouts",
                     tle::createTritonTleOptimizeExclusiveCumsumLayouts);
  ADD_PASS_WRAPPER_0("add_lower_exclusive_cumsum",
                     tle::createTritonTleLowerExclusiveCumsum);
  ADD_PASS_WRAPPER_0("add_lower_async_load",
                     tle::createTritonTleLowerAsyncLoad);
  ADD_PASS_WRAPPER_0("add_lower_pipe_to_nvws",
                     tle::createTritonTleLowerPipeToNvws);
  ADD_PASS_WRAPPER_0("add_lower_tma_copy", tle::createTritonTleLowerTmaCopy);
  ADD_PASS_WRAPPER_0("add_schedule_tma_store_sync",
                     tle::createTritonTleScheduleTmaStoreSync);

  ADD_PASS_WRAPPER_0("add_lower_extract_tile",
                     tle::createTritonTleLowerExtractTile);

  ADD_PASS_WRAPPER_0("add_lower_insert_tile",
                     tle::createTritonTleLowerInsertTile);
}

void init_tle_raw_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<tle::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                               py::dynamic_attr())
      .def(
          "get_results",
          [](tle::DSLRegionOp &op) -> std::vector<OpResult> {
            auto results_range = op->getResults();
            return std::vector<OpResult>(results_range.begin(),
                                         results_range.end());
          },
          ret::reference)
      .def("dump", &tle::DSLRegionOp::dump);

  py::class_<tle::YieldOp>(m, "YieldOp", py::module_local(), py::dynamic_attr())
      .def("dump", &tle::YieldOp::dump);

  auto *builder_cls = ir::getBuilderClass();
  builder_cls->def("compute_alias_operand_indices",
                   &computeAliasOperandIndices);
  builder_cls->def("create_tle_raw_region_by_llvm_func",
                   &createTLERawRegionByLLVMFunc);
  builder_cls->def("get_context", &TritonOpBuilder::getContext);
}

void init_tle_raw_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_tle_convert_arg_to_memdesc",
                     mlir::triton::tle::createTleConvertArgToMemDesc);
  ADD_PASS_WRAPPER_0("add_tle_remove_redundant_copy",
                     mlir::triton::tle::createTleRemoveRedundantCopy);
  ADD_PASS_WRAPPER_0("add_tle_dsl_region_inline",
                     mlir::triton::tle::createTleDSLRegionInline);
}

void init_llvm(py::module &&m) {
  m.def("parse_llvm_ir",
        [](std::string_view text, llvm::LLVMContext &llvmContext,
           mlir::MLIRContext &mlirContext) -> mlir::ModuleOp {
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(text);
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> llvmModule =
              llvm::parseIR(buffer->getMemBufferRef(), error, llvmContext);
          if (!llvmModule) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          return mlir::translateLLVMIRToModule(std::move(llvmModule),
                                               &mlirContext)
              ->clone();
        });
}

void init_triton_tle(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
#ifdef __FLIR_TILEIR__
    registry.insert<mlir::triton::tile::TileIRDialect>();
    context.appendDialectRegistry(registry);
#endif
    context.loadAllAvailableDialects();
  });
#ifdef __FLIR_TILEIR__
  m.def("load_tile_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::tile::TileIRDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  m.def("lower_gpu_tileir_to_ttir",
        [](ModuleOp &module) { lowerGpuTileIRToTTIR(module); });
#endif

  init_triton_tle_ir(m.def_submodule("ir"));
  init_triton_tle_passes(m.def_submodule("passes"));
  init_tle_raw_ir(m.def_submodule("raw_ir"));
  init_tle_raw_passes(m.def_submodule("raw_passes"));
  init_llvm(m.def_submodule("llvm"));
}
