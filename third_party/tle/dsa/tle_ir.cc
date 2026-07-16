// Copyright 2026- Xcoresigma Technology Co., Ltd

#include <algorithm>
#include <cctype>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "tle/dsa/dialect/include/IR/Dialect.h"

#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "ir.h"

using namespace mlir;
namespace py = pybind11;

constexpr unsigned kIntegerAttrBitWidth = 64;

struct DSAOpBuilder : public TritonOpBuilder {};

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

// Convert an address-space attribute to a TileIR MemorySpace enum. Keep this
// independent from target-specific dialect classes so TLE can be built for
// CUDA/NVIDIA as well as Ascend.
static mlir::triton::tile::MemorySpace attrToMemSpace(Attribute attr) {
  using MS = mlir::triton::tile::MemorySpace;
  auto text = attrToLowerString(attr);
  if (text.find("register") != std::string::npos)
    return MS::Register;
  if (text.find("shared") != std::string::npos)
    return MS::Shared;
  if (text.find("global") != std::string::npos)
    return MS::Global;
  if (text.find("local") != std::string::npos)
    return MS::Local;
  if (text.find("l0a") != std::string::npos)
    return MS::L0A;
  if (text.find("l0b") != std::string::npos)
    return MS::L0B;
  if (text.find("l0c") != std::string::npos)
    return MS::L0C;
  if (text.find("l1") != std::string::npos)
    return MS::L1;
  if (text.find("gm") != std::string::npos)
    return MS::GM;
  if (text.find("ub") != std::string::npos)
    return MS::UB;
  return MS::UB;
}

static bool isTileBuffer(Value value) {
  return value && isa<mlir::triton::tile::BufType>(value.getType());
}

static bool isTTTensorOrPointer(Value value) {
  if (!value)
    return false;
  auto type = value.getType();
  return isa<RankedTensorType, mlir::triton::PointerType>(type);
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
                   [](Value result) { return result.use_empty(); })) {
    op->erase();
  }
}

static void lowerGpuTileIRToTTIR(ModuleOp module) {
  OpBuilder builder(module.getContext());
  llvm::DenseMap<Value, Value> bufferValues;
  SmallVector<Operation *> eraseOps;

  module.walk([&](Operation *op) {
    if (auto copyOp = dyn_cast<mlir::triton::tile::CopyOp>(op)) {
      Value src = materializeTileBuffer(copyOp.getSrc(), bufferValues);
      if (!src)
        src = copyOp.getSrc();
      Value dst = copyOp.getDst();
      if (isTileBuffer(dst)) {
        bufferValues[dst] = src;
        eraseOps.push_back(op);
        return;
      }
      if (isTTTensorOrPointer(dst) && isTTTensorOrPointer(src)) {
        builder.setInsertionPoint(op);
        builder.create<mlir::triton::StoreOp>(
            op->getLoc(), dst, src, mlir::triton::CacheModifier::NONE,
            mlir::triton::EvictionPolicy::NORMAL);
        eraseOps.push_back(op);
      }
      return;
    }

    if (auto storeTensorOp = dyn_cast<mlir::triton::tile::StoreTensorOp>(op)) {
      bufferValues[storeTensorOp.getDst()] = storeTensorOp.getSrc();
      eraseOps.push_back(op);
      return;
    }

    if (auto subviewOp = dyn_cast<mlir::triton::tile::SubViewOp>(op)) {
      if (Value src = materializeTileBuffer(subviewOp.getSource(), bufferValues))
        bufferValues[subviewOp.getResult()] = src;
      eraseOps.push_back(op);
      return;
    }

    if (auto toTensorOp = dyn_cast<mlir::triton::tile::ToTensorOp>(op)) {
      Value value = materializeTileBuffer(toTensorOp.getSrc(), bufferValues);
      if (!value) {
        toTensorOp.emitError("cannot lower tile.to_tensor without a preceding "
                             "tile.copy or tile.store_tensor");
        return;
      }

      builder.setInsertionPoint(op);
      Value replacement = value;
      if (isa<mlir::triton::PointerType>(value.getType())) {
        auto load = builder.create<mlir::triton::LoadOp>(
            op->getLoc(), value, mlir::triton::CacheModifier::NONE,
            mlir::triton::EvictionPolicy::NORMAL, false);
        replacement = load.getResult();
      }
      toTensorOp.getResult().replaceAllUsesWith(replacement);
      eraseOps.push_back(op);
    }
  });

  for (Operation *op : llvm::reverse(eraseOps))
    eraseIfUnused(op);

  SmallVector<Operation *> cleanupOps;
  module.walk([&](Operation *op) {
    if (isa<mlir::triton::tile::AllocOp, mlir::triton::tile::SubViewOp,
            mlir::triton::tile::CopyOp, mlir::triton::tile::StoreTensorOp,
            mlir::triton::tile::ToTensorOp>(op)) {
      cleanupOps.push_back(op);
    }
  });
  for (Operation *op : llvm::reverse(cleanupOps))
    eraseIfUnused(op);
}

void init_triton_tle(py::module &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<triton::tle::TleDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  auto tle_cls = py::class_<DSAOpBuilder>(
      m, "tle_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      .def("restore_insertion_point",
           [](DSAOpBuilder &self, OpBuilder::InsertPoint pt) {
             self.restoreInsertionPoint(pt);
           })
      .def("set_loc",
           [](DSAOpBuilder &self, Location loc) { self.setLastLoc(loc); })
      .def("set_loc",
           [](DSAOpBuilder &self, const std::string &fileName, int line,
              int column) { self.setLastLoc(fileName, line, column); })
      .def("dsa_get_null_attr", [](DSAOpBuilder &self) { return Attribute(); })
      .def("dsa_get_string_attr",
           [](DSAOpBuilder &self, const std::string &name) -> Attribute {
             return self.getBuilder().getStringAttr(name);
           })
      .def("dsa_get_buffer_type",
           [](DSAOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             return MemRefType::get(shape, elementType,
                                    MemRefLayoutAttrInterface{}, memorySpace);
           })
      .def("dsa_get_buffer_type_with_strides",
           [](DSAOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const std::vector<int64_t> &strides,
              const Attribute &memorySpace) -> Type {
             // create a layout with strides, using dynamic offset
             auto layout = StridedLayoutAttr::get(
                 self.getBuilder().getContext(), ShapedType::kDynamic, strides);
             return MemRefType::get(shape, elementType, layout, memorySpace);
           })
      .def("create_dsa_alloc",
           [](DSAOpBuilder &self, Type memrefType) -> Value {
             return self.create<memref::AllocOp>(
                 mlir::cast<MemRefType>(memrefType));
           })
      // Add copy op
      .def("create_dsa_copy",
           [](DSAOpBuilder &self, Value &src, Value &dst,
              std::vector<Value> &shape, bool inter_no_alias) -> void {
             auto copyOp = self.create<triton::tle::DSACopyOp>(src, dst, shape);
             if (inter_no_alias) {
               copyOp->setAttr("inter_no_alias",
                               self.getBuilder().getBoolAttr(true));
             }
           })
      // Add op
      .def("create_dsa_add",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAAddOp>(lhs, rhs, res);
           })
      // Sub op
      .def("create_dsa_sub",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSASubOp>(lhs, rhs, res);
           })
      // Mul op
      .def("create_dsa_mul",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAMulOp>(lhs, rhs, res);
           })
      // Div op
      .def("create_dsa_div",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSADivOp>(lhs, rhs, res);
           })
      // Max op
      .def("create_dsa_max",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAMaxOp>(lhs, rhs, res);
           })
      // Min op
      .def("create_dsa_min",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAMinOp>(lhs, rhs, res);
           })
      // Dot op
      /// .def("create_dsa_dot",
      ///      [](DSAOpBuilder &self, Value &inA, Value &inB, Value &res,
      ///         std::vector<int64_t> &size, bool &initC, bool &traA, bool
      ///         &traB, bool &enable_hf32) -> void {
      ///        auto &builder = self.getBuilder();
      ///        auto sizeAttr = builder.getI64ArrayAttr(size);

      ///        // convert bool to boolattr.
      ///        auto initC_attr = builder.getBoolAttr(initC);
      ///        auto traA_attr = builder.getBoolAttr(traA);
      ///        auto traB_attr = builder.getBoolAttr(traB);
      ///        auto enable_hf32_attr = builder.getBoolAttr(enable_hf32);

      ///        self.create<triton::tle::DSADotOp>(inA, inB, res, sizeAttr,
      ///        initC_attr,
      ///                              traA_attr, traB_attr, enable_hf32_attr);
      ///      })
      .def("dsa_to_buffer",
           [](DSAOpBuilder &self, Value &src,
              const Attribute &addressSpace) -> Value {
             auto tensorType = dyn_cast<RankedTensorType>(src.getType());
             if (!tensorType) {
               llvm::report_fatal_error("to_buffer: src must be tensor type");
             }
             auto memrefType = MemRefType::get(
                 tensorType.getShape(), tensorType.getElementType(),
                 MemRefLayoutAttrInterface{}, addressSpace);
             return self.create<bufferization::ToMemrefOp>(memrefType, src);
           })
      .def("dsa_to_tensor",
           [](DSAOpBuilder &self, Value &src, bool writable) -> Value {
             const auto &memrefType = mlir::cast<MemRefType>(src.getType());
             auto hasAddressSpace = memrefType.getMemorySpace();
             if (hasAddressSpace) {
               return self.create<bufferization::ToTensorOp>(src, true,
                                                             writable);
             }
             return self.create<bufferization::ToTensorOp>(src, true, writable);
           })
      .def("create_dsa_extract_scalar",
           [](DSAOpBuilder &self, Value &src,
              std::vector<Value> &indices) -> Value {
             llvm::SmallVector<Value> arg_indices;
             for (const auto &i : indices) {
               auto iTy = i.getType();
               if (!iTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), i);
                 arg_indices.push_back(v);
               } else {
                 arg_indices.push_back(i);
               }
             }
             auto ret = self.create<tensor::ExtractOp>(src, arg_indices);
             return ret;
           })
      .def("create_dsa_extract_slice",
           [](DSAOpBuilder &self, Value &ful, std::vector<Value> &offs_vec,
              std::vector<int> &sizs_vec, std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
             }
             llvm::SmallVector<OpFoldResult> mixedOffsets;
             for (Value offset : offsets) {
               mixedOffsets.push_back(offset);
             }
             llvm::SmallVector<OpFoldResult> mixedSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               mixedSizes.push_back(self.getBuilder().getIndexAttr(s));
               retSizes.push_back(s);
             }
             llvm::SmallVector<OpFoldResult> mixedStrides;
             for (const auto &s : strd_vec) {
               mixedStrides.push_back(self.getBuilder().getIndexAttr(s));
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(
                 retTy, ful, mixedOffsets, mixedSizes, mixedStrides);
           })
      .def("create_dsa_insert_slice",
           [](DSAOpBuilder &self, Value &ful, Value &sub,
              std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
              std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
             }
             llvm::SmallVector<OpFoldResult> mixedOffsets;
             for (Value offset : offsets) {
               mixedOffsets.push_back(offset);
             }
             llvm::SmallVector<OpFoldResult> mixedSizes;
             for (const auto &s : sizs_vec) {
               mixedSizes.push_back(self.getBuilder().getIndexAttr(s));
             }
             llvm::SmallVector<OpFoldResult> mixedStrides;
             for (const auto &s : strd_vec) {
               mixedStrides.push_back(self.getBuilder().getIndexAttr(s));
             }
             auto ret = self.create<tensor::InsertSliceOp>(
                 sub, ful, mixedOffsets, mixedSizes, mixedStrides);
             return ret;
           })
      .def("create_dsa_subview",
           [](DSAOpBuilder &self, Value source, std::vector<Value> &offsets,
              const std::vector<int64_t> &sizes,
              const std::vector<int64_t> &strides) -> Value {
             SmallVector<mlir::OpFoldResult> mixedOffsets;
             auto *context = self.getBuilder().getContext();
             auto &builder = self.getBuilder();

             // Get source memref type for validation
             auto sourceType = mlir::cast<MemRefType>(source.getType());
             int64_t rank = sourceType.getRank();
             // Verify the number of parameters
             if (offsets.size() != rank || sizes.size() != rank ||
                 strides.size() != rank) {
               throw std::runtime_error("Number of offsets, sizes, and strides "
                                        "must match memref rank");
             }

             for (const auto &offset : offsets) {
               auto indexType = builder.getIndexType();
               if (offset.getType() != indexType) {
                 Value offset_val =
                     self.create<arith::IndexCastOp>(indexType, offset);
                 mixedOffsets.push_back(offset_val);
               } else {
                 mixedOffsets.push_back(offset);
               }
             }

             SmallVector<mlir::OpFoldResult> mixedSizes;
             SmallVector<mlir::OpFoldResult> mixedStrides;
             for (int64_t i = 0; i < rank; ++i) {
               int64_t size = sizes[i];
               int64_t stride = strides[i];
               int64_t srcDim = sourceType.getDimSize(i);

               // verify sizes cannot be negative or zero
               if (size <= 0) {
                 throw std::runtime_error("Expected sizes to be positive");
               }

               // verify strides cannot be negative or zero
               if (stride <= 0) {
                 throw std::runtime_error("Expected strides to be positive");
               }

               // getDimSize() returns -1 (ShapedType::kDynamic) for dynamic
               // dimensions
               if (!ShapedType::isDynamic(srcDim)) {
                 // verify the subview size does not exceed the source dimension
                 if (size > srcDim) {
                   throw std::runtime_error(
                       "Subview size cannot exceed source dimension size");
                 }

                 // verify strides cannot exceed the source dimension size
                 if (stride > srcDim) {
                   throw std::runtime_error(
                       "Stride cannot exceed source dimension size");
                 }
               }

               mixedSizes.push_back(IntegerAttr::get(
                   IntegerType::get(context, kIntegerAttrBitWidth), size));
               mixedStrides.push_back(IntegerAttr::get(
                   IntegerType::get(context, kIntegerAttrBitWidth), stride));
             }

             return self.create<memref::SubViewOp>(source, mixedOffsets,
                                                   mixedSizes, mixedStrides);
           });

  // ============================================================================
  // TileIR builder methods — create tile.* dialect ops
  // ============================================================================

  // Helper: load TileIR dialect into context
  m.def("load_tile_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<mlir::triton::tile::TileIRDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  m.def("lower_gpu_tileir_to_ttir",
        [](ModuleOp &module) { lowerGpuTileIRToTTIR(module); });

  // TileIR buffer / tensor type construction
  tle_cls.def("tile_get_buffer_type",
       [](DSAOpBuilder &self, std::vector<int64_t> &shape,
          Type &elementType, const Attribute &memorySpace) -> Type {
         auto memSpace = attrToMemSpace(memorySpace);
         auto *ctx = self.getBuilder().getContext();
         return mlir::triton::tile::BufType::get(ctx, shape, elementType, memSpace);
       })
  .def("tile_get_tensor_type",
       [](DSAOpBuilder &self, std::vector<int64_t> &shape,
          Type &elementType, const Attribute &memorySpace) -> Type {
         auto memSpace = attrToMemSpace(memorySpace);
         auto *ctx = self.getBuilder().getContext();
         return mlir::triton::tile::TensorType::get(ctx, shape, elementType, memSpace);
       })

  // tile.alloc — result type carries the memory space; pass it as the $space attr
  .def("create_tile_alloc",
       [](DSAOpBuilder &self, Type tileBufType) -> Value {
         auto bufType = mlir::cast<mlir::triton::tile::BufType>(tileBufType);
         return self.create<mlir::triton::tile::AllocOp>(
             tileBufType, bufType.getMemorySpace(),
             /*shape=*/mlir::ArrayAttr(), /*dtype=*/mlir::TypeAttr(),
             /*policy=*/mlir::triton::tile::PolicyAttr(),
             /*layout=*/mlir::triton::tile::LayoutAttr::get(
                 self.getBuilder().getContext(), mlir::triton::tile::Layout::ND),
             /*lifetime=*/mlir::triton::tile::LifetimeAttr(),
             /*comment=*/mlir::StringAttr());
       })
  // tile.copy — shape extents are informational at this layer; the op itself
  // takes only src/dst (+ optional engine/layout attrs).
  .def("create_tile_copy",
       [](DSAOpBuilder &self, Value &src, Value &dst,
          std::vector<Value> & /*shape*/, bool inter_no_alias) -> void {
         auto op = self.create<mlir::triton::tile::CopyOp>(
             src, dst, /*engine=*/mlir::triton::tile::EngineAttr(),
             /*src_layout=*/mlir::triton::tile::LayoutAttr::get(
                 self.getBuilder().getContext(), mlir::triton::tile::Layout::ND),
             /*dst_nz_layout=*/mlir::triton::tile::NZLayoutAttr(),
             /*transpose=*/mlir::UnitAttr(), /*comment=*/mlir::StringAttr());
         if (inter_no_alias) {
           op->setAttr("inter_no_alias", self.getBuilder().getBoolAttr(true));
         }
       })
  // tile.subview — result buffer type = sizes + source elt/space
  .def("create_tile_subview",
       [](DSAOpBuilder &self, Value source, std::vector<Value> &offsets,
          const std::vector<int64_t> &sizes,
          const std::vector<int64_t> &strides) -> Value {
         SmallVector<Value> indexOffsets;
         auto &builder = self.getBuilder();
         auto indexType = builder.getIndexType();
         for (Value offset : offsets) {
           if (offset.getType() != indexType) {
             offset = self.create<arith::IndexCastOp>(indexType, offset);
           }
           indexOffsets.push_back(offset);
         }
         auto *ctx = self.getBuilder().getContext();
         auto srcBuf = mlir::cast<mlir::triton::tile::BufType>(source.getType());
         auto resTy = mlir::triton::tile::BufType::get(
             ctx, sizes, srcBuf.getElementType(), srcBuf.getMemorySpace());
         auto op = self.create<mlir::triton::tile::SubViewOp>(
             resTy, source, indexOffsets,
             self.getBuilder().getI64ArrayAttr(sizes),
             self.getBuilder().getI64ArrayAttr(strides));
         return op.getResult();
       })
  // tile.to_tensor — result is a standard ranked tensor (so tt.dot etc. accept
  // it), mirroring the source buffer's shape and element type.
  .def("create_tile_to_tensor",
       [](DSAOpBuilder &self, Value &src, bool /*writable*/) -> Value {
         auto srcBuf = mlir::cast<mlir::triton::tile::BufType>(src.getType());
         auto resTy = mlir::RankedTensorType::get(srcBuf.getShape(),
                                                  srcBuf.getElementType());
         auto op = self.create<mlir::triton::tile::ToTensorOp>(resTy, src);
         return op.getResult();
       })
  // tile.store_tensor
  .def("create_tile_store_tensor",
       [](DSAOpBuilder &self, Value &src, Value &dst) -> void {
         self.create<mlir::triton::tile::StoreTensorOp>(src, dst);
       })
  // tile.set_flag
  .def("create_tile_set_flag",
       [](DSAOpBuilder &self, int64_t producer, int64_t consumer,
          int64_t event) -> void {
         self.create<mlir::triton::tile::SetFlagOp>(
             static_cast<mlir::triton::tile::Pipe>(producer),
             static_cast<mlir::triton::tile::Pipe>(consumer),
             static_cast<mlir::triton::tile::EventID>(event));
       })
  // tile.wait_flag
  .def("create_tile_wait_flag",
       [](DSAOpBuilder &self, int64_t producer, int64_t consumer,
          int64_t event) -> void {
         self.create<mlir::triton::tile::WaitFlagOp>(
             static_cast<mlir::triton::tile::Pipe>(producer),
             static_cast<mlir::triton::tile::Pipe>(consumer),
             static_cast<mlir::triton::tile::EventID>(event));
       })
  // tile.pipe_barrier
  .def("create_tile_pipe_barrier",
       [](DSAOpBuilder &self, int64_t pipe) -> void {
         self.create<mlir::triton::tile::PipeBarrierOp>(
             static_cast<mlir::triton::tile::Pipe>(pipe));
       })
  // tile.gm_offset — result pointer type matches the base
  .def("create_tile_gm_offset",
       [](DSAOpBuilder &self, Value &base, std::vector<Value> &indices,
          std::vector<Value> &strides) -> Value {
         SmallVector<Value> indexValues;
         SmallVector<Value> strideValues;
         auto &builder = self.getBuilder();
         auto indexType = builder.getIndexType();
         for (Value index : indices) {
           if (index.getType() != indexType) {
             index = self.create<arith::IndexCastOp>(indexType, index);
           }
           indexValues.push_back(index);
         }
         for (Value stride : strides) {
           if (stride.getType() != indexType) {
             stride = self.create<arith::IndexCastOp>(indexType, stride);
           }
           strideValues.push_back(stride);
         }
         auto op = self.create<mlir::triton::tile::GmOffsetOp>(
             base.getType(), base, indexValues, strideValues);
         return op.getResult();
       })
  // tile.cube_launch — async Cube matmul hook. The current TileIR op has no
  // token result, so the matching wait is emitted as a standalone op.
  .def("create_tile_cube_launch",
       [](DSAOpBuilder &self, Value &a, Value &b, Value &acc, Value &stageA,
          Value &stageB, Value &dst, bool transposeA, bool transposeB,
          bool init, std::string mma) -> void {
         auto &builder = self.getBuilder();
         auto unitAttr = builder.getUnitAttr();
         self.create<mlir::triton::tile::CubeLaunchOp>(
             a, b, acc, stageA, stageB, dst,
             transposeA ? unitAttr : mlir::UnitAttr(),
             transposeB ? unitAttr : mlir::UnitAttr(),
             init ? unitAttr : mlir::UnitAttr(),
             mma.empty() ? mlir::StringAttr() : builder.getStringAttr(mma),
             /*comment=*/mlir::StringAttr());
       })
  // tile.cube_wait
  .def("create_tile_cube_wait",
       [](DSAOpBuilder &self) -> void {
         self.create<mlir::triton::tile::CubeWaitOp>();
       });
}
