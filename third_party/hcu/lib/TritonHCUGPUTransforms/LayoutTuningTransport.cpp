#include "TritonHCUGPUTransforms/LayoutTuningTransport.h"
#include "TritonHCUGPUTransforms/LayoutExecutionContext.h"
#include "Analysis/HCUGPUAllocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include <cmath>
#include <map>

namespace layoutdp {
namespace {
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

llvm::json::Array mapping(const tt::LinearLayout &layout) {
  llvm::json::Array result;
  for (const auto &[name, bases] : layout.getBases()) {
    llvm::json::Array columns;
    for (const auto &basis : bases) {
      llvm::json::Array vector;
      for (int32_t v : basis) vector.push_back(int64_t(v));
      columns.push_back(std::move(vector));
    }
    result.push_back(llvm::json::Object{{"dimension", name.getValue().str()},
        {"size", int64_t(layout.getInDimSize(name))}, {"bases", std::move(columns)}});
  }
  return result;
}
std::string printed(Type type) {
  std::string result; llvm::raw_string_ostream os(result); os << type; return result;
}
} // namespace

llvm::json::Object addLayoutConversionTransport(
    mlir::ModuleOp module, llvm::json::Object explicitTransport,
    llvm::StringRef executionContext) {
  LayoutExecutionContext context(module, executionContext);
  llvm::json::Array operations, issues, routes;
  std::map<std::string, double> totals;
  for (const char *side : {"read", "write"})
    for (const char *unit : {"vector_groups", "word_work"})
      totals[std::string("transport_shared_") + side + "_" + unit] = 0;
  if (auto *rows = explicitTransport.getArray("operations"))
    for (auto &row : *rows) {
      if (auto *object = row.getAsObject()) (*object)["transport_kind"] = "explicit-local";
      operations.push_back(std::move(row));
    }
  if (auto *errors = explicitTransport.getArray("issues"))
    for (auto &error : *errors) issues.push_back(std::move(error));
  if (!explicitTransport.getBoolean("available").value_or(false) && issues.empty())
    issues.push_back("explicit local transfer facts unavailable");
  if (auto *values = explicitTransport.getObject("values")) {
    for (auto &[name, total] : totals) {
      auto v = values->getNumber(name);
      if (!v || !std::isfinite(*v) || *v < 0) {
        if (issues.empty()) issues.push_back("invalid explicit local transfer total");
      } else total = *v;
    }
  } else issues.push_back("missing explicit local transfer totals");
  llvm::json::Object explicitValues;
  if (auto *values = explicitTransport.getObject("values"))
    explicitValues = std::move(*values);
  if (!context.valid() && issues.empty()) issues.push_back(context.failure());
  unsigned ordinal = 0, ldsConversions = 0, zeroLdsConversions = 0;
  module.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    unsigned id = ordinal++;
    auto op = dyn_cast<ttg::ConvertLayoutOp>(operation);
    if (!op) return;
    auto src = op.getSrc().getType(), dst = op.getType();
    auto *ctx = op.getContext();
    auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
         warp = StringAttr::get(ctx, "warp"), block = StringAttr::get(ctx, "block"),
         reps = StringAttr::get(ctx, "reps"), offset = StringAttr::get(ctx, "offset"),
         iteration = StringAttr::get(ctx, "iteration");
    llvm::json::Object route{{"operation", int64_t(id)}, {"source_type", printed(src)},
        {"destination_type", printed(dst)}, {"available", false}};
    auto reject = [&](const char *reason) {
      route["reason"] = reason; issues.push_back(reason); routes.push_back(std::move(route));
    };
    auto conversion = mlir::minimalCvtLayout(src, dst);
    if (llvm::is_contained(conversion.getInDimNames(), block)) {
      reject("cross-CTA implicit shared conversion unsupported"); return;
    }
    if (!mlir::cvtNeedsSharedMemory(src, dst)) {
      ++zeroLdsConversions;
      route["available"] = true; route["requires_lds"] = false;
      route["route"] = mlir::cvtReordersRegisters(src, dst) ? "register-only-or-identity" : "warp-shuffle";
      routes.push_back(std::move(route)); return;
    }
    ++ldsConversions;
    Type element = src.getElementType();
    bool pointer = isa<tt::PointerType>(element);
    if (!pointer && !element.isIntOrFloat()) {
      reject("unsupported implicit shared payload type"); return;
    }
    unsigned payloadBits = pointer ? 64 : element.getIntOrFloatBitWidth();
    unsigned storageBits = std::max(8u, payloadBits);
    if (storageBits != 8 && storageBits != 16 && storageBits != 32 && storageBits != 64) {
      reject("implicit shared storage outside 8/16/32/64-bit contract"); return;
    }
    auto srcLayout = ttg::toLinearLayout(src), dstLayout = ttg::toLinearLayout(dst);
    auto within = [&](const tt::LinearLayout &ll) {
      return ll.sublayout({reg, lane, warp}, llvm::to_vector(ll.getOutDimNames()));
    };
    bool padded = op->hasAttr(tt::HCU::AttrSharedMemPadded);
    auto s = padded ? ttg::getLayoutWithinBlock(srcLayout) :
                     tt::actionRemoveBroadcastedRegs(within(srcLayout)).apply(within(srcLayout));
    auto d = padded ? ttg::getLayoutWithinBlock(dstLayout) :
                     tt::actionRemoveBroadcastedRegs(within(dstLayout)).apply(within(dstLayout));
    double weight = layoutExecutionWeight(op, &context);
    if (!std::isfinite(weight) || weight < 0) {
      reject("invalid implicit shared execution multiplicity"); return;
    }
    bool sidesAvailable = true;
    auto record = [&](bool read, const tt::LinearLayout &relative, unsigned vector,
                      unsigned repetitions, unsigned elementsPerRepetition,
                      unsigned originalElements) {
      if (!vector || !repetitions || !elementsPerRepetition || elementsPerRepetition % vector) {
        sidesAvailable = false; issues.push_back("invalid implicit shared vector/repetition grouping"); return;
      }
      uint64_t elements = uint64_t(elementsPerRepetition) * repetitions;
      double groups = double(elements) / vector;
      double words = double(elements) * storageBits / 32;
      llvm::json::Array outputs;
      for (auto name : relative.getOutDimNames()) outputs.push_back(name.getValue().str());
      llvm::json::Object row{{"operation", int64_t(id)}, {"operation_name", "ttg.convert_layout"},
          {"transport_kind", "implicit-layout-conversion"}, {"available", true}, {"read", read},
          {"strategy", padded ? "padded" : "swizzled"}, {"payload_bits", int64_t(payloadBits)},
          {"storage_bits", int64_t(storageBits)}, {"pointer_payload", pointer},
          {"original_register_elements", int64_t(originalElements)},
          {"broadcast_registers_removed", !padded}, {"register_elements_per_repetition", int64_t(elementsPerRepetition)},
          {"repetitions", int64_t(repetitions)}, {"transferred_register_elements", int64_t(elements)},
          {"vector_elements", int64_t(vector)}, {"vector_bits", int64_t(vector * storageBits)},
          {"vector_groups", groups}, {"word_work", words}, {"execution_weight", weight},
          {"relative_mapping", mapping(relative)}, {"relative_outputs", std::move(outputs)},
          {"exact_instruction_count", false}};
      std::string prefix = read ? "transport_shared_read_" : "transport_shared_write_";
      totals[prefix + "vector_groups"] += weight * groups;
      totals[prefix + "word_work"] += weight * words;
      operations.push_back(std::move(row));
    };
    if (padded) {
      auto config = tt::HCU::getScratchConfigForCvt(src, dst);
      if (config.repShape.empty() || !config.inVec || !config.outVec) {
        reject("missing padded implicit scratch configuration"); return;
      }
      auto shape = ttg::getShapePerCTA(src.getEncoding(), dst.getShape());
      SmallVector<unsigned> ctaShape(shape.begin(), shape.end());
      auto shared = ttg::chooseShemLayoutForRegToRegConversion(ctx, ctaShape, config.repShape, config.order);
      unsigned count = shared.getInDimSize(iteration);
      auto side = [&](bool read, const tt::LinearLayout &ll, unsigned vector) {
        auto relative = ll.invertAndCompose(shared);
        if (!relative.sublayoutIsZero({lane, warp, block}, {iteration}) ||
            !count || relative.getInDimSize(reg) % count) {
          sidesAvailable = false; issues.push_back("padded implicit iteration is not register-owned"); return;
        }
        record(read, relative, vector, count, relative.getInDimSize(reg) / count,
               read ? dstLayout.getInDimSize(reg) : srcLayout.getInDimSize(reg));
      };
      side(false, s, config.inVec); side(true, d, config.outVec);
    } else {
      auto shared = ttg::optimalSwizzlingLdSt(s, d, storageBits);
      unsigned count = shared.getInDimSize(reps);
      auto repetition = tt::LinearLayout::identity1D(count, reg, reps);
      auto side = [&](bool read, const tt::LinearLayout &ll) {
        auto total = ll.invertAndCompose(shared);
        auto permutation = tt::regPermForDivide(total, repetition, false);
        if (!permutation) {
          sidesAvailable = false; issues.push_back("implicit repetition permutation unavailable"); return;
        }
        auto divided = divideRight(permutation->apply(total), repetition);
        if (!divided) {
          sidesAvailable = false; issues.push_back("implicit repetition division unavailable"); return;
        }
        auto relative = divided->reshapeOuts({{offset, divided->getTotalOutDimSize()}});
        auto [vector, vectorPermutation] = tt::largestVectorisation(ctx, relative, storageBits, std::nullopt);
        record(read, relative, vector, count, relative.getInDimSize(reg),
               read ? dstLayout.getInDimSize(reg) : srcLayout.getInDimSize(reg));
      };
      side(false, s); side(true, d);
    }
    route["available"] = sidesAvailable; route["requires_lds"] = true;
    route["route"] = "lds"; route["strategy"] = padded ? "padded" : "swizzled";
    routes.push_back(std::move(route));
  });
  llvm::json::Object values;
  if (issues.empty()) for (const auto &[name, value] : totals) values[name] = value;
  return llvm::json::Object{{"schema", "bounded-shared-transport-closure"},
      {"coverage", "explicit-local-and-layout-conversion"}, {"available", issues.empty()},
      {"issues", std::move(issues)}, {"values", std::move(values)},
      {"operations", std::move(operations)}, {"conversion_routes", std::move(routes)},
      {"explicit_values", std::move(explicitValues)}, {"lds_conversions", int64_t(ldsConversions)},
      {"zero_lds_conversions", int64_t(zeroLdsConversions)}, {"analysis_clone_only", true},
      {"exact_instruction_count", false},
      {"units", "per-lane native transfer-template vector groups and 32-bit word equivalents, weighted by context/static loop multiplicity; no CTA/lane multiplier"},
      {"scope_exclusions", "reduce/scan/gather/atomic internal LDS remain in their existing feature families; not included a second time"},
      {"lowering_boundary", "after selected OptimizeLDSUsage; before LLVM target splitting, folding or instruction selection; not issued ISA counts"},
      {"pipeline", "shared-prefix+cleanup+opt-dot+RLC+reduce-data-duplication+optimize-LDS"}};
}
} // namespace layoutdp
