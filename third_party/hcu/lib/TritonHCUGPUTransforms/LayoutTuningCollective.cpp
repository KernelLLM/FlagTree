// Count the predicate-aware shared primitives in HCU Reduce/ScanOpToLLVM.
// Units are per CTA per invocation BEFORE LLVM CSE/vectorization/ISA selection.
// This deliberately does not reinterpret old nominal words as measured traffic.
#include "TritonHCUGPUTransforms/LayoutTuningCollective.h"
#include "TritonHCUGPUTransforms/LayoutExecutionContext.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace layoutdp {
namespace {
// Unlike the legacy multiplicity helper, an unknown loop or branch does NOT
// contribute one invocation. A known zero outer path still proves zero.
std::optional<double> exactFrequency(Operation *op,
                                    const LayoutExecutionContext &context) {
  double count = 1;
  bool unknown = !context.valid();
  for (Operation *child = op, *p = op->getParentOp(); p;
       child = p, p = p->getParentOp()) {
    if (child->getParentRegion() &&
        !child->getParentRegion()->hasOneBlock()) unknown = true;
    if (auto loop = dyn_cast<scf::ForOp>(p)) {
      if (auto n = context.trips(loop)) count *= *n;
      else unknown = true;
    } else if (auto branch = dyn_cast<scf::IfOp>(p)) {
      if (auto value = context.integer(branch.getCondition())) {
        bool thenRegion = child->getParentRegion() == &branch.getThenRegion();
        if ((!value->isZero()) != thenRegion) count = 0;
      } else unknown = true;
    } else if (isa<scf::WhileOp>(p) ||
               p->getName().getStringRef().starts_with("ttg.warp_specialize")) {
      unknown = true;
    } else if (p->getNumRegions() &&
               p->getName().getStringRef() != "tt.func" &&
               !isa<ModuleOp, scf::ExecuteRegionOp>(p)) {
      // Unknown region owners (parallel/forall/index-switch/custom control)
      // are not transparent one-invocation containers. A single-block
      // execute_region is transparent; its multi-block case is flagged above.
      unknown = true;
    }
  }
  if (!std::isfinite(count) || count < 0 || count > 9007199254740991.) return {};
  if (count == 0 && context.valid()) return 0.;
  return unknown ? std::nullopt : std::optional<double>(count);
}

std::vector<int32_t> coordinates(const tt::LinearLayout &layout,
                                 unsigned reg, unsigned lane, unsigned warp) {
  SmallVector<std::pair<StringAttr, int32_t>> input;
  for (auto name : layout.getInDimNames()) {
    unsigned value = name.getValue() == "register" ? reg :
                     name.getValue() == "lane" ? lane :
                     name.getValue() == "warp" ? warp : 0;
    input.push_back({name, int32_t(value)});
  }
  std::vector<int32_t> output;
  for (auto [name, value] : layout.apply(input)) output.push_back(value);
  return output;
}

using Mask = std::vector<bool>;
Mask allThreads(unsigned threads) { return Mask(threads, true); }

struct Stages {
  llvm::json::Array rows;
  std::map<std::string, double> values;
  unsigned warpSize;
  Stages(unsigned warpSize) : warpSize(warpSize) {
    for (auto side : {"read", "write"})
      for (auto unit : {"scalar_sites", "active_lane_accesses", "active_lane_bytes", "active_warp_issues"})
        values[std::string("collective_shared_") + side + "_" + unit] = 0;
    values["collective_shared_block_sync_sites"] = 0;
  }
  void transfer(llvm::StringRef phase, bool read, unsigned operand, unsigned bits,
                uint64_t sites, const Mask &mask, llvm::StringRef predicate) {
    uint64_t lanes = 0, warps = 0;
    llvm::json::Array masks;
    for (unsigned first = 0; first < mask.size(); first += warpSize) {
      uint64_t word = 0;
      for (unsigned lane = 0; lane < warpSize; ++lane)
        if (mask[first + lane]) { word |= uint64_t(1) << lane; ++lanes; }
      warps += word != 0;
      masks.push_back(llvm::utohexstr(word, true));
    }
    double accesses = double(sites) * lanes;
    std::string prefix = read ? "collective_shared_read_" : "collective_shared_write_";
    values[prefix + "scalar_sites"] += sites;
    values[prefix + "active_lane_accesses"] += accesses;
    values[prefix + "active_lane_bytes"] += accesses * (bits / 8);
    values[prefix + "active_warp_issues"] += double(sites) * warps;
    rows.push_back(llvm::json::Object{
        {"phase", phase.str()}, {"kind", read ? "read" : "write"},
        {"operand", int64_t(operand)}, {"storage_bits", int64_t(bits)},
        {"vector_elements", 1}, {"scalar_sites", int64_t(sites)},
        {"active_lanes_per_site", int64_t(lanes)},
        {"active_warps_per_site", int64_t(warps)},
        {"active_lane_accesses", accesses},
        {"active_lane_bytes", accesses * (bits / 8)},
        {"active_warp_issues", double(sites) * warps},
        {"predicate", predicate.str()}, {"warp_masks_hex", std::move(masks)}});
  }
  void barrier(unsigned sites) { values["collective_shared_block_sync_sites"] += sites; }
};
} // namespace

llvm::json::Object analyzeLayoutCollectiveSharedPlan(
    ModuleOp module, llvm::StringRef executionPayload) {
  LayoutExecutionContext context(module, executionPayload);
  llvm::json::Array operations, issues, explicitIds, otherLDS;
  std::map<std::string, double> totals = Stages(64).values;
  auto target = module->getAttrOfType<StringAttr>("ttg.target");
  bool targetKnown = target && target.getValue() == "hip:gfx936" &&
      ttg::TritonGPUDialect::getThreadsPerWarp(module) == 64 &&
      ttg::TritonGPUDialect::getNumCTAs(module) == 1;
  if (!context.valid()) issues.push_back(context.failure());
  if (!targetKnown) issues.push_back("requires verified HCU gfx936 wave64 single-CTA lowering contract");
  unsigned ordinal = 0;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    unsigned id = ordinal++;
    if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp>(op) ||
        (isa<ttg::LocalAllocOp>(op) && cast<ttg::LocalAllocOp>(op).getSrc()))
      explicitIds.push_back(int64_t(id));
    if (isa<ttg::ConvertLayoutOp, tt::AtomicRMWOp, tt::AtomicCASOp>(op))
      otherLDS.push_back(llvm::json::Object{{"operation", int64_t(id)},
          {"operation_name", op->getName().getStringRef().str()},
          {"reason", "may use separate shared lowering; outside reduce/scan ownership"}});
    bool reduce = isa<tt::ReduceOp>(op), scan = isa<tt::ScanOp>(op);
    if (!reduce && !scan) return;
    auto frequency = exactFrequency(op, context);
    llvm::json::Object row{{"operation", int64_t(id)},
        {"operation_name", op->getName().getStringRef().str()},
        {"id_scope", "same-post-prefix-module-as-explicit-shared"},
        {"available", false}, {"geometry_available", false},
        {"frequency_available", frequency.has_value()},
        {"frequency", frequency ? llvm::json::Value(*frequency) : llvm::json::Value(nullptr)}};
    auto fail = [&](llvm::StringRef why) {
      row["reason"] = why.str();
      issues.push_back(llvm::json::Object{{"operation", int64_t(id)}, {"reason", why.str()}});
      operations.push_back(std::move(row));
    };
    if (!targetKnown) { fail("unverified target lowering"); return; }
    if (!op->getNumOperands()) { fail("empty collective"); return; }
    auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!type || !isa_and_nonnull<ttg::DistributedEncodingTrait>(type.getEncoding())) {
      fail("non-distributed collective payload"); return;
    }
    unsigned warpSize = 64, numWarps = ttg::lookupNumWarps(op), threads = warpSize * numWarps;
    if (!numWarps || threads > 2048) { fail("thread enumeration budget"); return; }
    auto layout = ttg::toLinearLayout(type);
    auto attr = [&](const char *n) { return StringAttr::get(module.getContext(), n); };
    if (layout.getInDimSize(attr("block")) != 1 ||
        layout.getInDimSize(attr("lane")) != warpSize ||
        layout.getInDimSize(attr("warp")) != numWarps ||
        layout.getInDimSizeLog2(attr("register")) > 12) {
      fail("multi-CTA, inconsistent launch or register analysis budget"); return;
    }
    unsigned axis = reduce ? cast<tt::ReduceOp>(op).getAxis() : cast<tt::ScanOp>(op).getAxis();
    row["axis"] = int64_t(axis); row["threads_per_cta"] = int64_t(threads);
    row["warp_size"] = int64_t(warpSize);
    std::vector<unsigned> widths;
    bool supported = true;
    llvm::json::Array payloads;
    for (Type operandType : op->getOperandTypes()) {
      auto tensor = dyn_cast<RankedTensorType>(operandType);
      if (!tensor || tensor.getShape() != type.getShape() || tensor.getEncoding() != type.getEncoding()) { supported = false; break; }
      Type element = tensor.getElementType();
      if (!element.isIntOrFloat()) { supported = false; break; }
      unsigned bits = element.getIntOrFloatBitWidth();
      if (bits != 8 && bits != 16 && bits != 32 && bits != 64) { supported = false; break; }
      widths.push_back(bits); payloads.push_back(int64_t(bits));
    }
    // The numeric combiner must not hide calls, memory or nested control flow.
    op->getRegion(0).walk([&](Operation *node) {
      auto name = node->getName().getStringRef();
      if (isa<tt::ReduceReturnOp, tt::ScanReturnOp>(node)) return;
      if (node->getNumRegions() || (!name.starts_with("arith.") && !name.starts_with("math."))) supported = false;
    });
    if (!supported) { fail("unverified payload storage width or combiner side effects"); return; }
    row["payload_and_storage_bits"] = std::move(payloads);
    row["vector_policy"] = "scalar typed shared primitives per tuple operand; no post-LLVM vectorization inferred";
    Stages stages(warpSize);
    if (reduce) {
      auto reduction = cast<tt::ReduceOp>(op); ReduceOpHelper helper(reduction);
      if (!helper.isReduceWithinCTA()) { fail("cross-CTA reduction"); return; }
      bool crossWarp = !helper.isWarpSynchronous(); row["cross_warp"] = crossWarp;
      row["lowering_path"] = crossWarp ? "HCU-generic-cross-warp" : "HCU-warp-only-no-shared";
      row["target_cta_or_fastpath"] = "HCU TargetInfo inherits unsupported CTA-OR; no alternate shared-free CTA path";
      if (crossWarp) {
        std::set<std::vector<int32_t>> groups;
        for (unsigned reg = 0; reg < layout.getInDimSize(attr("register")); ++reg) {
          auto offset = coordinates(layout, reg, 0, 0); offset[axis] = 0; groups.insert(offset);
        }
        auto free = layout.getFreeVariableMasks();
        unsigned laneFree = free.lookup(attr("lane")), warpFree = free.lookup(attr("warp"));
        Mask first(threads, false), partialRead(threads, false), partialWrite(threads, false);
        uint64_t scratch = 1;
        for (unsigned extent : helper.getScratchRepShape()) scratch *= extent;
        unsigned interWarp = helper.getInterWarpSizeWithUniqueData();
        uint64_t rounds = std::max<uint64_t>(scratch / threads, 1);
        if (!interWarp || scratch > uint64_t(1) << 30) { fail("invalid or oversized scratch plan"); return; }
        for (unsigned thread = 0; thread < threads; ++thread) {
          unsigned lane = thread % warpSize, warp = thread / warpSize;
          auto laneCoords = coordinates(layout, 0, lane, 0);
          first[thread] = !(lane & laneFree) && !(warp & warpFree) && laneCoords[axis] == 0;
          partialRead[thread] = thread < scratch;
          partialWrite[thread] = thread < scratch && lane % interWarp == 0;
        }
        row["accumulator_groups"] = int64_t(groups.size());
        row["scratch_elements_per_operand"] = int64_t(scratch);
        row["scratch_bytes"] = int64_t(helper.getScratchSizeInBytes());
        row["partial_rounds"] = int64_t(rounds);
        for (unsigned i = 0; i < widths.size(); ++i) {
          stages.transfer("warp_partial_store", false, i, widths[i], groups.size(), first,
              "representative_lane && representative_warp && lane_axis==0");
          stages.transfer("cta_partial_load", true, i, widths[i], rounds, partialRead, "thread_id<scratch_elements");
          stages.transfer("cta_partial_store", false, i, widths[i], rounds, partialWrite,
              "thread_id<scratch_elements && lane_id%inter_warp_size==0");
          Type result = reduction.getResult()[i].getType();
          unsigned resultElements = isa<RankedTensorType>(result) ? ttg::getTotalElemsPerThread(result) : 1;
          stages.transfer("result_broadcast_load", true, i, widths[i], resultElements, allThreads(threads), "all threads; repeated result register loads retained before LLVM CSE");
        }
        stages.barrier(2);
      } else row["scratch_bytes"] = int64_t(0);
    } else {
      auto scanOp = cast<tt::ScanOp>(op); ScanLoweringHelper helper(scanOp);
      if (!helper.isSupported()) { fail("scan lowering rejects this encoding"); return; }
      unsigned axisWarps = helper.getAxisNumWarpsWithUniqueData();
      row["cross_warp"] = axisWarps > 1; row["reverse"] = scanOp.getReverse();
      row["lowering_path"] = axisWarps > 1 ? "HCU-cross-warp-scan" : "HCU-warp-only-scan-no-shared";
      if (axisWarps > 1) {
        unsigned axisElements = helper.getAxisNumElementsPerThread();
        unsigned sourceElements = ttg::getTotalElemsPerThread(type);
        if (!axisElements || sourceElements % axisElements) { fail("invalid scan chunk grouping"); return; }
        unsigned chunks = sourceElements / axisElements;
        auto free = layout.getFreeVariableMasks();
        unsigned laneFree = free.lookup(attr("lane")), warpFree = free.lookup(attr("warp"));
        Mask write(threads, false);
        // Delinearization removes free bits. Maximal axis coordinate selects
        // exactly the scanDim-1 lane used by storeWarpAccumulator.
        int32_t last = 0;
        for (unsigned lane = 0; lane < warpSize; ++lane)
          if (!(lane & laneFree)) last = std::max(last, coordinates(layout, 0, lane, 0)[axis]);
        for (unsigned thread = 0; thread < threads; ++thread) {
          unsigned lane = thread % warpSize, warp = thread / warpSize;
          write[thread] = !(lane & laneFree) && !(warp & warpFree) && coordinates(layout, 0, lane, 0)[axis] == last;
        }
        row["chunks"] = int64_t(chunks); row["axis_warps"] = int64_t(axisWarps);
        row["scratch_elements_per_operand"] = int64_t(helper.getScratchSizeInElems());
        row["scratch_bytes"] = int64_t(helper.getScratchSizeInBytes());
        for (unsigned i = 0; i < widths.size(); ++i) {
          stages.transfer("scan_warp_partial_store", false, i, widths[i], chunks, write,
              "representative_lane && representative_warp && lane_axis==scan_dim-1");
          stages.transfer("scan_all_warp_partials_load", true, i, widths[i], uint64_t(chunks) * axisWarps,
              allThreads(threads), "all threads; loop over axis_warps for every chunk");
        }
        stages.barrier(1);
      } else row["scratch_bytes"] = int64_t(0);
    }
    llvm::json::Object invocation, weighted;
    for (auto [key, value] : stages.values) {
      invocation[key] = value;
      if (frequency) { weighted[key] = value * *frequency; totals[key] += value * *frequency; }
    }
    row["geometry_available"] = true; row["stages"] = std::move(stages.rows);
    row["per_invocation"] = std::move(invocation); row["frequency_weighted"] = std::move(weighted);
    if (!frequency) { fail("unknown invocation frequency/control path; no one-trip imputation"); return; }
    row["available"] = true; operations.push_back(std::move(row));
  });
  llvm::json::Object values;
  if (issues.empty()) for (auto [key, value] : totals) values[key] = value;
  return llvm::json::Object{{"schema", "hcu-collective-shared-plan-v217"},
      {"available", issues.empty()}, {"issues", std::move(issues)},
      {"values", std::move(values)}, {"operations", std::move(operations)},
      {"explicit_shared_operation_ids", std::move(explicitIds)},
      {"other_possible_shared_operations", std::move(otherLDS)},
      {"all_program_shared_covered", false}, {"analysis_clone_only", true},
      {"units", "per-CTA predicate-aware scalar lowering primitives; active lane bytes and active warp issues before LLVM optimization; loop frequency applied once, grid not multiplied"},
      {"ownership", "only actual tt.reduce/tt.scan definitions in same module as explicit local transfers; excludes legacy post-dot proxy and conversion/atomic result shared costs"},
      {"exact_instruction_count", false}, {"measured_memory_traffic", false}};
}
} // namespace layoutdp
