// No scalar resource penalty is installed here. These features must be
// calibrated against backend artifacts, separately from memory/convert prices.
#include "TritonHCUGPUTransforms/LayoutTuningResources.h"
#include "Analysis/HCUGPUAllocation.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace layoutdp {
namespace {
std::string typeString(Type type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}
struct Footprint {
  uint64_t elements = 1, unique = 1, packed = 0, unpacked = 0;
  unsigned bits = 0;
  bool uniform = false, constant = false, known = true;
  std::string kind;
};
Footprint footprint(Value v, tt::ModuleAxisInfoAnalysis &axes) {
  Footprint f;
  Type type = v.getType(), element = type;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    element = tensor.getElementType();
    if (!tensor.getEncoding() || !isa<ttg::DistributedEncodingTrait>(tensor.getEncoding())) {
      f.known = false; f.kind = "unmodeled-tensor"; return f;
    }
    auto layout = ttg::toLinearLayout(tensor);
    auto unique = tt::actionRemoveBroadcastedRegs(layout).apply(layout);
    f.unique = unique.getInDimSize(StringAttr::get(v.getContext(), "register"));
    f.elements = ttg::getTotalElemsPerThread(tensor);
    if (auto *info = axes.getAxisInfo(v)) {
      auto c = info->getConstancy();
      f.uniform = c.size() == tensor.getRank();
      for (unsigned i = 0; i < c.size() && f.uniform; ++i)
        f.uniform &= c[i] >= tensor.getShape()[i];
    }
    f.kind = f.uniform ? "uniform-tensor" : "distributed-tensor";
  } else if (isa<ttg::MemDescType>(type)) {
    f.kind = "memdesc-address-proxy"; f.bits = 64;
    f.packed = f.unpacked = 2; return f;
  } else {
    f.kind = "scalar";
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Operation *owner = arg.getOwner()->getParentOp();
      auto visibility = owner->getAttrOfType<StringAttr>("sym_visibility");
      auto function = dyn_cast<FunctionOpInterface>(owner);
      f.uniform = function && !function.getFunctionBody().empty() &&
                  arg.getOwner() == &function.getFunctionBody().front() &&
                  (!visibility || visibility.getValue() == "public");
    }
    if (auto *op = v.getDefiningOp()) {
      auto name = op->getName().getStringRef();
      f.uniform = name == "tt.get_program_id" || name == "tt.get_num_programs";
    }
  }
  if (isa<tt::PointerType>(element)) f.bits = 64;
  else if (auto numeric = dyn_cast<IntegerType>(element)) f.bits = std::max(8u, numeric.getWidth());
  else if (auto numeric = dyn_cast<FloatType>(element)) f.bits = std::max(8u, numeric.getWidth());
  else if (isa<IndexType>(element)) f.bits = 64;
  else { f.known = false; f.kind = "unmodeled-type"; return f; }
  f.constant = isa_and_nonnull<arith::ConstantOp>(v.getDefiningOp());
  if (f.constant) {
    f.uniform = !isa<RankedTensorType>(type);
    if (auto constant = v.getDefiningOp<arith::ConstantOp>())
      if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue())) f.uniform = dense.isSplat();
  }
  uint64_t n = f.uniform ? 1 : f.unique;
  f.packed = (n * f.bits + 31) / 32;
  f.unpacked = n * ((f.bits + 31) / 32);
  return f;
}
struct Ids {
  DenseMap<Operation *, unsigned> ops;
  DenseMap<Value, unsigned> values;
  DenseMap<Block *, unsigned> blocks;
  std::vector<Operation *> orderedOps;
  std::vector<Value> orderedValues;
  explicit Ids(ModuleOp module) {
    auto add = [&](Value v) {
      if (values.count(v)) return;
      values[v] = orderedValues.size(); orderedValues.push_back(v);
    };
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      ops[op] = orderedOps.size(); orderedOps.push_back(op);
      if (auto *block = op->getBlock()) {
        if (!blocks.count(block)) blocks[block] = blocks.size();
        for (Value arg : block->getArguments()) add(arg);
      }
      for (Value result : op->getResults()) add(result);
    });
  }
};
using LiveSets = std::vector<std::set<unsigned>>;
LiveSets liveSets(ModuleOp module, const Ids &ids) {
  Liveness native(module);
  LiveSets live(ids.orderedOps.size());
  for (unsigned vi = 0; vi < ids.orderedValues.size(); ++vi) {
    Value value = ids.orderedValues[vi];
    if (value.use_empty()) continue;
    auto points = native.resolveLiveness(value);
    for (Operation *op : points) {
      if (!ids.ops.count(op)) continue;
      live[ids.ops.lookup(op)].insert(vi);
      // Region captures that survive the parent op remain live through its
      // children, including loop backedges. Never propagate region results
      // backwards into their own definitions or union sibling local values.
      Operation *definition = value.getDefiningOp();
      if (definition == op || (definition && op->isProperAncestor(definition))) continue;
      if (auto arg = dyn_cast<BlockArgument>(value))
        if (op == arg.getOwner()->getParentOp() || op->isProperAncestor(arg.getOwner()->getParentOp())) continue;
      if (op->getNumRegions()) op->walk<WalkOrder::PreOrder>([&](Operation *child) {
        live[ids.ops.lookup(child)].insert(vi);
      });
    }
    // Native liveness can omit uses lifted through some region interfaces.
    for (OpOperand &use : value.getUses()) live[ids.ops.lookup(use.getOwner())].insert(vi);
    if (auto *definition = value.getDefiningOp()) live[ids.ops.lookup(definition)].insert(vi);
  }
  return live;
}
struct Peak {
  uint64_t packed = 0, unpacked = 0, vector = 0, uniform = 0, temporary = 0;
  unsigned op = 0;
  std::set<unsigned> values;
};
llvm::json::Object peakJSON(const Peak &p, const Ids &ids) {
  llvm::json::Array values;
  for (auto value : p.values) values.push_back(int64_t(value));
  return llvm::json::Object{{"packed_words", int64_t(p.packed)}, {"unpacked_words", int64_t(p.unpacked)},
      {"vector_packed_words", int64_t(p.vector)}, {"uniform_packed_words", int64_t(p.uniform)},
      {"extra_temporary_words_proxy", int64_t(p.temporary)}, {"operation", int64_t(p.op)},
      {"operation_name", ids.orderedOps[p.op]->getName().getStringRef().str()}, {"live_values", std::move(values)}};
}

llvm::json::Object sharedScenario(ModuleOp module, const Ids &ids, const LiveSets &live,
                                   bool forcePadding) {
  auto scratch = [&](Operation *op) -> unsigned {
    if (forcePadding)
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
        return tt::HCU::getConvertLayoutScratchInBytes(cvt.getSrc().getType(), cvt.getType(), true);
    return tt::HCU::HCUAllocationAnalysisScratchSizeFn(op);
  };
  ModuleAllocation allocation(module, scratch);
  llvm::json::Array functions;
  uint64_t globalPeak = 0, globalAllocated = allocation.getSharedMemorySize();
  module.walk([&](FunctionOpInterface func) {
    Allocation *alloc = allocation.getFuncData(func);
    if (!alloc) return;
    std::map<size_t, std::set<unsigned>> buffers;
    uint64_t peak = 0, peakExtent = 0, scratchSum = 0, scratchMax = 0;
    unsigned peakOp = ids.ops.lookup(func.getOperation());
    func.walk([&](Operation *op) {
      unsigned oi = ids.ops.lookup(op);
      std::set<size_t> atPoint;
      auto temporary = alloc->getBufferId(op);
      if (temporary != Allocation::InvalidBufferId) {
        atPoint.insert(temporary);
        auto bytes = alloc->getAllocatedSize(temporary);
        scratchSum += bytes; scratchMax = std::max<uint64_t>(scratchMax, bytes);
      }
      for (unsigned vi : live[oi])
        for (auto buffer : alloc->getBufferIds(ids.orderedValues[vi])) atPoint.insert(buffer);
      uint64_t bytes = 0, extent = 0;
      for (auto buffer : atPoint) {
        buffers[buffer].insert(oi);
        bytes += alloc->getAllocatedSize(buffer);
        extent = std::max<uint64_t>(extent, alloc->getOffset(buffer) + alloc->getAllocatedSize(buffer));
      }
      if (bytes > peak) { peak = bytes; peakOp = oi; }
      peakExtent = std::max(peakExtent, extent);
    });
    llvm::json::Array records;
    for (auto &[buffer, points] : buffers) {
      llvm::json::Array operations;
      for (auto point : points) operations.push_back(int64_t(point));
      records.push_back(llvm::json::Object{{"buffer", int64_t(buffer)},
          {"bytes", int64_t(alloc->getAllocatedSize(buffer))}, {"offset", int64_t(alloc->getOffset(buffer))},
          {"live_operations", std::move(operations)}});
    }
    globalPeak = std::max(globalPeak, peak);
    functions.push_back(llvm::json::Object{{"function_operation", int64_t(ids.ops.lookup(func.getOperation()))},
        {"allocated_bytes", int64_t(alloc->getSharedMemorySize())}, {"peak_live_buffer_bytes", int64_t(peak)},
        {"peak_live_operation", int64_t(peakOp)}, {"peak_live_extent_bytes", int64_t(peakExtent)},
        {"scratch_static_sum_bytes", int64_t(scratchSum)}, {"scratch_max_bytes", int64_t(scratchMax)},
        {"buffers", std::move(records)}});
  });
  return llvm::json::Object{{"available", true}, {"allocated_bytes", int64_t(globalAllocated)},
      {"peak_live_buffer_bytes", int64_t(globalPeak)}, {"functions", std::move(functions)},
      {"source", "native HCU ModuleAllocation + alias-aware live sets"}};
}
} // namespace

llvm::json::Object analyzeLayoutResources(ModuleOp module, const llvm::json::Array &conversions) {
  Ids ids(module);
  auto live = liveSets(module, ids);
  tt::ModuleAxisInfoAnalysis axes(module);
  std::vector<Footprint> footprints;
  std::vector<unsigned> parent(ids.orderedValues.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned i) { while (parent[i] != i) i = parent[i]; return i; };
  auto join = [&](Value a, Value b) {
    unsigned x = root(ids.values.lookup(a)), y = root(ids.values.lookup(b));
    parent[std::max(x, y)] = std::min(x, y);
  };
  unsigned unknown = 0;
  for (Value v : ids.orderedValues) {
    footprints.push_back(footprint(v, axes)); unknown += !footprints.back().known;
  }
  // AxisInfo is primarily an integer/address analysis and may lose float
  // constancy at arith.addf. Prove uniformity for known pure arithmetic;
  // never infer it for arbitrary inline-asm/extern "elementwise" operations.
  bool changed;
  do {
    changed = false;
    for (Operation *op : ids.orderedOps) {
      auto name = op->getName().getStringRef();
      bool pureArithmetic = (name.starts_with("arith.") || name.starts_with("math.")) &&
                            op->hasTrait<OpTrait::Elementwise>();
      bool copy = name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" || name == "tt.addptr";
      if ((!pureArithmetic && !copy) || op->getNumRegions() || !op->getNumOperands()) continue;
      if (!llvm::all_of(op->getOperands(), [&](Value v) { return footprints[ids.values.lookup(v)].uniform; })) continue;
      for (Value result : op->getResults()) {
        auto &f = footprints[ids.values.lookup(result)];
        if (!f.known || f.uniform) continue;
        f.uniform = true; f.packed = f.unpacked = (f.bits + 31) / 32;
        if (isa<RankedTensorType>(result.getType())) f.kind = "uniform-tensor";
        changed = true;
      }
    }
  } while (changed);
  module.walk([&](Operation *op) {
    if (op->getNumResults() != 1 || !op->getNumOperands()) return;
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      if (mlir::cvtReordersRegisters(cvt.getSrc().getType(), cvt.getType())) join(cvt.getSrc(), cvt.getResult());
    } else {
      auto name = op->getName().getStringRef();
      if (name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims")
        join(op->getOperand(0), op->getResult(0));
    }
  });
  DenseMap<unsigned, uint64_t> temporaryWords;
  bool unsupportedConversion = false;
  for (const auto &item : conversions) {
    auto *c = item.getAsObject();
    if (!c) continue;
    auto route = c->getString("route").value_or("");
    unsigned oi = c->getInteger("operation").value_or(0);
    if (route == "lds") temporaryWords[oi] = 2; // address temporaries, not an index DAG
    else if (route == "warp-shuffle") temporaryWords[oi] = c->getInteger("packed_register_words").value_or(0);
    else if (route == "cross-cta-unsupported") unsupportedConversion = true;
  }
  Peak peakPacked, peakUnpacked, peakWithTemps;
  std::map<unsigned, Peak> functionPeaks;
  uint64_t maxSingle = 0;
  for (unsigned oi = 0; oi < live.size(); ++oi) {
    struct Group { uint64_t packed = 0, unpacked = 0; bool uniform = true; };
    std::map<unsigned, Group> groups;
    for (unsigned vi : live[oi]) {
      const auto &f = footprints[vi];
      auto &g = groups[root(vi)];
      g.packed = std::max(g.packed, f.packed); g.unpacked = std::max(g.unpacked, f.unpacked);
      g.uniform &= f.uniform;
      maxSingle = std::max(maxSingle, f.unpacked);
    }
    Peak point; point.op = oi; point.values = live[oi];
    for (auto &[id, g] : groups) {
      point.packed += g.packed; point.unpacked += g.unpacked;
      (g.uniform ? point.uniform : point.vector) += g.packed;
    }
    point.temporary = temporaryWords.lookup(oi);
    if (point.packed > peakPacked.packed) peakPacked = point;
    if (point.unpacked > peakUnpacked.unpacked) peakUnpacked = point;
    if (point.packed + point.temporary > peakWithTemps.packed + peakWithTemps.temporary) peakWithTemps = point;
    Operation *func = ids.orderedOps[oi];
    while (func && !isa<FunctionOpInterface>(func)) func = func->getParentOp();
    if (func) {
      auto &peak = functionPeaks[ids.ops.lookup(func)];
      if (point.unpacked > peak.unpacked) peak = point;
    }
  }
  llvm::json::Array values;
  for (unsigned vi = 0; vi < footprints.size(); ++vi) {
    const auto &f = footprints[vi];
    values.push_back(llvm::json::Object{{"value", int64_t(vi)}, {"alias_group", int64_t(root(vi))},
        {"type", typeString(ids.orderedValues[vi].getType())}, {"kind", f.kind}, {"known", f.known},
        {"bitwidth", int64_t(f.bits)}, {"elems_per_thread", int64_t(f.elements)},
        {"unique_elems_per_thread", int64_t(f.unique)}, {"uniform", f.uniform}, {"constant", f.constant},
        {"packed_words", int64_t(f.packed)}, {"unpacked_words", int64_t(f.unpacked)},
        {"block", int64_t(ids.blocks.lookup(ids.orderedValues[vi].getParentBlock()))}});
  }
  llvm::json::Array functions;
  for (auto &[fi, peak] : functionPeaks) {
    auto *func = ids.orderedOps[fi];
    auto name = func->getAttrOfType<StringAttr>("sym_name");
    functions.push_back(llvm::json::Object{{"function_operation", int64_t(fi)},
        {"name", name ? name.str() : ""}, {"unpacked_peak", peakJSON(peak, ids)}});
  }
  llvm::json::Object ldsCurrent, ldsPadded;
  if (!unsupportedConversion) {
    ldsCurrent = sharedScenario(module, ids, live, false);
    ldsPadded = sharedScenario(module, ids, live, true);
  } else {
    ldsCurrent = llvm::json::Object{{"available", false}, {"reason", "cross-CTA conversion unsupported"}};
    ldsPadded = llvm::json::Object{{"available", false}, {"reason", "cross-CTA conversion unsupported"}};
  }
  return llvm::json::Object{{"schema", 1}, {"id_scope", "materialized-candidate"}, {"calibrated", false},
      {"operation_count", int64_t(ids.orderedOps.size())}, {"block_count", int64_t(ids.blocks.size())},
      {"scoring_weight", 0.0}, {"scoring_policy", "features-only until joint calibration; do not double-charge legacy prices"},
      {"packed_peak", peakJSON(peakPacked, ids)}, {"unpacked_peak", peakJSON(peakUnpacked, ids)},
      {"with_conversion_temporaries_peak", peakJSON(peakWithTemps, ids)},
      {"functions", std::move(functions)},
      {"max_single_value_unpacked_words", int64_t(maxSingle)}, {"unmodeled_values", int64_t(unknown)},
      {"values", std::move(values)}, {"lds_current", std::move(ldsCurrent)}, {"lds_all_padded", std::move(ldsPadded)},
      {"limitations", "TTGIR proxy, not VGPR/SGPR or occupancy. Packing is hypothetical; constants remain counted. Captures conservatively live through regions; no sibling-local union. Lowering/scheduling/loop-carried aliasing and non-conversion temporaries may differ. Later LDS strategy/splits may change allocation; alternatives are not additive."}};
}

llvm::json::Object analyzeLayoutResourceContract(ModuleOp module) {
  Ids ids(module);
  auto live = liveSets(module, ids);
  tt::ModuleAxisInfoAnalysis axes(module);
  std::vector<Footprint> footprints;
  std::vector<unsigned> parent(ids.orderedValues.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](unsigned i) { while (parent[i] != i) i = parent[i]; return i; };
  auto join = [&](Value a, Value b) {
    auto x = root(ids.values.lookup(a)), y = root(ids.values.lookup(b));
    parent[std::max(x, y)] = std::min(x, y);
  };
  for (Value value : ids.orderedValues) footprints.push_back(footprint(value, axes));
  // Match the legacy alias and uniformity rules exactly before any exclusion.
  bool changed;
  do {
    changed = false;
    for (Operation *op : ids.orderedOps) {
      auto name = op->getName().getStringRef();
      bool pure = (name.starts_with("arith.") || name.starts_with("math.")) &&
                  op->hasTrait<OpTrait::Elementwise>();
      bool copy = name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" || name == "tt.addptr";
      if ((!pure && !copy) || op->getNumRegions() || !op->getNumOperands()) continue;
      if (!llvm::all_of(op->getOperands(), [&](Value v) { return footprints[ids.values.lookup(v)].uniform; })) continue;
      for (Value result : op->getResults()) {
        auto &f = footprints[ids.values.lookup(result)];
        if (!f.known || f.uniform) continue;
        f.uniform = true; f.packed = f.unpacked = (f.bits + 31) / 32;
        changed = true;
      }
    }
  } while (changed);
  module.walk([&](Operation *op) {
    if (op->getNumResults() != 1 || !op->getNumOperands()) return;
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      if (mlir::cvtReordersRegisters(cvt.getSrc().getType(), cvt.getType())) join(cvt.getSrc(), cvt.getResult());
    } else {
      auto name = op->getName().getStringRef();
      if (name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims")
        join(op->getOperand(0), op->getResult(0));
    }
  });
  std::map<unsigned, bool> constants, coordinates;
  unsigned unknown = 0;
  for (unsigned i = 0; i < footprints.size(); ++i) {
    unknown += !footprints[i].known;
    constants[root(i)] |= footprints[i].constant;
    if (auto *op = ids.orderedValues[i].getDefiningOp()) {
      auto name = op->getName().getStringRef();
      coordinates[root(i)] = coordinates[root(i)] || name == "tt.make_range" ||
                            name == "tt.get_program_id" || name == "tt.get_num_programs";
    }
  }
  llvm::json::Object variants;
  const char *names[] = {"all_values", "without_constants", "without_constants_or_coordinates"};
  for (unsigned mode = 0; mode < 3; ++mode) {
    Peak packed, unpacked; uint64_t maxSingle = 0;
    for (unsigned oi = 0; oi < live.size(); ++oi) {
      struct Group { uint64_t packed = 0, unpacked = 0; bool uniform = true; };
      std::map<unsigned, Group> groups;
      Peak point; point.op = oi;
      for (unsigned vi : live[oi]) {
        auto group = root(vi);
        if ((mode > 0 && constants[group]) || (mode > 1 && coordinates[group])) continue;
        const auto &f = footprints[vi]; auto &g = groups[group];
        g.packed = std::max(g.packed, f.packed); g.unpacked = std::max(g.unpacked, f.unpacked);
        g.uniform &= f.uniform; point.values.insert(vi);
        maxSingle = std::max(maxSingle, f.unpacked);
      }
      for (auto &[id, g] : groups) {
        point.packed += g.packed; point.unpacked += g.unpacked;
        (g.uniform ? point.uniform : point.vector) += g.packed;
      }
      if (point.packed > packed.packed) packed = point;
      if (point.unpacked > unpacked.unpacked) unpacked = point;
    }
    variants[names[mode]] = llvm::json::Object{{"packed_peak", peakJSON(packed, ids)},
        {"unpacked_peak", peakJSON(unpacked, ids)}, {"max_single_value_unpacked_words", int64_t(maxSingle)}};
  }
  return llvm::json::Object{{"available", unknown == 0}, {"unmodeled_values", int64_t(unknown)},
      {"schema", "diagnostic-resource-live-set-contract"}, {"variants", std::move(variants)},
      {"all_native_live_points_recomputed", true}, {"analysis_only", true},
      {"units", "hypothetical 32-bit words per thread; alias max within each live point"},
      {"is_hardware_register_count", false}, {"used_by_model", false}};
}
} // namespace layoutdp
