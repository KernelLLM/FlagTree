// Frozen 183-feature model beam search with transactional layout rewriting.
#include "TritonHCUGPUTransforms/LayoutTuning.h"
#include "TritonHCUGPUTransforms/LayoutTuningFacts.h"
#include "TritonHCUGPUTransforms/LayoutTuningSSA.h"
#include "TritonHCUGPUTransforms/LayoutTuningLearned.h"
#include "TritonHCUGPUTransforms/LayoutTuningUnified.h"
#include "TritonHCUGPUTransforms/LayoutTuningDomain.h"
#include "TritonHCUGPUTransforms/LayoutTuningResources.h"
#include "TritonHCUGPUTransforms/LayoutTuningIndex.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Analysis/HCUGPUAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace hcu = mlir::triton::hcugpu;

namespace layoutdp {
namespace {
Attribute encoding(Value v) {
  if (auto t = dyn_cast<RankedTensorType>(v.getType()))
    return t.getEncoding();
  if (auto t = dyn_cast<ttg::MemDescType>(v.getType()))
    return t.getEncoding();
  return {};
}
std::string printed(Attribute a) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (a) a.print(os);
  return s;
}
std::string printed(Type t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  t.print(os);
  return s;
}
Attribute rootEncoding(Attribute a) {
  while (a) {
    auto slice = dyn_cast<ttg::SliceEncodingAttr>(a);
    if (!slice) break;
    a = slice.getParent();
  }
  return a;
}
Attribute replaceRoot(Attribute like, Attribute root) {
  if (auto slice = dyn_cast<ttg::SliceEncodingAttr>(like))
    return ttg::SliceEncodingAttr::get(like.getContext(), slice.getDim(),
              cast<ttg::DistributedEncodingTrait>(replaceRoot(slice.getParent(), root)));
  return root;
}

// Canonicalize redundant order spellings only when the complete LinearLayout
// (including register order, broadcasts and CTA mapping) is exactly equal.
// Work on an analysis clone: executable identities and historical timings stay
// valid. Do not merge by shape, register count or a binary/case identity.
llvm::json::Object normalizeSemanticFeatures(ModuleOp module) {
  DenseMap<Type, Type> types;
  DenseMap<Attribute, SmallVector<RankedTensorType>> families;
  DenseSet<Attribute> coupledDotParents;
  auto familyKey = [&](ttg::BlockedEncodingAttr blocked) {
    SmallVector<unsigned> order(blocked.getOrder().size());
    std::iota(order.begin(), order.end(), 0);
    return ttg::BlockedEncodingAttr::get(
        module.getContext(), blocked.getSizePerThread(),
        blocked.getThreadsPerWarp(), blocked.getWarpsPerCTA(), order,
        blocked.getCTALayout());
  };
  auto collect = [&](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.getEncoding()) return;
    // At the new early insertion point, dot operands still refer to blocked
    // parents. Equal LinearLayouts do not imply identical parent attributes:
    // independently changing only a blocked result breaks tt.dot verification.
    if (auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(rootEncoding(tensor.getEncoding())))
      if (auto parent = dyn_cast<ttg::BlockedEncodingAttr>(rootEncoding(dot.getParent())))
        coupledDotParents.insert(familyKey(parent));
    auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
        rootEncoding(tensor.getEncoding()));
    if (blocked && blocked.getOrder().size() <= 4)
      if (!llvm::is_contained(families[familyKey(blocked)], tensor))
        families[familyKey(blocked)].push_back(tensor);
  };
  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Value arg : block.getArguments()) collect(arg.getType());
    for (Value result : op->getResults()) collect(result.getType());
    if (auto function = dyn_cast<tt::FuncOp>(op)) {
      for (Type t : function.getFunctionType().getInputs()) collect(t);
      for (Type t : function.getFunctionType().getResults()) collect(t);
    }
  });
  DenseMap<Attribute, Attribute> canonicalRoots;
  for (auto &entry : families) {
    if (coupledDotParents.contains(entry.first)) continue;
    auto blocked = cast<ttg::BlockedEncodingAttr>(entry.first);
    SmallVector<unsigned> order(blocked.getOrder().size());
    std::iota(order.begin(), order.end(), 0);
    do {
      auto candidate = ttg::BlockedEncodingAttr::get(
          module.getContext(), blocked.getSizePerThread(),
          blocked.getThreadsPerWarp(), blocked.getWarpsPerCTA(), order,
          blocked.getCTALayout());
      // All shapes and slice views sharing these hierarchy fields must agree.
      // Shape-local canonicalization alone can violate broadcast interfaces.
      if (llvm::all_of(entry.second, [&](RankedTensorType type) {
            auto changed = type.cloneWithEncoding(
                replaceRoot(type.getEncoding(), candidate));
            return ttg::toLinearLayout(type) == ttg::toLinearLayout(changed);
          })) {
        canonicalRoots[entry.first] = candidate;
        break;
      }
    } while (std::next_permutation(order.begin(), order.end()));
  }
  unsigned changedValues = 0, removed = 0;
  auto canonical = [&](Type original) -> Type {
    auto found = types.find(original);
    if (found != types.end()) return found->second;
    Type result = original;
    auto tensor = dyn_cast<RankedTensorType>(original);
    if (tensor && tensor.getEncoding()) {
      auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
          rootEncoding(tensor.getEncoding()));
      if (blocked && blocked.getOrder().size() <= 4) {
        if (auto base = canonicalRoots.lookup(familyKey(blocked)))
          result = tensor.cloneWithEncoding(
              replaceRoot(tensor.getEncoding(), base));
      }
    }
    types[original] = result;
    return result;
  };
  auto update = [&](Value value) {
    Type type = canonical(value.getType());
    changedValues += type != value.getType();
    value.setType(type);
  };
  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Value arg : block.getArguments()) update(arg);
    for (Value result : op->getResults()) update(result);
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
        constant->setAttr("value", dense.reshape(
            cast<ShapedType>(constant.getType())));
    if (auto function = dyn_cast<tt::FuncOp>(op)) {
      auto type = function.getFunctionType();
      SmallVector<Type> inputs, outputs;
      for (Type t : type.getInputs()) inputs.push_back(canonical(t));
      for (Type t : type.getResults()) outputs.push_back(canonical(t));
      function.setType(FunctionType::get(module.getContext(), inputs, outputs));
    }
  });
  DenseMap<Block *, DenseMap<std::pair<Value, Type>, Value>> available;
  SmallVector<ttg::ConvertLayoutOp> redundant;
  module.walk<WalkOrder::PreOrder>([&](ttg::ConvertLayoutOp cvt) {
    // Same-block, SSA-source-identical reuse only; never hoist across regions.
    if (cvt.getSrc().getType() == cvt.getType()) {
      cvt.replaceAllUsesWith(cvt.getSrc());
      redundant.push_back(cvt);
      return;
    }
    auto &block = available[cvt->getBlock()];
    std::pair<Value, Type> key{cvt.getSrc(), cvt.getType()};
    auto it = block.find(key);
    if (it == block.end()) block[key] = cvt.getResult();
    else {
      cvt.replaceAllUsesWith(it->second);
      redundant.push_back(cvt);
    }
  });
  for (auto cvt : redundant) { cvt.erase(); ++removed; }
  return llvm::json::Object{
      {"version", "exact-linearlayout-order-reuse-v25"},
      {"changed_values", int64_t(changedValues)},
      {"removed_conversions", int64_t(removed)},
      {"analysis_clone_only", true},
      {"scope", "blocked/slice order equivalence; same-block conversion reuse"}};
}
bool explicitLayout(Operation *op) {
  for (NamedAttribute attr : op->getAttrs())
    if (attr.getName().strref().starts_with("tle.explicit_")) return true;
  return false;
}
bool memoryOp(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return n == "tt.load" || n == "tt.store" || n == "hcug.buffer_load" ||
         n == "hcug.buffer_store";
}
bool sourceOp(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return n == "arith.constant" || n == "tt.splat" || n == "tt.make_range";
}
bool uniformOp(Operation *op) {
  return memoryOp(op) || op->hasTrait<OpTrait::Elementwise>() ||
         op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
         op->hasTrait<OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
         op->getName().getStringRef() == "tt.broadcast";
}
bool transferOp(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return n == "tt.expand_dims" || n == "tt.trans" || n == "tt.reshape" ||
         n == "tt.reduce" || n == "tt.join" || n == "tt.split";
}
bool controlOp(Operation *op) { return isa<scf::ForOp, scf::IfOp>(op); }
bool convertOp(Operation *op) { return isa<ttg::ConvertLayoutOp>(op); }

// A producer has one encoding. Each operand use may request another encoding;
// conversions are values of their own, never competing assignments to SSA.
struct Plan : std::vector<Attribute> {
  std::vector<std::vector<Attribute>> requests;
  std::vector<Attribute> mfmaWarpHints;
  bool operator==(const Plan &other) const {
    return static_cast<const std::vector<Attribute> &>(*this) ==
               static_cast<const std::vector<Attribute> &>(other) && requests == other.requests &&
           mfmaWarpHints == other.mfmaWarpHints;
  }
  bool operator!=(const Plan &other) const { return !(*this == other); }
};
struct UseInfo { unsigned value, op, operand; };
struct Edge {
  unsigned from, to;
  Operation *op;
  bool identity, forward;
};
struct ValueInfo {
  Value value;
  std::string fixed;
};

// This model is independent of prices, labels and the search algorithm.
class LayoutModel {
public:
  ModuleOp mod;
  std::vector<ValueInfo> values;
  DenseMap<Value, unsigned> ids;
  DenseMap<Operation *, unsigned> opIds;
  std::vector<Operation *> operations;
  std::vector<UseInfo> uses;
  std::vector<unsigned> boundaryOps;
  std::vector<unsigned> matrixOps;
  std::vector<Attribute> localTemplates;
  std::vector<Edge> edges;
  std::vector<std::vector<unsigned>> adjacency;
  std::vector<unsigned> anchors;
  DenseSet<Operation *> supported;
  Plan seed;

  explicit LayoutModel(ModuleOp module, bool matrixDomain = false) : mod(module) {
    mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
      opIds[op] = operations.size();
      operations.push_back(op);
      for (Value result : op->getResults()) addValue(result, "");
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            addValue(arg, "unmodeled block argument / function ABI");
    });
    seed.requests.resize(operations.size());
    seed.mfmaWarpHints.resize(operations.size());
    localTemplates.resize(operations.size());
    for (unsigned oi = 0; oi < operations.size(); ++oi) {
      Operation *op = operations[oi];
      seed.mfmaWarpHints[oi] = op->getAttr("hcu.layout.mfma_warps_per_cta");
      if (matrixDomain) {
        if (auto dot = dyn_cast<tt::DotOp>(op)) {
          auto type = dot.getType();
          auto a = dot.getA().getType().getElementType();
          auto b = dot.getB().getType().getElementType();
          unsigned warps = ttg::lookupNumWarps(op);
          if (warps && warps <= 8 && !(warps & (warps - 1)) &&
              type.getRank() == 2 && type.getElementType().isF32() &&
              isa_and_nonnull<ttg::BlockedEncodingAttr>(type.getEncoding()) &&
              ((a.isBF16() && b.isBF16()) || (a.isF16() && b.isF16())))
            matrixOps.push_back(oi);
        }
      }
      seed.requests[oi].resize(op->getNumOperands());
      for (OpOperand &use : op->getOpOperands()) {
        auto it = ids.find(use.get());
        if (it == ids.end()) continue;
        unsigned port = use.getOperandNumber();
        uses.push_back({it->second, oi, port});
        // Existing conversions accept the producer's actual encoding. Their
        // result type already expresses the requested destination.
        if (!convertOp(op)) seed.requests[oi][port] = encoding(use.get());
      }
      if (!uniformOp(op) || sourceOp(op) || convertOp(op) || explicitLayout(op) || op->getNumRegions()) continue;
      Attribute like;
      bool eligible = true;
      auto inspect = [&](Value v) {
        auto attr = encoding(v);
        if (!attr) return;
        if (!isa<RankedTensorType>(v.getType()) || !isa<ttg::BlockedEncodingAttr>(rootEncoding(attr))) {
          eligible = false; return;
        }
        if (like && like != attr) eligible = false;
        like = attr;
      };
      for (Value v : op->getOperands()) inspect(v);
      for (Value v : op->getResults()) inspect(v);
      if (eligible && like) {
        localTemplates[oi] = like;
        boundaryOps.push_back(oi);
      }
    }
    std::stable_sort(boundaryOps.begin(), boundaryOps.end(), [&](unsigned a, unsigned b) {
      return memoryOp(operations[a]) > memoryOp(operations[b]);
    });
    adjacency.resize(values.size());
    mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        supported.insert(op);
        auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        if (!yield) return;
        supported.insert(yield);
        for (unsigned i = 0; i < loop.getNumResults(); ++i) {
          Value arg = loop.getRegionIterArgs()[i];
          if (ids.count(arg)) values[ids.lookup(arg)].fixed.clear();
          tie(loop.getInitArgs()[i], arg, op);
          tie(arg, yield.getOperand(i), op);
          tie(arg, loop.getResult(i), op);
        }
      } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
        supported.insert(op);
        for (Region &region : op->getRegions()) {
          if (region.empty()) continue;
          auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
          if (!yield) continue;
          supported.insert(yield);
          for (unsigned i = 0; i < branch.getNumResults(); ++i)
            tie(branch.getResult(i), yield.getOperand(i), op);
        }
      } else if (sourceOp(op) || convertOp(op)) {
        supported.insert(op);
      } else if (transferOp(op)) {
        supported.insert(op);
        for (Value input : op->getOperands())
          for (Value output : op->getResults())
            connect(input, output, op, false);
      } else if (uniformOp(op) && !op->getNumRegions()) {
        supported.insert(op);
        SmallVector<Value> related;
        for (Value v : op->getOperands()) if (ids.count(v)) related.push_back(v);
        for (Value v : op->getResults()) if (ids.count(v)) related.push_back(v);
        for (unsigned i = 1; i < related.size(); ++i) tie(related[0], related[i], op);
      }
      if (!supported.count(op))
        for (Value result : op->getResults())
          if (ids.count(result))
            values[ids.lookup(result)].fixed = "unsupported producer: " + op->getName().getStringRef().str();
      if (explicitLayout(op)) {
        for (Value v : op->getResults()) pin(v, "explicit TLE layout");
        for (Value v : op->getOperands()) pin(v, "explicit TLE layout");
      }
      if (memoryOp(op) || op->getName().getStringRef() == "tt.reduce") {
        for (Value v : op->getOperands()) addAnchor(v);
        for (Value v : op->getResults()) addAnchor(v);
      }
    });
    // One anchor per connected constraint component. Repeating every pointer,
    // mask and data value in the same chain would waste the proposal budget.
    DenseSet<unsigned> covered;
    std::vector<unsigned> unique;
    for (unsigned anchor : anchors) {
      if (!covered.insert(anchor).second) continue;
      unique.push_back(anchor);
      std::deque<unsigned> work{anchor};
      while (!work.empty()) {
        unsigned id = work.front(); work.pop_front();
        for (unsigned ei : adjacency[id])
          if (covered.insert(edges[ei].to).second) work.push_back(edges[ei].to);
      }
    }
    anchors = std::move(unique);
  }

  void addValue(Value value, StringRef fixed) {
    if (!encoding(value) || ids.count(value)) return;
    unsigned id = values.size();
    ids[value] = id;
    values.push_back({value, fixed.str()});
    seed.push_back(encoding(value));
    if (!isa<RankedTensorType>(value.getType())) values.back().fixed = "memdesc interface";
  }
  void pin(Value v, const std::string &reason) {
    auto it = ids.find(v);
    if (it != ids.end()) values[it->second].fixed = reason;
  }
  void addAnchor(Value v) {
    auto it = ids.find(v);
    if (it == ids.end() || !values[it->second].fixed.empty() ||
        !isa<ttg::BlockedEncodingAttr>(rootEncoding(seed[it->second]))) return;
    if (!llvm::is_contained(anchors, it->second)) anchors.push_back(it->second);
  }
  void connect(Value a, Value b, Operation *op, bool identity) {
    if (!ids.count(a) || !ids.count(b)) return;
    unsigned x = ids.lookup(a), y = ids.lookup(b);
    if (x == y) return;
    adjacency[x].push_back(edges.size());
    edges.push_back({x, y, op, identity, true});
    adjacency[y].push_back(edges.size());
    edges.push_back({y, x, op, identity, false});
  }
  void tie(Value a, Value b, Operation *op) { connect(a, b, op, true); }

  std::optional<Plan> propose(const Plan &base, unsigned anchor, Attribute enc) const {
    DenseMap<unsigned, Attribute> assigned;
    std::deque<unsigned> work;
    auto assign = [&](unsigned id, Attribute attr) {
      if (!attr || (!values[id].fixed.empty() && attr != seed[id])) return false;
      auto it = assigned.find(id);
      if (it != assigned.end()) return it->second == attr;
      // Every changed register value must remain a rank-compatible blocked/slice
      // encoding. MMA/shared changes are a separately gated future extension.
      if (attr != seed[id]) {
        auto type = dyn_cast<RankedTensorType>(values[id].value.getType());
        auto root = dyn_cast<ttg::BlockedEncodingAttr>(rootEncoding(attr));
        if (!type || !root) return false;
        unsigned rank = root.getSizePerThread().size();
        for (Attribute a = attr; auto slice = dyn_cast<ttg::SliceEncodingAttr>(a); a = slice.getParent()) {
          if (!rank || slice.getDim() >= rank) return false;
          --rank;
        }
        if (rank != static_cast<unsigned>(type.getRank())) return false;
      }
      assigned[id] = attr;
      work.push_back(id);
      return true;
    };
    if (!assign(anchor, enc)) return std::nullopt;
    while (!work.empty()) {
      unsigned id = work.front(); work.pop_front();
      for (unsigned ei : adjacency[id]) {
        const Edge &e = edges[ei];
        Attribute next = e.identity ? assigned.lookup(id)
                                   : (e.forward ? inferDstEncoding(e.op, assigned.lookup(id))
                                                : inferSrcEncoding(e.op, assigned.lookup(id)));
        if (!assign(e.to, next)) return std::nullopt;
      }
    }
    Plan result = base;
    for (auto &kv : assigned) result[kv.first] = kv.second;
    for (const auto &use : uses) {
      auto it = assigned.find(use.value);
      Operation *consumer = operations[use.op];
      if (it != assigned.end() && supported.count(consumer) && !convertOp(consumer))
        result.requests[use.op][use.operand] = it->second;
    }
    return result;
  }

  // Cross disconnected constraint components transactionally. Single-anchor
  // moves can otherwise leave compensating conversions in a loop/interface,
  // so no prefix of a useful complete assignment survives the beam.
  std::optional<Plan> proposeCoherentSPT(const Plan &base, unsigned spt) const {
    Plan result=base;
    for (unsigned id=0;id<values.size();++id) {
      if (!values[id].fixed.empty()) continue;
      auto root=dyn_cast<ttg::BlockedEncodingAttr>(rootEncoding(base[id]));
      if (!root || root.getOrder().empty()) continue;
      SmallVector<unsigned> sizes(root.getSizePerThread().begin(),root.getSizePerThread().end());
      sizes[root.getOrder().front()]=spt;
      uint64_t product=1;for (unsigned size:sizes) product*=size;
      if (product>16) return std::nullopt;
      result[id]=replaceRoot(base[id],ttg::BlockedEncodingAttr::get(
          root.getContext(),sizes,root.getThreadsPerWarp(),root.getWarpsPerCTA(),root.getOrder(),root.getCTALayout()));
    }
    // Do not route this complete assignment through the one-anchor inference
    // graph: its pairwise equality constraints can reject jointly legal type
    // changes. Preserve pinned/unknown interfaces, update supported uses, then
    // let transactional materialization and the actual IR verifier decide.
    for (const auto &use:uses)
      if (supported.count(operations[use.op]) && !convertOp(operations[use.op]))
        result.requests[use.op][use.operand]=result[use.value];
    return result;
  }

  Attribute localEncoding(const Plan &plan, unsigned oi) const {
    for (Value result : operations[oi]->getResults())
      if (ids.count(result)) return plan[ids.lookup(result)];
    for (Attribute request : plan.requests[oi]) if (request) return request;
    return localTemplates[oi];
  }

  std::optional<Plan> proposeLocal(const Plan &base, unsigned oi, Attribute attr) const {
    if (oi >= localTemplates.size() || !localTemplates[oi] || !attr) return std::nullopt;
    Plan result = base;
    Operation *op = operations[oi];
    for (Value value : op->getResults()) {
      auto it = ids.find(value);
      if (it == ids.end()) continue;
      unsigned id = it->second;
      if (!values[id].fixed.empty() && attr != seed[id]) return std::nullopt;
      result[id] = attr;
    }
    for (unsigned port = 0; port < result.requests[oi].size(); ++port)
      if (seed.requests[oi][port]) result.requests[oi][port] = attr;
    return result;
  }

  llvm::json::Array dumpUseAssignment(const Plan &plan, bool changesOnly = false) const {
    llvm::json::Array result;
    for (const auto &use : uses) {
      Attribute required = plan.requests[use.op][use.operand];
      if (changesOnly && required == seed.requests[use.op][use.operand]) continue;
      result.push_back(llvm::json::Object{{"value", int64_t(use.value)}, {"consumer", int64_t(use.op)},
          {"operand", int64_t(use.operand)}, {"requested_encoding", printed(required)},
          {"producer_encoding", printed(plan[use.value])},
          {"needs_conversion", required && required != plan[use.value]}});
    }
    return result;
  }

  std::vector<Attribute> matrixDomain(unsigned oi) const {
    std::vector<Attribute> result{seed.mfmaWarpHints[oi]};
    unsigned warps = ttg::lookupNumWarps(operations[oi]);
    for (unsigned m = 1; m <= warps; m *= 2) {
      Attribute hint = DenseI32ArrayAttr::get(mod->getContext(),
          {static_cast<int32_t>(m), static_cast<int32_t>(warps / m)});
      if (!llvm::is_contained(result, hint)) result.push_back(hint);
    }
    return result;
  }

  llvm::json::Array dumpMatrixAssignment(const Plan &plan, bool changesOnly = false) const {
    llvm::json::Array result;
    for (unsigned oi = 0; oi < plan.mfmaWarpHints.size(); ++oi) {
      if (changesOnly && plan.mfmaWarpHints[oi] == seed.mfmaWarpHints[oi]) continue;
      if (!plan.mfmaWarpHints[oi] && !seed.mfmaWarpHints[oi]) continue;
      result.push_back(llvm::json::Object{{"operation", int64_t(oi)},
          {"mfma_warps_per_cta", printed(plan.mfmaWarpHints[oi])}});
    }
    return result;
  }

  // Complete, transactional materialization. Unknown consumers keep their
  // interface encoding using a per-use conversion inserted at a dominating
  // position immediately before the use (no cross-region conversion reuse).
  OwningOpRef<ModuleOp> materialize(const Plan &plan, std::string &reason) const {
    IRMapping mapping;
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(mod->clone(mapping)));
    if (plan.size() != values.size()) { reason = "incomplete plan"; return {}; }
    for (unsigned i = 0; i < values.size(); ++i) {
      Value v = mapping.lookupOrDefault(values[i].value);
      if (plan[i] == seed[i]) continue;
      auto type = dyn_cast<RankedTensorType>(v.getType());
      if (!type || !values[i].fixed.empty()) { reason = "fixed value changed"; return {}; }
      v.setType(type.cloneWithEncoding(plan[i]));
      if (auto constant = v.getDefiningOp<arith::ConstantOp>()) {
        auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
        if (dense) constant->setAttr("value", dense.reshape(cast<ShapedType>(v.getType())));
      }
    }
    if (plan.requests.size() != operations.size() || plan.mfmaWarpHints.size() != operations.size()) {
      reason = "incomplete use/matrix plan"; return {};
    }
    for (unsigned oi = 0; oi < operations.size(); ++oi) {
      Operation *old = operations[oi];
      Operation *op = mapping.lookupOrDefault(old);
      if (plan.mfmaWarpHints[oi] != seed.mfmaWarpHints[oi]) {
        if (!llvm::is_contained(matrixOps, oi)) { reason = "fixed matrix hint changed"; return {}; }
        if (plan.mfmaWarpHints[oi]) op->setAttr("hcu.layout.mfma_warps_per_cta", plan.mfmaWarpHints[oi]);
        else op->removeAttr("hcu.layout.mfma_warps_per_cta");
      }
      if (plan.requests[oi].size() != old->getNumOperands()) { reason = "incomplete operand plan"; return {}; }
      for (unsigned port = 0; port < old->getNumOperands(); ++port) {
        Value after = op->getOperand(port);
        Attribute required = plan.requests[oi][port];
        if (!required || encoding(after) == required) continue;
        auto type = dyn_cast<RankedTensorType>(after.getType());
        if (!type) { reason = "cannot convert a memdesc interface"; return {}; }
        OpBuilder builder(op);
        auto cvt = ttg::ConvertLayoutOp::create(builder, op->getLoc(),
                                               type.cloneWithEncoding(required), after);
        op->setOperand(port, cvt.getResult());
      }
    }
    SmallVector<ttg::ConvertLayoutOp> redundant;
    DenseMap<Block *, DenseMap<std::pair<Value, Type>, Value>> available;
    // The original assignment is an ordinary scoring candidate. Normalize
    // conversions uniformly, regardless of how this assignment was reached.
    clone->walk<WalkOrder::PreOrder>([&](ttg::ConvertLayoutOp cvt) {
      if (explicitLayout(cvt)) return;
      if (cvt.getSrc().getType() == cvt.getType()) {
        cvt.replaceAllUsesWith(cvt.getSrc()); redundant.push_back(cvt); return;
      }
      auto &block = available[cvt->getBlock()];
      std::pair<Value, Type> key{cvt.getSrc(), cvt.getType()};
      auto it = block.find(key);
      if (it != block.end()) {
        cvt.replaceAllUsesWith(it->second); redundant.push_back(cvt);
      } else block[key] = cvt.getResult();
    });
    for (auto cvt : redundant) cvt.erase();
    // Capture candidate verifier diagnostics in the report; never emit errors
    // for rejected candidates into the user's otherwise valid compilation.
    ScopedDiagnosticHandler handler(mod->getContext(), [&](Diagnostic &diag) {
      llvm::raw_string_ostream os(reason); diag.print(os); return success();
    });
    if (failed(verify(*clone))) return {};
    return clone;
  }

  llvm::json::Object dump() const {
    llvm::json::Array vs, es, uses, ops;
    for (unsigned i = 0; i < values.size(); ++i) {
      llvm::json::Object row{{"id", int64_t(i)}, {"type", printed(values[i].value.getType())},
                             {"encoding", printed(seed[i])}, {"fixed", values[i].fixed}};
      if (auto result = dyn_cast<OpResult>(values[i].value)) {
        row["producer"] = int64_t(opIds.lookup(result.getOwner()));
        row["result"] = int64_t(result.getResultNumber());
      } else if (auto arg = dyn_cast<BlockArgument>(values[i].value)) {
        Operation *owner = arg.getOwner()->getParentOp();
        row["region_owner"] = int64_t(opIds.lookup(owner));
        for (unsigned r = 0; r < owner->getNumRegions(); ++r) {
          unsigned b = 0;
          for (Block &block : owner->getRegion(r)) {
            if (&block == arg.getOwner()) {
              row["region"] = int64_t(r); row["block"] = int64_t(b);
            }
            ++b;
          }
        }
        row["argument"] = int64_t(arg.getArgNumber());
      }
      vs.push_back(std::move(row));
    }
    std::vector<Operation *> ordered(opIds.size());
    for (auto &kv : opIds) ordered[kv.second] = kv.first;
    for (Operation *op : ordered) {
      ops.push_back(llvm::json::Object{{"id", int64_t(opIds.lookup(op))},
                                      {"name", op->getName().getStringRef().str()},
                                      {"supported", supported.count(op) != 0}});
      for (OpOperand &use : op->getOpOperands()) {
        auto it = ids.find(use.get());
        if (it == ids.end()) continue;
        uses.push_back(llvm::json::Object{{"value", int64_t(it->second)},
             {"consumer", int64_t(opIds.lookup(op))}, {"operand", int64_t(use.getOperandNumber())},
             {"boundary", !supported.count(op)}, {"required_encoding", printed(encoding(use.get()))}});
      }
    }
    for (const Edge &e : edges) if (e.forward)
      es.push_back(llvm::json::Object{{"from", int64_t(e.from)}, {"to", int64_t(e.to)},
           {"op", int64_t(opIds.lookup(e.op))}, {"kind", e.identity ? "identity" : "transfer"}});
    llvm::json::Array boundaries;
    for (unsigned oi : boundaryOps) boundaries.push_back(int64_t(oi));
    llvm::json::Array matrices;
    for (unsigned oi : matrixOps) matrices.push_back(int64_t(oi));
    return llvm::json::Object{{"values", std::move(vs)}, {"ops", std::move(ops)},
                              {"local_boundary_ops", std::move(boundaries)},
                              {"mfma_warp_ops", std::move(matrices)},
                              {"constraints", std::move(es)}, {"uses", std::move(uses)}};
  }
};

// Enumerate field mutations independently of price coverage. The original
// attribute (including CTA partition and slice parents) is always preserved.
// Absolute frozen-model score; no structural cost or seed-relative score.
struct Score {
  bool modelOnly = true, jointActive = true;
  llvm::json::Object layoutFacts, jointRanking;
  double jointTotal = std::numeric_limits<double>::infinity();
  std::string jointFailure;
  double ranking() const { return jointTotal; }
  const std::string &rankingFailure() const { return jointFailure; }
};
llvm::json::Value number(double n) {
  return std::isfinite(n) ? llvm::json::Value(n) : llvm::json::Value(nullptr);
}
llvm::json::Object scoreDetails(const Score &score) {
  return llvm::json::Object{{"ranking", llvm::json::Object(score.jointRanking)},
      {"layout_facts", llvm::json::Object(score.layoutFacts)},
      {"seed_reference_input", false}, {"fallback_allowed", false}};
}

std::string planKey(const Plan &plan) {
  std::string key;
  for (Attribute attr : plan) { key += printed(attr); key.push_back('\n'); }
  key += "--uses--\n";
  for (const auto &ports : plan.requests) {
    for (Attribute attr : ports) { key += printed(attr); key.push_back('|'); }
    key.push_back('\n');
  }
  // Keep the original key spelling when this optional domain is unused.
  if (llvm::any_of(plan.mfmaWarpHints, [](Attribute a) { return bool(a); })) {
    key += "--mfma-warps--\n";
    for (Attribute hint : plan.mfmaWarpHints) { key += printed(hint); key.push_back('|'); }
  }
  return key;
}
} // namespace

// Preorder is identical to the resource/index diagnostics, and every record
// comes from this very module. Python never guesses cross-clone identities.
static llvm::json::Object exportCurrentLayoutSSA(
    ModuleOp module, const llvm::json::Object &resources) {
  DenseMap<Operation *, unsigned> operationIds;
  DenseMap<Value, unsigned> valueIds;
  DenseMap<Block *, unsigned> blockIds;
  SmallVector<Operation *> operations;
  SmallVector<Block *> blocks;
  auto value = [&](Value v) {
    if (!valueIds.count(v)) valueIds[v] = valueIds.size();
  };
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    operationIds[op] = operations.size(); operations.push_back(op);
    if (Block *block = op->getBlock()) {
      if (!blockIds.count(block)) {
        blockIds[block] = blocks.size(); blocks.push_back(block);
      }
      for (Value arg : block->getArguments()) value(arg);
    }
    for (Value result : op->getResults()) value(result);
  });
  llvm::json::Array operationRows, blockRows;
  for (Operation *op : operations) {
    llvm::json::Array operands, results, regions;
    for (Value v : op->getOperands()) operands.push_back(int64_t(valueIds.lookup(v)));
    for (Value v : op->getResults()) results.push_back(int64_t(valueIds.lookup(v)));
    unsigned index = 0;
    for (Region &region : op->getRegions()) {
      llvm::json::Array body;
      for (Block &block : region) {
        if (!blockIds.count(&block))
          return llvm::json::Object{{"available", false}, {"reason", "unobserved empty block"}};
        body.push_back(int64_t(blockIds.lookup(&block)));
      }
      regions.push_back(llvm::json::Object{{"region", int64_t(index++)}, {"blocks", std::move(body)}});
    }
    llvm::json::Object row{{"operation", int64_t(operationIds.lookup(op))},
        {"name", op->getName().getStringRef().str()}, {"operands", std::move(operands)},
        {"results", std::move(results)}, {"regions", std::move(regions)}};
    row["block"] = op->getBlock() ? llvm::json::Value(int64_t(blockIds.lookup(op->getBlock()))) : llvm::json::Value(nullptr);
    operationRows.push_back(std::move(row));
  }
  for (Block *block : blocks) {
    llvm::json::Array arguments, body;
    for (Value v : block->getArguments()) arguments.push_back(int64_t(valueIds.lookup(v)));
    for (Operation &op : *block) body.push_back(int64_t(operationIds.lookup(&op)));
    Operation *parent = block->getParentOp(); unsigned regionIndex = 0;
    for (Region &region : parent->getRegions()) {
      if (&region == block->getParent()) break;
      ++regionIndex;
    }
    blockRows.push_back(llvm::json::Object{{"block", int64_t(blockIds.lookup(block))},
        {"arguments", std::move(arguments)}, {"operations", std::move(body)},
        {"parent_operation", int64_t(operationIds.lookup(parent))}, {"parent_region", int64_t(regionIndex)}});
  }
  const auto *values = resources.getArray("values");
  if (!values || values->size() != valueIds.size())
    return llvm::json::Object{{"available", false}, {"reason", "resource Value cardinality differs"}};
  llvm::json::Array valueRows;
  for (const auto &row : *values) valueRows.push_back(row);
  // Export the actual payload rank, not the expanded parent rank of a slice.
  // Matrix columns and AxisInfo axes therefore share one coordinate system.
  tt::ModuleAxisInfoAnalysis axes(module);
  llvm::json::Array coordinates;
  for (Operation *op : operations) {
    Value data, address;
    bool load = false;
    if (auto access = dyn_cast<tt::LoadOp>(op)) {
      data = access.getResult(); address = access.getPtr(); load = true;
    } else if (auto access = dyn_cast<tt::StoreOp>(op)) {
      data = access.getValue(); address = access.getPtr();
    } else continue;
    llvm::json::Object row{{"operation", int64_t(operationIds.lookup(op))},
        {"load", load}, {"data_value", int64_t(valueIds.lookup(data))},
        {"address_value", int64_t(valueIds.lookup(address))}, {"available", true}};
    if (auto type = dyn_cast<RankedTensorType>(data.getType())) {
      row["kind"] = "tensor";
      const auto *info = axes.getAxisInfo(address);
      if (!type.getEncoding() || !isa<ttg::DistributedEncodingTrait>(type.getEncoding()) ||
          !info || info->getContiguity().size() != unsigned(type.getRank())) {
        row["available"] = false; row["reason"] = "missing actual-rank layout or AxisInfo";
      } else {
        auto layout = ttg::toLinearLayout(type);
        llvm::json::Array inputs, outputs, shape, contiguity;
        for (int64_t dim : type.getShape()) shape.push_back(dim);
        for (int64_t n : info->getContiguity()) contiguity.push_back(n);
        for (auto name : layout.getOutDimNames()) outputs.push_back(name.getValue().str());
        for (const auto &[name, bases] : layout.getBases()) {
          llvm::json::Array vectors;
          for (const auto &basis : bases) {
            llvm::json::Array vector;
            for (int32_t n : basis) vector.push_back(int64_t(n));
            vectors.push_back(std::move(vector));
          }
          inputs.push_back(llvm::json::Object{{"dimension", name.getValue().str()},
              {"size", int64_t(layout.getInDimSize(name))}, {"bases", std::move(vectors)}});
        }
        row["mapping"] = llvm::json::Object{{"inputs", std::move(inputs)}, {"outputs", std::move(outputs)}};
        row["shape"] = std::move(shape); row["contiguity"] = std::move(contiguity);
      }
    } else row["kind"] = "scalar";
    coordinates.push_back(std::move(row));
  }
  return llvm::json::Object{{"schema", "native-ssa-region-graph-v199"}, {"available", true},
      {"operations", std::move(operationRows)}, {"blocks", std::move(blockRows)}, {"values", std::move(valueRows)},
      {"coordinate_mappings", std::move(coordinates)}, {"coordinate_axis_scope", "actual-payload-rank"},
      {"id_scope", "one-normalized-materialized-module"}, {"ids_are_inference_features", false},
      {"native_associations_checked", true}, {"exported_from_facts_clone", true}};
}

static llvm::json::Object analyzeCurrentLayoutFactsImpl(ModuleOp candidate,
    unsigned numStages, bool diagnostics, llvm::StringRef executionContext,
    bool includeSSA, bool includeResourceContract = false) {
  OwningOpRef<ModuleOp> normalized = cast<ModuleOp>(candidate->clone());
  auto normalization = normalizeSemanticFeatures(*normalized);
  if (failed(verify(*normalized)))
    return llvm::json::Object{{"feature_version", layoutFactsVersion},
        {"pipeline_stage", layoutTuningStage}, {"complete", false},
        {"issues", llvm::json::Array{"normalized analysis clone failed verification"}},
        {"values", llvm::json::Object{}}, {"feature_names", llvm::json::Array{}},
        {"semantic_normalization", std::move(normalization)}};
  auto facts = analyzeLayoutFacts(*normalized, numStages, diagnostics, executionContext);
  if (includeResourceContract)
    facts["resource_contract"] = analyzeLayoutResourceContract(*normalized);
  if (includeSSA) {
    if (const auto *resources = facts.getObject("resources"))
      facts["ssa_graph"] = exportCurrentLayoutSSA(*normalized, *resources);
    else
      facts["ssa_graph"] = llvm::json::Object{{"available", false}, {"reason", "missing diagnostic resources"}};
  }
  facts["semantic_normalization"] = std::move(normalization);
  return facts;
}

llvm::json::Object analyzeCurrentLayoutFacts(ModuleOp candidate,
    unsigned numStages, bool diagnostics, llvm::StringRef executionContext) {
  return analyzeCurrentLayoutFactsImpl(candidate, numStages, diagnostics, executionContext, false);
}

llvm::json::Object analyzeCurrentLayoutSSAFacts(ModuleOp candidate,
    unsigned numStages, llvm::StringRef executionContext) {
  return analyzeCurrentLayoutFactsImpl(candidate, numStages, true, executionContext, true);
}

llvm::json::Object analyzeCurrentLayoutResourceContractFacts(ModuleOp candidate,
    unsigned numStages, llvm::StringRef executionContext, bool includeSSA) {
  return analyzeCurrentLayoutFactsImpl(candidate, numStages, true, executionContext,
                                       includeSSA, true);
}

LogicalResult tuneLayouts(ModuleOp mod, const LayoutTuningOptions &options,
                          std::string *reportJSON) {
  using Clock = std::chrono::steady_clock;
  auto passStart = Clock::now();
  auto elapsed = [](Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); };
  double fullScoreSeconds=0, proposalSeconds=0;
  uint64_t fullScoreCalls=0, proposalCalls=0, poolGenerated=0, heuristicCalls=0;
  if (options.scorer != "model" || options.mode != "search" ||
      options.algorithm != "beam" || options.candidatePolicy != "balanced-coherent")
    return mod.emitError("this pass supports only frozen-model beam search with balanced-coherent candidates");
  if (options.mfmaWarpCandidates && options.projection.arch != "gfx936")
    return mod.emitError("the frozen MFMA candidate domain requires gfx936");
  if (!options.maxCandidates || !options.beamWidth || !options.maxTrials ||
      options.candidatePool < options.maxCandidates || options.candidatePool > 512)
    return mod.emitError("invalid layout beam search budget");
  constexpr bool factsOnly = false;
  std::optional<LayoutLearnedModel> learned;
  {
    if (options.jointModelPath.empty() || options.jointModelSHA256.size() != 64 ||
        options.jointPolicy != layoutLearnedPolicy)
      return mod.emitError("model pass requires a stage-bound generic-current-layout-v114 model, SHA256 and policy; older facts require retraining");
    auto buffer=llvm::MemoryBuffer::getFile(options.jointModelPath);
    if(!buffer)return mod.emitError("cannot read learned layout model");
    llvm::SHA256 sha;sha.update((*buffer)->getBuffer());
    if(llvm::toHex(sha.final(),true)!=options.jointModelSHA256)
      return mod.emitError("learned layout model SHA256 mismatch");
    auto document=llvm::json::parse((*buffer)->getBuffer());
    if(!document){llvm::consumeError(document.takeError());return mod.emitError("invalid learned layout model JSON");}
    LayoutLearnedModel parsed;std::string error;
    if(!readLayoutLearnedModel(*document,parsed,error))return mod.emitError(error);
    learned=std::move(parsed);
  }
  LayoutModel model(mod, options.mfmaWarpCandidates);
  llvm::json::Object report{{"schema", 2}, {"pass", options.scorer == "model" ? "tritonhcugpu-layout-model-tuning" : "tritonhcugpu-layout-program-cost-tuning"},
                            {"mode", options.mode}, {"algorithm", options.algorithm}, {"model", model.dump()}};
  report["pipeline_stage"] = layoutTuningStage;
  report["scoring_contract"] = options.scorer;
  report["fallback_allowed"] = false;
  unsigned trials = 0, rejected = 0, changed = 0, coherentTrials = 0;
  std::string reason, firstRejection;
  Plan selected = model.seed;
  auto original = model.materialize(model.seed, reason);
  if (!original) return mod.emitError("layout seed replay failed: ") << reason;
  auto evaluate = [&](const Plan &plan, ModuleOp candidate) {
    auto scoreStart=Clock::now(); ++fullScoreCalls;
    Score score;score.modelOnly=true;score.jointActive=true;
    auto projection = options.projection;
    projection.numStages = options.numStages;
    score.layoutFacts=learned && learned->unified183 ?
        projectLayoutUnifiedFacts(candidate,projection) : projectLayoutFacts(candidate,projection);
    if(learned) {
      score.jointRanking=scoreLayoutFacts(*learned,score.layoutFacts,options.jointModelSHA256);
      score.jointTotal=score.jointRanking.getNumber("total").value_or(std::numeric_limits<double>::infinity());
      score.jointFailure=score.jointRanking.getString("failure").value_or("").str();
    } else {
      score.jointFailure="facts-only analysis; no model loaded or score claimed";
      score.jointRanking=llvm::json::Object{{"scorer","facts-only"},{"total",nullptr},
          {"failure",score.jointFailure},{"fallback_allowed",false}};
    }
    fullScoreSeconds+=elapsed(scoreStart);
    return score;
  };
  auto domainFor = [&](const Plan &plan, unsigned target, bool local) {
    Attribute like=local ? model.localEncoding(plan,target) : plan[target];
    auto start=Clock::now(); ++proposalCalls;
    std::vector<Attribute> result{like};
    auto root=dyn_cast<ttg::BlockedEncodingAttr>(rootEncoding(like));
    if (!root) return result;
    BlockedLayoutChoice choice{{root.getSizePerThread().begin(),root.getSizePerThread().end()},
      {root.getThreadsPerWarp().begin(),root.getThreadsPerWarp().end()},
      {root.getWarpsPerCTA().begin(),root.getWarpsPerCTA().end()}, {root.getOrder().begin(),root.getOrder().end()}};
    auto pool=generateBlockedProposalPool(choice,options.candidatePool);
    poolGenerated+=pool.size();
    std::vector<Attribute> attrs;
    std::vector<LayoutProposalMetrics> metrics(pool.size());
    for (unsigned i=0;i<pool.size();++i) {
      const auto &c=pool[i];
      Attribute attr=replaceRoot(like,ttg::BlockedEncodingAttr::get(mod.getContext(),c.sizePerThread,c.threadsPerWarp,c.warpsPerCTA,c.order,root.getCTALayout()));
      attrs.push_back(attr);
    }
    result.clear();
    for (unsigned i:selectBlockedProposals(pool,metrics,options.maxCandidates,false,target))
      result.push_back(attrs[i]);
    proposalSeconds+=elapsed(start); return result;
  };
  Score seedScore=evaluate(model.seed,*original),bestScore=seedScore;
  unsigned searchBudget=options.maxTrials;
  bool learnedActive=bool(learned);
  const char *finalScorer=learned ? layoutLearnedPolicy : factsOnly ? "facts-only" : "ProgramCost";
  report["active_scorer"]=finalScorer;
  report["model_active"]=bool(learned);
  report["workload_context_used_by_scorer"]=false;
  llvm::json::Array trialDetails;
  unsigned selectedTrial = 0;
  bool traceFailed = false;
  if (!options.tracePath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(options.tracePath))
      return mod.emitError("cannot create layout trace directory: ") << ec.message();
    std::error_code ec;
    llvm::raw_fd_ostream input(options.tracePath + "/pass-input.ttgir", ec, llvm::sys::fs::OF_Text);
    if (ec) return mod.emitError("cannot write layout pass input: ") << ec.message();
    mod.print(input, OpPrintingFlags().enableDebugInfo()); input << "\n";
  }
  auto record = [&](unsigned id, unsigned parent, unsigned anchor, Attribute requested,
                    StringRef status, StringRef diagnostic, const Plan *plan,
                    const Score *score, ModuleOp candidate, StringRef move = "component", int dpWindow = -1,
                    const std::vector<unsigned> *oracleAssignment = nullptr) {
    if (options.tracePath.empty()) return;
    llvm::json::Object row{{"id", int64_t(id)}, {"parent", int64_t(parent)},
                          {"status", status.str()}, {"reason", diagnostic.str()}, {"move_kind", move.str()}};
    if (dpWindow >= 0) row["dp_window"] = int64_t(dpWindow);
    if (oracleAssignment) {
      llvm::json::Array digits; for (unsigned digit : *oracleAssignment) digits.push_back(int64_t(digit));
      row["oracle_assignment"] = std::move(digits);
    }
    if (requested) {
      row["anchor"] = int64_t(anchor);
      row["requested_encoding"] = printed(requested);
    }
    if (score) row["score"] = scoreDetails(*score);
    if (plan) {
      llvm::json::Array changes;
      for (unsigned i = 0; i < plan->size(); ++i)
        if ((*plan)[i] != model.seed[i])
          changes.push_back(llvm::json::Object{{"value", int64_t(i)},
                                              {"encoding", printed((*plan)[i])}});
      row["changes"] = std::move(changes);
      row["use_changes"] = model.dumpUseAssignment(*plan, true);
      row["mfma_warp_changes"] = model.dumpMatrixAssignment(*plan, true);
    }
    if (candidate) {
      std::string file = "candidate-" + std::to_string(id) + ".ttgir";
      std::error_code ec;
      llvm::raw_fd_ostream out(options.tracePath + "/" + file, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        mod.emitError("cannot write layout candidate: ") << ec.message(); traceFailed = true;
      } else {
        // Preserve original source/NameLocs. Dropping them makes reparsing
        // assign snapshot-file locations and can change the generated binary.
        candidate.print(out, OpPrintingFlags().enableDebugInfo()); out << "\n";
        row["ir_file"] = file;
      }
    }
    trialDetails.push_back(std::move(row));
  };
  record(0, 0, 0, {}, "seed", "", &model.seed, &seedScore, *original, "seed");
  {
    struct State { Plan plan; Score score; std::string key; unsigned trial; };
    std::vector<State> beam{{model.seed, seedScore, planKey(model.seed), 0}};
    std::set<std::string> seen{beam.front().key};
    std::set<std::string> expanded;
    if (learned && options.algorithm=="beam" && options.candidatePolicy=="balanced-coherent") {
      for (unsigned spt:{1u,2u,4u,8u,16u}) {
        if (trials>=searchBudget) break;
        reason.clear();
        std::optional<Plan> proposed;
        {
          ScopedDiagnosticHandler handler(mod.getContext(),[&](Diagnostic &diag) {
            llvm::raw_string_ostream os(reason);diag.print(os);return success();
          });
          proposed=model.proposeCoherentSPT(model.seed,spt);
        }
        if (!proposed) {
          ++trials;++coherentTrials;++rejected;
          record(trials,0,0,{},"constraint-rejected","coherent SPT conflicts with a fixed/rank interface",nullptr,nullptr,{},"coherent-blocked-spt");
          continue;
        }
        auto key=planKey(*proposed);if (!seen.insert(key).second) continue;
        ++trials;++coherentTrials;
        auto clone=model.materialize(*proposed,reason);
        if (!clone) {
          ++rejected;record(trials,0,0,{},"verifier-rejected",reason,&*proposed,nullptr,{},"coherent-blocked-spt");continue;
        }
        auto score=evaluate(*proposed,*clone);
        if (!std::isfinite(score.ranking())) {
          ++rejected;record(trials,0,0,{},"cost-unknown",score.rankingFailure(),&*proposed,&score,{},"coherent-blocked-spt");continue;
        }
        record(trials,0,0,{},"verified","",&*proposed,&score,*clone,"coherent-blocked-spt");
        if (score.ranking()+1e-9<bestScore.ranking()) {
          bestScore=score;selected=*proposed;selectedTrial=trials;
        }
        beam.push_back({std::move(*proposed),std::move(score),std::move(key),trials});
      }
      std::stable_sort(beam.begin(),beam.end(),[](const State &a,const State &b) {
        return a.score.ranking()<b.score.ranking() || (a.score.ranking()==b.score.ranking() && a.key<b.key);
      });
      if (beam.size()>options.beamWidth) beam.resize(options.beamWidth);
    }
    while (trials < searchBudget) {
      bool progressed = false;
      std::vector<State> next = beam;
      for (const State &state : beam) {
        if (!expanded.insert(state.key).second) continue;
        progressed = true;
        struct Move { bool local; unsigned anchor; Attribute attr; bool matrix = false; };
        std::vector<Move> componentMoves, localMoves, moves;
        // Reach matrix distributions before spending the budget on ordinary
        // blocked/use neighbors. Every distribution uses the same full scorer.
        for (unsigned oi : model.matrixOps)
          for (Attribute hint : model.matrixDomain(oi))
            if (hint != state.plan.mfmaWarpHints[oi]) moves.push_back({false, oi, hint, true});
        for (unsigned anchor : model.anchors)
          for (Attribute attr : domainFor(state.plan,anchor,false))
            if (attr != state.plan[anchor]) componentMoves.push_back({false, anchor, attr});
        // Round-robin over operations so a small budget reaches multiple use
        // boundaries rather than exhausting all neighbors of one load first.
        std::vector<std::vector<Attribute>> localDomains;
        for (unsigned oi : model.boundaryOps)
          localDomains.push_back(domainFor(state.plan,oi,true));
        for (unsigned neighbor = 1; neighbor < options.maxCandidates; ++neighbor)
          for (unsigned i = 0; i < localDomains.size(); ++i)
            if (neighbor < localDomains[i].size())
              localMoves.push_back({true, model.boundaryOps[i], localDomains[i][neighbor]});
        for (unsigned i = 0; i < std::max(componentMoves.size(), localMoves.size()); ++i) {
          if (i < componentMoves.size()) moves.push_back(componentMoves[i]);
          if (i < localMoves.size()) moves.push_back(localMoves[i]);
        }
        for (const Move &move : moves) {
            unsigned anchor = move.anchor;
            Attribute attr = move.attr;
            StringRef kind = move.matrix ? "mfma-warps" : move.local ? "local-use-boundary" : "component";
            if (trials >= searchBudget) break;
            reason.clear();
            std::optional<Plan> proposed;
            {
              ScopedDiagnosticHandler handler(mod.getContext(), [&](Diagnostic &diag) {
                llvm::raw_string_ostream os(reason); diag.print(os); return success();
              });
              if (move.matrix) {
                proposed = state.plan;
                proposed->mfmaWarpHints[anchor] = attr;
              } else proposed = move.local ? model.proposeLocal(state.plan, anchor, attr)
                                           : model.propose(state.plan, anchor, attr);
            }
            if (!proposed) {
              ++trials; ++rejected;
              if (firstRejection.empty()) firstRejection = reason.empty() ? "constraint conflict or fixed interface" : reason;
              record(trials, state.trial, anchor, attr, "constraint-rejected",
                     reason.empty() ? "constraint conflict or fixed interface" : reason, nullptr, nullptr, {}, kind);
              continue;
            }
            auto key = planKey(*proposed);
            if (!seen.insert(key).second) continue;
            ++trials;
            reason.clear();
            auto clone = model.materialize(*proposed, reason);
            if (!clone) {
              ++rejected;
              if (firstRejection.empty()) firstRejection = reason;
              record(trials, state.trial, anchor, attr, "verifier-rejected", reason, &*proposed, nullptr, {}, kind);
              continue;
            }
            auto score = evaluate(*proposed, *clone);
            if (!std::isfinite(score.ranking())) {
              ++rejected;
              record(trials, state.trial, anchor, attr, "cost-unknown", learnedActive ? score.rankingFailure() : "non-finite estimate", &*proposed, &score, {}, kind);
              continue;
            }
            record(trials, state.trial, anchor, attr, "verified", "", &*proposed, &score, *clone, kind);
            if (score.ranking() + 1e-9 < bestScore.ranking()) {
              bestScore = score; selected = *proposed; selectedTrial = trials;
            }
            next.push_back({std::move(*proposed), std::move(score), std::move(key), trials});
        }
        if (trials >= searchBudget) break;
      }
      std::stable_sort(next.begin(), next.end(), [](const State &a, const State &b) {
        return a.score.ranking() < b.score.ranking() || (a.score.ranking() == b.score.ranking() && a.key < b.key);
      });
      unsigned width = options.beamWidth;
      if (next.size() > width) next.resize(width);
      beam = std::move(next);
      if (!progressed) break;
    }
  }
  if (traceFailed) return failure();
  for (unsigned i = 0; i < selected.size(); ++i) changed += selected[i] != model.seed[i];
  reason.clear();
  auto final = model.materialize(selected, reason);
  if (!final) return mod.emitError("selected layout plan failed verification: ") << reason;
  auto committedScore = evaluate(selected, *final);
  if (std::abs(committedScore.ranking() - bestScore.ranking()) > 1.e-6 ||
      llvm::json::Value(scoreDetails(committedScore)) != llvm::json::Value(scoreDetails(bestScore)))
    return mod.emitError("selected conversion plan differs from evaluated candidate");
  report["candidate_domain"] = llvm::json::Object{
      {"policy",options.candidatePolicy}, {"pool_cap",int64_t(options.candidatePool)},
      {"subset_cap",int64_t(options.maxCandidates)}, {"calls",int64_t(proposalCalls)},
      {"raw_proposals",int64_t(poolGenerated)}, {"heuristic_calls",int64_t(heuristicCalls)},
      {"coherent_spt_trials",int64_t(coherentTrials)},
      {"coherent_spt_policy","model beam only; legal whole-component products; same full scorer and global trial budget"},
      {"heuristic", "Pareto priority of current complete-plan memory/conversion/reduction/register proxies; no scalar weights"},
      {"full_domain_enumerated",false}, {"final_scorer_unchanged",true}};
  report["compile_timing"] = llvm::json::Object{
      {"schema","native-layout-pass-timing-v186"}, {"unit","seconds"},
      {"pass_until_report_seconds",elapsed(passStart)}, {"full_score_seconds",fullScoreSeconds},
      {"proposal_selection_seconds",proposalSeconds}, {"full_score_calls",int64_t(fullScoreCalls)},
      {"kernel_runtime_included",false}};
  report["trials"] = int64_t(trials);
  report["rejected"] = int64_t(rejected);
  report["changed_values"] = int64_t(changed);
  report["budget_exhausted"] = trials >= options.maxTrials && options.mode == "search";
  report["first_rejection"] = firstRejection;
  report["seed_cost"] = number(seedScore.ranking());
  report["selected_cost"] = number(bestScore.ranking());
  report["seed_score"] = scoreDetails(seedScore);
  report["selected_score"] = scoreDetails(bestScore);
  report["selected_trial"] = int64_t(selectedTrial);
  report["selection_status"] = factsOnly ? "facts-only-no-ranking" :
      std::isfinite(bestScore.ranking()) ? "finite-complete-candidate" :
      options.scorer == "model" ? "coverage-failure-no-fallback" : "no-supported-candidate-original-preserved";
  if (!options.tracePath.empty()) {
    llvm::json::Object trace{{"schema", 2}, {"selected_trial", int64_t(selectedTrial)},
                             {"trials", std::move(trialDetails)}};
    std::error_code ec;
    llvm::raw_fd_ostream out(options.tracePath + "/trials.json", ec, llvm::sys::fs::OF_Text);
    if (ec) return mod.emitError("cannot write layout trial trace: ") << ec.message();
    out << llvm::json::Value(std::move(trace)) << "\n";
  }
  report["cost_sources"] = llvm::json::Object{};
  llvm::json::Array assignments;
  for (unsigned i = 0; i < selected.size(); ++i)
    assignments.push_back(llvm::json::Object{{"value", int64_t(i)}, {"encoding", printed(selected[i])}});
  report["assignment"] = std::move(assignments);
  report["use_assignment"] = model.dumpUseAssignment(selected);
  report["mfma_warp_domain_enabled"] = options.mfmaWarpCandidates;
  report["mfma_warp_assignment"] = model.dumpMatrixAssignment(selected);
  report["changed_mfma_warp_hints"] = int64_t(model.dumpMatrixAssignment(selected, true).size());
  report["changed_uses"] = int64_t(model.dumpUseAssignment(selected, true).size());
  if (reportJSON) {
    llvm::raw_string_ostream out(*reportJSON);
    out << llvm::json::Value(llvm::json::Object(report));
    out.flush();
  }
  if (!options.dumpPath.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream out(options.dumpPath, ec, llvm::sys::fs::OF_Text);
    if (ec) return mod.emitError("cannot write layout tuning report: ") << ec.message();
    out << llvm::json::Value(std::move(report)) << "\n";
  }
  if (options.scorer == "model" && !factsOnly && !std::isfinite(bestScore.ranking()))
    return mod.emitError("model pass could not score a complete candidate; coverage failure, no ProgramCost fallback: ") << bestScore.rankingFailure();
  // Commit only after verification, score consistency and diagnostic I/O.
  if ((options.mode == "search" || options.mode == "replay") && selected != model.seed &&
      (!learnedActive || std::isfinite(bestScore.ranking()))) mod.getBodyRegion().takeBody(final->getBodyRegion());
  llvm::errs() << "[layout-tuning] mode=" << options.mode << " values=" << selected.size()
               << " trials=" << trials << " rejected=" << rejected << " changed=" << changed << "\n";
  return success();
}
} // namespace layoutdp
