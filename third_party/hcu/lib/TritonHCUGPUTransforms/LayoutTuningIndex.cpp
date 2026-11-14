// Native address/mask SSA slice plus explicit region/recurrence boundaries.
// No textual IR parsing, speculative CSE, or additional scalar cost here.
#include "TritonHCUGPUTransforms/LayoutTuningIndex.h"
#include "TritonHCUGPUTransforms/LayoutTuningMemory.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Analysis/HCUGPUAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/GenericSwizzling.h"
#include <cmath>
#include <map>
#include <set>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace hcu = mlir::triton::hcugpu;
namespace layoutdp {
namespace {
std::string printed(Type type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}
std::optional<APInt> integerBits(Value v) {
  auto c = v.getDefiningOp<arith::ConstantOp>();
  if (!c) return {};
  if (auto i = dyn_cast<IntegerAttr>(c.getValue())) return i.getValue();
  if (auto d = dyn_cast<DenseIntElementsAttr>(c.getValue()))
    if (d.isSplat()) return d.getSplatValue<APInt>();
  return {};
}
std::optional<int64_t> integer(Value v) {
  if (auto bits = integerBits(v))
    if (bits->getBitWidth() <= 64) return bits->getSExtValue();
  return {};
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
llvm::json::Object frequency(Operation *op, const Ids &ids) {
  double weight = 1;
  llvm::json::Array loops, branches;
  bool dynamic = false;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(p)) {
      auto a = integer(loop.getLowerBound()), b = integer(loop.getUpperBound()), s = integer(loop.getStep());
      std::optional<double> trip;
      if (a && b && s && *s > 0) {
        // Double arithmetic avoids signed overflow in very large bounds.
        trip = std::max(0.0, std::ceil((double(*b) - double(*a)) / double(*s)));
        weight *= *trip;
      } else dynamic = true;
      llvm::json::Object item{{"operation", int64_t(ids.ops.lookup(p))}};
      item["static_trip_count"] = trip ? llvm::json::Value(*trip) : llvm::json::Value(nullptr);
      loops.push_back(std::move(item));
    } else if (isa<scf::WhileOp>(p)) {
      dynamic = true;
      loops.push_back(llvm::json::Object{{"operation", int64_t(ids.ops.lookup(p))}, {"static_trip_count", nullptr}});
    } else if (isa<scf::IfOp>(p)) branches.push_back(int64_t(ids.ops.lookup(p)));
  }
  return llvm::json::Object{{"static_execution_weight", weight}, {"dynamic_loop_count_unknown", dynamic},
      {"enclosing_loops", std::move(loops)}, {"conditional_regions", std::move(branches)},
      {"interpretation", "static multiplicity only; unknown loop counts use one; branch probabilities not modeled"}};
}
bool expression(Operation *op) {
  auto name = op->getName().getStringRef();
  if (name.starts_with("arith.") || name.starts_with("math.")) return !op->getNumRegions();
  return name == "tt.addptr" || name == "tt.make_range" || name == "tt.get_program_id" ||
      name == "tt.get_num_programs" || name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.trans" || name == "tt.reshape" ||
      name == "tt.join" || name == "tt.split" || name == "ttg.convert_layout" ||
      name == "tt.ptr_to_int" || name == "tt.int_to_ptr" ||
      name == "ttg.memdesc_index" || name == "ttg.memdesc_subslice" ||
      name == "ttg.memdesc_trans" || name == "ttg.memdesc_reshape";
}
std::string category(Operation *op) {
  auto n = op->getName().getStringRef();
  if (n == "arith.constant") return "constant";
  if (n == "tt.get_program_id" || n == "tt.get_num_programs") return "program-coordinate";
  if (n == "tt.make_range") return "layout-coordinate";
  if (n == "tt.addptr") return "pointer-add";
  if (n == "tt.ptr_to_int" || n == "tt.int_to_ptr") return "pointer-integer-cast";
  if (n == "ttg.convert_layout") return "conversion-owned";
  if (n == "arith.divui" || n == "arith.divsi" || n == "arith.remui" || n == "arith.remsi" || n == "arith.ceildivui" || n == "arith.ceildivsi" || n == "arith.floordivsi") {
    auto divisor = integerBits(op->getOperand(1));
    if (!divisor) return "div-rem-dynamic";
    if (divisor->isZero()) return "div-rem-zero-undefined";
    bool unsignedOp = n == "arith.divui" || n == "arith.remui" || n == "arith.ceildivui";
    if ((unsignedOp || !divisor->isNegative()) && divisor->isPowerOf2()) return "div-rem-positive-power-of-two";
    return "div-rem-other-constant";
  }
  if (n == "arith.cmpi" || n == "arith.cmpf" || n == "arith.select") return "predicate-select";
  if (n == "arith.extsi" || n == "arith.extui" || n == "arith.trunci" || n == "arith.index_cast" || n == "arith.index_castui") return "integer-width-change";
  if (n == "tt.splat" || n == "tt.broadcast" || n == "tt.expand_dims" || n == "tt.trans" || n == "tt.reshape" || n == "tt.join" || n == "tt.split") return "shape-view";
  if (n.starts_with("arith.")) return "arithmetic";
  if (n.starts_with("math.")) return "math";
  return "descriptor-view";
}

// These are physical mapping features, NOT an instruction estimate. Lowering
// folds register constants, flattens lane/warp bases, then matrixVectorProd
// optimizes bit segments. LLVM can share/fold further. Keep the raw matrix.
llvm::json::Object mapping(const tt::LinearLayout &ll) {
  auto reg = StringAttr::get((*ll.getInDimNames().begin()).getContext(), "register");
  llvm::json::Array inputs, outputs;
  unsigned dynamicBits = 0, dynamicTerms = 0, xorJoins = 0;
  std::vector<unsigned> terms(ll.getNumOutDims(), 0);
  for (auto name : ll.getOutDimNames()) outputs.push_back(name.getValue().str());
  for (const auto &[name, bases] : ll.getBases()) {
    llvm::json::Array vectors;
    for (const auto &basis : bases) {
      llvm::json::Array vector;
      bool nonzero = false;
      for (unsigned i = 0; i < basis.size(); ++i) {
        vector.push_back(int64_t(basis[i]));
        if (basis[i] && (name.getValue() == "lane" || name.getValue() == "warp")) {
          ++terms[i]; ++dynamicTerms; nonzero = true;
        }
      }
      dynamicBits += nonzero;
      vectors.push_back(std::move(vector));
    }
    inputs.push_back(llvm::json::Object{{"dimension", name.getValue().str()}, {"size", int64_t(ll.getInDimSize(name))}, {"bases", std::move(vectors)}});
  }
  for (unsigned t : terms) xorJoins += t ? t - 1 : 0;
  auto unique = tt::actionRemoveBroadcastedRegs(ll).apply(ll);
  return llvm::json::Object{{"inputs", std::move(inputs)}, {"outputs", std::move(outputs)},
      {"register_elements", int64_t(ll.getInDimSize(reg))},
      {"unique_register_elements", int64_t(unique.getInDimSize(reg))},
      {"lane_warp_nonzero_input_bits", int64_t(dynamicBits)}, {"lane_warp_output_terms", int64_t(dynamicTerms)},
      {"lane_warp_xor_joins_proxy", int64_t(xorJoins)}, {"exact_instruction_count", false},
      {"lowering_basis", "applyLinearLayout: register constants folded; lane/warp matrixVectorProd; block coordinate separately exposed"}};
}
llvm::json::Object mapping(RankedTensorType type) { return mapping(ttg::toLinearLayout(type)); }

llvm::json::Object swizzledAddress(tt::LinearLayout cvt, unsigned bits, unsigned repetitions) {
  auto *ctx = (*cvt.getInDimNames().begin()).getContext();
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
       warp = StringAttr::get(ctx, "warp"), offset = StringAttr::get(ctx, "offset");
  auto [vector, permutation] = tt::largestVectorisation(ctx, cvt, bits, std::nullopt);
  cvt = permutation.apply(cvt);
  auto tile = tt::LinearLayout::identity1D(vector, reg, offset);
  auto quotient = divideLeft(cvt, tile);
  if (!quotient) return llvm::json::Object{{"available", false}, {"reason", "native vector tile not divisible"}};
  auto reps = tt::zerosLike(tile) * *quotient;
  auto addr = tt::LinearLayout({{lane, reps.getBases().lookup(lane)}, {warp, reps.getBases().lookup(warp)}}, reps.getOutDims(), false);
  auto [additive, strides] = tt::actionAdditiveStrides(reps, addr, 0);
  auto bytes = tt::zerosLike(tt::LinearLayout::identity1D(bits / 8, reg, offset)) * addr;
  uint64_t vectors = uint64_t(cvt.getInDimSize(reg)) / vector * repetitions;
  uint64_t groups = uint64_t(cvt.getInDimSize(reg)) / additive * repetitions;
  llvm::json::Array dag;
  dag.push_back(llvm::json::Object{{"id", 0}, {"op", "linear-layout-byte-base"}, {"inputs", llvm::json::Array{}}, {"instances_before_cse", int64_t(repetitions)}});
  dag.push_back(llvm::json::Object{{"id", 1}, {"op", "xor-register-immediate"}, {"inputs", llvm::json::Array{0}}, {"instances_before_cse", int64_t(groups)}});
  dag.push_back(llvm::json::Object{{"id", 2}, {"op", "add-vector-immediate"}, {"inputs", llvm::json::Array{1}}, {"instances_before_cse", int64_t(vectors)}});
  dag.push_back(llvm::json::Object{{"id", 3}, {"op", "gep-byte"}, {"inputs", llvm::json::Array{2}}, {"instances_before_cse", int64_t(vectors)}});
  return llvm::json::Object{{"available", true}, {"vector_elements", int64_t(vector)}, {"additive_elements", int64_t(additive)},
      {"repetitions", int64_t(repetitions)}, {"vector_addresses_before_cse", int64_t(vectors)}, {"byte_base_mapping", mapping(bytes)},
      {"address_template_dag", std::move(dag)}, {"exact_instruction_count", false},
      {"source", "native lowerLdStShared/lowerLdSt vectorisation + additive strides; constant folding/CSE/immediate folding can remove templates"}};
}
llvm::json::Array scratchAddresses(ttg::ConvertLayoutOp op) {
  auto *ctx = op.getContext();
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
       warp = StringAttr::get(ctx, "warp"), reps = StringAttr::get(ctx, "reps"),
       offset = StringAttr::get(ctx, "offset"), iteration = StringAttr::get(ctx, "iteration");
  auto src = ttg::toLinearLayout(op.getSrc().getType()), dst = ttg::toLinearLayout(op.getType());
  unsigned bits = tt::getBitwidth(op.getType());
  auto within = [&](const tt::LinearLayout &ll) {
    auto block = ll.sublayout({reg, lane, warp}, llvm::to_vector(ll.getOutDimNames()));
    return tt::actionRemoveBroadcastedRegs(block).apply(block);
  };
  auto s = within(src), d = within(dst);
  auto shared = ttg::optimalSwizzlingLdSt(s, d, bits);
  unsigned count = shared.getInDimSize(reps);
  auto repetition = tt::LinearLayout::identity1D(count, reg, reps);
  auto swizzle = [&](tt::LinearLayout ll) {
    auto total = ll.invertAndCompose(shared);
    auto perm = tt::regPermForDivide(total, repetition, false);
    if (!perm) return llvm::json::Object{{"available", false}, {"reason", "native repetition permutation unavailable"}};
    auto divided = divideRight(perm->apply(total), repetition);
    if (!divided) return llvm::json::Object{{"available", false}, {"reason", "native repetition division unavailable"}};
    return swizzledAddress(divided->reshapeOuts({{offset, divided->getTotalOutDimSize()}}), bits, count);
  };
  llvm::json::Array scenarios;
  scenarios.push_back(llvm::json::Object{{"strategy", "swizzled"}, {"store", swizzle(s)}, {"load", swizzle(d)}, {"alternatives_not_additive", true}});
  auto config = tt::HCU::getScratchConfigForCvt(op.getSrc().getType(), op.getType());
  auto shape = ttg::getShapePerCTA(op.getSrc().getType().getEncoding(), op.getType().getShape());
  SmallVector<unsigned> ctaShape(shape.begin(), shape.end());
  auto padded = ttg::chooseShemLayoutForRegToRegConversion(ctx, ctaShape, config.repShape, config.order);
  unsigned iterations = padded.getInDimSize(iteration);
  unsigned stride = config.repShape[config.order[0]];
  unsigned padding = config.paddedRepShape[config.order[0]] - stride;
  auto paddedSide = [&](tt::LinearLayout ll, unsigned vector) {
    auto address = ll.invertAndCompose(padded);
    uint64_t vectors = uint64_t(address.getInDimSize(reg)) / vector;
    llvm::json::Array dag;
    dag.push_back(llvm::json::Object{{"id", 0}, {"op", "linear-layout-element-base"}, {"inputs", llvm::json::Array{}}, {"instances_before_cse", 1}});
    dag.push_back(llvm::json::Object{{"id", 1}, {"op", "xor-register-immediate"}, {"inputs", llvm::json::Array{0}}, {"instances_before_cse", int64_t(vectors)}});
    unsigned last = 1;
    if (padding) {
      for (auto name : {"lshr-padding-stride", "shl-padding-size", "add-padding"}) {
        llvm::json::Array inputs{int64_t(last)};
        if (StringRef(name) == "add-padding") inputs.push_back(1);
        dag.push_back(llvm::json::Object{{"id", int64_t(last + 1)}, {"op", name}, {"inputs", std::move(inputs)}, {"instances_before_cse", int64_t(vectors)}});
        ++last;
      }
    }
    dag.push_back(llvm::json::Object{{"id", int64_t(last + 1)}, {"op", "gep-element-scaled"}, {"inputs", llvm::json::Array{int64_t(last)}}, {"instances_before_cse", int64_t(vectors)}});
    return llvm::json::Object{{"available", true}, {"vector_elements", int64_t(vector)}, {"repetitions", int64_t(iterations)},
        {"vector_addresses_before_cse", int64_t(vectors)}, {"element_offset_mapping", mapping(address)},
        {"padding_stride_elements", int64_t(stride)}, {"padding_size_elements", int64_t(padding)},
        {"address_template_dag", std::move(dag)}, {"exact_instruction_count", false}, {"source", "native HCU ConvertLayoutForcedPadding address template before LLVM folding"}};
  };
  scenarios.push_back(llvm::json::Object{{"strategy", "padded"}, {"store", paddedSide(src, config.inVec)}, {"load", paddedSide(dst, config.outVec)}, {"alternatives_not_additive", true}});
  return scenarios;
}
struct Edge { Value value; std::string kind; };
struct Root { Operation *op; Value value; std::string role; };
bool functionEntry(Block *block) {
  auto function = dyn_cast<FunctionOpInterface>(block->getParentOp());
  return function && !function.getFunctionBody().empty() &&
         block == &function.getFunctionBody().front();
}
class Analysis {
public:
  ModuleOp module;
  Ids ids;
  tt::ModuleAxisInfoAnalysis axes;
  std::vector<Root> roots;
  std::map<unsigned, std::vector<Edge>> edges;
  std::map<unsigned, std::set<unsigned>> membership;
  DenseMap<Value, bool> uniform;
  bool extended = false;
  explicit Analysis(ModuleOp module, bool extended = false)
      : module(module), ids(module), axes(module), extended(extended) {
    for (Value v : ids.orderedValues) {
      bool u = false;
      if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
        if (auto *info = axes.getAxisInfo(v)) {
          auto c = info->getConstancy(); u = c.size() == type.getRank();
          for (unsigned i = 0; i < c.size() && u; ++i) u &= c[i] >= type.getShape()[i];
        }
      } else if (auto arg = dyn_cast<BlockArgument>(v)) {
        auto *p = arg.getOwner()->getParentOp();
        auto visibility = p->getAttrOfType<StringAttr>("sym_visibility");
        // Only entry ABI arguments are uniform. Non-entry CFG block arguments
        // may carry lane-varying values even in a public kernel.
        u = functionEntry(arg.getOwner()) && (!visibility || visibility.getValue() == "public");
      }
      if (auto *op = v.getDefiningOp()) {
        auto n = op->getName().getStringRef();
        if (n == "tt.get_program_id" || n == "tt.get_num_programs") u = true;
        if (auto c = dyn_cast<arith::ConstantOp>(op)) {
          u = !isa<RankedTensorType>(v.getType());
          if (auto d = dyn_cast<DenseElementsAttr>(c.getValue())) u = d.isSplat();
        }
      }
      uniform[v] = u;
    }
    bool changed;
    do {
      changed = false;
      for (Operation *op : ids.orderedOps) {
        auto n = op->getName().getStringRef();
        bool pure = (n.starts_with("arith.") || n.starts_with("math.")) && op->hasTrait<OpTrait::Elementwise>();
        bool copy = n == "tt.addptr" || n == "tt.splat" || n == "tt.broadcast" || n == "tt.expand_dims" || n == "ttg.convert_layout";
        if ((pure || copy) && op->getNumOperands() && llvm::all_of(op->getOperands(), [&](Value v) { return uniform.lookup(v); }))
          for (Value v : op->getResults()) if (!uniform.lookup(v)) { uniform[v] = true; changed = true; }
        if (auto loop = dyn_cast<scf::ForOp>(op))
          if (uniform.lookup(loop.getLowerBound()) && uniform.lookup(loop.getUpperBound()) && uniform.lookup(loop.getStep()) && !uniform.lookup(loop.getInductionVar())) {
            uniform[loop.getInductionVar()] = true; changed = true;
          }
      }
    } while (changed);
  }
  void root(Operation *op, Value v, StringRef role) { if (v) roots.push_back({op, v, role.str()}); }
  void findRoots() {
    for (Operation *op : ids.orderedOps) {
      if (auto access = dyn_cast<tt::LoadOp>(op)) { root(op, access.getPtr(), "global-pointer"); root(op, access.getMask(), "mask"); }
      else if (auto access = dyn_cast<tt::StoreOp>(op)) { root(op, access.getPtr(), "global-pointer"); root(op, access.getMask(), "mask"); }
      else if (auto access = dyn_cast<tt::AtomicRMWOp>(op)) { root(op, access.getPtr(), "atomic-pointer"); root(op, access.getMask(), "mask"); }
      else if (auto access = dyn_cast<tt::AtomicCASOp>(op)) root(op, access.getPtr(), "atomic-pointer");
      else if (auto access = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        root(op, access.getSrc(), "global-pointer"); root(op, access.getMask(), "mask"); root(op, access.getResult(), "shared-destination");
      }
      else if (auto access = dyn_cast<hcu::BufferOpInterface>(op)) {
        root(op, access.getPtr(), "buffer-base"); root(op, access.getOffsets(), "buffer-offset-elements"); root(op, access.getStride(), "buffer-stride-bytes");
        if (auto a = dyn_cast<hcu::BufferLoadOp>(op)) root(op, a.getMask(), "mask");
        if (auto a = dyn_cast<hcu::BufferStoreOp>(op)) root(op, a.getMask(), "mask");
        if (auto a = dyn_cast<hcu::BufferAtomicRMWOp>(op)) root(op, a.getMask(), "mask");
        if (auto a = dyn_cast<hcu::BufferLoadToLocalOp>(op)) { root(op, a.getMask(), "mask"); root(op, a.getDest(), "shared-destination"); }
      } else if (auto access = dyn_cast<ttg::LocalLoadOp>(op)) root(op, access.getSrc(), "shared-descriptor");
      else if (auto access = dyn_cast<ttg::LocalStoreOp>(op)) root(op, access.getDst(), "shared-descriptor");
    }
  }
  std::vector<Edge> dependencies(Value v) {
    std::vector<Edge> result;
    auto add = [&](Value input, StringRef kind) { result.push_back({input, kind.str()}); };
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        unsigned i = arg.getArgNumber();
        if (i == 0) {
          add(loop.getLowerBound(), "induction-boundary"); add(loop.getUpperBound(), "induction-boundary"); add(loop.getStep(), "induction-boundary");
        } else {
          add(loop.getInitArgs()[i - 1], "loop-init-boundary");
          add(loop.getBody()->getTerminator()->getOperand(i - 1), "loop-recurrence-boundary");
        }
      }
      return result;
    }
    auto *op = v.getDefiningOp();
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      add(loop.getInitArgs()[i], "loop-init-boundary");
      add(loop.getBody()->getTerminator()->getOperand(i), "loop-exit-boundary");
      add(loop.getLowerBound(), "loop-control-boundary"); add(loop.getUpperBound(), "loop-control-boundary"); add(loop.getStep(), "loop-control-boundary");
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      add(branch.getCondition(), "operand");
      add(branch.thenBlock()->getTerminator()->getOperand(i), "region-yield");
      add(branch.elseBlock()->getTerminator()->getOperand(i), "region-yield");
    } else if (extended && isa<tt::LoadOp, hcu::BufferLoadOp>(op)) {
      // Device-loaded contents remain unknown. Record their address/mask
      // provenance as explicit memory boundaries, never as entry constants.
      if (auto load = dyn_cast<tt::LoadOp>(op)) {
        add(load.getPtr(), "memory-address-boundary");
        if (load.getMask()) add(load.getMask(), "memory-mask-boundary");
      } else if (auto load = dyn_cast<hcu::BufferLoadOp>(op)) {
        add(load.getPtr(), "memory-address-boundary");
        add(load.getOffsets(), "memory-address-boundary");
        if (load.getMask()) add(load.getMask(), "memory-mask-boundary");
      }
    } else if (expression(op)) {
      for (Value input : op->getOperands()) add(input, "operand");
    }
    return result;
  }
  void visit(Value v, unsigned rootId) {
    unsigned vi = ids.values.lookup(v);
    if (!membership[vi].insert(rootId).second) return;
    if (!edges.count(vi)) edges[vi] = dependencies(v);
    for (const Edge &edge : edges[vi]) visit(edge.value, rootId);
  }
  // All associations below use actual Values/Operations in this analysis.
  // Never join the model's memory-operation numbers to materialized IDs.
  llvm::json::Object slice(Value start) {
    std::set<unsigned> seen, operations, arguments, pids, sizes;
    std::map<std::string, unsigned> counts;
    unsigned opaque = 0, boundaries = 0, loadedValues = 0;
    std::function<void(Value)> walk = [&](Value v) {
      unsigned vi = ids.values.lookup(v);
      if (!seen.insert(vi).second) return;
      auto *op = v.getDefiningOp();
      if (!op) {
        auto arg = cast<BlockArgument>(v);
        if (functionEntry(arg.getOwner())) arguments.insert(vi);
        else if (!isa<scf::ForOp>(arg.getOwner()->getParentOp())) ++opaque;
      } else {
        if (operations.insert(ids.ops.lookup(op)).second) ++counts[category(op)];
        auto n = op->getName().getStringRef();
        if (n == "tt.get_program_id" || n == "tt.get_num_programs") {
          auto axis = op->getAttrOfType<IntegerAttr>("axis");
          if (!axis || axis.getInt() < 0 || axis.getInt() > 2) ++opaque;
          else (n == "tt.get_program_id" ? pids : sizes).insert(axis.getInt());
        }
        bool loaded = extended && isa<tt::LoadOp, hcu::BufferLoadOp>(op);
        loadedValues += loaded;
        if (!expression(op) && !isa<scf::ForOp, scf::IfOp>(op) && !loaded) ++opaque;
      }
      for (const auto &edge : dependencies(v)) {
        boundaries += StringRef(edge.kind).ends_with("boundary");
        walk(edge.value);
      }
    };
    walk(start);
    llvm::json::Array pidAxes, sizeAxes, values, ops;
    for (auto axis : pids) pidAxes.push_back(int64_t(axis));
    for (auto axis : sizes) sizeAxes.push_back(int64_t(axis));
    for (auto vi : seen) values.push_back(int64_t(vi));
    for (auto oi : operations) ops.push_back(int64_t(oi));
    llvm::json::Object categories;
    for (const auto &[name, count] : counts) categories[name] = int64_t(count);
    llvm::json::Object result{{"available", opaque == 0}, {"opaque_frontiers", int64_t(opaque)},
        {"boundary_edges", int64_t(boundaries)}, {"entry_argument_count", int64_t(arguments.size())},
        {"program_id_axes", std::move(pidAxes)}, {"num_programs_axes", std::move(sizeAxes)},
        {"operation_categories", std::move(categories)}, {"values", std::move(values)}, {"operations", std::move(ops)},
        {"interpretation", "syntactic SSA reachability, not a proof of address equality, cancellation, inter-CTA reuse or cache hits"}};
    if (extended) {
      result["memory_derived_values"] = int64_t(loadedValues);
      result["loaded_contents_known"] = false;
    }
    return result;
  }
  llvm::json::Object extendedDataSlice(Value start, bool forward) {
    std::set<unsigned> seen, operations, sharedBoundaries, regionBoundaries;
    std::map<std::string, unsigned> counts;
    llvm::json::Array flow;
    unsigned opaque = 0;
    std::function<void(Value)> walk;
    auto edge = [&](Value from, Value to, StringRef kind) {
      flow.push_back(llvm::json::Object{{"from", int64_t(ids.values.lookup(from))},
          {"to", int64_t(ids.values.lookup(to))}, {"kind", kind.str()}});
      walk(to);
    };
    auto count = [&](Operation *op) {
      if (operations.insert(ids.ops.lookup(op)).second)
        ++counts[op->getName().getStringRef().str()];
    };
    auto terminal = [&](Operation *op) {
      bool shared = isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::LocalAllocOp,
                        ttg::AsyncCopyGlobalToLocalOp>(op);
      if (shared) sharedBoundaries.insert(ids.ops.lookup(op));
      return shared || isa<tt::ReduceOp, tt::LoadOp, tt::StoreOp,
                           hcu::BufferLoadOp, hcu::BufferStoreOp>(op) ||
             op->getName().getStringRef().contains("dot");
    };
    walk = [&](Value value) {
      if (!seen.insert(ids.values.lookup(value)).second) return;
      if (!forward) {
        if (auto *op = value.getDefiningOp()) {
          count(op);
          if (terminal(op)) return;
          if (isa<scf::ForOp, scf::IfOp>(op)) {
            regionBoundaries.insert(ids.ops.lookup(op));
            for (const auto &dep : dependencies(value)) edge(value, dep.value, dep.kind);
          } else if (expression(op)) {
            for (Value input : op->getOperands()) edge(value, input, "producer");
          } else ++opaque;
        } else {
          auto arg = cast<BlockArgument>(value);
          if (functionEntry(arg.getOwner())) return;
          if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
            regionBoundaries.insert(ids.ops.lookup(loop));
            for (const auto &dep : dependencies(value)) edge(value, dep.value, dep.kind);
          } else ++opaque;
        }
        return;
      }
      for (OpOperand &use : value.getUses()) {
        Operation *op = use.getOwner();
        count(op);
        if (terminal(op)) continue;
        if (auto yield = dyn_cast<scf::YieldOp>(op)) {
          Operation *parent = op->getParentOp();
          unsigned index = use.getOperandNumber();
          regionBoundaries.insert(ids.ops.lookup(parent));
          if (isa<scf::IfOp, scf::ForOp>(parent) && index < parent->getNumResults()) {
            edge(value, parent->getResult(index), "region-result");
            if (auto loop = dyn_cast<scf::ForOp>(parent))
              edge(value, loop.getRegionIterArgs()[index], "loop-recurrence-boundary");
          } else ++opaque;
        } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
          regionBoundaries.insert(ids.ops.lookup(op));
          unsigned index = use.getOperandNumber();
          if (index >= 3 && index - 3 < loop.getNumResults()) {
            edge(value, loop.getRegionIterArgs()[index - 3], "loop-init-boundary");
            edge(value, loop.getResult(index - 3), "loop-exit-boundary");
          } else {
            // Bounds control all results; do not guess a trip count.
            for (Value result : op->getResults()) edge(value, result, "loop-control-boundary");
          }
        } else if (isa<scf::IfOp>(op)) {
          regionBoundaries.insert(ids.ops.lookup(op));
          for (Value result : op->getResults()) edge(value, result, "conditional-control");
        } else if (expression(op)) {
          for (Value result : op->getResults()) edge(value, result, "consumer");
        } else ++opaque;
      }
    };
    walk(start);
    llvm::json::Object byName;
    llvm::json::Array ops, values;
    for (const auto &[name, count] : counts) byName[name] = int64_t(count);
    for (auto oi : operations) ops.push_back(int64_t(oi));
    for (auto vi : seen) values.push_back(int64_t(vi));
    return llvm::json::Object{{"available", opaque == 0}, {"opaque_frontiers", int64_t(opaque)},
        {"direction", forward ? "consumers" : "producers"}, {"operation_counts", std::move(byName)},
        {"operations", std::move(ops)}, {"values", std::move(values)}, {"value_edges", std::move(flow)},
        {"shared_memory_boundaries", int64_t(sharedBoundaries.size())},
        {"region_boundaries", int64_t(regionBoundaries.size())}, {"branch_probabilities_known", false},
        {"interpretation", "SSA-port data reachability through if/yield/for; first global/shared memory/reduce/dot boundary; unknown CFG/calls remain opaque; no dynamic execution or memory-value inference"}};
  }
  llvm::json::Object dataSlice(Value start, bool forward) {
    if (extended) return extendedDataSlice(start, forward);
    std::set<unsigned> seen, operations;
    std::map<std::string, unsigned> counts;
    unsigned opaque = 0;
    std::function<void(Value)> walk = [&](Value v) {
      if (!seen.insert(ids.values.lookup(v)).second) return;
      auto step = [&](Operation *op) {
        if (!operations.insert(ids.ops.lookup(op)).second) return;
        auto n = op->getName().getStringRef();
        ++counts[n.str()];
        // Stop at the first semantic producer/consumer. Traversing a reduce
        // would mix its scalar combiner with the tensor data path.
        bool terminal = isa<tt::ReduceOp, tt::LoadOp, tt::StoreOp,
                            hcu::BufferLoadOp, hcu::BufferStoreOp>(op) || n.contains("dot");
        if (terminal) return;
        if (!expression(op)) { ++opaque; return; }
        if (forward) for (Value result : op->getResults()) walk(result);
        else for (Value input : op->getOperands()) walk(input);
      };
      if (forward) for (Operation *user : v.getUsers()) step(user);
      else if (auto *op = v.getDefiningOp()) step(op);
      else if (!functionEntry(cast<BlockArgument>(v).getOwner())) ++opaque;
    };
    walk(start);
    llvm::json::Object byName;
    llvm::json::Array ops;
    for (const auto &[name, count] : counts) byName[name] = int64_t(count);
    for (auto oi : operations) ops.push_back(int64_t(oi));
    return llvm::json::Object{{"available", opaque == 0}, {"opaque_frontiers", int64_t(opaque)},
        {"direction", forward ? "consumers" : "producers"}, {"operation_counts", std::move(byName)},
        {"operations", std::move(ops)}, {"interpretation", "unique reachable operations up to first memory/reduce/dot boundary; not dynamic use counts"}};
  }
  std::optional<MemoryAccessWork> linearRankOneWork(RankedTensorType type,
                                                   Value address, Value base,
                                                   llvm::json::Object &facts) {
    auto addressType = dyn_cast<RankedTensorType>(address.getType());
    if (type.getRank() != 1 || !addressType || addressType.getRank() != 1 ||
        !type.getElementType().isIntOrFloat() ||
        !isa<ttg::DistributedEncodingTrait>(type.getEncoding())) return {};
    auto layout = ttg::toLinearLayout(type);
    auto *ctx = module.getContext();
    auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
         warp = StringAttr::get(ctx, "warp"), block = StringAttr::get(ctx, "block");
    if (layout.getInDimSize(block) != 1) return {};
    layout = tt::actionRemoveBroadcastedRegs(layout).apply(layout);
    unsigned registers = layout.getInDimSize(reg), lanes = layout.getInDimSize(lane),
             warps = layout.getInDimSize(warp);
    if (registers > 64 || lanes > 64 || warps > 16) return {};
    auto *info = axes.getAxisInfo(address);
    if (!info || info->getContiguity().size() != 1 || info->getDivisibility().size() != 1) return {};
    unsigned bytes = std::max(1u, type.getElementTypeBitWidth()/8);
    if (bytes > 16 || (bytes & (bytes-1))) return {};
    int64_t contig = info->getContiguity()[0], align = info->getDivisibility()[0];
    if (isa<tt::PointerType>(addressType.getElementType()))
      align = std::max<int64_t>(1, align/bytes);
    if (base) {
      auto *bi = axes.getAxisInfo(base);
      align = std::min<int64_t>(align, bi ? std::max<int64_t>(1, bi->getDivisibility(0)/bytes) : 1);
    }
    if (contig < 1 || align < 1) return {};
    unsigned alignmentBytes = bytes;
    while (alignmentBytes < 32 && std::min(contig, align) % (2*alignmentBytes/bytes) == 0)
      alignmentBytes *= 2;
    facts = llvm::json::Object{{"contiguity_elements", contig},
        {"alignment_step_bytes", int64_t(alignmentBytes)}, {"element_bytes", int64_t(bytes)}};
    double work = 0;
    unsigned maxLanes = 0, maxSegments = 0;
    for (unsigned r = 0; r < registers; ++r) {
      double worstWarp = 0;
      for (unsigned w = 0; w < warps; ++w) {
        std::map<int64_t, std::set<int64_t>> groups;
        std::set<int64_t> coordinates;
        for (unsigned l = 0; l < lanes; ++l) {
          auto output = layout.apply({{reg, int32_t(r)}, {lane, int32_t(l)},
                                      {warp, int32_t(w)}, {block, 0}});
          int64_t coordinate = output[0].second;
          coordinates.insert(coordinate);
          groups[coordinate/contig].insert(coordinate%contig);
        }
        unsigned totalSegments = 0;
        for (const auto &group : groups) {
          unsigned worst = 0;
          for (unsigned offset = 0; offset < 32; offset += alignmentBytes) {
            std::set<uint64_t> segments;
            for (int64_t coordinate : group.second)
              for (unsigned byte = 0; byte < bytes; ++byte)
                segments.insert((offset+coordinate*bytes+byte)/32);
            worst = std::max<unsigned>(worst, segments.size());
          }
          totalSegments += worst;
        }
        // Distinct unknown-stride contiguous runs stay separate. All-active,
        // scalarized structural work only; no mask contents or vectorization
        // inferred. This is the same 32-byte modeling granularity as v3.
        worstWarp = std::max(worstWarp, 32.0*totalSegments/(coordinates.size()*bytes));
        maxLanes = std::max<unsigned>(maxLanes, coordinates.size());
        maxSegments = std::max(maxSegments, totalSegments);
      }
      work += worstWarp;
    }
    if (!std::isfinite(work) || work <= 0) return {};
    return MemoryAccessWork{work, double(registers), registers/work, 1,
                            maxLanes, maxSegments, 0};
  }
  llvm::json::Object accessRelations() {
    llvm::json::Array rows;
    bool allKnown = true;
    for (Operation *op : ids.orderedOps) {
      Value data, address, base, mask;
      bool load = false;
      if (auto a = dyn_cast<tt::LoadOp>(op)) { load = true; data = a.getResult(); address = a.getPtr(); mask = a.getMask(); }
      else if (auto a = dyn_cast<tt::StoreOp>(op)) { data = a.getValue(); address = a.getPtr(); mask = a.getMask(); }
      else if (auto a = dyn_cast<hcu::BufferLoadOp>(op)) { load = true; data = a.getResult(); address = a.getOffsets(); base = a.getPtr(); mask = a.getMask(); }
      else if (auto a = dyn_cast<hcu::BufferStoreOp>(op)) { data = a.getValue(); address = a.getOffsets(); base = a.getPtr(); mask = a.getMask(); }
      else {
        // Do not silently claim complete coverage of unsupported memory ops.
        if (isa<hcu::BufferOpInterface, tt::AtomicRMWOp, tt::AtomicCASOp,
                ttg::AsyncCopyGlobalToLocalOp>(op)) {
          allKnown = false;
          rows.push_back(llvm::json::Object{{"operation", int64_t(ids.ops.lookup(op))},
              {"available", false}, {"reason", "unsupported-global-access-kind"}});
        }
        continue;
      }
      llvm::json::Object row{{"operation", int64_t(ids.ops.lookup(op))}, {"load", load},
          {"data_value", int64_t(ids.values.lookup(data))}, {"address_value", int64_t(ids.values.lookup(address))},
          {"frequency", frequency(op, ids)}, {"address", slice(address)}, {"data_path", dataSlice(data, load)}};
      if (base) row["base"] = slice(base);
      if (mask) row["mask"] = slice(mask);
      row["masked"] = bool(mask);
      auto type = dyn_cast<RankedTensorType>(data.getType());
      Attribute enc = type ? type.getEncoding() : Attribute();
      while (auto s = dyn_cast_or_null<ttg::SliceEncodingAttr>(enc)) enc = s.getParent();
      auto b = dyn_cast_or_null<ttg::BlockedEncodingAttr>(enc);
      MemoryPattern p;
      if (type && b) {
        p.shape.assign(type.getShape().begin(), type.getShape().end());
        p.elementBytes = std::max(1u, type.getElementTypeBitWidth() / 8);
        p.sizePerThread.assign(b.getSizePerThread().begin(), b.getSizePerThread().end());
        p.threadsPerWarp.assign(b.getThreadsPerWarp().begin(), b.getThreadsPerWarp().end());
        p.warpsPerCTA.assign(b.getWarpsPerCTA().begin(), b.getWarpsPerCTA().end());
        p.order.assign(b.getOrder().begin(), b.getOrder().end());
        if (auto *info = axes.getAxisInfo(address)) {
          p.contiguity.assign(info->getContiguity().begin(), info->getContiguity().end());
          p.alignment.assign(info->getDivisibility().begin(), info->getDivisibility().end());
          auto at = dyn_cast<RankedTensorType>(address.getType());
          bool pointer = at && isa<tt::PointerType>(at.getElementType());
          int64_t baseAlignment = 1;
          if (base) if (auto *bi = axes.getAxisInfo(base)) baseAlignment = std::max<int64_t>(1, bi->getDivisibility(0) / p.elementBytes);
          for (auto &a : p.alignment) {
            if (pointer) a = std::max<int64_t>(1, a / p.elementBytes);
            if (base) a = std::min(a, baseAlignment);
          }
        }
        if (mask) {
          p.maskAlignment.assign(type.getRank(), 1);
          if (auto *info = axes.getAxisInfo(mask)) p.maskAlignment.assign(info->getConstancy().begin(), info->getConstancy().end());
        }
        for (Attribute a = type.getEncoding(); auto s = dyn_cast<ttg::SliceEncodingAttr>(a); a = s.getParent()) {
          auto dim = s.getDim();
          p.shape.insert(p.shape.begin() + dim, 1);
          if (!p.contiguity.empty()) p.contiguity.insert(p.contiguity.begin() + dim, 1);
          if (!p.alignment.empty()) p.alignment.insert(p.alignment.begin() + dim, 1);
          if (!p.maskAlignment.empty()) p.maskAlignment.insert(p.maskAlignment.begin() + dim, 1);
        }
      }
      auto work = type && b ? candidateMemoryWork(p) : std::nullopt;
      bool linearGeometry = extended && type && !b;
      llvm::json::Object linearFacts;
      if (linearGeometry) work = linearRankOneWork(type, address, base, linearFacts);
      bool scalarSite = extended && !type && data.getType().isIntOrFloat();
      auto array = [](const std::vector<int64_t> &v) { llvm::json::Array a; for (auto x : v) a.push_back(x); return a; };
      row["pattern"] = llvm::json::Object{{"shape", array(p.shape)}, {"size_per_thread", array(p.sizePerThread)},
          {"threads_per_warp", array(p.threadsPerWarp)}, {"warps_per_cta", array(p.warpsPerCTA)}, {"order", array(p.order)},
          {"contiguity", array(p.contiguity)}, {"alignment", array(p.alignment)}, {"mask_alignment", array(p.maskAlignment)},
          {"element_bytes", int64_t(p.elementBytes)}};
      if (work) {
        row["geometry"] = llvm::json::Object{{"work", work->work}, {"instructions", work->instructions},
            {"vector_elements", int64_t(work->vectorElements)}, {"unique_lanes", int64_t(work->uniqueLanes)},
            {"lane_stride_bytes", int64_t(work->laneStrideBytes)}, {"segments", int64_t(work->segments)},
            {"segment_utilization", work->segmentUtilization}};
        row["data_mapping"] = mapping(type);
        if (linearGeometry) {
          auto *geometry = row.getObject("geometry");
          geometry->erase("lane_stride_bytes");
          (*geometry)["method"] = "scalarized-linearlayout-rank1-v1";
          (*geometry)["segment_granularity_bytes"] = 32;
          (*geometry)["hardware_transactions_known"] = false;
          (*geometry)["mask_contents_known"] = false;
          (*geometry)["unknown_stride_runs_coalesced"] = false;
          (*geometry)["inputs"] = std::move(linearFacts);
        }
      }
      if (extended) {
        row["data_type"] = printed(data.getType());
        row["scalar_access_site"] = scalarSite;
      }
      if (scalarSite) {
        // Logical scalar access sites, not estimated physical transactions or
        // active lanes. Keep their units explicit and separately projectable.
        row["geometry"] = llvm::json::Object{{"work", 1.0}, {"instructions", 1.0},
            {"units", "logical-scalar-access-sites"},
            {"element_bytes", int64_t(std::max(1u, data.getType().getIntOrFloatBitWidth() / 8))},
            {"physical_transactions_known", false}, {"active_lanes_known", false}};
      }
      bool known = bool(work) || scalarSite;
      for (const char *key : {"address", "base", "mask", "data_path"})
        if (const auto *s = row.getObject(key)) known &= s->getBoolean("available").value_or(false);
      row["available"] = known;
      allKnown &= known;
      rows.push_back(std::move(row));
    }
    return llvm::json::Object{{"schema", 1}, {"version", extended ? "native-access-relations-v2" : "native-access-relations-v1"},
        {"id_scope", "materialized-candidate"}, {"available", allKnown}, {"used_by_scorer", false}, {"accesses", std::move(rows)},
        {"association", "native Operation/Value identity within this materialized module; numeric IDs are audit references only"},
        {"limitations", "syntactic address and data reachability only; static multiplicity, unknown dynamic counts; no reuse/cache/latency proof; unknown facts reject feature extraction"}};
  }
  llvm::json::Object report(const llvm::json::Array &conversions) {
    findRoots();
    for (unsigned i = 0; i < roots.size(); ++i) visit(roots[i].value, i);
    llvm::json::Array rootRows, nodes, operations, implicit;
    for (unsigned i = 0; i < roots.size(); ++i) rootRows.push_back(llvm::json::Object{{"id", int64_t(i)}, {"operation", int64_t(ids.ops.lookup(roots[i].op))},
        {"operation_name", roots[i].op->getName().getStringRef().str()}, {"value", int64_t(ids.values.lookup(roots[i].value))}, {"role", roots[i].role}});
    std::map<unsigned, std::set<unsigned>> operationRoots;
    std::map<unsigned, uint64_t> operationElements;
    unsigned frontiers = 0, boundaryEdges = 0;
    for (const auto &[vi, members] : membership) {
      Value v = ids.orderedValues[vi];
      Operation *op = v.getDefiningOp();
      Block *block = op ? op->getBlock() : cast<BlockArgument>(v).getOwner();
      Type element = v.getType();
      uint64_t elements = 1;
      if (auto tensor = dyn_cast<RankedTensorType>(element)) {
        element = tensor.getElementType();
        if (tensor.getEncoding() && isa<ttg::DistributedEncodingTrait>(tensor.getEncoding())) {
          auto ll = ttg::toLinearLayout(tensor);
          elements = tt::actionRemoveBroadcastedRegs(ll).apply(ll).getInDimSize(StringAttr::get(module.getContext(), "register"));
        } else elements = 0;
      }
      unsigned bits = 0;
      if (isa<tt::PointerType, IndexType>(element)) bits = 64;
      else if (auto i = dyn_cast<IntegerType>(element)) bits = i.getWidth();
      else if (auto f = dyn_cast<FloatType>(element)) bits = f.getWidth();
      bool frontier = op ? !expression(op) && !isa<scf::ForOp, scf::IfOp>(op) &&
          !(extended && isa<tt::LoadOp, hcu::BufferLoadOp>(op)) :
          !functionEntry(block) && !isa<scf::ForOp>(block->getParentOp());
      frontiers += frontier;
      llvm::json::Array deps, boundaries, rootIds;
      for (const auto &edge : edges[vi]) {
        llvm::json::Object e{{"value", int64_t(ids.values.lookup(edge.value))}, {"kind", edge.kind}};
        if (StringRef(edge.kind).ends_with("boundary")) { boundaries.push_back(std::move(e)); ++boundaryEdges; }
        else deps.push_back(std::move(e));
      }
      for (auto i : members) rootIds.push_back(int64_t(i));
      llvm::json::Object node{{"value", int64_t(vi)}, {"type", printed(v.getType())}, {"block", int64_t(ids.blocks.lookup(block))},
          {"bitwidth", int64_t(bits)}, {"bitwidth_known", bits != 0}, {"cta_uniform_proven", uniform.lookup(v)},
          {"unique_elements_per_thread", int64_t(elements)}, {"work_elements_proxy", int64_t(uniform.lookup(v) ? 1 : elements)},
          {"dependencies", std::move(deps)}, {"boundary_dependencies", std::move(boundaries)}, {"roots", std::move(rootIds)},
          {"opaque_frontier", frontier}};
      node["operation"] = op ? llvm::json::Value(int64_t(ids.ops.lookup(op))) : llvm::json::Value(nullptr);
      node["operation_name"] = op ? op->getName().getStringRef().str() : "block-argument";
      if (auto c = integer(v)) node["integer_splat_or_scalar"] = *c;
      nodes.push_back(std::move(node));
      if (!op || !expression(op)) continue;
      unsigned oi = ids.ops.lookup(op);
      operationRoots[oi].insert(members.begin(), members.end());
      operationElements[oi] = std::max<uint64_t>(operationElements[oi], uniform.lookup(v) ? 1 : elements);
      if (isa<tt::MakeRangeOp>(op))
        implicit.push_back(llvm::json::Object{{"operation", int64_t(oi)}, {"value", int64_t(vi)}, {"kind", "make-range-coordinate"},
            {"owner", "index-expression"}, {"mapping", mapping(cast<RankedTensorType>(v.getType()))}});
    }
    std::map<std::string, double> work;
    unsigned shared = 0;
    for (const auto &[oi, members] : operationRoots) {
      auto *op = ids.orderedOps[oi];
      auto kind = category(op);
      auto freq = frequency(op, ids);
      double proxy = operationElements[oi] * *freq.getNumber("static_execution_weight");
      work[kind] += proxy;
      llvm::json::Array rootIds;
      for (auto i : members) rootIds.push_back(int64_t(i));
      shared += members.size() > 1;
      std::string owner = kind == "conversion-owned" ? "conversion" : "index-expression";
      operations.push_back(llvm::json::Object{{"operation", int64_t(oi)}, {"operation_name", op->getName().getStringRef().str()},
          {"category", kind}, {"owner", owner}, {"roots", std::move(rootIds)}, {"frequency", std::move(freq)},
          {"work_elements_proxy", int64_t(operationElements[oi])}, {"weighted_work_proxy", proxy}});
    }
    // Conversions use the *actual conversion plan*, never pretend their data
    // operands are address expressions. Logical matrices are not the swizzled
    // scratch matrix; route-specific vector/repetition features remain here.
    for (const auto &item : conversions) {
      const auto &c = *item.getAsObject();
      unsigned oi = *c.getInteger("operation");
      auto cvt = cast<ttg::ConvertLayoutOp>(ids.orderedOps[oi]);
      llvm::json::Object row{{"operation", int64_t(oi)}, {"kind", "conversion-address-or-shuffle"}, {"owner", "conversion"},
          {"route", c.getString("route")->str()}, {"source_mapping", mapping(cvt.getSrc().getType())}, {"destination_mapping", mapping(cvt.getType())},
          {"exact_scratch_address_dag", false}, {"frequency", frequency(cvt, ids)}};
      for (auto key : {"strategy", "strategy_uncertainty", "swizzled_repetitions", "padded_store_vector", "padded_load_vector", "padded_vector_ops_proxy", "mixed_transpositions", "lane_permutation", "packing_log2"})
        if (auto *v = c.get(key)) row[key] = *v;
      if (c.getString("route") == "lds") row["scratch_address_scenarios"] = scratchAddresses(cvt);
      implicit.push_back(std::move(row));
    }
    llvm::json::Object summary;
    for (auto &[key, value] : work) summary[key] = value;
    return llvm::json::Object{{"schema", 1}, {"id_scope", "materialized-candidate"}, {"scoring_weight", 0}, {"calibrated", false},
        {"roots", std::move(rootRows)}, {"nodes", std::move(nodes)}, {"operations", std::move(operations)}, {"implicit_lowering", std::move(implicit)},
        {"access_relations", accessRelations()},
        {"weighted_work_by_category", std::move(summary)}, {"shared_operations", int64_t(shared)}, {"opaque_frontiers", int64_t(frontiers)}, {"boundary_edges", int64_t(boundaryEdges)},
        {"cost_ownership", llvm::json::Object{{"memory_prices", "whole microbenchmark; index/predicate contribution not isolated; do not add index cost blindly"},
            {"conversion_prices", "whole conversion microbenchmark including address/sync; structural route also owns address proxies"},
            {"existing_arithmetic", "can already charge index arith ops; must partition before calibration"}, {"new_index_features", "diagnostic only; all weights zero"}}},
        {"limitations", "SSA sharing only, no speculative expression CSE/hoisting; ordinary dependencies form a DAG, loop/induction boundaries separate; while/CFG/call/unknown/data producers are explicit opaque frontiers; uniform proofs conservative; work proxies not instruction counts/latency; dynamic trip counts and branch probabilities unknown, static branch work summed; conversion scratch templates are alternative current lowering paths before folding/CSE, later OptimizeLDSUsage can change/split strategy; exact non-conversion shared-memory addresses not reconstructed"}};
  }
};
} // namespace
llvm::json::Object analyzeLayoutIndex(ModuleOp module, const llvm::json::Array &conversions,
                                     bool extendedAccess) {
  auto legacy = Analysis(module).report(conversions);
  if (extendedAccess)
    legacy["access_graph_v2"] = Analysis(module, true).report(conversions);
  return legacy;
}
} // namespace layoutdp
