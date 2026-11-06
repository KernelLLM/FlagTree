#include "TritonHCUGPUTransforms/LayoutTuningProjection.h"
#include "TritonHCUGPUTransforms/LayoutTuningSSA.h"
#include "TritonHCUGPUTransforms/LayoutTuningCollective.h"
#include "TritonHCUGPUTransforms/LayoutTuningFacts.h"
#include "TritonHCUGPUTransforms/LayoutTuningResources.h"
#include "TritonHCUGPUTransforms/LayoutTuningTransport.h"
#include "TritonHCUGPUTransforms/LayoutExecutionContext.h"
#include "TritonHCUGPUTransforms/Passes.h"
#include "TritonHCUGPUToLLVM/Passes.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>

#ifdef __FLAGTREE_RLC_ENHANCE__
namespace mlir::triton::gpu {
std::unique_ptr<mlir::Pass>
createTritonGPURemoveLayoutConversionsEnhanced(bool enhance);
}
#endif

namespace layoutdp {
namespace {
// The normal TTGIR inliner leaves some helpers until LLVM alwaysinline.
// Expand direct, nonrecursive,
// noinline=false bodies only on the analysis clone, preserving CFG/effects.
bool expandAnalysisCalls(mlir::ModuleOp module) {
  using namespace mlir;
  namespace tt = mlir::triton;
  llvm::DenseSet<Operation *> visiting, visited;
  std::function<bool(tt::FuncOp)> acyclic = [&](tt::FuncOp f) {
    if (visited.contains(f)) return true;
    if (!visiting.insert(f).second) return false;
    bool valid = true;
    f.walk([&](tt::CallOp call) {
      auto callee = dyn_cast_or_null<tt::FuncOp>(cast<CallOpInterface>(call.getOperation()).resolveCallable());
      if (!callee || callee.isExternal()) return;
      if (auto attr = callee->getAttrOfType<BoolAttr>("noinline"); attr && attr.getValue()) return;
      valid &= acyclic(callee);
    });
    visiting.erase(f); visited.insert(f); return valid;
  };
  for (auto f : module.getOps<tt::FuncOp>()) if (!f.isPrivate() && !acyclic(f)) return false;
  bool progress = true;
  while (progress) {
    progress = false;
    SmallVector<tt::CallOp> calls;
    module.walk([&](tt::CallOp call) {
      if (!call->getParentOfType<tt::FuncOp>().isPrivate()) calls.push_back(call);
    });
    for (auto call : calls) {
      auto f = dyn_cast_or_null<tt::FuncOp>(cast<CallOpInterface>(call.getOperation()).resolveCallable());
      if (!f || f.isExternal()) continue;
      if (auto attr = f->getAttrOfType<BoolAttr>("noinline"); attr && attr.getValue()) continue;
      uint64_t count = 0; module.walk([&](Operation *) { ++count; });
      if (count > 200000) return false;
      auto &body = f.getBody();
      if (body.empty() || body.front().getNumArguments() != call.getNumOperands()) return false;
      IRMapping mapping;
      for (auto [arg, operand] : llvm::zip(body.front().getArguments(), call.getOperands())) mapping.map(arg, operand);
      OpBuilder builder(call);
      if (body.hasOneBlock()) {
        auto ret = dyn_cast<tt::ReturnOp>(body.front().getTerminator());
        if (!ret) return false;
        for (auto &op : body.front().without_terminator()) builder.clone(op, mapping);
        SmallVector<Value> results;
        for (Value v : ret.getOperands()) results.push_back(mapping.lookupOrDefault(v));
        call.replaceAllUsesWith(results); call.erase();
      } else {
        auto exec = scf::ExecuteRegionOp::create(builder, call.getLoc(), call.getResultTypes());
        auto &region = exec.getRegion();
        for (Block &old : body) {
          auto *fresh = new Block(); region.push_back(fresh); mapping.map(&old, fresh);
          if (&old == &body.front()) continue;
          for (BlockArgument arg : old.getArguments()) mapping.map(arg, fresh->addArgument(arg.getType(), arg.getLoc()));
        }
        for (Block &old : body) {
          builder.setInsertionPointToEnd(mapping.lookup(&old));
          for (Operation &op : old) {
            if (auto ret = dyn_cast<tt::ReturnOp>(op)) {
              SmallVector<Value> values;
              for (Value v : ret.getOperands()) values.push_back(mapping.lookupOrDefault(v));
              scf::YieldOp::create(builder, ret.getLoc(), values);
            } else builder.clone(op, mapping);
          }
        }
        call.replaceAllUsesWith(exec.getResults()); call.erase();
      }
      progress = true;
    }
  }
  return succeeded(verify(module));
}

// Match the ordinary local shared lowering's layout composition, broadcast
// removal and vector grouping. These are physical transfer primitives, NOT
// emitted ISA counts: target transpose patterns and LLVM may change lowering.
llvm::json::Object sharedTransport(mlir::ModuleOp module,
                                  llvm::StringRef executionContext,
                                  bool descriptorControlFlow = false) {
  using namespace mlir;
  namespace tt = mlir::triton;
  namespace ttg = mlir::triton::gpu;
  LayoutExecutionContext context(module, executionContext);
  llvm::json::Array rows, issues;
  std::map<std::string, double> totals;
  for (const char *side : {"read", "write"})
    for (const char *unit : {"vector_groups", "word_work"})
      totals[std::string("transport_shared_") + side + "_" + unit] = 0;
  auto printed = [](Type type) {
    std::string text; llvm::raw_string_ostream os(text); os << type; return text;
  };
  if (!context.valid()) issues.push_back(context.failure());
  unsigned ordinal = 0;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    unsigned id = ordinal++;
    Value value; ttg::MemDescType desc; bool read = false;
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      value = load.getResult(); desc = load.getSrc().getType(); read = true;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
      value = store.getSrc(); desc = store.getDst().getType();
    } else if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (!alloc.getSrc()) return;
      value = alloc.getSrc(); desc = alloc.getType();
    } else {
      // SCF carries descriptor references through block arguments/results;
      // the container/terminator itself does not transfer tensor payload.
      // Walk still visits every local access inside each region. Keep this
      // opt-in for the current-module audit so the v204 API stays identical.
      if (descriptorControlFlow && isa<scf::ForOp, scf::WhileOp, scf::IfOp,
          scf::ExecuteRegionOp, scf::YieldOp, scf::ConditionOp>(op)) return;
      // Unknown operations with shared descriptors must not silently report
      // zero. Descriptor views, deallocation and barriers do not move payload.
      auto name = op->getName().getStringRef();
      if (name.starts_with("ttg.memdesc_") || name == "ttg.local_dealloc" ||
          name == "ttg.local_barrier" || name == "ttg.local_alloc") return;
      bool shared = false;
      for (Type type : op->getOperandTypes())
        if (auto d = dyn_cast<ttg::MemDescType>(type))
          shared |= isa<ttg::SharedMemorySpaceAttr>(d.getMemorySpace());
      for (Type type : op->getResultTypes())
        if (auto d = dyn_cast<ttg::MemDescType>(type))
          shared |= isa<ttg::SharedMemorySpaceAttr>(d.getMemorySpace());
      if (shared) issues.push_back(name.str() + ": shared payload path not modeled");
      return;
    }
    double weight = layoutExecutionWeight(op, &context);
    llvm::json::Object row{{"operation", int64_t(id)},
        {"operation_name", op->getName().getStringRef().str()},
        {"read", read}, {"available", false},
        {"data_type", printed(value.getType())}, {"descriptor_type", printed(desc)},
        {"execution_weight", weight}, {"exact_instruction_count", false}};
    auto fail = [&](const char *why) {
      row["reason"] = why;
      issues.push_back(op->getName().getStringRef().str() + ": " + why);
      rows.push_back(std::move(row));
    };
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !isa<ttg::DistributedEncodingTrait>(type.getEncoding()) ||
        !isa<ttg::SharedMemorySpaceAttr>(desc.getMemorySpace())) {
      fail("requires distributed tensor and local shared descriptor"); return;
    }
    auto element = desc.getElementType();
    if (element != type.getElementType() || !element.isIntOrFloat()) {
      fail("unsupported payload conversion or non-numeric storage"); return;
    }
    unsigned bits = element.getIntOrFloatBitWidth();
    if (bits != 8 && bits != 16 && bits != 32 && bits != 64) {
      fail("storage width outside explicit 8/16/32/64 bit contract"); return;
    }
    auto enc = desc.getEncoding();
    auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(enc);
    if (!padded && !isa<ttg::SwizzledSharedEncodingAttr,
        ttg::SharedLinearEncodingAttr, ttg::AMDRotatingSharedEncodingAttr>(enc)) {
      fail("shared encoding not supported by bounded local view"); return;
    }
    auto regLayout = ttg::toLinearLayout(type);
    auto sharedLayout = padded ? padded.getLinearComponent() : ttg::toLinearLayout(desc);
    auto cvt = regLayout.invertAndCompose(sharedLayout);
    auto *ctx = module.getContext();
    auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane"),
         warp = StringAttr::get(ctx, "warp"), offset = StringAttr::get(ctx, "offset"),
         block = StringAttr::get(ctx, "block");
    if (!cvt.isTrivialOver({block})) {
      fail("cross-CTA shared transport not supported"); return;
    }
    cvt = cvt.sublayout({reg, lane, warp}, {offset});
    unsigned originalElements = cvt.getInDimSize(reg);
    cvt = tt::actionRemoveBroadcastedRegs(cvt).apply(cvt);
    std::optional<int> cap;
    if (padded) cap = padded.getMinInterval();
    auto [vector, permutation] = tt::largestVectorisation(ctx, cvt, bits, cap);
    unsigned elements = cvt.getInDimSize(reg);
    if (!vector || elements % vector || !std::isfinite(weight) || weight < 0) {
      fail("invalid native vector grouping or execution multiplicity"); return;
    }
    llvm::json::Array inputs;
    for (const auto &[name, bases] : cvt.getBases()) {
      llvm::json::Array vectors;
      for (const auto &basis : bases) {
        llvm::json::Array coordinates;
        for (int32_t x : basis) coordinates.push_back(int64_t(x));
        vectors.push_back(std::move(coordinates));
      }
      inputs.push_back(llvm::json::Object{{"dimension", name.getValue().str()},
          {"size", int64_t(cvt.getInDimSize(name))}, {"bases", std::move(vectors)}});
    }
    double groups = double(elements) / vector, words = double(elements) * bits / 32;
    row["available"] = true; row["storage_bits"] = int64_t(bits);
    row["original_register_elements"] = int64_t(originalElements);
    row["unique_register_elements"] = int64_t(elements);
    row["vector_elements"] = int64_t(vector);
    row["vector_bits"] = int64_t(vector * bits);
    row["vector_groups"] = groups; row["word_work"] = words;
    row["relative_mapping"] = std::move(inputs);
    std::string prefix = read ? "transport_shared_read_" : "transport_shared_write_";
    totals[prefix + "vector_groups"] += weight * groups;
    totals[prefix + "word_work"] += weight * words;
    rows.push_back(std::move(row));
  });
  llvm::json::Object values;
  // Partial sums are diagnostic only. Missing facts are never zero imputed.
  if (issues.empty()) for (const auto &[name, value] : totals) values[name] = value;
  return llvm::json::Object{{"schema", "bounded-shared-transport-v204"},
      {"available", issues.empty()}, {"issues", std::move(issues)},
      {"values", std::move(values)}, {"operations", std::move(rows)},
      {"exact_instruction_count", false}, {"analysis_clone_only", true},
      {"frequency_policy", "existing context/static loop multiplicities; unknown loops one, branch probabilities unknown"},
      {"pipeline", "shared-prefix+cleanup+opt-dot+RLC+reduce-data-duplication+optimize-LDS"}};
}
} // namespace
llvm::json::Object analyzeCurrentLayoutCollectiveFacts(
    mlir::ModuleOp candidate, llvm::StringRef executionContext) {
  mlir::OwningOpRef<mlir::ModuleOp> clone =
      mlir::cast<mlir::ModuleOp>(candidate->clone());
  if (mlir::failed(mlir::verify(*clone)) || !expandAnalysisCalls(*clone))
    return llvm::json::Object{{"schema", "hcu-collective-shared-plan-v217"},
        {"available", false}, {"issues", llvm::json::Array{"invalid module or analysis call expansion"}},
        {"values", llvm::json::Object{}}, {"analysis_clone_only", true}};
  mlir::PassManager cleanup(candidate.getContext());
  cleanup.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(cleanup.run(*clone)))
    return llvm::json::Object{{"available", false}, {"issues", llvm::json::Array{"symbol cleanup failed"}},
        {"values", llvm::json::Object{}}, {"analysis_clone_only", true}};
  return analyzeLayoutCollectiveSharedPlan(*clone, executionContext);
}

llvm::json::Object analyzeCurrentLayoutSharedTransportFacts(
    mlir::ModuleOp candidate, llvm::StringRef executionContext) {
  mlir::OwningOpRef<mlir::ModuleOp> clone =
      mlir::cast<mlir::ModuleOp>(candidate->clone());
  auto fail = [](const char *why) {
    return llvm::json::Object{{"schema", "native-current-shared-transport-v207"},
        {"available", false}, {"issues", llvm::json::Array{why}},
        {"values", llvm::json::Object{}}, {"analysis_clone_only", true}};
  };
  if (mlir::failed(mlir::verify(*clone)) || !expandAnalysisCalls(*clone))
    return fail("current-module verification or bounded call expansion failed");
  mlir::PassManager cleanup(candidate.getContext());
  cleanup.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(cleanup.run(*clone)))
    return fail("current-module symbol cleanup failed");
  auto result = sharedTransport(*clone, executionContext, true);
  result["schema"] = "native-current-shared-transport-v207";
  result["pipeline"] = "current-module-clone+bounded-call-expansion+symbol-dce; no early projection";
  return result;
}

llvm::json::Object analyzeCurrentLayoutTransportClosureFacts(
    mlir::ModuleOp candidate, llvm::StringRef executionContext) {
  mlir::OwningOpRef<mlir::ModuleOp> clone = mlir::cast<mlir::ModuleOp>(candidate->clone());
  if (mlir::failed(mlir::verify(*clone)) || !expandAnalysisCalls(*clone))
    return llvm::json::Object{{"schema", "bounded-shared-transport-closure"},
        {"available", false}, {"issues", llvm::json::Array{"invalid module or analysis call expansion"}},
        {"values", llvm::json::Object{}}, {"analysis_clone_only", true}};
  mlir::PassManager cleanup(candidate.getContext());
  cleanup.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(cleanup.run(*clone)))
    return llvm::json::Object{{"schema", "bounded-shared-transport-closure"},
        {"available", false}, {"issues", llvm::json::Array{"current-module cleanup failed"}},
        {"values", llvm::json::Object{}}, {"analysis_clone_only", true}};
  auto result = addLayoutConversionTransport(*clone, sharedTransport(*clone, executionContext, true), executionContext);
  result["pipeline"] = "current-module-clone+bounded-call-expansion+symbol-dce; no early projection";
  return result;
}
void addLayoutThreadLocalityAndTLE(mlir::OpPassManager &pm) {
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm.addPass(mlir::triton::tle::createTritonTleSelectEncodings());
  pm.addPass(mlir::triton::tle::createTritonTleInsertLocalPointerBarriers());
  pm.addPass(mlir::triton::tle::createTritonTleOptimizeLocalPointerLoads());
  pm.addPass(mlir::triton::tle::createTritonTleOptimizeLocalPointerStores());
}

void addLayoutMatmulPrefix(mlir::OpPassManager &pm,
                          const LayoutProjectionConfig &config) {
  mlir::TritonHCUGPUAccelerateMatmulOptions options;
  options.archGenerationName = config.arch;
  options.matrixInstructionSize = config.matrixInstructionSize;
  options.kPack = config.kPack;
  options.mmacLayoutForce = config.mmacLayoutForce;
  pm.addPass(mlir::createTritonHCUGPUAccelerateMatmul(options));
#ifdef __FLAGTREE_RLC_ENHANCE__
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversionsEnhanced(false));
#else
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
#endif
}

void addLayoutEvaluationPrefix(mlir::OpPassManager &pm,
                               const LayoutProjectionConfig &config) {
  addLayoutThreadLocalityAndTLE(pm);
  addLayoutMatmulPrefix(pm, config);
}

static llvm::json::Object projectLayoutFactsImpl(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config, bool diagnostics, bool includeSSA,
    bool includeTransport = false, bool includeCollective = false,
    bool includeResourceContract = false, bool includeTransportClosure = false) {
  llvm::json::Object configuration{{"arch", config.arch},
      {"matrix_instr_nonkdim", config.matrixInstructionSize},
      {"kpack", config.kPack}, {"mmac_layout_force", config.mmacLayoutForce},
      {"num_stages", int64_t(config.numStages)}};
  configuration["execution_context"] = config.executionContext;
  std::string binding;
  llvm::raw_string_ostream out(binding);
  out << layoutProjectionVersion << '\n' << layoutProjectionAnalysisStage << '\n'
      << llvm::json::Value(llvm::json::Object(configuration));
  out.flush();
  llvm::SHA256 sha;
  sha.update(binding);
  auto fingerprint = llvm::toHex(sha.final(), true);
  auto target = candidate->getAttrOfType<mlir::StringAttr>("ttg.target");
  auto bind = [&](llvm::json::Object facts) {
    facts["schema"] = layoutProjectionVersion;
    facts["feature_version"] = layoutProjectedFactsVersion;
    facts["pipeline_stage"] = layoutTuningStage;
    facts["analysis_stage"] = layoutProjectionAnalysisStage;
    facts["producer_fingerprint"] = fingerprint;
    facts["producer_configuration"] = llvm::json::Object(configuration);
    facts["backend_feasibility"] = "unknown";
    facts["numerical_validity"] = "unknown";
    facts["analysis_clone_only"] = true;
    return facts;
  };
  auto fail = [&](const char *why) {
    return bind(llvm::json::Object{{"complete", false},
        {"target", target ? target.getValue().str() : ""},
        {"issues", llvm::json::Array{why}}, {"values", llvm::json::Object{}},
        {"feature_names", llvm::json::Array{}}, {"projection_succeeded", false},
        {"seed_reference_input", false}, {"prices_used", false},
        {"timing_inputs_used", false}});
  };
  if (config.arch.empty() || config.kPack < 1 || config.matrixInstructionSize < 0 ||
      !target || target.getValue() != "hip:" + config.arch)
    return fail("missing or inconsistent explicit projection configuration/target");
  mlir::OwningOpRef<mlir::ModuleOp> clone =
      mlir::cast<mlir::ModuleOp>(candidate->clone());
  if (mlir::failed(mlir::verify(*clone)))
    return fail("current candidate failed verification");
  mlir::PassManager pm(candidate.getContext());
  addLayoutEvaluationPrefix(pm, config);
  if (mlir::failed(pm.run(*clone))) return fail("analysis prefix failed; no fallback");
  if (!expandAnalysisCalls(*clone)) return fail("analysis call expansion rejected recursion, budget or invalid CFG");
  mlir::PassManager cleanup(candidate.getContext());
  cleanup.addPass(mlir::createSymbolDCEPass());
  cleanup.addPass(mlir::createCSEPass());
  cleanup.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
  cleanup.addPass(mlir::createCSEPass());
  if (mlir::failed(cleanup.run(*clone)))
    return fail("analysis prefix/cleanup failed; no fallback");
  auto facts = includeResourceContract ? analyzeCurrentLayoutResourceContractFacts(*clone, config.numStages, config.executionContext, includeSSA)
                         : includeSSA ? analyzeCurrentLayoutSSAFacts(*clone, config.numStages, config.executionContext)
                         : analyzeCurrentLayoutFacts(*clone, config.numStages, diagnostics, config.executionContext);
  // Dot-operand optimization may turn an apparently harmless early layout
  // change into narrow shared-memory transfers. Keep the original facts and
  // add a second, explicitly bounded current-candidate view of this native
  // transform. The selection pass remains at its original early slot.
  auto lookahead = mlir::cast<mlir::ModuleOp>(clone->clone());
  mlir::OwningOpRef<mlir::ModuleOp> lookaheadOwner = lookahead;
  mlir::PassManager dotPrefix(candidate.getContext());
  mlir::triton::hcugpu::TritonHCUGPUOptimizeDotOperandsOptions dotOptions;
  dotOptions.archGenerationName = config.arch;
  dotPrefix.addPass(mlir::triton::hcugpu::createTritonHCUGPUOptimizeDotOperands(dotOptions));
#ifdef __FLAGTREE_RLC_ENHANCE__
  dotPrefix.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversionsEnhanced(false));
#else
  dotPrefix.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
#endif
  if (mlir::failed(dotPrefix.run(lookahead)))
    return fail("dot-operand analysis lookahead failed; no fallback");
  // Both diagnostic descendants start with exactly this native prefix.
  // Fork the already transformed module before their distinct LDS tails;
  // do not rerun dot-operand optimization and RLC on a second base clone.
  mlir::OwningOpRef<mlir::ModuleOp> transport;
  if (includeTransport)
    transport = mlir::cast<mlir::ModuleOp>(lookahead->clone());
  mlir::PassManager lookaheadTail(candidate.getContext());
  lookaheadTail.addPass(mlir::triton::HCU::createOptimizeLDSUsagePass(config.arch, 0));
  if (mlir::failed(lookaheadTail.run(lookahead)))
    return fail("dot-operand LDS analysis tail failed; no fallback");
  auto later = includeResourceContract ? analyzeCurrentLayoutResourceContractFacts(lookahead, config.numStages, config.executionContext, false)
      : analyzeCurrentLayoutFacts(lookahead, config.numStages, true, config.executionContext);
  if (includeResourceContract) {
    if (!facts.getObject("resource_contract") || !later.getObject("resource_contract"))
      return fail("missing normalized diagnostic resource facts; no fallback");
    auto pre = std::move(*facts.getObject("resource_contract"));
    auto post = std::move(*later.getObject("resource_contract"));
    later.erase("resource_contract");
    facts["resource_contract"] = llvm::json::Object{
        {"pre_dot", std::move(pre)}, {"post_dot", std::move(post)}};
  }
  auto *values = facts.getObject("values");
  const auto *lateValues = later.getObject("values");
  if (!values || !lateValues) return fail("missing dot-operand lookahead facts");
  for (const char *key : {"lds_vector_ops", "lds_allocated_bytes", "packed_live_words",
       "conversion_temporary_words", "lds_load_word_sites", "lds_store_word_sites",
       "lds_block_sync_sites", "lds_warp_sync_sites", "shuffle_word_ops"}) {
    auto value = lateValues->getNumber(key);
    if (!value) return fail("missing dot-operand lookahead work");
    (*values)[std::string("post_dot_") + key] = *value;
  }
  if (auto *names = facts.getArray("feature_names")) {
    names->clear();
    std::vector<std::string> sorted;
    for (const auto &item : *values) sorted.push_back(item.first.str());
    std::sort(sorted.begin(), sorted.end());
    for (const auto &name : sorted) names->push_back(name);
  }
  if (diagnostics) facts["dot_operand_lookahead"] = std::move(later);
  if (includeTransport) {
    mlir::PassManager transportPrefix(candidate.getContext());
    transportPrefix.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
    transportPrefix.addPass(mlir::triton::HCU::createOptimizeLDSUsagePass(config.arch, 0));
    if (mlir::failed(transportPrefix.run(*transport)))
      return fail("bounded shared transport prefix failed; no fallback");
    auto transfers = sharedTransport(*transport, config.executionContext, includeTransportClosure);
    facts["shared_transport"] = includeTransportClosure ?
        addLayoutConversionTransport(*transport, std::move(transfers), config.executionContext) : std::move(transfers);
    if (includeCollective)
      facts["collective_shared_plan"] = analyzeLayoutCollectiveSharedPlan(*transport, config.executionContext);
  }
  facts["projection_succeeded"] = true;
  return bind(std::move(facts));
}

llvm::json::Object projectLayoutFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config, bool diagnostics) {
  return projectLayoutFactsImpl(candidate, config, diagnostics, false);
}

llvm::json::Object projectLayoutSSAFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config) {
  return projectLayoutFactsImpl(candidate, config, true, true);
}

llvm::json::Object projectLayoutTransportFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config) {
  return projectLayoutFactsImpl(candidate, config, true, true, true);
}

llvm::json::Object projectLayoutCollectiveFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config) {
  return projectLayoutFactsImpl(candidate, config, true, true, true, true);
}
llvm::json::Object projectLayoutResourceFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config) {
  return projectLayoutFactsImpl(candidate, config, true, true, true, true, true);
}
llvm::json::Object projectLayoutTransportClosureFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config) {
  return projectLayoutFactsImpl(candidate, config, true, true, true, true, false, true);
}
} // namespace layoutdp
