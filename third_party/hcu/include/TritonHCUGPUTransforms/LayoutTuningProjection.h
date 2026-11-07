// Shared backend prefix and read-only, current-candidate fact producer.
#ifndef TRITON_HCU_LAYOUT_TUNING_PROJECTION_H
#define TRITON_HCU_LAYOUT_TUNING_PROJECTION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace layoutdp {
inline constexpr const char *layoutProjectionVersion =
    "hcu-current-candidate-projection-v114";
inline constexpr const char *layoutProjectedFactsVersion =
    "projected-layout-facts-v114";
inline constexpr const char *layoutProjectionAnalysisStage =
    "native-prefix-through-first-post-slot-RLC+bounded-direct-call-expansion+symbol-dce+cse-licm-cse+exact-linearlayout-order-reuse-v25+execution-context-v1+distributed-reduce-gather-cat-v159+dot-operand-lds-lookahead-v163+declared-order-register-groups-v165+collective-primitive-pooling-v173";

struct LayoutProjectionConfig {
  std::string arch;
  int matrixInstructionSize = 0, kPack = 1, mmacLayoutForce = -1;
  unsigned numStages = 0;
  std::string executionContext;
};

// Keep model-specific hint validation outside the upstream matmul pass.
mlir::LogicalResult validateLayoutMFMAHints(mlir::ModuleOp module,
    llvm::StringRef arch, bool requireConsumed = false);

// These exact two builders are also called by HIPBackend.make_ttgir.
void addLayoutThreadLocalityAndTLE(mlir::OpPassManager &pm);
void addLayoutMatmulPrefix(mlir::OpPassManager &pm,
                          const LayoutProjectionConfig &config);
void addLayoutEvaluationPrefix(mlir::OpPassManager &pm,
                               const LayoutProjectionConfig &config);

// Input is one complete candidate at layoutTuningStage, never a seed delta.
// Clones before all transforms. No search domain, weights, prices or timing.
// Failure is explicit; completeness, codegen and numerical validity are distinct.
llvm::json::Object projectLayoutFacts(mlir::ModuleOp candidate,
                                    const LayoutProjectionConfig &config,
                                    bool diagnostics = false);
llvm::json::Object projectLayoutResourceFacts(
    mlir::ModuleOp candidate, const LayoutProjectionConfig &config);
} // namespace layoutdp
#endif
