// Cost-independent facts of one complete current candidate. No seed or prices.
#ifndef TRITON_HCU_LAYOUT_TUNING_FACTS_H
#define TRITON_HCU_LAYOUT_TUNING_FACTS_H

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"

namespace mlir::triton { class ModuleAxisInfoAnalysis; }
namespace layoutdp {
class LayoutExecutionContext;
inline constexpr const char *layoutFactsVersion = "layout-facts-v114";
inline constexpr const char *layoutTuningStage =
    "after-f32-dot-tc-remove-layout-conversions-before-optimize-thread-locality-v1";

// Search-only physical proxy. The type override never reaches learned facts.
llvm::json::Object analyzeLayoutAccessHeuristic(
    mlir::Value value, mlir::Value address, mlir::Value mask, mlir::Value base,
    mlir::triton::ModuleAxisInfoAnalysis &axes, mlir::RankedTensorType candidateType);
llvm::json::Object analyzeLayoutConversionFacts(
    mlir::triton::gpu::ConvertLayoutOp op);
llvm::json::Object analyzeLayoutReductionFacts(
    mlir::ModuleOp module,
    const llvm::DenseMap<mlir::Operation *, unsigned> &operationIds,
    const LayoutExecutionContext *context = nullptr, bool modelExtensions = false);
llvm::json::Object analyzeLayoutFacts(mlir::ModuleOp module, unsigned numStages,
                                    bool diagnostics = true,
                                    llvm::StringRef executionContext = "");
// Direct read-only semantic normalization and facts, without LayoutModel/domain.
llvm::json::Object analyzeCurrentLayoutFacts(mlir::ModuleOp candidate,
                                           unsigned numStages,
                                           bool diagnostics = false,
                                           llvm::StringRef executionContext = "");
} // namespace layoutdp
#endif
