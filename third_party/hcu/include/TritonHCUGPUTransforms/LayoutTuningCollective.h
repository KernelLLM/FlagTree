// Opt-in lowering-plan facts. No scorer, weights, seed or emitted ISA model.
#ifndef TRITON_HCU_LAYOUT_TUNING_COLLECTIVE_H
#define TRITON_HCU_LAYOUT_TUNING_COLLECTIVE_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
namespace layoutdp {
llvm::json::Object analyzeLayoutCollectiveSharedPlan(
    mlir::ModuleOp module, llvm::StringRef executionContext);
} // namespace layoutdp
#endif
