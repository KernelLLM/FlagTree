// Optional diagnostic export from the SAME normalized current-candidate clone.
// Never enables a scorer or changes the production insertion point.
#ifndef TRITON_HCU_LAYOUT_TUNING_SSA_H
#define TRITON_HCU_LAYOUT_TUNING_SSA_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
namespace layoutdp {
struct LayoutProjectionConfig;
llvm::json::Object analyzeCurrentLayoutSSAFacts(mlir::ModuleOp candidate,
    unsigned numStages, llvm::StringRef executionContext);
// Diagnostic resource alternatives share the exact normalized facts clone.
llvm::json::Object analyzeCurrentLayoutResourceContractFacts(mlir::ModuleOp candidate,
    unsigned numStages, llvm::StringRef executionContext, bool includeSSA);
llvm::json::Object projectLayoutSSAFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config);
// Opt-in bounded shared transport view. Legacy APIs and their 139 values are
// unchanged. The production selection pass is not moved or enabled here.
llvm::json::Object projectLayoutTransportFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config);
// Opt-in, predicate-aware reduce/scan shared plan on the SAME late clone as
// explicit shared transfers. Legacy APIs and all old numeric columns unchanged.
llvm::json::Object projectLayoutCollectiveFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config);
llvm::json::Object analyzeCurrentLayoutCollectiveFacts(
    mlir::ModuleOp candidate, llvm::StringRef executionContext);
// Diagnostic comparison on an already transformed module; deliberately does
// not run the early projection on cached final TTGIR.
llvm::json::Object analyzeCurrentLayoutSharedTransportFacts(
    mlir::ModuleOp candidate, llvm::StringRef executionContext);
// Same four transport columns with explicit semantic binding. Old entry points
// retain explicit-only meanings; no model weights are reused automatically.
llvm::json::Object projectLayoutTransportClosureFacts(
    mlir::ModuleOp candidate, const LayoutProjectionConfig &config);
llvm::json::Object analyzeCurrentLayoutTransportClosureFacts(
    mlir::ModuleOp candidate, llvm::StringRef executionContext);
} // namespace layoutdp
#endif
