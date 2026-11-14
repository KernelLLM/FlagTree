// Structural resources of a fully materialized candidate, not hardware VGPRs.
#ifndef TRITON_HCU_LAYOUT_TUNING_RESOURCES_H
#define TRITON_HCU_LAYOUT_TUNING_RESOURCES_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace layoutdp {
llvm::json::Object analyzeLayoutResources(mlir::ModuleOp module,
                                         const llvm::json::Array &conversions);
// Diagnostic alternatives only. No legacy fact or scorer is changed.
llvm::json::Object analyzeLayoutResourceContract(mlir::ModuleOp module);
}
#endif
