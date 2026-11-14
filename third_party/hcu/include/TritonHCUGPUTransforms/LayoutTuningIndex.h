#ifndef TRITON_HCU_LAYOUT_TUNING_INDEX_H
#define TRITON_HCU_LAYOUT_TUNING_INDEX_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace layoutdp {
// Diagnostic features on the actual materialized candidate. Not a latency
// model: memory/conversion prices can already include address generation.
llvm::json::Object analyzeLayoutIndex(mlir::ModuleOp module,
                                     const llvm::json::Array &conversions,
                                     bool extendedAccess = false);
}
#endif
