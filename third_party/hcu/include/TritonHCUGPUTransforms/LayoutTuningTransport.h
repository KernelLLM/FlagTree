// Opt-in transport closure: existing explicit transfers plus implicit LDS
// layout conversions. Same four columns, explicit semantic binding, no prices.
#ifndef TRITON_HCU_LAYOUT_TUNING_TRANSPORT_H
#define TRITON_HCU_LAYOUT_TUNING_TRANSPORT_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace layoutdp {
llvm::json::Object addLayoutConversionTransport(
    mlir::ModuleOp module, llvm::json::Object explicitTransport,
    llvm::StringRef executionContext);
}
#endif
