#ifndef TRITON_HCU_LAYOUT_EXECUTION_CONTEXT_H
#define TRITON_HCU_LAYOUT_EXECUTION_CONTEXT_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace layoutdp {
// Runtime bindings only resolve scalar SSA expressions. No names/families,
// pointer addresses, tensor contents, measured times or seed inputs.
class LayoutExecutionContext {
public:
  LayoutExecutionContext(mlir::ModuleOp module, llvm::StringRef payload);
  bool valid() const { return error.empty(); }
  const std::string &failure() const { return error; }
  std::optional<llvm::APInt> integer(mlir::Value value, unsigned depth = 0) const;
  std::optional<double> trips(mlir::scf::ForOp loop) const;
  llvm::json::Object scaleFacts(mlir::ModuleOp module) const;
private:
  bool present = false, gridKnown = false;
  int64_t grid[3] = {0, 0, 0};
  double tensorCount = 0, tensorElements = 0, tensorBytes = 0,
         tensorSpanBytes = 0, contiguousAxes = 0, unknownTensors = 0;
  std::string error;
  llvm::DenseMap<mlir::Value, llvm::APInt> scalars;
  mutable llvm::DenseMap<mlir::Value, std::optional<llvm::APInt>> memo;
};
double layoutExecutionWeight(mlir::Operation *op,
                             const LayoutExecutionContext *context = nullptr);
} // namespace layoutdp
#endif
