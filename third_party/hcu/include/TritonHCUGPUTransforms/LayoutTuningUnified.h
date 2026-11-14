// Complete current-candidate input producer for the unified learned model.
#ifndef TRITON_HCU_LAYOUT_TUNING_UNIFIED_H
#define TRITON_HCU_LAYOUT_TUNING_UNIFIED_H
#include "TritonHCUGPUTransforms/LayoutTuningProjection.h"
#include <vector>
namespace layoutdp {
inline constexpr const char *layoutUnifiedContract = "absolute-current-tile-warp-layout";
// Pins the frozen reference extraction semantics, not any coefficients/version.
inline constexpr const char *layoutUnifiedContractSHA =
    "ef9ea80fdf58cdb994ecf31ac747623c445031e45de1faa7a1f15c0ed43b7561";
const std::vector<std::string> &layoutUnifiedFeatureNames();
llvm::json::Object enrichLayoutUnifiedFacts(llvm::json::Object facts,
                                           bool diagnostics = false);
llvm::json::Object projectLayoutUnifiedFacts(mlir::ModuleOp candidate,
    const LayoutProjectionConfig &config, bool diagnostics = false);
} // namespace layoutdp
#endif
