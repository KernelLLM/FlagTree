#ifndef TRITON_HCU_LAYOUT_TUNING_H
#define TRITON_HCU_LAYOUT_TUNING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "TritonHCUGPUTransforms/LayoutTuningProjection.h"
#include <string>

namespace layoutdp {
struct LayoutTuningOptions {
  std::string scorer = "model";
  std::string mode = "search";
  std::string algorithm = "beam";
  std::string dumpPath, tracePath;
  unsigned numStages = 0; // Actual launch pipeline stages.
  LayoutProjectionConfig projection;
  // Opt-in matrix distribution domain. Total warps and tile stay fixed.
  bool mfmaWarpCandidates = false;
  // All three required together. Off by default; never auto-installed.
  std::string jointModelPath, jointModelSHA256, jointPolicy;
  unsigned maxTrials = 64, maxCandidates = 12, beamWidth = 3;
  std::string candidatePolicy = "balanced-coherent";
  unsigned candidatePool = 128;
};
mlir::LogicalResult tuneLayouts(mlir::ModuleOp mod,
                                const LayoutTuningOptions &options,
                                std::string *reportJSON = nullptr);
} // namespace layoutdp
#endif
