#include "TritonHCUGPUTransforms/Passes.h"
#include "TritonHCUGPUTransforms/LayoutTuning.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cstdlib>
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_TRITONHCUGPULAYOUTMODELTUNING
#include "TritonHCUGPUTransforms/Passes.h.inc"
namespace {
bool runLayoutConfiguration(ModuleOp module, layoutdp::LayoutTuningOptions config) {
    auto env = [](const char *name, std::string &value) {
      if (const char *s = std::getenv(name)) value = s;
    };
    env("FLAGTREE_LAYOUT_TUNING_MODE", config.mode);
    env("FLAGTREE_LAYOUT_TUNING_CANDIDATE_POLICY", config.candidatePolicy);
    env("FLAGTREE_LAYOUT_TUNING_ALGORITHM", config.algorithm);
    env("FLAGTREE_LAYOUT_TUNING_JOINT_MODEL", config.jointModelPath);
    env("FLAGTREE_LAYOUT_TUNING_JOINT_MODEL_SHA256", config.jointModelSHA256);
    env("FLAGTREE_LAYOUT_TUNING_JOINT_POLICY", config.jointPolicy);
    env("FLAGTREE_LAYOUT_TUNING_DUMP", config.dumpPath);
    env("FLAGTREE_LAYOUT_TUNING_TRACE", config.tracePath);
    if (const char *s = std::getenv("FLAGTREE_LAYOUT_TUNING_MFMA_WARP_DOMAIN")) {
      if (std::string(s) != "0" && std::string(s) != "1") {
        module.emitError("MFMA warp domain must be 0 or 1");
        return false;
      }
      config.mfmaWarpCandidates = std::string(s) == "1";
    }
    auto budget = [&](const char *name, unsigned &value) {
      const char *s = std::getenv(name);
      if (!s) return true;
      char *end = nullptr;
      unsigned long n = std::strtoul(s, &end, 10);
      if (!*s || *s == '-' || *end || n > std::numeric_limits<unsigned>::max()) {
        module.emitError("invalid layout tuning budget: ") << name;
        return false;
      }
      value = n;
      return true;
    };
    return budget("FLAGTREE_LAYOUT_TUNING_TRIALS", config.maxTrials) &&
        budget("FLAGTREE_LAYOUT_TUNING_CANDIDATES", config.maxCandidates) &&
        budget("FLAGTREE_LAYOUT_TUNING_CANDIDATE_POOL", config.candidatePool) &&
        budget("FLAGTREE_LAYOUT_TUNING_BEAM", config.beamWidth) &&
        succeeded(layoutdp::tuneLayouts(module, config));
}

struct LayoutModelTuningPass : impl::TritonHCUGPULayoutModelTuningBase<LayoutModelTuningPass> {
  using Base::Base;
  void runOnOperation() override {
    layoutdp::LayoutTuningOptions config;
    config.scorer = "model";
    config.projection.arch = projectionArch;
    config.projection.executionContext = executionContext;
    config.projection.matrixInstructionSize = projectionMatrixInstructionSize;
    config.projection.kPack = projectionKPack;
    config.projection.mmacLayoutForce = projectionMmacLayoutForce;
    config.numStages = numStages;
    config.mode = mode;
    config.algorithm = algorithm;
    config.dumpPath = dumpPath;
    config.tracePath = tracePath;
    config.maxTrials = maxTrials;
    config.maxCandidates = maxCandidates;
    config.beamWidth = beamWidth;
    config.jointModelPath = jointModelPath;
    config.jointModelSHA256 = jointModelSHA256;
    config.jointPolicy = jointPolicy;
    if (!runLayoutConfiguration(getOperation(), std::move(config))) signalPassFailure();
  }
};

} // namespace
} // namespace mlir
