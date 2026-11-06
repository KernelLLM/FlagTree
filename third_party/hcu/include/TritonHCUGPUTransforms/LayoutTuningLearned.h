// Stage-bound learned scorer. Never evaluates or falls back to ProgramCost.
#ifndef TRITON_HCU_LAYOUT_TUNING_LEARNED_H
#define TRITON_HCU_LAYOUT_TUNING_LEARNED_H

#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

namespace layoutdp {
inline constexpr const char *layoutLearnedPolicy = "generic-current-layout-v114";
struct LayoutLearnedModel {
  struct Interaction {
    unsigned left, right;
    double center, scale, coefficient, lower, upper;
    std::string name;
  };
  std::string id, target;
  std::vector<std::string> features;
  std::vector<double> center, scale, coefficients, lower, upper;
  std::vector<Interaction> interactions;
  double intercept = 0;
  bool unified183 = false;
  std::string inputContractSHA;
};
bool readLayoutLearnedModel(const llvm::json::Value &document,
                           LayoutLearnedModel &model, std::string &reason);
llvm::json::Object scoreLayoutFacts(const LayoutLearnedModel &model,
                                   const llvm::json::Object &facts,
                                   const std::string &deploymentSHA);
} // namespace layoutdp
#endif
