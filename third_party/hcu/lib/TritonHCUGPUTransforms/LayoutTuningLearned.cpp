#include "TritonHCUGPUTransforms/LayoutTuningLearned.h"
#include "TritonHCUGPUTransforms/LayoutTuningFacts.h"
#include "TritonHCUGPUTransforms/LayoutTuningProjection.h"
#include "TritonHCUGPUTransforms/LayoutTuningUnified.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace layoutdp {
bool readLayoutLearnedModel(const llvm::json::Value &document,
                           LayoutLearnedModel &model, std::string &reason) {
  auto fail = [&](const char *message) { reason = message; return false; };
  const auto *doc = document.getAsObject();
  if (!doc || doc->getString("schema") != layoutLearnedPolicy ||
      doc->getString("feature_version") != layoutProjectedFactsVersion)
    return fail("expected generic-current-layout-v114/projected-layout-facts-v114 artifact; older facts require retraining");
  if (doc->getString("pipeline_stage") != layoutTuningStage)
    return fail("learned model pipeline stage mismatch");
  if (doc->getString("producer_schema") != layoutProjectionVersion ||
      doc->getString("analysis_stage") != layoutProjectionAnalysisStage)
    return fail("learned model projection producer/analysis stage mismatch");
  auto id = doc->getString("model_id"), target = doc->getString("target");
  auto *features = doc->getArray("features");
  auto intercept = doc->getNumber("intercept");
  if (!id || id->empty() || !target || target->empty() || !features ||
      features->empty() || features->size() > 4096 || !intercept ||
      !std::isfinite(*intercept)) return fail("invalid learned model metadata");
  LayoutLearnedModel parsed;
  parsed.id = id->str(); parsed.target = target->str();
  parsed.intercept = *intercept;
  std::set<std::string> unique;
  for (const auto &item : *features) {
    auto name = item.getAsString();
    if (!name || name->empty() || !unique.insert(name->str()).second)
      return fail("invalid/duplicate learned feature name");
    parsed.features.push_back(name->str());
  }
  if (parsed.features==layoutUnifiedFeatureNames()) {
    const auto *contract=doc->getObject("input_contract");
    if (!contract || contract->getString("schema")!=layoutUnifiedContract ||
        doc->getString("input_contract_sha256")!=layoutUnifiedContractSHA)
      return fail("unified183 extraction contract mismatch");
    parsed.unified183=true;parsed.inputContractSHA=layoutUnifiedContractSHA;
  }
  auto vector = [&](const char *key, std::vector<double> &out,
                    bool positive = false) {
    const auto *values = doc->getArray(key);
    if (!values || values->size() != features->size()) return false;
    for (const auto &item : *values) {
      auto value = item.getAsNumber();
      if (!value || !std::isfinite(*value) || (positive && *value <= 0)) return false;
      out.push_back(*value);
    }
    return true;
  };
  if (!vector("center", parsed.center) || !vector("scale", parsed.scale, true) ||
      !vector("coefficients", parsed.coefficients) ||
      !vector("lower", parsed.lower) || !vector("upper", parsed.upper))
    return fail("invalid learned model vectors");
  for (unsigned i = 0; i < parsed.features.size(); ++i)
    if (parsed.lower[i] > parsed.upper[i]) return fail("invalid learned feature bounds");
  if (doc->get("interactions")) {
    const auto *terms=doc->getArray("interactions");
    if (!parsed.unified183 || !terms || terms->size()>32 ||
        doc->getString("interaction_transform")!="log1p-product")
      return fail("invalid learned interaction contract");
    std::set<std::pair<unsigned,unsigned>> seen;
    for (const auto &value:*terms) {
      const auto *term=value.getAsObject();if (!term || term->size()!=7) return fail("invalid learned interaction fields");
      auto left=term->getString("left"),right=term->getString("right");
      auto center=term->getNumber("center"),scale=term->getNumber("scale"),coefficient=term->getNumber("coefficient");
      auto lower=term->getNumber("lower"),upper=term->getNumber("upper");
      if (!left || !right || !center || !scale || !coefficient || !lower || !upper ||
          !std::isfinite(*center) || !std::isfinite(*scale) || *scale<=0 || !std::isfinite(*coefficient) ||
          !std::isfinite(*lower) || !std::isfinite(*upper) || *lower>*upper)
        return fail("invalid learned interaction parameters");
      auto l=std::find(parsed.features.begin(),parsed.features.end(),left->str());
      auto r=std::find(parsed.features.begin(),parsed.features.end(),right->str());
      if (l==parsed.features.end() || r==parsed.features.end()) return fail("unknown learned interaction feature");
      unsigned li=l-parsed.features.begin(),ri=r-parsed.features.begin();
      if (!seen.emplace(std::min(li,ri),std::max(li,ri)).second) return fail("duplicate learned interaction");
      parsed.interactions.push_back({li,ri,*center,*scale,*coefficient,*lower,*upper,
          "log1p("+left->str()+")*log1p("+right->str()+")"});
    }
  } else if (doc->get("interaction_transform")) return fail("interaction transform without terms");
  model = std::move(parsed);
  return true;
}

llvm::json::Object scoreLayoutFacts(const LayoutLearnedModel &model,
                                   const llvm::json::Object &facts,
                                   const std::string &deploymentSHA) {
  llvm::json::Object result{{"scorer", layoutLearnedPolicy},
      {"model_id", model.id}, {"deployment_sha256", deploymentSHA},
      {"pipeline_stage", layoutTuningStage}, {"feature_version", layoutProjectedFactsVersion},
      {"producer_schema", layoutProjectionVersion}, {"analysis_stage", layoutProjectionAnalysisStage},
      {"seed_reference_input", false}, {"fallback_allowed", false},
      {"units", "ordinal-score-not-kernel-duration"}};
  auto fail = [&](const char *reason) {
    result["failure"] = reason; result["total"] = nullptr;
    return std::move(result);
  };
  if (facts.getString("feature_version") != layoutProjectedFactsVersion ||
      facts.getString("pipeline_stage") != layoutTuningStage ||
      facts.getString("schema") != layoutProjectionVersion ||
      facts.getString("analysis_stage") != layoutProjectionAnalysisStage)
    return fail("facts stage/schema mismatch");
  if (facts.getString("target") != model.target) return fail("model target mismatch");
  if (model.unified183 && facts.getString("input_contract_sha256")!=model.inputContractSHA)
    return fail("unified183 facts input contract mismatch");
  if (!facts.getBoolean("complete").value_or(false)) return fail("incomplete layout facts");
  const auto *values = facts.getObject("values");
  if (!values) return fail("missing layout facts");
  double score = model.intercept;
  llvm::json::Array inputs, outside;
  std::vector<double> logs;
  for (unsigned i = 0; i < model.features.size(); ++i) {
    auto value = values->getNumber(model.features[i]);
    if (!value || !std::isfinite(*value) || *value < 0)
      return fail("required layout fact missing or non-finite");
    // The current schema fixes log1p before train-only centering/scaling.
    double x = (std::log1p(*value) - model.center[i]) / model.scale[i];
    logs.push_back(std::log1p(*value));
    inputs.push_back(x);
    if (x < model.lower[i] || x > model.upper[i]) outside.push_back(model.features[i]);
    score += x * model.coefficients[i];
  }
  llvm::json::Array interactionInputs;
  for (const auto &term:model.interactions) {
    double x=(logs[term.left]*logs[term.right]-term.center)/term.scale;
    interactionInputs.push_back(x);
    if (x<term.lower || x>term.upper) outside.push_back(term.name);
    score+=x*term.coefficient;
  }
  if (!std::isfinite(score)) return fail("non-finite learned score");
  result["total"] = score;
  result["failure"] = "";
  result["inputs"] = std::move(inputs);
  if (!model.interactions.empty()) result["interaction_inputs"] = std::move(interactionInputs);
  result["outside_training_bounds"] = std::move(outside);
  result["support_policy"] = "valid facts are scored; distribution shift is reported, never a switch of scorer";
  return result;
}
} // namespace layoutdp
