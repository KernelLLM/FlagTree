/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "TritonDistributed/Conversion/TritonDistributedToHIVM/Passes.h"
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONDISTRIBUTEDTOHIVM
#include "TritonDistributed/Conversion/TritonDistributedToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isSupportedCoreRatio(int32_t cube, int32_t vector) {
  // Keep in sync with _normalize_core_ratio in python/triton_dist/jit.py.
  static constexpr std::pair<int32_t, int32_t> supportedCoreRatios[] = {
      {0, 1}, {1, 0}, {1, 1}, {1, 2}};
  return llvm::is_contained(supportedCoreRatios, std::make_pair(cube, vector));
}

struct ConvertTritonDistributedToHIVM
    : public triton::impl::ConvertTritonDistributedToHIVMBase<
          ConvertTritonDistributedToHIVM> {
  using ConvertTritonDistributedToHIVMBase::ConvertTritonDistributedToHIVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // Check if the kernel contains tl.dot. Without tl.dot,
    // the kernel would be pure AIV kernel.
    bool existDot = false;
    mod.walk([&](triton::DotOp dotOp) {
      existDot = true;
      return WalkResult::interrupt();
    });
    mod.walk([&](triton::DotScaledOp dotScaledOp) {
      existDot = true;
      return WalkResult::interrupt();
    });
    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    // Transfer TD's distributed.core_ratio into hivm.core_ratio.
    if (auto attr = mod->getAttr("distributed.core_ratio")) {
      auto ratio = dyn_cast<StringAttr>(attr);
      if (!ratio) {
        mod.emitError("distributed.core_ratio must be a \"<cube>:<vector>\" "
                      "string");
        return signalPassFailure();
      }
      auto [cubeStr, vectorStr] = ratio.getValue().split(':');
      int32_t cube, vector;
      if (cubeStr.getAsInteger(10, cube) ||
          vectorStr.getAsInteger(10, vector)) {
        mod.emitError("invalid distributed.core_ratio '")
            << ratio.getValue() << "', expected \"<cube>:<vector>\"";
        return signalPassFailure();
      }

      if (!isSupportedCoreRatio(cube, vector)) {
        mod.emitError("unsupported distributed.core_ratio '")
            << cube << ":" << vector << "'";
        return signalPassFailure();
      }
      if (cube == 0 && existDot) {
        mod.emitError("invalid core-ratio '")
            << cube << ":" << vector << "' with a MIX kernel";
        return signalPassFailure();
      }

      auto ratioAttr = DenseI32ArrayAttr::get(context, {cube, vector});
      mod.walk([&](triton::FuncOp func) {
        func->setAttr("hivm.core_ratio", ratioAttr);
      });
      mod->removeAttr("distributed.core_ratio");
    }

    // Populate patterns to convert distributed ops to hivm.custom
    mlir::triton::ASCEND::populateDistributedOpToHIVMPatterns(patterns, benefit,
                                                              existDot);

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace
