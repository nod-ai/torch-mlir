//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/ONNXToTorch/ONNXToTorch.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/FrontendTypeConversion.h"

#include "torch-mlir/Dialect/ONNX/IR/ONNXOps.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------
// Patterns for individual ops should live in one of the other files, and
// added via the relevant `populate*PatternsAndLegality` functions.
// This file is just for the pass definition itself.

namespace {
class ConvertONNXToTorch
    : public ConvertONNXToTorchBase<ConvertONNXToTorch> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<torch::Torch::TorchDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
    TorchConversion::getFrontendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<torch::Torch::TorchDialect,
                           torch::TorchConversion::TorchConversionDialect>();
    target.addIllegalDialect<mlir::ONNXDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupFrontendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    onnx_to_torch::populateBasicPatternsAndLegality(typeConverter,
                                                      patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertONNXToTorchPass() {
  return std::make_unique<ConvertONNXToTorch>();
}
