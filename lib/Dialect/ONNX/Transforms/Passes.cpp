//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/Dialect/ONNX/IR/ONNXDialect.hpp"
#include "torch-mlir/Dialect/ONNX/Transforms/Passes.hpp"
#include "torch-mlir/Conversion/ONNXToTorch/ONNXToTorch.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace mlir::torch;

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/ONNX/Transforms/Passes.h.inc"
} // end namespace

void onnx_mlir::registerONNXToTorchPasses() {
  ::registerPasses();
  PassPipelineRegistration<torch::Torch::TorchLoweringPipelineOptions>(
      "onnx-to-torch-pipeline",
      "Pipeline converting ONNX to Torch dialect.",
      onnx_mlir::createONNXToTorchPasses);
}

void onnx_mlir::createONNXToTorchPasses(
    OpPassManager &pm, const torch::Torch::TorchLoweringPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<func::FuncOp>(createConvertONNXToTorchPass());
  pm.addPass(TorchConversion::createFuncFrontendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(
        TorchConversion::createFinalizingFrontendTypeConversionPass());
  pm.addPass(mlir::onnx::createEraseONNXEntryPointPass());
  pm.addPass(Torch::createRefineFuncValueSemanticsPass());
}
