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
#include "torch-mlir/Dialect/ONNX/IR/ONNXDialect.hpp"
#include "torch-mlir/Dialect/ONNX/Transforms/Passes.hpp"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/ONNX/Transforms/Passes.h.inc"
} // end namespace

void onnx_mlir::registerONNXSimplificationPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<torch::Torch::TorchLoweringPipelineOptions>(
      "onnx-simplification-pipeline",
      "Pipeline simplifying ONNX dialect.",
      onnx_mlir::createONNXSimplificationPasses);
}

void onnx_mlir::createONNXSimplificationPasses(
    OpPassManager &pm, const torch::Torch::TorchLoweringPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addPass(mlir::createSymbolDCEPass());
}
