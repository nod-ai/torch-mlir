//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/ONNX/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
//#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
//#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
//#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
//#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

//#include "src/Compiler/CompilerOptions.hpp"
//#include "src/Compiler/CompilerPasses.hpp"
//#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/ONNX/Transforms/Passes.h.inc"
} // end namespace

void mlir::torch::registerONNXSimplificationPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<Torch::TorchLoweringPipelineOptions>(
      "onnx-simplification-pipeline",
      "Pipeline simplifying ONNX dialect.",
      onnx::createONNXSimplificationPasses);
}

void onnx::createONNXSimplificationPasses(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());

  //if (onnxOpTransformThreshold > 0) {
  //  pm.addPass(onnx_mlir::createONNXOpTransformPass(
  //      onnxOpTransformThreshold, onnxOpTransformReport));
  //} else {
  //  for (int i = 0; i < repeatOnnxTransform; i++) {
  //    pm.addPass(mlir::createCanonicalizerPass());
  //    pm.addPass(onnx_mlir::createShapeInferencePass());
  //    pm.addNestedPass<func::FuncOp>(
  //        onnx_mlir::createConstPropONNXToONNXPass());
  //  }
  //}

  // Clean dead code.
  //pm.addPass(mlir::createSymbolDCEPass());

}
