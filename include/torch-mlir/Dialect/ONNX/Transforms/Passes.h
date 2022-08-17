//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_ONNX_TRANSFORMS_PASSES_H
#define TORCHMLIR_DIALECT_ONNX_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace torch {
namespace onnx {

/// Creates a pipeline that simplifies onnx dialect
//void createONNXSimplificationPasses(
//    OpPassManager &pm,
//    const torch::Torch::TorchLoweringPipelineOptions &options);
} // namespace onnx

/// Registers ONNX simplification pipeline
//void registerONNXSimplificationPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_ONNX_TRANSFORMS_PASSES_H
