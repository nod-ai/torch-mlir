//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_ONNX_TRANSFORMS_PASSDETAIL_H
#define TORCHMLIR_DIALECT_ONNX_TRANSFORMS_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace onnx {

#define GEN_PASS_CLASSES
#include "torch-mlir/Dialect/ONNX/Transforms/Passes.h.inc"

} // namespace onnx
} // end namespace mlir

#endif // TORCHMLIR_DIALECT_ONNX_TRANSFORMS_PASSDETAIL_H
