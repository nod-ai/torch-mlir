//===- ONNXImporter.cpp - C Interface for MLIR Registration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/IR.h"
#include "mlir-c/Support.h"
#include "torch-mlir-c/ONNXImporter.h"
#include "torch-mlir/Dialect/ONNX/Builder/FrontendDialectTransformer.hpp"

MlirModule torchMlirImportONNX(MlirContext context, MlirStringRef modelStringRef) {
  std::string model_fname(modelStringRef.data, modelStringRef.length);
  onnx_mlir::ImportOptions options;
  options.useOnnxModelTypes = true;
  std::string errorStr;
  mlir::OwningOpRef<mlir::ModuleOp> importModule;
  ImportFrontendModelFile(
          model_fname, *unwrap(context), importModule, &errorStr, options);

  if (!importModule)
    return MlirModule{nullptr};
  return MlirModule{importModule.release().getOperation()};
}
