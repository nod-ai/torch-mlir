/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Clip.cpp - Shape Inference for Clip Op ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements shape inference for the ONNX Clip operator.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/ONNX/IR/ONNXOpsHelper.hpp"
#include "torch-mlir/Dialect/ONNX/IR/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXClipOpShapeHelper::computeShape(
    ONNXClipOpAdaptor operandAdaptor) {
  Value input = operandAdaptor.input();
  MemRefBoundsIndexCapture bounds(input);
  int64_t rank = bounds.getRank();

  DimsExpr outputDims(rank);
  for (int64_t i = 0; i < rank; ++i)
    outputDims[i] = bounds.getDim(i);
  dimsForOutput() = outputDims;

  return success();
}

} // namespace onnx_mlir
