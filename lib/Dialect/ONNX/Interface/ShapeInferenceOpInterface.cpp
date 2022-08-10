/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ShapeInferenceOpInterface.cpp - Definition for ShapeInference ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the implementations of the shape inference interfaces
// defined in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/ONNX/Interface/ShapeInferenceOpInterface.hpp"

namespace mlir {

/// Include the auto-generated declarations.
#include "torch-mlir/Dialect/ONNX/Interface/ShapeInferenceOpInterface.cpp.inc"

} // end namespace mlir
