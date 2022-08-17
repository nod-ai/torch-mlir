/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- ShapeInferenceOpInterface.hpp - Definition for ShapeInference ---===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "torch-mlir/Dialect/ONNX/Interface/ShapeInferenceOpInterface.hpp.inc"
