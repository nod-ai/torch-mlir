/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ResultTypeInferenceOpInterface.hpp --------------===//
//===------- Infer Data Type for Result of Op Interface Definition -------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of the data type reference for op
// interface.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "torch-mlir/Dialect/ONNX/Interface/ResultTypeInferenceOpInterface.hpp.inc"

} // end namespace mlir
