/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ONNXDialect.cpp - ONNX Operations -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/DialectImplementation.h"

#include "torch-mlir/Dialect/ONNX/IR/ONNXDialect.hpp"

using namespace mlir;

// Code for ONNX_Dialect class
#include "torch-mlir/Dialect/ONNX/IR/ONNXDialect.cpp.inc"
