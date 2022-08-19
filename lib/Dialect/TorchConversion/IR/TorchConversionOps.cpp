//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch;

static bool haveSameSizeAndElementType(TensorType lhs, TensorType rhs) {
  if (lhs.hasRank() != rhs.hasRank())
    return false;
  bool sameSize = lhs.hasRank() ? lhs.getShape().equals(rhs.getShape()) : true;
  bool sameElementType = lhs.getElementType() == rhs.getElementType();
  return sameElementType && sameSize;
}


//===----------------------------------------------------------------------===//
// ToNonValueTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ToNonValueTensorOp::verify() {
  auto resultType =
      getResult().getType().cast<Torch::NonValueTensorType>().getWithValueSemantics().toBuiltinTensor();
  auto operandType = getOperand().getType().cast<TensorType>();
  if (!haveSameSizeAndElementType(resultType, operandType)) {
    return emitError()
           << "operand and result must have the same size and dtype";
  }
  return success();
}

LogicalResult ToNonValueTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType =
      Torch::NonValueTensorType::getFromBuiltinTensor(operands[0].getType().cast<TensorType>());
  if (!resultType)
    return failure();
  inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// FromNonValueTensorOp
//===----------------------------------------------------------------------===//

LogicalResult FromNonValueTensorOp::verify() {
  auto resultType = getResult().getType().cast<TensorType>();
  auto operandType =
      getOperand().getType().cast<Torch::NonValueTensorType>().getWithValueSemantics().toBuiltinTensor();
  if (!haveSameSizeAndElementType(resultType, operandType)) {
    return emitError()
           << "operand and result must have the same size and dtype";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ToBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ToBuiltinTensorOp::verify() {
  auto resultType = getResult().getType().cast<TensorType>();
  auto operandType =
      getOperand().getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  if (!haveSameSizeAndElementType(resultType, operandType)) {
    return emitError()
           << "operand and result must have the same size and dtype";
  }
  return success();
}

LogicalResult ToBuiltinTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType =
      operands[0].getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  if (!resultType)
    return failure();
  inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// FromBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult FromBuiltinTensorOp::verify() {
  auto resultType =
      getResult().getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  auto operandType = getOperand().getType().cast<TensorType>();
  if (!haveSameSizeAndElementType(resultType, operandType)) {
    return emitError()
           << "operand and result must have the same size and dtype";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FromI64Op
//===----------------------------------------------------------------------===//

OpFoldResult FromI64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// ToI64Op
//===----------------------------------------------------------------------===//

OpFoldResult ToI64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.cpp.inc"
