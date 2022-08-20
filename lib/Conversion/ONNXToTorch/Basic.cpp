//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/ONNXToTorch/ONNXToTorch.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/ONNX/IR/ONNXOps.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertONNXConstantOp : public OpConversionPattern<ONNXConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.value().has_value())
      return rewriter.notifyMatchFailure(op, "unimplemented: non-dense values are unsupported");
    ElementsAttr value = adaptor.valueAttr();
    auto newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<NonValueTensorLiteralOp>(op, newResultType, value);
    return success();
  }
};
} // namespace

namespace {
class ConvertONNXAddOp : public OpConversionPattern<ONNXAddOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value one = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    auto newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, newResultType, adaptor.A(), adaptor.B(), one);
    return success();
  }
};
} // namespace

void mlir::torch::onnx_to_torch::populateBasicPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<ONNXConstantOp>();
  patterns.add<ConvertONNXConstantOp>(typeConverter, context);
  target.addIllegalOp<ONNXAddOp>();
  patterns.add<ConvertONNXAddOp>(typeConverter, context);
}
