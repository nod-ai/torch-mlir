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

namespace {
class ConvertONNXConvOp : public OpConversionPattern<ONNXConvOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto context = op.getContext();
    if (op.auto_pad() != llvm::StringRef("NOTSET"))
      return rewriter.notifyMatchFailure(op, "unimplemented: padding only supported through padding lists");

    // tensor rank needed for default values of dilations and strides
    //Value input = adaptor.X();
    //int64_t rank = getTensorRank(input);

    // padding defaults to [0, ..., 0]
    // TODO: Verify padding values and check if a padding op is needed
    SmallVector<Value> padding;
    if (op.pads().has_value()) {
      //ArrayRef<int64_t> newPads(adaptor.pads(), adaptor.pads()->size()/2);
      for (int64_t padIndex : llvm::seq(0, (int)op.pads().value().size()/2))
        padding.push_back(rewriter.create<ConstantIntOp>(loc, op.pads().value()[padIndex].cast<IntegerAttr>()));
    } else {
      //if (rank < 0)
      //  return rewriter.notifyMatchFailure(op, "unimplemented: unranked input tensor without explicit pads");
      //for (auto i : llvm::seq(0, rank))
      //  padding.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0)));
      padding.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0)));
    }
    Value paddingList = rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(Torch::IntType::get(context)), padding);

    // dilations defaults to [1, ..., 1]
    SmallVector<Value> dilations;
    if (op.dilations().has_value()) {
      for (auto dilation : op.dilations().value())
        dilations.push_back(rewriter.create<ConstantIntOp>(loc, dilation.cast<IntegerAttr>()));
    } else {
      //if (rank < 0)
      //  return rewriter.notifyMatchFailure(op, "unimplemented: unranked input tensor without explicit dilations");
      //for (auto i : llvm::seq(0, rank))
      //  dilations.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1)));
      dilations.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1)));
    }
    Value dilationsList = rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(Torch::IntType::get(context)), dilations);

    // strides defaults to [1, ..., 1]
    SmallVector<Value> strides;
    if (op.strides().has_value()) {
      for (auto stride : op.strides().value())
        strides.push_back(rewriter.create<ConstantIntOp>(loc, stride.cast<IntegerAttr>()));
    } else {
      //if (rank < 0)
      //  return rewriter.notifyMatchFailure(op, "unimplemented: unranked input tensor without explicit strides");
      //for (auto i : llvm::seq(0, rank))
      //  strides.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1)));
      strides.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1)));
    }
    Value stridesList = rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(Torch::IntType::get(context)), strides);

    // ONNXConvTranspose is a different op
    Value transposed = rewriter.create<ConstantBoolOp>(loc, false);

    // Output padding defaults to 0 for now TODO: actually handle output padding
    SmallVector<Value> output_padding;
    output_padding.push_back(rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0)));
    Value outputPaddingList = rewriter.create<PrimListConstructOp>(
            loc, Torch::ListType::get(Torch::IntType::get(context)), output_padding);

    Value groups = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(op.group()));

    auto newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(op, newResultType, adaptor.X(), adaptor.W(),
            adaptor.B(), stridesList, paddingList, dilationsList, transposed,
            outputPaddingList, groups);
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
