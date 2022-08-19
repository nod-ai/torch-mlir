//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/FrontendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

void mlir::torch::TorchConversion::getFrontendTypeConversionDependentDialects(
    DialectRegistry &registry) {
  registry.insert<TorchConversionDialect>();
}

//===----------------------------------------------------------------------===//
// Type conversion setup.
//===----------------------------------------------------------------------===//

static void
setupTensorToNonValueTensorConversion(ConversionTarget &target,
                                          TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToNonValueTensorOp,
                    TorchConversion::FromNonValueTensorOp>();
  typeConverter.addConversion(
      [](TensorType type) -> Optional<Type> {
        return Torch::NonValueTensorType::getFromBuiltinTensor(type);
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            Torch::NonValueTensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<ToNonValueTensorOp>(loc, type, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder, TensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BaseTensorType>());
    return builder.create<FromNonValueTensorOp>(loc, type, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

//static void setupI1ToTorchBoolConversion(ConversionTarget &target,
//                                         TypeConverter &typeConverter) {
//  target.addLegalOp<TorchConversion::ToI1Op, TorchConversion::FromI1Op>();
//  typeConverter.addConversion([](IntegerType type) -> Optional<Type> {
//    return Torch::BoolType::get(type.getContext());
//  });
//  typeConverter.addTargetMaterialization([](OpBuilder &builder,
//                                            IntegerType type, ValueRange inputs,
//                                            Location loc) -> Optional<Value> {
//    // Other builtin integer types could be handled by other materializers.
//    if (!(type.getWidth() == 1 && type.isSignless()))
//      return None;
//    assert(inputs.size() == 1);
//    assert(inputs[0].getType().isa<Torch::BoolType>());
//    return builder.create<ToI1Op>(loc, inputs[0]).getResult();
//  });
//  auto sourceMaterialization = [](OpBuilder &builder, Torch::BoolType type,
//                                  ValueRange inputs, Location loc) -> Value {
//    assert(inputs.size() == 1);
//    assert(inputs[0].getType().isa<IntegerType>());
//    return builder.create<FromI1Op>(loc, inputs[0]);
//  };
//  typeConverter.addSourceMaterialization(sourceMaterialization);
//  typeConverter.addArgumentMaterialization(sourceMaterialization);
//}
//
//static void setupI64ToTorchIntConversion(ConversionTarget &target,
//                                         TypeConverter &typeConverter) {
//  target.addLegalOp<TorchConversion::ToI64Op, TorchConversion::FromI64Op>();
//  typeConverter.addConversion([](IntegerType type) -> Optional<Type> {
//    return Torch::IntType::get(type.getContext(), 64);
//  });
//  typeConverter.addTargetMaterialization([](OpBuilder &builder,
//                                            IntegerType type, ValueRange inputs,
//                                            Location loc) -> Optional<Value> {
//    assert(inputs.size() == 1);
//    assert(inputs[0].getType().isa<IntegerType>());
//    return builder.create<FromI64Op>(loc, inputs[0]).getResult();
//  });
//  auto sourceMaterialization = [](OpBuilder &builder, Torch::IntType type,
//                                  ValueRange inputs, Location loc) -> Value {
//    assert(inputs.size() == 1);
//    assert(inputs[0].getType().isa<IntegerType>());
//    return builder.create<ToI64Op>(loc, inputs[0]).getResult();
//  };
//  typeConverter.addSourceMaterialization(sourceMaterialization);
//  typeConverter.addArgumentMaterialization(sourceMaterialization);
//}
//
//static void setupF64ToTorchFloatConversion(ConversionTarget &target,
//                                           TypeConverter &typeConverter) {
//  target.addLegalOp<TorchConversion::ToF64Op, TorchConversion::FromF64Op>();
//  typeConverter.addConversion([](Float64Type type) -> Optional<Type> {
//    return Torch::FloatType::get(type.getContext());
//  });
//  typeConverter.addTargetMaterialization([](OpBuilder &builder,
//                                            Float64Type type, ValueRange inputs,
//                                            Location loc) -> Optional<Value> {
//    assert(inputs.size() == 1);
//    assert(inputs[0].getType().isa<Float64Type>());
//    return builder.create<FromF64Op>(loc, inputs[0]).getResult();
//  });
//  auto sourceMaterialization = [](OpBuilder &builder, Torch::FloatType type,
//                                  ValueRange inputs, Location loc) -> Value {
//    assert(inputs.size() == 1);
//    assert(inputs[0].getType().isa<Torch::FloatType>());
//    return builder.create<ToF64Op>(loc, inputs[0]).getResult();
//  };
//  typeConverter.addSourceMaterialization(sourceMaterialization);
//  typeConverter.addArgumentMaterialization(sourceMaterialization);
//}

void mlir::torch::TorchConversion::setupFrontendTypeConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  setupTensorToNonValueTensorConversion(target, typeConverter);
//  setupI1ToTorchBoolConversion(target, typeConverter);
//  setupI64ToTorchIntConversion(target, typeConverter);
//  setupF64ToTorchFloatConversion(target, typeConverter);
}
