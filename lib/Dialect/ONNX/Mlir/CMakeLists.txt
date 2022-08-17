# SPDX-License-Identifier: Apache-2.0

add_mlir_library(TorchMLIRONNXMlir
  IndexExpr.cpp
  IndexExprDetail.cpp
  DialectBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Mlir

  LINK_LIBS PUBLIC
  MLIRMathDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRLLVMIRTransforms
  )
