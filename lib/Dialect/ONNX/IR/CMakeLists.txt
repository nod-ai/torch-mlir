# SPDX-License-Identifier: Apache-2.0

add_mlir_library(TorchMLIRONNXDialect
  DialectBuilder.cpp
  ONNXDialect.cpp
  ONNXOps.cpp
  ONNXOpsHelper.cpp
  ONNXEinsumOpHelper.cpp
  ONNXTypes.cpp
  ONNXShapeHelper.cpp
  Rewrite.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/IR

  DEPENDS
  MLIRONNXIncGen
  MLIRONNXRewriteIncGen
  MLIRONNXHasOnnxSubgraphOpInterfaceIncGen
  MLIRONNXResultTypeInferenceOpInterfaceIncGen
  MLIRONNXShapeInferenceOpInterfaceIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  TorchMLIRONNXSupport
  TorchMLIRONNXMlir
  TorchMLIRONNXInterface
  #OMMlirDialects
  #OMMlirUtilities
  onnx
)

torch_mlir_target_includes(TorchMLIRONNXDialect)
