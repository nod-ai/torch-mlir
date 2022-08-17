# SPDX-License-Identifier: Apache-2.0

add_mlir_library(TorchMLIRONNXInterface
  ShapeInferenceOpInterface.cpp
  ResultTypeInferenceOpInterface.cpp
  HasOnnxSubgraphOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Interface

  DEPENDS
  MLIRIR
  MLIRONNXHasOnnxSubgraphOpInterfaceIncGen
  MLIRONNXResultTypeInferenceOpInterfaceIncGen
  MLIRONNXShapeInferenceOpInterfaceIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
)
