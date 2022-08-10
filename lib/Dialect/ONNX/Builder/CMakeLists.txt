# SPDX-License-Identifier: Apache-2.0

# This is supposed to cause onnx to be built. More importantly, some variable definitions
# when building onnx such as -DONNX_ML=1 -DONNX_NAMESPACE=onnx will be carried over
# when compiling FrontendDialectHelper.cpp, etc. because they are properly exported
# as part of the onnx target and target_link_libraries interprets them correctly.
add_mlir_library(TorchMLIRONNXBuilder
  FrontendDialectHelper.cpp
  FrontendDialectTransformer.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Builder
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  TorchMLIRONNXInterface
  TorchMLIRONNXDialect
  onnx
  )
