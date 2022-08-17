# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(TorchMLIRONNXSupport
  TypeUtilities.cpp
  Diagnostic.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Support

  LINK_LIBS PUBLIC
  MLIRIR
  )
