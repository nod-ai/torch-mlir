add_mlir_library(TorchMLIRONNXPasses
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Transforms

  DEPENDS
  TorchMLIRONNXPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncTransforms
)
