get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_mlir_library(TorchMLIRONNXPasses
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Transforms

  DEPENDS
  TorchMLIRONNXPassIncGen
  ${OMLibs}

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncTransforms
  ${OMLibs}
)
