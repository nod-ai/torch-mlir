add_mlir_library(TorchMLIRONNXPasses
  Passes.cpp
  Decompose.cpp
  DecomposeEinsum.cpp
  EraseONNXEntryPoint.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/Transforms

  DEPENDS
  TorchMLIRONNXPassIncGen
  TorchMLIRONNXDecomposeIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncTransforms
  MLIRTransformUtils
  onnx
)

torch_mlir_target_includes(TorchMLIRONNXPasses)
