add_mlir_conversion_library(TorchMLIRONNXToTorch
  ONNXToTorch.cpp
  Basic.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/ONNXToTorch

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TorchMLIRTorchDialect
  TorchMLIRONNXDialect
  TorchMLIRTorchConversionPasses
)

torch_mlir_target_includes(TorchMLIRONNXToTorch)
