add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)

set(linked_libs
  MLIRFuncDialect
  MLIRIR
  MLIRSupport

  TorchMLIRTorchPasses
  TorchMLIRTorchConversionDialect

  TorchMLIRTorchDialect
  TorchMLIRTorchConversionPasses

  TorchMLIRTMTensorPasses
  TorchMLIRTMTensorDialect

  TorchMLIRConversionPasses
  TorchMLIRRefBackend
  
  TorchMLIRONNXPasses
  TorchMLIRONNXDialect)

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${linked_libs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
