add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(RefBackend)

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  DEPENDS
  ${OMLibs}

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
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

  ${OMLibs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
