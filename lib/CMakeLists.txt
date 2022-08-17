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
  TorchMLIRRefBackend)

if(TORCH_MLIR_ENABLE_ONNX)
  get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)
  list(APPEND linked_libs
    ${OMLibs}
    TorchMLIRONNXPasses)
endif()

add_mlir_library(TorchMLIRInitAll
  InitAll.cpp

  DEPENDS
  ${OMLibs}

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${linked_libs}
)

torch_mlir_target_includes(TorchMLIRInitAll)
