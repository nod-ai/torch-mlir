# SPDX-License-Identifier: Apache-2.0

add_mlir_library(TorchMLIRONNXDialect
  DialectBuilder.cpp
  ONNXDialect.cpp
  ONNXOps.cpp
  ONNXOpsHelper.cpp
  ONNXEinsumOpHelper.cpp
  ONNXTypes.cpp
  Rewrite.cpp

  ShapeInference/ArgMinMax.cpp
  ShapeInference/AveragePool.cpp
  ShapeInference/CategoryMapper.cpp  
  ShapeInference/Clip.cpp  
  ShapeInference/Compress.cpp
  ShapeInference/Concat.cpp
  ShapeInference/Conv.cpp
  ShapeInference/DepthToSpace.cpp
  ShapeInference/Expand.cpp
  ShapeInference/Flatten.cpp
  ShapeInference/Gather.cpp
  ShapeInference/GatherElements.cpp  
  ShapeInference/GatherND.cpp  
  ShapeInference/Gemm.cpp
  ShapeInference/LRN.cpp
  ShapeInference/MatMul.cpp
  ShapeInference/MaxPool.cpp
  ShapeInference/ONNXShapeHelper.cpp  
  ShapeInference/OneHot.cpp
  ShapeInference/Pad.cpp
  ShapeInference/Reduction.cpp
  ShapeInference/Reshape.cpp
  ShapeInference/ReverseSequence.cpp
  ShapeInference/RoiAlign.cpp
  ShapeInference/Shape.cpp
  ShapeInference/Slice.cpp
  ShapeInference/SpaceToDepth.cpp
  ShapeInference/Split.cpp
  ShapeInference/Squeeze.cpp
  ShapeInference/Tile.cpp
  ShapeInference/TopK.cpp
  ShapeInference/Transpose.cpp
  ShapeInference/Unsqueeze.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/ONNX/IR

  DEPENDS
  MLIRONNXIncGen
  MLIRONNXRewriteIncGen
  MLIRONNXHasOnnxSubgraphOpInterfaceIncGen
  MLIRONNXResultTypeInferenceOpInterfaceIncGen
  MLIRONNXShapeInferenceOpInterfaceIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  TorchMLIRONNXSupport
  TorchMLIRONNXMlir
  TorchMLIRONNXInterface
  #OMMlirDialects
  #OMMlirUtilities
  onnx
)

torch_mlir_target_includes(TorchMLIRONNXDialect)
