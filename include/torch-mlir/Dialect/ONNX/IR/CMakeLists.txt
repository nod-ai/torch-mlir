set(LLVM_TARGET_DEFINITIONS ONNX.td)
mlir_tablegen(ONNXOps.hpp.inc -gen-op-decls)
mlir_tablegen(ONNXOps.cpp.inc -gen-op-defs)
mlir_tablegen(ONNXDialect.hpp.inc -gen-dialect-decls -dialect=onnx)
mlir_tablegen(ONNXDialect.cpp.inc -gen-dialect-defs -dialect=onnx)
mlir_tablegen(ONNXTypes.hpp.inc -gen-typedef-decls -typedefs-dialect=onnx)
mlir_tablegen(ONNXTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=onnx)
mlir_tablegen(ONNXAttributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=onnx)
mlir_tablegen(ONNXAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=onnx)
add_public_tablegen_target(MLIRONNXIncGen)
add_dependencies(mlir-headers MLIRONNXIncGen)

set(LLVM_TARGET_DEFINITIONS Rewrite.td)
mlir_tablegen(rewriter.inc -gen-rewriters)
add_public_tablegen_target(MLIRONNXRewriteIncGen)

#add_mlir_doc(ONNXDialect ONNXDialect ONNX/ -gen-dialect-doc)
#add_mlir_doc(ONNXOps ONNXOps ONNX/ -gen-op-doc)

configure_file(ONNXOps.td.inc.dc.in
  ${CMAKE_CURRENT_BINARY_DIR}/ONNXOps.td.inc.dc
  @ONLY
  )
