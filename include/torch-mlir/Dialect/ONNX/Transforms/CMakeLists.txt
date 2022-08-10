set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(TorchMLIRONNXPassIncGen)

set(LLVM_TARGET_DEFINITIONS Decompose.td)
mlir_tablegen(ONNXDecompose.inc -gen-rewriters)
add_public_tablegen_target(TorchMLIRONNXDecomposeIncGen)

add_mlir_doc(Passes TorchMLIRONNXTransforms ./ -gen-pass-doc)
