/*===-- torch-mlir-c/ONNXImporter.h - Dialect functions  ----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef TORCHMLIR_C_ONNXIMPORTER_H
#define TORCHMLIR_C_ONNXIMPORTER_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirModule torchMlirImportONNX(MlirContext context, MlirStringRef modelStringRef);

#ifdef __cplusplus
}
#endif

#endif  // TORCHMLIR_C_ONNXIMPORTER_H
