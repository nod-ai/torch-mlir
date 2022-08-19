//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_ONNX_PASSES
#define TORCHMLIR_ONNX_PASSES

include "mlir/Pass/PassBase.td"

def EraseONNXEntryPoint
    : Pass<"onnx-erase-entry-point", "ModuleOp"> {
  let summary = "Erase the `onxx.EntryPoint` op.";
  let constructor =
    "mlir::onnx::createEraseONNXEntryPointPass()";
  let description = [{
    The onnx entry point is currently being removed.
  }];
}

#endif // TORCHMLIR_ONNX_PASSES
