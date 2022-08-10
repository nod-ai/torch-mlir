# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch_mlir

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input + input

model = ToyModel()

module = torch_mlir.compile(model, torch.ones(1, 3, 224, 224), output_type="onnx")
print(module)
