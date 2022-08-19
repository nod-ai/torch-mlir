# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y

model = ToyModel()

x = torch.ones(1, 3)
y = torch.ones(1, 3)

module = torch_mlir.compile(model, (x, x), output_type="onnx")
print(module)

backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
jit_module = backend.load(compiled)

print(jit_module.main_graph(x.numpy(), y.numpy()))
