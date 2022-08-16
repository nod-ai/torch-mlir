# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys

from PIL import Image
import requests

import torch
import torchvision.models as models
from torchvision import transforms

import torch_mlir

def load_and_preprocess_image(url: str):
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    img = Image.open(requests.get(url, headers=headers,
                                  stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels

image_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"

print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
labels = load_labels()

resnet18 = models.resnet18(pretrained=True)
resnet18.train(False)

module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type="onnx")

import subprocess
import tempfile
import warnings

temp_module = tempfile.NamedTemporaryFile(
        mode="wt", suffix="_to_onnx.mlir", prefix="tmp_torch_"
)
temp_module.write(str(module.operation.get_asm()))

command = ['torch-mlir-opt']
command += [temp_module.name, '--mlir-elide-elementsattrs-if-larger=32']

try:
    subprocess.run(command)
except FileNotFoundError as e:
    module.dump()
    warnings.warn("Couldn't find 'torch-mlir-opt' in the PATH so the module was dumped. Please add it to the path to elide large constants.")
