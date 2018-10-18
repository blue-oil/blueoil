# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Test file for OnnxIO."""
from os import path
import unittest

from core.graph import Graph


def onnx_is_available() -> bool:
    available = True
    try:
        __import__('onnx')
    except ImportError:
        available = False

    return available


class TestOnnxIO(unittest.TestCase):
    """Test class for OnnxIO."""

    @unittest.skipUnless(onnx_is_available(), "ONNX is not available (reinstall with --enable-onnx)")
    def test_onnx_import(self) -> None:
        """Test code for importing ONNX file with OnnxIO."""
        from frontend.onnx import OnnxIO

        onnx_path = path.join('examples',
                              'classification',
                              'lmnet_quantize_cifar10_stride_2.20180523.3x3')

    @unittest.skipUnless(onnx_is_available(), "ONNX is not available (reinstall with --enable-onnx)")
    def future_test_onnx_import_lmnet_classification(self) -> None:
        """Test code for importing lmnet classification via ONNX."""
        # Current lmnet classification onnx has several problems,
        # so we leave this test for the near future.
        from frontend.onnx import OnnxIO

        onnx_path = path.join('examples',
                              'classification',
                              'lmnet_quantize_cifar10_stride_2.20180523.3x3',
                              # 'lmnet_quantize_cifar10_max_pooling',
                              'onnx_minimal_graph_with_shape.pb')

        onnx_io = OnnxIO()
        model = onnx_io.read(onnx_path)

        print("ONNX import test (using lmnet classification) passed!")

    @unittest.skipUnless(onnx_is_available(), "ONNX is not available (reinstall with --enable-onnx)")
    def test_onnx_import_pytorch_alexnet(self) -> None:
        """Test code for importing PyTorch AlexNet via ONNX."""
        from frontend.onnx import OnnxIO
        # install torch and torchvisiion anyway
        from pip._internal import main as pipmain
        pipmain(['install', 'torch', 'torchvision'])

        from torch.autograd import Variable
        import torch.onnx
        import torchvision

        dummy_input = Variable(torch.randn(10, 3, 224, 224))
        model = torchvision.models.alexnet(pretrained=True)

        # providing these is optional, but makes working with the
        # converted model nicer.
        input_names = ["learned_%d" % i for i in range(16)] + ["actual_input_1"]
        output_names = ["output1"]

        onnx_path = path.join('examples',
                              'classification',
                              'alexnet.pb')

        torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names,
                          output_names=output_names)

        onnx_io = OnnxIO()
        model = onnx_io.read(onnx_path)

        # for debugging
        # json_path = path.join('examples',
        #                       'classification',
        #                       'alexnet.json')
        # model = onnx_io.read(onnx_path, json_path)

        graph: Graph = model.graph
        outputs = graph.get_outputs()

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, [10, 1000])
        print("ONNX import test (using PyTorch alexnet) passed!")


if __name__ == '__main__':
    unittest.main()
