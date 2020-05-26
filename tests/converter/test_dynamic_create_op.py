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
"""Test file for dynamic creation of operators."""
import importlib
import unittest
from typing import List

import numpy as np

from blueoil.converter.core.data_types import Float32
from blueoil.converter.core.operators import Constant


class TestDynamicCreateOp(unittest.TestCase):
    """Test class for dynamic creation of operators."""

    def test_dynamic_create_unary(self) -> None:
        """Test code for unary operators."""
        unary_ops = [
            'Identity',
            'BinaryMeanScalingQuantizer',
            'Transpose',
            'LinearMidTreadHalfQuantizer',
            'MaxPool',
            'AveragePool',
            'Reshape',
            'Softmax'
        ]

        # unary input
        shape = [1, 3, 3, 3]
        x = Constant(
            'const',
            Float32(),
            np.zeros(shape)
        )

        name = 'test'
        dtype = Float32()

        for op in unary_ops:
            shape = [1, 3, 3, 3]
            module = importlib.import_module('blueoil.converter.core.operators')
            try:
                op_def = getattr(module, op)
                input_ops = {n: x for n in op_def.input_names}
                shape = self.reverse_shape(shape) if op == 'Transpose' \
                    else [1, 2, 2, 3] if op == 'MaxPool' or op == 'AveragePool' \
                    else shape
                args = [name, shape, dtype, input_ops]
                obj = op_def(*args)
                self.assertEqual(obj.name, name)
            except Exception as e:
                print(f'failed in testing {op}.')
                raise e

        print("Dynamic unary operator load test passed!")

    def reverse_shape(self, shape: List[int]) -> List[int]:
        return [shape[i] for i in range(len(shape) - 1, -1, -1)]

    def test_dynamic_create_binary(self) -> None:
        """Test code for binary operators."""
        x = Constant(
            'const1',
            Float32(),
            np.zeros([1, 3, 3, 3])
        )

        w = Constant(
            'const2',
            Float32(),
            np.zeros([1, 2, 2, 3])
        )

        binary_ops = [
            'Conv',
            'Add'
        ]

        name = 'test'
        dtype = Float32()

        for op in binary_ops:
            shape = [1, 3, 3, 3]
            module = importlib.import_module('blueoil.converter.core.operators')
            try:
                op_def = getattr(module, op)
                shape = [1, 2, 2, 3] if op == 'Conv' else shape
                input_ops = {n: opw for n, opw in zip(op_def.input_names, [x, w])} \
                    if op == 'Conv' else {n: x for n in op_def.input_names}
                args = [name, shape, dtype, input_ops]
                obj = op_def(*args)
                self.assertEqual(obj.name, name)
            except Exception as e:
                print(f'failed in testing {op}.')
                raise e

        print("Dynamic binary operator load test passed!")

    def test_dynamic_create_batchnorm(self) -> None:
        """Test code for n-ary operators (BatchNormalization)."""
        x = Constant(
            'const',
            Float32(),
            np.zeros([1, 3, 3, 3])
        )

        nary_ops = [
            'BatchNormalization'
        ]

        name = 'test'
        shape = [1, 3, 3, 3]
        dtype = Float32()

        for op in nary_ops:
            module = importlib.import_module('blueoil.converter.core.operators')
            try:
                op_def = getattr(module, op)
                input_ops = {n: x for n in op_def.input_names}
                args = [name, shape, dtype, input_ops]
                obj = op_def(*args)
                self.assertEqual(obj.name, name)
            except Exception as e:
                print(f'failed in testing {op}.')
                raise e

        print("Dynamic batchnorm operator load test passed!")


if __name__ == '__main__':
    unittest.main()
