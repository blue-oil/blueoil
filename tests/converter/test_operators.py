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
"""Test file for operators."""
import unittest
from typing import Dict

import numpy as np

from blueoil.converter.core.data_types import Float32
from blueoil.converter.core.operators import Constant, Conv, Input, MaxPool, Operator


class TestOperators(unittest.TestCase):
    """Test class for operators."""

    def test_maxpool(self) -> None:
        """Test code for MaxPool."""
        # get MaxPool's input names
        i_names = MaxPool.input_names
        self.assertEqual(i_names, ['X'])

        # set x to MaxPool m's input
        x = Constant(
            'const',
            Float32(),
            np.zeros([1, 3, 3, 3])
        )
        inputs: Dict[str, Operator] = {i_names[0]: x}
        MaxPool(
            "MaxPool",
            [1, 2, 2, 3],
            Float32(),
            inputs,
            kernel_shape=[2, 2]
        )

        print("MaxPool test passed!")

    def test_conv(self) -> None:
        """Test code for Conv."""
        # get Conv's input names
        i_names = Conv.input_names
        self.assertTrue({'X', 'W'}.issubset(set(i_names)))

        # set x to MaxPool m's input
        x = Input(
            'input',
            [1, 3, 3, 3],
            Float32(),
        )
        w = Constant(
            'weight',
            Float32(),
            np.zeros([1, 2, 2, 5])
        )
        inputs: Dict[str, Operator] = {i_names[0]: x, i_names[1]: w}
        c = Conv(
            "conv1",
            [1, 2, 2, 3],
            Float32(),
            inputs,
            kernel_shape=[2, 2]
        )

        self.assertEqual(c.batchsize, 1)
        self.assertEqual(c.height, 2)
        self.assertEqual(c.width, 2)
        self.assertEqual(c.channel, 3)
        self.assertEqual(c.kernel_height, 2)
        self.assertEqual(c.kernel_width, 2)

        print("Conv test passed!")


if __name__ == '__main__':
    unittest.main()
