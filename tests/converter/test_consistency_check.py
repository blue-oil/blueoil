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
"""Test file for consistency checking in core.operators"""
import importlib
import unittest
from typing import List, cast

import numpy as np

from blueoil.converter.core.data_types import Float32
from blueoil.converter.core.operators import Add, Constant, Conv, Input, MaxPool, Operator


class TestConsistencyCheck(unittest.TestCase):
    """Test class for dynamic creation of operators."""

    def test_add_consistency1(self) -> None:
        """Test code for 'Add', which succeeds."""
        a = Constant(
            'const1',
            Float32(),
            np.zeros([1, 3, 3])
        )
        b = Constant(
            'const2',
            Float32(),
            np.zeros([3])
        )
        input_ops = {'A': cast(Operator, a), 'B': cast(Operator, b)}
        add = Add(
            'add1',
            [1, 3, 3],
            Float32(),
            input_ops
        )

        print("Consistency test for 'Add' #1 passed!")

    def test_add_consistency2(self) -> None:
        """Test code for 'Add', which fails."""
        a = Constant(
            'const1',
            Float32(),
            np.zeros([1, 3, 3])
        )
        b = Constant(
            'const2',
            Float32(),
            np.zeros([2])
        )
        input_ops = {'A': cast(Operator, a), 'B': cast(Operator, b)}
        try:
            add = Add(
                'add1',
                [1, 3, 3],
                Float32(),
                input_ops
            )
        except AssertionError:
            print("Consistency test for 'Add' #2 passed!")

        else:
            self.assertTrue(False, "Consistency test for 'Add' #2 failed.")

    def test_pool_consistency(self) -> None:
        """Test code for Pool."""
        x = Constant(
            'const1',
            Float32(),
            np.zeros([1, 3, 3, 3])
        )
        input_ops = {'X': cast(Operator, x)}

        add = MaxPool(
            'max_pool1',
            [1, 2, 2, 3],
            Float32(),
            input_ops,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2]
        )

        print("Consistency test for pooling operator passed!")

    def test_conv_consistency(self) -> None:
        """Test code for Conv."""
        x = Input(
            'const1',
            [1, 3, 3, 3],
            Float32(),
        )
        w = Constant(
            'weight',
            Float32(),
            np.zeros([1, 2, 2, 3])
        )
        input_ops = {'X': cast(Operator, x), 'W': cast(Operator, w)}

        add = Conv(
            'conv_under_test',
            [1, 3, 3, 3],
            Float32(),
            input_ops,
            pads=[1, 1, 2, 2],
            strides=[2, 2]
        )

        print("Consistency test for conv operator passed!")


if __name__ == '__main__':
    unittest.main()
