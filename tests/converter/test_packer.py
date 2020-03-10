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
"""Unittest for Packer."""

import unittest

import numpy as np

from blueoil.converter.modules.packer import Packer


class TestPacker(unittest.TestCase):
    """Test class for Packer."""

    def test_bw1_dividable_by_wordsize(self):
        """Test for when the input tensor size is able to divide by wordsize (1 bit version)."""
        packer = Packer(1, 32)

        test_input = np.zeros([32], dtype=np.float32)
        test_input[0:6] = [0, 1, 0, 1, 0, 1]

        test_output = packer.run(test_input)

        self.assertEqual(test_output[0], 42)

    def test_bw2_dividable_by_wordsize(self):
        """Test for when the input tensor size is able to divide by wordsize (2 bit version)."""
        packer = Packer(2, 32)

        test_input = np.zeros([32], dtype=np.float32)
        test_input[0:6] = [0, 3, 0, 3, 0, 3]

        test_output = packer.run(test_input)
        expected_output = [42, 42]

        np.testing.assert_array_equal(test_output[0], expected_output)

    def test_bw1_not_dividable_by_wordsize(self):
        """Test for when the input tensor size is not able to divide by wordsize (1 bit version)."""
        packer = Packer(1, 37)

        test_input = np.zeros([37], dtype=np.float32)
        test_input[0::2] = 1

        test_output = packer.run(test_input)
        expected_output = [1431655765]

        np.testing.assert_array_equal(test_output[0], expected_output)

    def test_bw2_not_dividable_by_wordsize(self):
        """Test for when the input tensor size is not able to divide by wordsize (2 bit version)."""
        packer = Packer(2, 37)

        test_input = np.zeros([32], dtype=np.float32)
        test_input[0:6] = [0, 3, 0, 3, 0, 3]
        test_input[0::3] = 2

        test_output = packer.run(test_input)
        expected_output = [34, 1227133547]

        np.testing.assert_array_equal(test_output[0], expected_output)

    def test_raise_exception_bitwidth(self):
        """Raise an exception if an input value is larger than bitwidth."""
        packer = Packer(2, 32)

        test_input = np.zeros([64], dtype=np.float32)
        test_input[0:-1:2] = 1
        test_input[0:-1:4] = 4

        with self.assertRaises(ValueError):
            packer.run(test_input)

    def test_raise_exception_wordsize(self):
        """Raise an exception if an input value is not multiple of word size."""
        packer = Packer(2, 32)

        test_input = np.zeros([83], dtype=np.float32)
        test_input[0:-1:2] = 1
        test_input[0:-1:4] = 4

        with self.assertRaises(ValueError):
            packer.run(test_input)


if __name__ == '__main__':
    unittest.main(verbosity=2)
