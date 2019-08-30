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
"""Packer module."""
import numpy as np


class Packer:

    def __init__(self,
                 bitwidth: int,
                 wordsize: int) -> None:
        """Initialize packer object.

        Parameters
        ----------
        bitwidth: int
            Bitwidth of a kernel

        wordsize: int
            Wordsize

        """
        super().__init__()

        self.bitwidth = bitwidth
        self.wordsize = wordsize

    def pack_to_word(self, v, powers=None):
        if not powers:
            return np.dot(v, self.powers)
        else:
            return np.dot(v, powers)

    def run(self, tensor: np.ndarray, data_format: str = 'NHWC') -> np.ndarray:
        """Pack a tensor.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor.

        data_format : str
            Order of dimension. This defaults to 'NHWC', where 'N' is
            the number of kernels, 'H' and 'W' are the height and width,
            and 'C' is the depth / the number of channels.

        Returns
        -------
        output_tensor : np.ndarray
            Quantized tensor.
        """

        wordsize = self.wordsize

        output_size = tensor.size // wordsize
        if tensor.size % wordsize != 0:
            output_size += 1

        tensor_flat = tensor.flatten(order='C').astype(np.uint32)
        output = np.zeros(output_size, dtype=np.uint32)
        oi = 0

        # generate powers (1,2,4,8....) here to pack binary values fast
        self.powers = np.power(2, np.arange(wordsize))
        for i in range(0, tensor.size // wordsize):
            iw = i * wordsize
            sliced_tensor = tensor_flat[iw:iw + wordsize]

            for _ in range(0, self.bitwidth):
                output[oi] = self.pack_to_word(np.bitwise_and(sliced_tensor, 1))
                oi += 1
                sliced_tensor = np.right_shift(sliced_tensor, 1)

        if tensor.size % wordsize != 0:
            iw = (tensor.size // wordsize + 1) * wordsize
            sliced_tensor = tensor_flat[iw:-1]
            powers = np.power(2, np.arange(sliced_tensor.size))
            for _ in range(0, self.bitwidth):
                output[oi] = self.pack_to_word(np.bitwise_and(sliced_tensor, 1), powers)
                oi += 1
                sliced_tensor = np.right_shift(sliced_tensor, 1)

        return output.reshape([1, output_size])
