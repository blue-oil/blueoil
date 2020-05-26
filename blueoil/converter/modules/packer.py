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
    """Packer class packs small integer values to dense unsigned integer (uint8 or uint32)."""

    def __init__(self,
                 bitwidth: int,
                 wordsize: int) -> None:
        """Initialize packer object.

        Args:
            bitwidth (int): Bitwidth of a kernel
            wordsize (int): Wordsize

        """
        super().__init__()

        self.bitwidth = bitwidth
        self.wordsize = wordsize
        # generate powers of 2 (1,2,4,8....) here to pack binary values fast
        self.powers = np.power(2, np.arange(wordsize)).astype(np.uint32)

    def _pack_to_word(self, v):
        wordsize = self.wordsize
        powers = self.powers
        if v.size != wordsize:
            powers = np.power(2, np.arange(v.size)).astype(np.uint32)
        return np.dot(v, powers).astype(np.uint32)

    def run(self, tensor: np.ndarray, data_format: str = 'NHWC') -> np.ndarray:
        """Pack a tensor.

        Args:
            tensor (np.ndarray): Input tensor.
            data_format (str): Order of dimension. This defaults to 'NHWC', where 'N' is
                the number of kernels, 'H' and 'W' are the height and
                width, and 'C' is the depth / the number of channels.

        Returns:
            np.ndarray: Quantized tensor.

        """

        wordsize = self.wordsize

        if (tensor >= (2 ** self.bitwidth)).any():
            raise ValueError("all value of input tensor must be less than bit width ({})".format(self.bitwidth))

        output_size = tensor.size // wordsize
        output_size += 1 if tensor.size % wordsize != 0 else 0
        output_size *= self.bitwidth

        tensor_flat = tensor.flatten(order='C').astype(np.uint32)
        output = np.zeros(output_size, dtype=np.uint32)
        oi = 0
        for i in range(0, tensor.size, wordsize):
            if i + wordsize < tensor.size:
                sliced_tensor = tensor_flat[i:i + wordsize]
            else:
                sliced_tensor = tensor_flat[i:]

            for _ in range(0, self.bitwidth):
                output[oi] = self._pack_to_word(np.bitwise_and(sliced_tensor, 1))
                oi += 1
                sliced_tensor = np.right_shift(sliced_tensor, 1)

        return output.reshape([1, output_size])
