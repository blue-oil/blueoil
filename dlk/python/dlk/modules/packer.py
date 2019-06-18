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
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

from modules.base_module import BaseModule


class Packer(BaseModule):

    def __init__(self,
                 nbit_qkernel: int,
                 nbit_qword: int,
                 multiplexing_mode: int = 2,
                 extra_bit_value: bool = True) -> None:
        """Initialize packer object.

        Parameters
        ----------
        nbit_qkernel: int
            Bitwidth of a kernel

        nbit_qword: int
            Wordsize

        multiplexing_mode: int
            This defaults to 2.

        extra_bit_value: bool
            This defaults to True.
        """
        super().__init__()
        self.packer = None
        self.map_wordsize_type = {8: np.uint8, 32: np.uint32}

        self.lib.packer_create.argtypes = []
        self.lib.packer_create.restype = ct.c_void_p

        self.lib.packer_delete.argtypes = [ct.c_void_p]
        self.lib.packer_delete.restype = None

        self.lib.packer_set_bitwidth.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.packer_set_bitwidth.restype = None

        self.lib.packer_set_wordsize.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.packer_set_wordsize.restype = None

        self.lib.packer_set_multiplexing_mode.argtypes = [
            ct.c_void_p,
            ct.c_int
        ]
        self.lib.packer_set_multiplexing_mode.restype = None

        self.lib.packer_set_extra_bit_value.argtypes = [ct.c_void_p, ct.c_bool]
        self.lib.packer_set_extra_bit_value.restype = None

        self.lib.packer_get_output_size.argtypes = [
            ct.c_void_p, ct.c_int, ct.c_int
        ]
        self.lib.packer_get_output_size.restype = ct.c_int

        self.lib.packer_get_wordsize.argtypes = [ct.c_void_p]
        self.lib.packer_get_wordsize.restype = ct.c_int

        self.lib.packer_run.argtypes = [
            ct.c_void_p,
            ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
            ct.c_int,
            ct.c_void_p
        ]
        self.lib.packer_run.restype = None

        self.packer = self.lib.packer_create()

        self.set_bitwidth(nbit_qkernel)
        self.set_wordsize(nbit_qword)
        self.set_multiplexing_mode(multiplexing_mode)
        self.set_extra_bit_value(extra_bit_value)

    def __del__(self):
        if self.packer:
            self.lib.packer_delete(self.packer)
            self.packer = None
            self.lib = None

    def set_bitwidth(self, bitwidth):
        self.lib.packer_set_bitwidth(self.packer, bitwidth)

    def set_wordsize(self, wordsize):
        self.lib.packer_set_wordsize(self.packer, wordsize)

    def set_multiplexing_mode(self, mode):
        self.lib.packer_set_multiplexing_mode(self.packer, mode)

    def set_extra_bit_value(self, value):
        self.lib.packer_set_extra_bit_value(self.packer, value)

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
        # nk = data_format.index('N') if 'N' in data_format \
        #     else data_format.index('O')

        number_of_kernels = 1  # tensor.shape[nk]
        output_size = self.lib.packer_get_output_size(self.packer,
                                                      tensor.size,
                                                      number_of_kernels)
        wordsize = self.lib.packer_get_wordsize(self.packer)
        output_tensor = np.zeros(output_size, self.map_wordsize_type[wordsize])
        tensor_flat = tensor.flatten(order='C')

        self.lib.packer_run(
            self.packer,
            tensor_flat,
            tensor.size,
            output_tensor.ctypes.data_as(ct.c_void_p)
        )

        output_tensor = np.reshape(output_tensor, [number_of_kernels, output_size // number_of_kernels])
        return output_tensor
