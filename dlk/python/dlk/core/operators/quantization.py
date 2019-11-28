# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
"""Definition of operators."""
import copy
import functools
import warnings
from termcolor import colored
from abc import abstractmethod
from itertools import dropwhile
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from core.view import View
from utils import classproperty

from .base import *

if TYPE_CHECKING:
    import core.operators as ops

Ops = Dict[str, 'Operator']
OutOps = Dict[str, List['Operator']]

warning_sign = colored('WRN', 'red', attrs=['blink'])


class Quantizer(Operator):
    """Base class for quantizers."""

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init this quantization operator."""
        self._scaling_factor = np.float32(0)
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def equals(self, other: Any) -> bool:
        sup = super().equals(other)
        return sup and np.isclose(self.scaling_factor, other.scaling_factor)

    @property
    def nbit(self) -> int:
        raise NotImplementedError('Quantizer does not have bit value defined')

    @property
    def max_v(self) -> float:
        raise NotImplementedError('Quantizer does not have max value defined')

    @property
    def scaling_factor(self) -> np.float32:
        return self._scaling_factor

    @property
    def preserve_quantization(self) -> bool:
        return False

    @scaling_factor.setter
    def scaling_factor(self, val: np.float32) -> None:
        self._scaling_factor = val

    @abstractmethod
    def binarizer(self, data: np.ndarray) -> np.ndarray:
        """Maps the quantized values into >= 0 integer values.

        This is actually an abstract method and should be overridden.
        """
        raise NotImplementedError(
            f'operator {self.op_type} need to implement the binarizer method')


class QTZ_binary_mean_scaling(Quantizer):
    """Quantization operator using binary scaling.

    Input
    -----
    input
        Input tensor, which must have float values.

    Output
    ------
    output
        Quantized tensor

    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the quantization operator."""
        self._scaling_factor = 0
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(self._input_ops['input'].shape == self.shape,
                     f'Shape mismatch at {self.op_type}" {self.name}"')

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    def run_forward(self) -> np.ndarray:
        in_data = self.input_ops['input'].data
        self._scaling_factor = np.mean(np.abs(in_data))
        self._data = np.sign(in_data)

        return self._data * self._scaling_factor

    def run_forward_no_scaling_factor(self) -> np.ndarray:
        in_data = self.input_ops['input'].data
        self._scaling_factor = np.mean(np.abs(in_data))
        self._data = np.sign(in_data)

        return self._data

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['input']

    def binarizer(self, data: np.ndarray) -> np.ndarray:
        """Maps the quantized values into >= 0 integer values."""
        bdata = copy.deepcopy(data)
        bdata[bdata < 0] = 0
        return bdata


class QTZ_linear_mid_tread_half(Quantizer):
    """Quantization operator with 'linear mid tread half'.

    Input
    -----
    X
        Input tensor

    Y
        Constant

    Z
        Another constant

    """
    _input_names = ['X', 'Y', 'Z']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init this quantization operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        x_shape = self._input_ops['X'].shape
        message = 'QTZ_linear_mid_tread_half operator has inconsistency in shapes: '
        message += f'input "X" has {x_shape}, while it has {self.shape}'
        self._assert(x_shape == self.shape, message)

    def run(self, **kwargs) -> Dict:
        """Return the result of forward calculation of an activation quantizer.

        Currently this function is only used by threshold skipping optimization pass
        for recursively calculating thresholds of the skipping patterns.
        """
        bit = self._input_ops['Y'].data
        max_value = np.float64(self._input_ops['Z'].data)
        in_data = np.float64(kwargs['data'])

        n = 2 ** bit - 1
        np.clip(in_data, 0, max_value, out=in_data)
        kwargs['data'] = np.floor(in_data * n / max_value + 0.5).astype(np.int32)
        return kwargs

    def de_run(self, **kwargs) -> Dict:
        """Return the result of reversed calculation of an activation quantizer.

        Currently this function is only used by threshold skipping optimization pass
        for recursively calculating thresholds of the skipping patterns.
        """
        bit = self._input_ops['Y'].data
        max_value = np.float64(self._input_ops['Z'].data)
        in_data = np.float64(kwargs['data'])

        n = 2 ** bit - 1
        kwargs['data'] = (in_data * np.float64(max_value)) / np.float64(n)
        return kwargs

    def run_forward(self) -> np.ndarray:
        """General function for this quantization operator.

        This function returns numpy array.
        """
        data_dict = self.run(data=self._input_ops['X'].data)
        self._data = data_dict['data']
        return self._data

    @property
    def nbit(self) -> int:
        return self._input_ops['Y'].data[0]

    @property
    def max_v(self) -> float:
        return self._input_ops['Z'].data[0]

    @property
    def is_monotonic(self) -> bool:
        return True

    @property
    def _dispatch_name(self) -> str:
        return self.op_type

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['X']

    def binarizer(self, data: np.ndarray) -> np.ndarray:
        """Maps the quantized values into >= 0 integer values."""
        return data


class QTZ_binary_channel_wise_mean_scaling(Quantizer):
    """Quantization operator using binary channel wise scaling.

    Input
    -----
    input
        Input tensor, which must have float values.

    Output
    ------
    output
        Quantized tensor

    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the quantization operator."""
        self._scaling_factor = 0
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(self._input_ops['input'].shape == self.shape,
                     f'Shape mismatch at {self.op_type}" {self.name}"')

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['input']

    @property
    def is_monotonic(self) -> bool:
        return False

    def run_forward(self) -> np.ndarray:
        in_data = self.input_ops['input'].data
        self._scaling_factor = np.mean(np.abs(in_data), axis=(1, 2, 3)).astype(np.float32)
        self._data = np.sign(in_data)

        scaling = copy.deepcopy(self._scaling_factor)
        extra_dims = tuple(np.ones((len(self._data.shape) - len(scaling.shape)), dtype=np.int32))
        scaling = scaling.reshape(scaling.shape + extra_dims)

        return scaling * self._data

    def run_forward_no_scaling_factor(self) -> np.ndarray:
        in_data = self.input_ops['input'].data
        self._scaling_factor = np.mean(np.abs(in_data), axis=(1, 2, 3)).astype(np.float32)
        self._data = np.sign(in_data)
        return self._data

    def binarizer(self, data: np.ndarray) -> np.ndarray:
        """Maps the quantized values into >= 0 integer values."""
        bdata = copy.deepcopy(data)
        bdata[bdata < 0] = 0
        return bdata


class Lookup(Quantizer):
    r"""Lookup operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['input', 'lsb', 'msb']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return True

    @property
    def nbit(self) -> int:
        return 2

    @property
    def max_v(self) -> float:
        return 2.0


