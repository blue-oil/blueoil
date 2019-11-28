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


class Softmax(Operator):
    r"""Softmax operator.

    The operator computes the softmax (normalized exponential) values for each layer in the
    batch of the given input. The input is a 2-D tensor (Tensor) of size
    (batch_size x input_feature_dimensions). The output tensor has the same shape and contains
    the softmax values of the corresponding input.

    X does not need to explicitly be a 2D vector; rather, it will be coerced into one.
    For an arbitrary n-dimensional tensor X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and
    k is the axis provided, then X will be coerced into a 2-dimensional tensor with dimensions
    [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default case where axis=1, this means
    the X tensor will be coerced into a 2D tensor of dimensions [a_0, a_1 * ... * a_{n-1}],
    where a_0 is often the batch size. In this situation, we must have
    a_0 = N and a_1 * ... * a_{n-1} = D. Each of these dimensions must be matched correctly,
    or else the operator will throw errors.

    Inputs
    ------
    input
        The input tensor that's coerced into a 2D matrix of size (NxD) as described above.

    Outputs
    -------
    output
        The output values with the same shape as input tensor.

    """

    _input_names = ['input']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(self.input_ops['input'].shape == self.shape)

    @property
    def is_monotonic(self) -> bool:
        return False

    def run_forward(self) -> np.ndarray:
        in_data = self.input_ops['input'].data
        exp = np.exp(in_data - np.max(in_data))
        self._data = exp / np.expand_dims(exp.sum(axis=-1), -1)
        return self._data

    @property
    def preserve_quantization(self) -> bool:
        return False


class Relu(Operator):
    """Relu class.

    Relu takes one input data (Tensor) and produces one output data (Tensor)
    where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.

    Inputs
    ------
    X
        Input tensor

    Outputs
    -------
    Y
        Output tensor

    """

    _input_names = ['X']
    _output_names = ['Y']

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(self.input_ops['X'].shape == self.shape)

    @property
    def is_monotonic(self) -> bool:
        return False

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['X']

    @property
    def preserve_quantization(self) -> bool:
        return False


class LeakyRelu(Operator):
    """Leaky relu class.

    Leaky_relu takes one input data (Tensor) and produces one output data (Tensor)
    where the function, y = max(input * alpha, input), is applied to the tensor elementwise.

    Inputs
    ------
    X
        Input tensor

    Outputs
    -------
    Y
        Output tensor

    """

    _input_names = ['X']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 alpha: float = 0.2) -> None:
        """Init the LeakyRelu operator."""
        self.alpha = alpha
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(self.input_ops['X'].shape == self.shape)

    def run_forward(self) -> np.ndarray:
        in_data = self.input_ops['X'].data
        self._data = np.maximum(in_data * self.alpha, in_data)
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return False

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['X']

    @property
    def preserve_quantization(self) -> bool:
        return False


