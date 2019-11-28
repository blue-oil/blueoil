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


class Variable(Operator):
    """Variable class, which must be Input, Output or a constant."""

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 data: np.ndarray,
                 dimension_format: str = 'NHWC') -> None:
        """Init the variable."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)
        self._data = data

    @property
    def is_variable(self) -> bool:
        """Return True, as this is a variable."""
        return True

    @property
    def is_monotonic(self) -> bool:
        return False

    def transpose(self, perm: List[int]) -> None:
        """Transpose the shape and format. This operation is destructive."""
        super().transpose(perm)
        self._data = self._data.transpose(perm)

    @property
    def data(self) -> np.ndarray:
        """Return data."""
        return self._data

    @data.setter
    def data(self, val: np.ndarray) -> None:
        self._data = val

    @property
    def preserve_quantization(self) -> bool:
        return False


class Input(Variable):
    """Input class. This is a placeholder."""

    _input_names: List[str] = []
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 dimension_format: str = 'NHWC') -> None:
        """Init the input variable."""
        data = np.zeros(shape, dtype=dtype.nptype())
        super().__init__(name, shape, dtype, {}, data, dimension_format=dimension_format)


class Constant(Variable):
    """Constant class. This object has data inside."""

    _input_names: List[str] = []
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 dtype: DataType,
                 data: np.ndarray,
                 dimension_format: str = 'OHWI',
                 transposed_dimension_format: str = 'OHWI',
                 packed: bool = False,
                 actual_shape: List[int] = [],
                 transposed_data: List[int] = None,
                 transposed_shape: List[int] = None,
                 kn2row_data: List[int] = None,
                 kn2row_dimension_format: str = 'HWOI',
                 kn2row_shape: List[int] = None,) -> None:
        """Init the variable.

        If the constant is hard quantized, data is packed and the actual shape
        must be expressed with `actual_shape`.
        """
        shape = list(data.shape) if not packed else actual_shape
        self._packed = packed
        self._transposed_data = transposed_data
        self._transposed_shape = transposed_shape
        self._transposed_dimension_format = transposed_dimension_format
        self._kn2row_data = kn2row_data
        self._kn2row_dimension_format = kn2row_dimension_format
        self._kn2row_shape = kn2row_shape
        super().__init__(name, shape, dtype, {}, data, dimension_format=dimension_format)

    def run_forward(self) -> np.ndarray:
        return self._data

    @property
    def is_packed(self) -> bool:
        return self._packed

    @property
    def transposed_data(self) -> List[int]:
        """Return transposed data."""
        return self._transposed_data

    @property
    def transposed_dimension_format(self) -> str:
        return self._transposed_dimension_format

    @property
    def transposed_shape(self) -> List[int]:
        return self._transposed_shape

    @property
    def kn2row_data(self) -> List[int]:
        return self._kn2row_data

    @property
    def kn2row_dimension_format(self) -> str:
        return self._kn2row_dimension_format

    @property
    def kn2row_shape(self) -> List[int]:
        return self._kn2row_shape


class Output(Variable):
    """Output class."""

    _input_names = ['input']
    _output_names: List[str] = []

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = ''
                 ) -> None:
        """Init the output variable."""
        data = np.zeros(shape, dtype=dtype.nptype())
        super().__init__(name, shape, dtype, input_ops, data, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(len(self._input_ops) == 1, f'output {self.name} has {len(self._input_ops)} inputs.')
        self._assert(self._input_ops['input'].shape == self.shape,
                     f'Shape mismatch at {self.op_type} "{self.name}"')
        self._assert(self._input_ops['input'].dtype == self.dtype,
                     f'Type mismatch at {self.op_type} "{self.name}"')



