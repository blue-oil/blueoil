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


class Identity(Operator):
    """Identity operator.

    Inputs
    ------
    input
        Input tensor

    Output
    ------
    output
        Tensor to copy input

    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self, name: str, shape: List[int], dtype: DataType, input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the identity operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(self._input_ops['input'].shape == self.shape,
                     f'Shape mismatch at {self.op_type} "{self.name}"')

    def run_forward(self) -> np.ndarray:
        self._data = self._input_ops['input'].data
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return self.input_ops['input'].is_monotonic

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['input']

    @property
    def preserve_quantization(self) -> bool:
        return True


class BatchNormalization(Operator):
    """Batch normalization operator.

    Carries out batch normalization as described in the paper https://arxiv.org/abs/1502.03167.

    Inputs
    ------
    X
        The input 4-dimensional tensor.

    scale
        The scale as a 1-dimensional tensor of size C to be applied to the output.

    B
        The bias as a 1-dimensional tensor of size C to be applied to the output.

    mean
        The estimated mean (testing) as a 1-dimensional tensor of size C.

    var
        The estimated variance (testing) as a 1-dimensional tensor of size C.

    Outputs
    -------
    Y
        The output 4-dimensional tensor of the same shape as X.

    Attributes (Optional constructor parameters)
    ----------
    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    epsilon : float
        The epsilon value to use to avoid division by zero, default is 1e-5f.

    is_test : bool
        If set to True, run spatial batch normalization in test mode, default is False.

    """

    _input_names = ['X', 'scale', 'B', 'mean', 'var']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 epsilon: float = float(10 ** -5),
                 is_test: bool = False) -> None:
        """Init the batch normalization operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)
        self._epsilon = epsilon
        self.is_test = is_test
        # self.momentum = momentum
        # self.spatial = spatial

    def _check_consistency(self) -> None:
        super()._check_consistency()
        x_shape = self._input_ops['X'].shape
        message = 'BatchNorm operator has inconsistency in shapes: '
        message += f'input "X" has {x_shape}, while it has {self.shape}'
        self._assert(x_shape == self.shape, message)

    def run(self, **kwargs) -> Dict:
        """Return the forward calculation results of batch normalization.

        Currently this function is only used by threshold skipping optimization pass
        for recursively calculating thresholds of the skipping patterns.
        """
        scale = np.float64(self._input_ops['scale'].data)
        beta = np.float64(self._input_ops['B'].data)
        mean = np.float64(self._input_ops['mean'].data)
        var = np.float64(self._input_ops['var'].data)

        x_norm = (kwargs['data'] - mean) / np.sqrt(var + self.epsilon)
        kwargs['data'] = scale * x_norm + beta
        return kwargs

    def de_run(self, **kwargs) -> Dict:
        """Return the reversed calculation results of batch normalization.

        Currently this function is only used by threshold skipping optimization pass
        for recursively calculating thresholds of the skipping patterns.
        """
        scale = np.float64(self._input_ops['scale'].data)
        beta = np.float64(self._input_ops['B'].data)
        mean = np.float64(self._input_ops['mean'].data)
        var = np.float64(self._input_ops['var'].data)

        kwargs['data'] = (((kwargs['data'] - beta) / scale) * np.sqrt(var + self.epsilon)) + mean
        return kwargs

    def run_forward(self) -> np.ndarray:
        kwdata = {'data': self.input_ops['X'].data}
        data_dict = self.run(**kwdata)
        self._data = data_dict['data']
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return True

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['X']

    @property
    def _dispatch_name(self) -> str:
        return "batch_normalization"

    @property
    def preserve_quantization(self) -> bool:
        return False


class Dropout(Operator):
    """Dropout operator.

    Dropout takes one input data (Tensor) and produces two Tensor outputs, output (Tensor)
    and mask (Tensor). Y will either be a random dropout, or a simple copy of the input.
    Note that our implementation of Dropout does scaling in the training phase, so during
    testing nothing needs to be done. This operator has optional inputs/outputs.

    Inputs
    ------
    data
        The input data as Tensor.

    Outputs (1 - 2)
    -------
    output
        The output.

    mask (optional)
        The output mask.

    Attributes
    ----------
    ratio: float
        (float, default 0.5) the ratio of random dropout

    """

    _input_names = ['data']
    _output_names = ['output', 'mask']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'HWCN',
                 ratio: float = 0.5) -> None:
        """Init the reshape operator."""
        self._ratio = ratio
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    @property
    def ratio(self):
        return self._ratio

    @property
    def is_monotonic(self) -> bool:
        return False

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['data']

    @property
    def preserve_quantization(self) -> bool:
        return False


class Gather(Operator):
    r"""Gather operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['x', 'out_idx']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


class Unique(Operator):
    r"""Unique operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['x']
    _output_names = ['y', 'idx']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


class Cast(Operator):
    r"""Cast operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['x']
    _output_names = ['y']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return False


class Prod(Operator):
    r"""Prod operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['input', 'indices']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return False


class BatchNormalizationOptimized(Operator):
    """Optimized batch normalization operator.
    This operator for only inference.
    Inputs
    ------
    X
        The input 4-dimensional tensor.
    scale
        The scale as a 1-dimensional tensor of size C to be applied to the output.
    bias
        The bias as a 1-dimensional tensor of size C to be applied to the output.
    Outputs
    -------
    Y
        The output 4-dimensional tensor of the same shape as X.
    """

    _input_names = ['X', 'scale', 'bias']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the optimized batch normalization operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        x_shape = self._input_ops['X'].shape
        message = 'BatchNorm operator has inconsistency in shapes: '
        message += f'input "X" has {x_shape}, while it has {self.shape}'
        self._assert(x_shape == self.shape, message)

    def run(self, **kwargs) -> Dict:
        """Return the forward calculation results of batch normalization.
        Currently this function is only used by threshold skipping optimization pass
        for recursively calculating thresholds of the skipping patterns.
        """
        scale = np.float64(self._input_ops['scale'].data)
        bias = np.float64(self._input_ops['bias'].data)

        kwargs['data'] = scale * kwargs['data'] + bias
        return kwargs

    def run_forward(self) -> np.ndarray:
        kwdata = {'data': self.input_ops['X'].data}
        data_dict = self.run(**kwdata)
        self._data = data_dict['data']
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return True

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['X']

    @property
    def _dispatch_name(self) -> str:
        return "batch_normalization"

    @property
    def preserve_quantization(self) -> bool:
        return False
