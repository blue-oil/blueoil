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


class SpaceToDepth(Operator):
    """Space to Depth operator.

    Input
    -----
    input
        Input tensor

    Output
    ------
    output
        A tensor with reduced height and width and increased depth

    Attributes (optional constructor parameters)
    ----------
    block_size : integer
        Input block size
    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 block_size: int = 2) -> None:
        """Init the quantization operator."""
        self._block_size = block_size
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        """
        This check the following constraints:
            Output depth must be
            1. (multiple of kernel_size^2 * 32) OR
            2. (kernel_size^2 * {8, 16}).
        """
        super()._check_consistency()
        if self.channel % 32 != 0:
            warnings.warn(warning_sign +
                          f" Output channels need to be multiple of 32 for {self.name} of {self.op_type}, "
                          f"but got output channel size of {self.channel}",
                          stacklevel=2)

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @property
    def block_size(self) -> np.int32:
        return self._block_size

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['input']

    @property
    def preserve_quantization(self) -> bool:
        return True


class Transpose(Operator):
    """Transpose operator.

    Transpose the input tensor similar to numpy.transpose. For example, when perm=[3, 1, 0, 2],
    given an input tensor of shape [1, 2, 3, 4], the output shape will be [4, 2, 1, 3].


    Inputs
    ------
    data
        An input tensor.

    Outputs
    -------
    transposed
        Transposed output.

    Attributes (optional constructor parameters)
    perm : list of ints
        A list of integers. By default, reverse the dimensions, otherwise permute the axes according
        to the values given.

    """

    _input_names = ['data']
    _output_names = ['transposed']

    def __init__(self, name: str, shape: List[int], dtype: DataType, input_ops: Ops,
                 perm: List[int] = [], dimension_format: str = 'NHWC') -> None:
        self._permutation = perm if perm else [i for i in range(len(shape) - 1, -1, -1)]
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(len(self.permutation) == len(self.shape),
                     f'Illegal permutation for Transpose: {self.permutation}.')
        self._assert(set(self.permutation) == set([i for i in range(len(self.shape))]),
                     f'Illegal permutation for Transpose: {self.permutation}.')
        transposed_shape = [self._input_ops['data'].shape[i] for i in self.permutation]
        self._assert(self.shape == transposed_shape,
                     f'Shape mismatch at {self.op_type} "{self.name}"')

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def permutation(self) -> List[int]:
        """Get transpose permutation in list of ints."""
        return self._permutation

    def run_forward(self) -> np.ndarray:
        self._data = self._input_ops['data'].data.transpose(self._permutation)
        return self._data

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        perm = attrs['perm']
        return [lists['data'][i] for i in perm]

    @property
    def preserve_quantization(self) -> bool:
        return True


class Reshape(Operator):
    """Reshape operator.

    Reshape the input tensor similar to numpy.reshape.

    It takes a tensor as input and an argument shape. It outputs the reshaped tensor.

    At most one dimension of the new shape can be -1. In this case, the value is inferred
    from the size of the tensor and the remaining dimensions. A dimension could also be 0,
    in which case the actual dimension value is unchanged (i.e. taken from the input tensor).

    Inputs
    ------
    data
        An input tensor.

    Outputs
    -------
    reshaped
        Reshaped data.

    """

    _input_names = ['data', 'shape']
    _output_names = ['reshaped']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the reshape operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        # check the size
        in_size = functools.reduce(lambda x, y: x * y, self.input_ops['data'].shape)
        out_size = functools.reduce(lambda x, y: x * y, self.shape)
        self._assert(in_size == out_size,
                     "the specified shape is inconsistent with that of the input.")

    def run_forward(self) -> np.ndarray:
        self._data = self.input_ops['data'].data.reshape(self.shape)
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


class Flatten(Operator):
    """Flatten class.

    Flattens the input tensor into a 2D matrix.
    If input tensor has shape [d_0, d_1, ... d_n] then the output will have shape
    [d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn].

    Inputs
    ------
    input
        A tensor of rank >= axis.

    Outputs
    -------
    output
        A 2D tensor with the contents of the input tensor, with input dimensions up to
        axis flattened to the outer dimension of the output and remaining input dimensions
        flattened into the inner dimension of the output.

    Attributes
    ----------
    axis : int
        (Default to 1) Indicate up to which input dimensions (exclusive)
        should be flattened to the outer dimension of the output.
        The value for axis must be in the range [0, R], where R is the rank
        of the input tensor. When axis = 0, the shape of the output tensor
        is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is
        (d_0, d_1, ... d_n).

    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'HWCN',
                 axis: int = 1) -> None:
        """Init the reshape operator."""
        self._axis = axis
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        axis = attrs['axis'] if attrs.get('axis') else 1

        in_shape = lists['input']
        fst = functools.reduce(lambda x, y: x * y,
                               [s for s in in_shape[:axis]]) if axis > 0 else 1
        snd = functools.reduce(lambda x, y: x * y,
                               [s for s in in_shape[axis:]])

        return [fst, snd]

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


class ConcatOnDepth(Operator):
    """Concatenation operator.

    Input
    -----
    input1
        Input tensor

    input2
        Input tensor

    input3
        Input tensor

    input4
        Input tensor

    input5
        Input tensor

    Output
    ------
    output
        A tensor which is the concatenation of the inputs in the depth axis

    Attributes (optional constructor parameters)
    ----------
    block_size : integer
        Input block size
    """

    _input_names = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the concat on depth operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


class DepthToSpace(Operator):
    """Depth to Space operator.

    Input
    -----
    input
        Input tensor

    Output
    ------
    output
        A tensor with increased height and width and decreased depth

    Attributes (optional constructor parameters)
    ----------
    block_size : integer
        Input block size
    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 block_size: int = 2) -> None:
        """Init the quantization operator."""
        self._block_size = block_size
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        """
        This check the following constraints:
            1. qunatized-packed data requires depth of input must be multiple of kernel_size^2 * 32
        """
        super()._check_consistency()
        if self.input_ops['input'].op_type == 'QTZ_linear_mid_tread_half' and \
                self.input_ops['input'].channel % 128 != 0:
            warnings.warn(warning_sign +
                          f" Input channels need to be multiple of kernel_size^2 * 32 for "
                          f"{self.name} of {self.op_type}, but got {self.input_ops['input'].channel}",
                          stacklevel=2)

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @property
    def block_size(self) -> np.int32:
        return self._block_size

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['input']

    @property
    def preserve_quantization(self) -> bool:
        return True


class ResizeNearestNeighbor(Operator):
    """Resize Nearest Neighbor operator.

    Input
    -----
    input
        Input tensor

    Output
    ------
    output
        A tensor with resized height and width and same depth

    Attributes (optional constructor parameters)
    ----------
        Align corners is not supported.
    """

    _input_names = ['input']
    _output_names = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC'
                 ) -> None:
        """Init the quantization operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        return lists['input']

    @property
    def preserve_quantization(self) -> bool:
        return True


class Split(Operator):
    """Split operator.

    Split a tensor into a list of tensors, along the specified 'axis'.

    Input
    -----
    input
        The tensor to split

    Output
    ------
    output
        Output forming list of tensors after split

    Attributes (optional constructor parameters)
    ----------
    axis : integer
        Axis to split on

    split : list of integer
        Length of each output

    """

    _input_names = ['A', 'B']
    _output_names = ['output1', 'output2', 'output3', 'output4', 'output5']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 num_split: int = 1) -> None:
        """Init the split operator."""
        self._split = num_split
        self._axis = input_ops['A'].data[0]
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(isinstance(self._split, int),
                     f'Attribute value incorrect at {self.op_type}" {self.name}"')
        self._assert(self._input_ops['B'].shape[self._axis] % self._split == 0,
                     f'Shape not divisible by {self._axis} at {self.op_type}" {self.name}"')

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def num_splits(self) -> int:
        return self._split

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        in_shape = lists['B']
        out_shape = in_shape

        split = attrs['split'] if attrs.get('split') else 1
        ch_idx = format.index('C')

        if in_shape[ch_idx] % split == 0:
            out_shape[ch_idx] = int(in_shape[ch_idx] / split)

        return out_shape

    @property
    def preserve_quantization(self) -> bool:
        return True


class Pad(Operator):
    """Pad operator.
    Pad a tensor. This operation pads a tensor according to the paddings specified
    Input
    -----
    A
        The input to be padded
    B
        The padding size, this (B) is a numpy array that supports "CONSTANT" mode in
        tensorflow during importing, it has shape of [n, 2], where n is the rank of
        input A, assume input A has dimension of D the padded size of each dimension D
        of the output C is given by the formula below:
                B[D, 0] + A.dim_size(D) + B[D, 1]
        Note. currently only the channel-wise paddings are supported.

    Output
    ------
    C
        A result after being padded. Has the same type as input tensor
    """

    _input_names = ['A', 'B']
    _output_names = ['C']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the Pad operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(np.all(self.input_ops['B'].data[:-1] == 0),
                     f'{self.op_type}" {self.name}" only supports channel-wise paddings')

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__

    @property
    def is_monotonic(self) -> bool:
        return False

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        a_shape = lists['A']
        padding_constant = attrs['padding_const']
        new_shape = []
        for padding, dim in zip(padding_constant, a_shape):
            if padding is None or dim is None or any((x is None for x in padding)):
                new_shape.append(None)
            else:
                new_shape.append(sum(padding) + dim)

        return new_shape

    @property
    def preserve_quantization(self) -> bool:
        return False


class Shape(Operator):
    r"""Shape operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['input']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return False


class StridedSlice(Operator):
    r"""StridedSlice operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['input', 'begin', 'end', 'strides']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return False



