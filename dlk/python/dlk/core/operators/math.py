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
from .base import *

import warnings
from itertools import dropwhile
from typing import Any, Dict, Optional

from core.view import View
from utils import classproperty

Ops = Dict[str, 'Operator']


class Add(Operator):
    """Add operator.

    Performs element-wise binary addition (with Numpy-style broadcasting support).
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    Inputs
    ------
    A
        First operand.

    B
        Second operand.

    Outputs
    -------
    C
        Result, has same element type as two inputs

    """

    _input_names = ['A', 'B']
    _output_names = ['C']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init add operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        # No. 1 ... Pad additional shape if the lengths are not equal
        a_shape = self.input_ops['A'].shape
        b_shape = self.input_ops['B'].shape
        len_diff = len(a_shape) - len(b_shape)
        if (len_diff != 0):
            head = [1 for i in range(abs(len_diff))]
            if len_diff > 0:
                b_shape = [*head, *b_shape]
            else:
                a_shape = [*head, *a_shape]
        # No. 2 ... Check the numbers at each dimension are the same, or one of them has 1
        length = len(a_shape)
        ash = self._input_ops["A"].shape
        bsh = self._input_ops["B"].shape
        for i in range(length):
            message = f'operands could not be broadcast together with shapes {ash} {bsh}'
            self._assert(a_shape[i] == b_shape[i] or a_shape[i] == 1 or b_shape[i] == 1, message)
        # No. 3 ... Check the output shape is consistent with the two operands
        output_shape = [max(a, b) for a, b in zip(a_shape, b_shape)]
        message = f'output shape {self.shape} does not match with two operands {ash} {bsh}'
        self._assert(output_shape == self.shape, message)

        # we only implement depth-wise broadcast on C
        ash_reduced = [x for x in dropwhile(lambda x: x == 1, ash)]
        bsh_reduced = [x for x in dropwhile(lambda x: x == 1, bsh)]
        self.is_depthwise = ((len(ash_reduced) == 1) or (len(bsh_reduced) == 1)) and\
                            (ash_reduced[-1] == bsh_reduced[-1])

    def run_forward(self) -> np.ndarray:
        a = self._input_ops['A'].data
        b = self._input_ops['B'].data
        self._data = a + b
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return False

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        # No. 1 ... Pad additional shape if the lengths are not equal
        a_shape = lists['A']
        b_shape = lists['B']
        len_diff = len(a_shape) - len(b_shape)
        if (len_diff != 0):
            head = [1 for i in range(abs(len_diff))]
            if len_diff > 0:
                b_shape = [*head, *b_shape]
            else:
                a_shape = [*head, *a_shape]
        # No. 2 ... Check the numbers at each dimension are the same, or one of them has 1
        length = len(a_shape)
        ash = lists["A"]
        bsh = lists["B"]
        for i in range(length):
            message = f'operands could not be broadcast together with shapes {ash} {bsh}'
            assert a_shape[i] == b_shape[i] or a_shape[i] == 1 or b_shape[i] == 1, message
        # No. 3 ... calc the output shape is consistent with the two operands
        output_shape = [max(a, b) for a, b in zip(a_shape, b_shape)]

        return output_shape

    @property
    def preserve_quantization(self) -> bool:
        return False


class Gemm(Operator):
    """Gemm operator.

    General Matrix multiplication:
    https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

    A' = transpose(A) if transA else A

    B' = transpose(B) if transB else B

    Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape
    [M, K] or [K, M], input tensor B has shape [K, N] or [N, K], input tensor C
    is broadcastable to shape [M, N], and output tensor Y has shape [M, N].
    A will be transposed before doing the computation if attribute transA is
    True, same for B and transB. This operator supports unidirectional
    broadcasting (tensor C should be unidirectional broadcastable to tensor A * B);
    for more details please check the doc.

    Inputs
    ------
    A
        Input tensor A. The shape of A should be [M, K] if transA is False, or
        [K, M] if transA is True.

    B
        Input tensor B. The shape of B should be (K, N) if transB is False, or
        [N, K] if transB is True.

    C
        Input tensor C. The shape of C should be unidirectional broadcastable
        to [M, N].

    Outputs
    -------
    Y
        Output tensor of shape [M, N].

    Attributes
    ----------
    alpha
        Scalar multiplier for the product of input tensors A * B

    beta
        Scalar multiplier for input tensor C

    transA
        Whether A should be transposed

    transB
        Whether B should be transposed

    """

    _input_names = ['A', 'B', 'C']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'HWCN',
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 transA: bool = False,
                 transB: bool = False,
                 ) -> None:
        """Init the reshape operator."""
        self._alpha = alpha
        self._beta = beta
        self._transA = transA
        self._transB = transB
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        transA = attrs['transA']
        transB = attrs['transB']

        M = lists['A'][0] if not transA else lists['A'][1]
        N = lists['A'][1] if not transB else lists['B'][0]

        return [M, N]

    @property
    def preserve_quantization(self) -> bool:
        return False


class Mul(Operator):
    """Mul operator.

    Performs element-wise binary multiplication (with Numpy-style broadcasting support).
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.

    Inputs
    ------
    A
        First operand.

    B
        Second operand.

    Outputs
    -------
    C
        Result, has same element type as two inputs

    """

    _input_names = ['A', 'B']
    _output_names = ['C']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:

        """Init add operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        # No. 1 ... Pad additional shape if the lengths are not equal
        a_shape = self.input_ops['A'].shape
        b_shape = self.input_ops['B'].shape
        len_diff = len(a_shape) - len(b_shape)
        if (len_diff != 0):
            head = [1 for i in range(abs(len_diff))]
            if len_diff > 0:
                b_shape = [*head, *b_shape]
            else:
                a_shape = [*head, *a_shape]
        # No. 2 ... Check the numbers at each dimension are the same, or one of them has 1
        length = len(a_shape)
        ash = self._input_ops["A"].shape
        bsh = self._input_ops["B"].shape
        for i in range(length):
            message = f'operands could not be broadcast together with shapes {ash} {bsh}'
            self._assert(a_shape[i] == b_shape[i] or a_shape[i] == 1 or b_shape[i] == 1, message)
        # No. 3 ... Check the output shape is consistent with the two operands
        output_shape = [max(a, b) for a, b in zip(a_shape, b_shape)]
        message = f'output shape {self.shape} does not match with two operands {ash} {bsh}'
        self._assert(output_shape == self.shape, message)

        # we only implement depth-wise broadcast on C
        ash_reduced = [x for x in dropwhile(lambda x: x == 1, ash)]
        bsh_reduced = [x for x in dropwhile(lambda x: x == 1, bsh)]
        self.is_depthwise = ((len(ash_reduced) == 1) or (len(bsh_reduced) == 1)) and \
                            (ash_reduced[-1] == bsh_reduced[-1])

    def run_forward(self) -> np.ndarray:
        a = self._input_ops['A'].data
        b = self._input_ops['B'].data
        self._data = a * b
        return self._data

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return False

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        # No. 1 ... Pad additional shape if the lengths are not equal
        a_shape = lists['A']
        b_shape = lists['B']
        len_diff = len(a_shape) - len(b_shape)
        if (len_diff != 0):
            head = [1 for i in range(abs(len_diff))]
            if len_diff > 0:
                b_shape = [*head, *b_shape]
            else:
                a_shape = [*head, *a_shape]
        # No. 2 ... Check the numbers at each dimension are the same, or one of them has 1
        length = len(a_shape)
        ash = lists["A"]
        bsh = lists["B"]
        for i in range(length):
            message = f'operands could not be broadcast together with shapes {ash} {bsh}'
            assert a_shape[i] == b_shape[i] or a_shape[i] == 1 or b_shape[i] == 1, message
        # No. 3 ... calc the output shape is consistent with the two operands
        output_shape = [max(a, b) for a, b in zip(a_shape, b_shape)]

        return output_shape


class Maximum(Operator):
    """Maximum operator.

    Performs element-wise max() operation.

    Inputs
    ------
    A
        First operand.

    B
        Second operand.

    Outputs
    -------
    C
        Result, has same shape and data type than the inputs

    """

    _input_names = ['A', 'B']
    _output_names = ['C']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init Maximum operator."""
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
        return False


class MatMul(Operator):
    """Matrix Multiplication operator.
    Matrix product. Multiplies matrix a by matrix b, producing a * b
    Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)`` for ``np.matmul``.
    Input
    -----
    A
        2-dimensional matrix A
    B
        2-dimensional matrix B
    Output
    ------
    C
        Matrix multiply results from A * B
    """

    _input_names = ['A', 'B']
    _output_names = ['C']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the MatMul operator."""
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        a_shape = self._input_ops["A"].shape
        b_shape = self._input_ops["B"].shape
        message = f'operands could not be scalar, use * instead'
        # scalars are rejected
        self._assert(len(a_shape) != 1 or len(b_shape) != 1, message)
        # Shape alignment
        message = f'operand shapes are not aligned'
        self._assert(a_shape[1] == b_shape[0], message)

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
        b_shape = lists['B']
        output_shape = [a_shape[0], b_shape[1]]
        return output_shape

    def run_forward(self) -> np.ndarray:
        a_data = self.input_ops['A'].data
        b_data = self.input_ops['B'].data
        self._data = np.matmul(a_data, b_data)
        return self._data

    @property
    def preserve_quantization(self) -> bool:
        return False


class Minimum(Operator):
    r"""Minimum operator.

    Inputs
    ------
    input
        The input tensor.

    Outputs
    -------
    output
        The output.

    """

    _input_names = ['x', 'y']
    _output_names = ['output']

    def _check_consistency(self) -> None:
        super()._check_consistency()

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


