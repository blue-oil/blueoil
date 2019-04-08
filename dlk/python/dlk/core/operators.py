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
"""Defenition of operators."""
import functools
import copy
from itertools import dropwhile
from typing import cast, Any, Dict, Optional, TYPE_CHECKING
from core.view import View
from utils import classproperty
from abc import abstractmethod
from .data_types import *

if TYPE_CHECKING:
    import core.operators as ops

Ops = Dict[str, 'Operator']
OutOps = Dict[str, List['Operator']]


class Operator(object):
    """Base class of operators."""

    _input_names: List[str] = ['input']
    _output_names: List[str] = ['output']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC') -> None:
        """Init the operator."""
        self._name: str = name
        self._input_ops: Ops = input_ops
        self._output_ops: OutOps = {}
        self._dtype = dtype
        self._data = np.zeros(shape, dtype=dtype.nptype())
        self.__update_shape(shape, dimension_format)
        self.view: View = View(self)
        self.__connect_to_outputs()
        self._check_consistency()
        self._rank = len(shape)
        self._available_buffer = ''

    def __update_shape(self, shape: List[int], dimension_format: str) -> None:
        self._shape: List[int] = shape
        self._dimension_format = dimension_format
        self.index_H = self._dimension_format.index('H') if 'H' in self._dimension_format else None
        self.index_W = self._dimension_format.index('W') if 'W' in self._dimension_format else None
        self.index_N = self._dimension_format.index('N') if 'N' in self._dimension_format \
            else self._dimension_format.index('O') if 'O' in self._dimension_format else None
        self.index_C = self._dimension_format.index('C') if 'C' in self._dimension_format \
            else self._dimension_format.index('I') if 'I' in self._dimension_format else None

    def __connect_to_outputs(self) -> None:
        """Connect input operators' outputs to this object."""
        for ip in self._input_ops.values():
            if ip.op_type == 'Split':
                for x in ip.output_names:
                    if x not in ip._output_ops.keys():
                        ip.add_output(x, self)
                        break
            else:
                key = ip.output_names[0]
                ip.add_output(key, self)

    def _assert(self, predicate: bool, message: str = '') -> None:
        """Assert a predicate. When it fails, raise an error.

        This is a substitute for an `assert` statement. The `assert` is
        not checked in byte-compiled code, but this is always checked.

        Parameters
        ----------
        predicate : bool
            Assertion to be true

        message : str
            Error message in the failure of the assertion

        """
        if not predicate:
            raise AssertionError(message) if message else AssertionError()

    def _check_consistency(self) -> None:
        """Check data consistency in the initialization."""
        # check the input ops
        self._assert(set(self._input_ops.keys()).issubset(set(self._input_names)),
                     f"Operator inputs must consist of {', '.join(self._input_names)}")

    def equals(self, other: Any) -> bool:
        """Return if these two objects are equivalent."""
        if other is None or not isinstance(other, Operator):
            print(f'{self.name} has different type.')
            return False

        eq_type = self.op_type == other.op_type
        if not eq_type:
            print(f'{self.name} and {other.name} have different type: {self.op_type} and {other.op_type}')

        eq_shape = self.shape == other.shape
        if not eq_shape:
            print(f'{self.name} and {other.name} have different shape: {self.shape} and {other.shape}')

        eq_dtype = self.dtype == other.dtype
        if not eq_dtype:
            print(f'{self.name} and {other.name} have different dtype: {self.dtype} and {other.dtype}')

        eq_dim = self._dimension_format.replace('I', 'C').replace('O', 'N') \
            == other._dimension_format.replace('I', 'C').replace('O', 'N')
        if not eq_dim:
            print(f'{self.name} and {other.name} have different dimension: {self.dimension} and {other.dimension}')

        eq_data = eq_shape and np.allclose(self.data, other.data)
        if not eq_data:
            print(f'{self.name} and {other.name} have different data: {self.data} and {other.data}')

        return eq_type and eq_shape and eq_dtype and eq_dim and eq_data

    @property
    def name(self) -> str:
        """Return name. This must be a unique name in the graph."""
        return self._name

    @property
    def op_type(self) -> str:
        """Return the operation type."""
        return type(self).__name__

    @property
    def input_ops(self) -> Ops:
        """Return a dict of input operators.

        Returns
        -------
        ops : Dict of operators
            Collection of input operators in a dictionary format.
            The keys are input symbols, which can be taken from `input_names` property.

        """
        return self._input_ops

    @classproperty
    def input_names(cls) -> List[str]:
        """Return the input key names the operator provides.

        For example, `Conv` has two inputs, 'X' for the input data and 'W' for the weight.
        So `Conv.input_names` returns the list `['X', 'W']`.

        Returns
        -------
        names : list of str
            List of key names

        """
        return cls._input_names

    @property
    def input_nodes(self) -> List['Operator']:
        """Return a list of input operators in proper order (original protobuf argument order).

        Returns
        -------
        ops : List of operators
            This list is already ordered following the order of the arguments in the original
             protobuf operators (positional order in the list of arguments).

        """
        return [self._input_ops[i] for i in self.input_names if self.input_ops.get(i)]

    @property
    def output_ops(self) -> OutOps:
        """Return a dict of output operators.

        Returns
        -------
        op_lists : Dict of list of operators
            Collection of (list of) output operators in a dictionary format.
            The keys are output symbols, which can be taken from `output_names` property.

        """
        return self._output_ops

    @property
    def output_op_list(self) -> List['Operator']:
        """Return a list of output operators.

        Returns
        -------
        ops : list of operators
            List of output operators.

        """
        return sum(list(self._output_ops.values()), [])

    @classproperty
    def output_names(cls) -> List[str]:
        """Return the output key names the operator provides.

        For example, `Conv` has one output 'Y'.
        So `Conv.output_names` returns the list `['Y']`.

        Returns
        -------
        names : list of str
            List of key names

        """
        return cls._output_names

    def add_input(self, ident: str, node: 'Operator') -> None:
        """Add input node.

        Parameters
        ----------
        ident : str
            key name of the input. This has to be in list `input_names`.

        node : Operator
            Node to be registered as the input.

        """
        self._assert(ident in self._input_names, "Illegal input name")
        self._input_ops[ident] = node

    def add_inputs(self, inputs: Ops) -> None:
        """Add input (possibly multiple) nodes at a once.

        Parameters
        ----------
        outputs : Dict of str to Operator
            Collection of pair of key name and a operator to be registered as the input.
            All the key names have to be in list `input_names`.

        """
        assert set(inputs.keys()).issubset(set(self._input_names)), "Illegal output names included"
        self._input_ops.update(inputs)

    def add_output(self, ident: str, node: 'Operator') -> None:
        """Add output node.

        Parameters
        ----------
        ident : str
            key name of the output. This has to be in list `output_names`.

        node : Operator
            Node to be registered as the output.

        """
        self._assert(ident in self._output_names, "Illegal output name")
        lst: Optional[List['Operator']] = self._output_ops.get(ident)
        if lst is not None:
            lst.append(node)
        else:
            self._output_ops[ident] = [node]

    def add_outputs(self, outputs: OutOps) -> None:
        """Add output (possibly multiple) nodes at a once.

        Parameters
        ----------
        outputs : Dict of str to list of Operators
            Collection of pair of key name and a list of operators to be registered as the output.
            All the key names have to be in list `output_names`.

        """
        assert set(outputs.keys()).issubset(set(self._output_names)), f"Illegal output names included"
        for n in outputs.keys():
            lst = self._output_ops.get(n)
            if lst is not None:
                lst += [x for x in outputs[n] if x not in lst]
            else:
                self._output_ops[n] = list(outputs[n])

        self._output_ops.update(outputs)

    def remove_input(self, ident: str) -> None:
        """Remove an input node.

        Parameters
        ----------
        ident : str
            Key name of the input node to be removed.
            This key is in `input_names`, not the name of the operator.

        """
        self._input_ops.pop(ident)

    def remove_output(self, ident: str) -> None:
        """Remove an output node.

        Parameters
        ----------
        ident : str
            Key name of the output node to be removed.
            This key is in `output_names`, not the name of the operator.

        """
        self._output_ops.pop(ident)

    @property
    def shape(self) -> List[int]:
        """Get the shape defined in this node."""
        return self._shape

    @shape.setter
    def shape(self, v: List[int]) -> None:
        """Set the shape defined in this node."""
        self._shape = v

    @property
    def dtype(self) -> DataType:
        """Get the data type defined in this node."""
        return self._dtype

    @dtype.setter
    def dtype(self, v: DataType) -> None:
        """Set the data type defined in this node."""
        self._dtype = v

    @property
    def ndims(self) -> int:
        """Get the number of dimension defined in this node."""
        return len(self._shape)

    @property
    def shape(self) -> List[int]:
        return self._shape

    @shape.setter
    def shape(self, v: List[int]) -> None:
        self._shape = v

    @property
    def dtype(self) -> DataType:
        return self._dtype

    @dtype.setter
    def dtype(self, v: DataType) -> None:
        self._dtype = v

    @property
    def ndims(self) -> int:
        return len(self._shape)

    @property
    def dimension(self) -> str:
        """Return dimension in string.

        This dimension consists of 'N', 'C', 'H', and 'W', where 'N' is the number of batch size,
        'C' is the number of channels, 'H' and 'C' are the height and the weight in the 2-D image.
        """
        return self._dimension_format

    @property
    def size(self) -> int:
        """Get the whole size of the output data."""
        import operator
        pred = functools.partial(functools.reduce, operator.mul)
        return int(pred(self._shape))  # type: ignore

    @property
    def is_variable(self) -> bool:
        """Return if this node is a variable node (i.e. Input or Output)."""
        return False

    @property
    def is_scalar(self) -> bool:
        """Return if this node is a scalar node (i.e. `size == 1`)."""
        return self.size == 1

    @property
    def height(self) -> int:
        """Get the size of height in the shape."""
        if self.index_H is not None:
            return self.shape[self.index_H]
        else:
            raise ValueError(f'Operator {self.name} does not have the height property.')

    @property
    def width(self) -> int:
        """Get the size of width in the shape."""
        if self.index_W is not None:
            return self.shape[self.index_W]
        else:
            raise ValueError(f'Operator {self.name} does not have the width property.')

    @property
    def channel(self) -> int:
        """Get the number of channels in the shape."""
        if self.index_C is not None:
            return self.shape[self.index_C]
        else:
            raise ValueError(f'Operator {self.name} does not have the channel property.')

    @property
    def batchsize(self) -> int:
        """Get the number of batch size in the shape."""
        if self.index_N is not None:
            return self.shape[self.index_N]
        else:
            raise ValueError(f'Operator {self.name} does not have the batchsize property.')

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def available_buffer(self) -> str:
        return self._available_buffer

    @available_buffer.setter
    def available_buffer(self, v: str) -> None:
        self._available_buffer = v

    def transpose(self, perm: List[int]) -> None:
        """Transpose the shape and format. This operation is destructive."""
        self._assert(len(set(perm)) == len(self._shape), "Illegal permutation spacified.")
        self._assert(max(perm) == len(self._shape) - 1, "Illegal permutation spacified.")
        self._assert(min(perm) == 0, "Illegal permutation spacified.")

        # change the shape
        new_shape: List[int] = [self._shape[i] for i in perm]

        # change the format
        new_format: str = functools.reduce(
            lambda x, y: x + y, [self._dimension_format[i] for i in perm])

        # update
        self.__update_shape(new_shape, new_format)

    @property
    def data(self) -> np.ndarray:
        """Get the output data.

        This value is valid only after `run_forward()` or some value has assigned with the setter.
        """
        return self._data

    @property
    def is_monotonic(self) -> bool:
        raise NotImplementedError(f'operator {self.name} is monotonic or not?')

    def run(self, **kwargs) -> Dict:
        """The intermediate runtime, run the operator with external data

        This is actually an abstract method and should be overrided.
        """
        raise NotImplementedError('run is not implemented yet')

    def run_forward(self) -> np.ndarray:
        """Run the operator, calculate and set the result.

        This is actually an abstract method and should be overrided.
        """
        raise NotImplementedError(
            f'operator {self.op_type} does not have runtime implemenatation yet.')

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__.lower()

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        """Infer its output shape from inputs' shapes.

        This is actually an abstract method and should be overrided.
        """
        raise NotImplementedError(f'operator {cls.__name__} cannot infer its shape.')

    @property
    def preserve_quantization(self) -> bool:
        """whether to preserve the operator for quantization"""
        raise NotImplementedError(
            f'Preservation for quantization of operator {self.op_type} is not defined.')


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
                 dimension_format: str = 'NHWC',
                 packed: bool = False,
                 actual_shape: List[int] = [],
                 transposed_data: List[int] = None) -> None:
        """Init the variable.

        If the constant is hard quantized, data is packed and the actual shape
        must be expressed with `actual_shape`.
        """
        shape = list(data.shape) if not packed else actual_shape
        self._packed = packed
        self._transposed_data = transposed_data
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
        """Init the ouput variable."""
        data = np.zeros(shape, dtype=dtype.nptype())
        super().__init__(name, shape, dtype, input_ops, data, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(len(self._input_ops) == 1, f'output {self.name} has {len(self._input_ops)} inputs.')
        self._assert(self._input_ops['input'].shape == self.shape,
                     f'Shape mismatch at {self.op_type} "{self.name}"')
        self._assert(self._input_ops['input'].dtype == self.dtype,
                     f'Type mismatch at {self.op_type} "{self.name}"')


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

        This is actually an abstract method and should be overrided.
        """
        raise NotImplementedError(
            f'operator {self.op_type} need to implement the binarizer method')


class QTZ_binary_mean_scaling(Quantizer):
    """Quantization operator using binary scaling.

    Input
    -----
    input
        Input tensor, which must have float falues.

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
        super()._check_consistency()

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


class Conv(Operator):
    """Convolution operator.

    The convolution operator consumes an input tensor and a weight, and computes the output.
    Currently this is only for 2-D images.

    Inputs
    ------
    X
        Input data tensor from previous layer. Note that this is for the 2D image.

    W
        The weight tensor that will be used in the convolutions.

    B (Optional)
        1D bias.

    Outputs
    -------
    Y
        Output data tensor that contains the result of the convolution.
        The output dimensions are functions of the kernel size, stride size, and pad lengths.

    Attributes (Optional constructer parameters)
    ----------
    kernel_shape : list of ints
        The shape of the convolution kernel. If not present, should be inferred from input W.

    kernel_dimensions : int
        The dimension of the input. The default value is 2, which means 2-D image.

    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    kernel_dim_format : str
        Dimension denotation, which must consists of 'H' and 'W', where 'H' and 'W' are the
        height and width of input image. The default is 'HW'.

    dilations : list of ints
        Dilation value along each axis of the filter. If not present, the dilation defaults to 1
        along each axis.

    pads : list of ints
        Padding for the beginning and ending along each axis, it can take any value greater than
        or equal to 0. The value represent the number of pixels added to the beginning and end
        part of the corresponding axis.
        `pads` format should be as follow [x1_begin, x2_begin, x1_end, x2_end], where
        xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number
        of pixels added at the end of axis `i`.
        If not present, the padding defaults to 0 along start and end of each axis.

    strides : list of ints
        Stride along each axis. If not present, the stride defaults to 1 along each axis.

    quantized : bool
        Whether it is quantized. If not present, the switch defaults to False.

    thresholds : list of floats
        Threshold values that are used in threshold skipping. If not present, this defaults to
        an empty list. Ignored if `quantized` is not true.

    """

    _input_names = ['X', 'W', 'B']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 kernel_shape: List[int] = [],
                 kernel_dimensions: int = 2,
                 dimension_format: str = 'NHWC',
                 kernel_dim_format: str = 'HW',
                 dilations: List[int] = [1, 1],
                 pads: List[int] = [0, 0, 0, 0],
                 strides: List[int] = [1, 1],
                 quantized: bool = False,
                 thresholds: List[float] = []) -> None:

        # currently, only 2-D is supported.
        if kernel_dimensions != 2:
            raise NotImplementedError(f"Convolution for {kernel_dimensions}-D is not defined!")

        self._num_dimensions = kernel_dimensions
        self._dilations = dilations
        self.kernel_index_H = kernel_dim_format.index('H') if 'H' in kernel_dim_format else None
        self.kernel_index_W = kernel_dim_format.index('W') if 'W' in kernel_dim_format else None
        if self.kernel_index_H is None or self.kernel_index_W is None:
            ValueError(f'kernel dimension format {kernel_dim_format} is not supported.')
        w = input_ops['W']
        k_list = [w.height, w.width]
        perm: List[int] = [self.kernel_index_H, self.kernel_index_W]  # type: ignore
        self.kernel_shape = kernel_shape if kernel_shape else [k_list[i] for i in perm]
        self._kernel_dim_format = kernel_dim_format
        self._pads = pads
        self._strides = strides
        self._is_quantized = quantized
        self._a_quantizer: List['Quantizer'] = []
        self._quantizer: Optional['Quantizer'] = None
        self._thresholds = thresholds
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)
        # if kernel shape is not assigned, estimate kernel shape from input W's shape

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(len(self.shape) == self._num_dimensions + 2,
                     f'{self.name} has illegal shape {self.shape}')
        self._assert(len(self.kernel_shape) == self._num_dimensions,
                     f'Illegal kernel shape: {self.kernel_shape} for {self._num_dimensions}-D.')
        self._assert(len(self._kernel_dim_format) == self._num_dimensions)
        self._assert(len(self.dilations) == self._num_dimensions)
        self._assert(len(self.pads) == self._num_dimensions + 2)
        self._assert(len(self.strides) == self._num_dimensions)
        self._assert(len(self.dimension) == len(self.shape))

        # check the shape consistency
        if not self._is_quantized:
            in_H = self._input_ops['W'].height
            in_W = self._input_ops['W'].width
            mH = f'The kernel height {self.kernel_height} does not match the weight height {in_H}'
            mH += f' at operator {self.name}.'
            mH += f'\nThe kernel shape is {self.kernel_shape}, and the weight shape is {self._input_ops["W"].shape}.'
            mH += f'¥nThe weight format is {self._input_ops["W"].dimension}.'
            self._assert(in_H == self.kernel_height, mH)
            mW = f'The kernel width {self.kernel_width} does not match the weight width {in_W} at operator {self.name}'
            mW += f'\nThe kernel shape is {self.kernel_shape}, and the weight shape is {self._input_ops["W"].shape}.'
            mW += f'\nThe weight format is {self._input_ops["W"].dimension}.'
            self._assert(in_W == self.kernel_width, mW)
        if self.kernel_index_H is not None and self.index_H is not None:
            pad_H = self.pads[self.kernel_index_H] + \
                self.pads[self.kernel_index_H + self._num_dimensions]
            stride_H = self.strides[self.kernel_index_H]
            dilation_H = self.dilations[self.kernel_index_H]
            # print(self.name, ' input dimension: ', self.input_ops['X'].dimension)
            # print(self.name, ' input shape: ', self.input_ops['X'].shape)
            # print(self.name, ' input height: ', self.input_ops['X'].height)
            # print(self.name, ' weight shape: ', self.input_ops['W'].shape)
            # print(self.name, ' weight height: ', self.input_ops['W'].height)
            # print(self.name, ' kernel height: ', self.kernel_height)
            # print(self.name, ' pad_H: ', pad_H)
            # print(self.name, ' stride_H: ', stride_H)
            # print(self.name, ' output height: ', self.height, ' (', self.index_H, ' in ', self.dimension, ')')
            output_H_base = self.input_ops['X'].height + pad_H - \
                (self.kernel_height + 2 * (dilation_H - 1))
            # print(self.name, ' output_H_base ', output_H_base)
            output_H, output_H_rest = divmod(output_H_base, stride_H)
            output_H += 1
            message = f'Conv operator {self.name} does not match the height:'
            message += f' inferred as {output_H} but got {self.height}.'
            self._assert(output_H == self.height, message)
            # self._assert(output_H_rest == 0,
            #              f'Conv operator {self.name} should adjust the height pad to plus {output_H_rest}.')
            if output_H_rest > 0:
                print(f'mispadding height at {self.name}: {output_H_rest}')

        if self.kernel_index_W is not None and self.index_W is not None:
            pad_W = self.pads[self.kernel_index_W] + \
                self.pads[self.kernel_index_W + self._num_dimensions]
            stride_W = self.strides[self.kernel_index_W]
            dilation_W = self.dilations[self.kernel_index_W]
            # print(self.name, ' input shape: ', self.input_ops['X'].shape)
            # print(self.name, ' input width: ', self.input_ops['X'].width)
            # print(self.name, ' weight shape: ', self.input_ops['W'].shape)
            # print(self.name, ' weight width: ', self.input_ops['W'].width)
            # print(self.name, ' pad_W: ', pad_W)
            # print(self.name, ' stride_W: ', stride_W)
            # print(self.name, ' output width: ', self.width, ' (', self.index_W, ' in ', self.dimension, ')')
            output_W_base = self.input_ops['X'].width + pad_W - \
                (self.kernel_width + 2 * (dilation_W - 1))
            output_W, output_W_rest = divmod(output_W_base, stride_W)
            output_W += 1
            message = f'Conv operator {self.name} does not match the width:'
            message += f' inferred as {output_W} but got {self.width} in {self.dimension} format.\n'
            message += f'The shape is {self.shape}.'
            self._assert(output_W == self.width, message)
            # self._assert(output_W_rest == 0,
            #              f'Conv operator {self.name} should adjust the width pad to plus {output_W_rest}.')
            if output_W_rest > 0:
                print(f'mispadding width at {self.name}: {output_W_rest}')

    @property
    def kernel_dimensions(self) -> int:
        """Get the number of dimensions."""
        return self._num_dimensions

    @property
    def dilations(self) -> List[int]:
        """Get dilations."""
        return self._dilations

    @property
    def pads(self) -> List[int]:
        """Get pads."""
        return self._pads

    @property
    def strides(self) -> List[int]:
        """Get strides."""
        return self._strides

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def is_quantized(self) -> bool:
        """Return if this operator is quantized.

        Currently it always returns False, as quantized version is not supported yet.
        """
        return self._is_quantized

    @is_quantized.setter
    def is_quantized(self, val: bool) -> None:
        self._is_quantized = val

    @property
    def scaling_factor(self) -> float:
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, val: float) -> None:
        self._scaling_factor = val

    @property
    def a_quantizer(self) -> List[Quantizer]:
        return list(self._a_quantizer)

    @a_quantizer.setter
    def a_quantizer(self, op_lst: List[Quantizer]) -> None:
        self._a_quantizer = list(op_lst)

    @property
    def quantizer(self) -> Optional[Quantizer]:
        return self._quantizer

    @quantizer.setter
    def quantizer(self, op: Optional[Quantizer]) -> None:
        self._quantizer = op

    @property
    def kernel_height(self) -> int:
        """Return the height in the kernel shape."""
        if self.kernel_index_H is not None:
            return self.kernel_shape[self.kernel_index_H]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_height property.')

    @property
    def kernel_width(self) -> int:
        """Return the weight in the kernel shape."""
        if self.kernel_index_W is not None:
            return self.kernel_shape[self.kernel_index_W]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_width property.')

    # @property
    # def kernel_channels(self) -> int:
    #     if not self.is_quantized:
    #         return self.kernel_shape[self.index_C]
    #     else:
    #         raise NotImplementedError

    # @property
    # def kernel_batchsize(self) -> int:
    #     if not self.is_quantized:
    #         return self.kernel_shape[self.index_N]
    #     else:
    #         raise NotImplementedError

    @property
    def has_thresholds(self) -> bool:
        return self.is_quantized and len(self._thresholds) > 0

    @property
    def thresholds(self) -> List[float]:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, val: List[float]) -> None:
        self._thresholds = val

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        """Infer its shape from inputs' shapes."""
        idx_N = input_formats[0].index('N')  # from input X
        idx_C = input_formats[1].index('O')  # from weight W
        idx_H = input_formats[0].index('H')  # from input X
        idx_W = input_formats[0].index('W')  # from input X

        N = lists['X'][idx_N]
        C = lists['W'][idx_C]

        # calc H and W
        dilations = attrs['dilations']
        pads = attrs['pads']
        strides = attrs['strides']
        kernel_shape = attrs['kernel_shape']

        # H
        pads_H = pads[0] + pads[2]
        input_H = lists['X'][idx_H] + pads_H
        window_H = kernel_shape[0] + 2 * (dilations[0] - 1)
        stride_H = strides[0]
        H, rest_H = divmod((input_H - window_H), stride_H)
        H += 1
        # assert rest_H == 0, f'differfence in height: {rest_H} at {cls.__name__}'

        # W
        pads_W = pads[1] + pads[3]
        input_W = lists['X'][idx_W] + pads_W
        window_W = kernel_shape[1] + 2 * (dilations[1] - 1)
        stride_W = strides[1]
        W, rest_W = divmod((input_W - window_W), stride_W)
        W += 1
        # assert rest_W == 0, f'differfence in width: {rest_W} at {cls.__name__}'

        NCHW = [N, C, H, W]
        return [NCHW[i] for i in [format.index(s) for s in 'NCHW']]

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
        kwargs['data'] = np.round(in_data * n / max_value).astype(np.int32)
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
                 input_ops: Ops) -> None:
        """Init add operator."""
        super().__init__(name, shape, dtype, input_ops)

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


class Pool(Operator):
    """Pooling operator.

    This is a base class and must not be instantiated directly.

    """

    _input_names = ['X']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 kernel_shape: List[int] = [2, 2],
                 kernel_dim_format: str = 'HW',
                 dimensions: int = 2,
                 pads: List[int] = [0, 0, 0, 0],
                 strides: List[int] = [1, 1]) -> None:
        """Init the pooling operator."""
        if dimensions != 2:
            raise NotImplementedError

        self.kernel_dims = dimensions
        self.kernel_dim_format = kernel_dim_format
        self.kernel_shape = kernel_shape
        self._pads = pads
        self.strides = strides
        self.kernel_index_H = kernel_dim_format.index('H') if 'H' in kernel_dim_format else None
        self.kernel_index_W = kernel_dim_format.index('W') if 'W' in kernel_dim_format else None
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()

        self._assert(len(self.kernel_shape) == self.kernel_dims, 'Illegal kernel shape.')
        self._assert(len(self.kernel_dim_format) == self.kernel_dims,
                     'Illegal kernel dimension format.')
        self._assert(len(self._pads) == self.kernel_dims + 2, 'Illegal pad definitions.')
        self._assert(len(self.strides) == self.kernel_dims, 'Illigal stride definitions.')

        # check the shape consistency
        if self.kernel_index_H is not None and self.index_H is not None:
            pad_H = self._pads[self.kernel_index_H] + \
                self._pads[self.kernel_index_H + self.kernel_dims]
            output_H_base = self.input_ops['X'].shape[self.index_H] + pad_H - self.kernel_height
            stride_H = self.strides[self.kernel_index_H]
            output_H, output_H_rest = divmod(output_H_base, stride_H)
            output_H += 1
            message = f'Pooling operator {self.name} does not match the height: {output_H} vs {self.height}.'
            self._assert(output_H == self.height, message)
            self._assert(output_H_rest == 0,
                         f'Pooling operator {self.name} should adjust the height pad to plus {output_H_rest}.')

        if self.kernel_index_W is not None and self.index_W is not None:
            pad_W = self._pads[self.kernel_index_W] + \
                self._pads[self.kernel_index_W + self.kernel_dims]
            output_W_base = self.input_ops['X'].shape[self.index_W] + pad_W - self.kernel_width
            stride_W = self.strides[self.kernel_index_W]
            output_W, output_W_rest = divmod(output_W_base, stride_W)
            output_W += 1
            message = f'Pooling operator {self.name} does not match the width: {output_W} vs {self.width}.'
            self._assert(output_W == self.width, message)
            self._assert(output_W_rest == 0,
                         f'Pooling operator {self.name} should adjust the width pad to plus {output_W_rest}.')

    @property
    def kernel_height(self) -> int:
        """Get the height in the kernel shape."""
        if self.kernel_index_H is not None:
            return self.kernel_shape[self.kernel_index_H]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_height property.')

    @property
    def kernel_width(self) -> int:
        """Get the Width in the kernel shape."""
        if self.kernel_index_W is not None:
            return self.kernel_shape[self.kernel_index_W]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_width property.')

    @property
    def pads(self) -> List[int]:
        return self._pads

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        """Infer its shape from inputs' shapes."""
        # attributes
        pads = attrs['pads']
        strides = attrs['strides']
        kernel_shape = attrs['kernel_shape']

        idx_N = input_formats[0].index('N')  # from input X
        idx_C = input_formats[0].index('C')  # from weight W
        idx_H = input_formats[0].index('H')  # from input X
        idx_W = input_formats[0].index('W')  # from input X

        N = lists['X'][idx_N]
        C = lists['X'][idx_C]

        # H
        pads_H = pads[0] + pads[2]
        input_H = lists['X'][idx_H] + pads_H
        window_H = kernel_shape[0]
        stride_H = strides[0]
        H, rest_H = divmod((input_H - window_H), stride_H)
        H += 1
        assert rest_H == 0, f'differfence in height: {rest_H} at {cls.__name__}'

        # W
        pads_W = pads[1] + pads[3]
        input_W = lists['X'][idx_W] + pads_W
        window_W = kernel_shape[1]
        stride_W = strides[1]
        W, rest_W = divmod((input_W - window_W), stride_W)
        W += 1
        assert rest_W == 0, f'differfence in width: {rest_W} at {cls.__name__}'

        NCHW = [N, C, H, W]
        perm = [format.index(s) for s in 'NCHW']
        return [NCHW[i] for i in perm]

    @property
    def preserve_quantization(self) -> bool:
        return False


class MaxPool(Pool):
    """Max pooling operator.

    MaxPool consumes an input tensor X and applies max pooling across the the tensor according
    to kernel sizes, stride sizes, and pad lengths. max pooling consisting of computing the max
    on all values of a subset of the input tensor according to the kernel size and downsampling
    the data into the output tensor Y for further processing.

    Inputs
    ------
    X
        Input data tensor from the previous operator.

    Outputs
    -------
    Y
        Output data tensor from max pooling across the input tensor.
        Dimensions will vary based on various kernel, stride, and pad sizes.
        Floor value of the dimension is used.

    Attributes (Optional constructor parameters)
    --------------------------------------------
    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    kernel_shape : list of ints
        The size of the kernel along each axis.

    kernel_dim_format : str
        Dimension denotation, which must consists of H', and 'W', where 'H' and 'W' are the
        height and width of input image. The default is 'HW'.

    dimensions : int
        Dimensions. This defaults to 2, which means 2-D image.
        Currently only 2 is available.

    pads : list of ints
        Padding for the beginning and ending along each axis, it can take any value greater
        than or equal to 0. The value represent the number of pixels added to the beginning
        and end part of the corresponding axis. `pads` format should be as follow
        [x1_begin, x2_begin, x1_end, x2_end], where xi_begin the number of pixels added at
        the beginning of axis `i` and xi_end, the number of pixels added at the end of axis
        `i`. If not present, the padding defaults to 0 along start and end of each axis.

    strides : list of ints
        Stride along each axis. If not present, the stride defaults to 1 along each axis.

    """

    @property
    def _dispatch_name(self) -> str:
        return 'max_pool'

    @property
    def is_monotonic(self) -> bool:
        return False


class AveragePool(Pool):
    """Average pooling operator.

    AveragePool consumes an input tensor X and applies average pooling across the the tensor
    according to kernel sizes, stride sizes, and pad lengths. average pooling consisting of
    computing the average on all values of a subset of the input tensor according to the
    kernel size and downsampling the data into the output tensor Y for further processing.

    Inputs
    ------
    X
        Input data tensor from the previous operator.

    Outputs
    -------
    Y
        Output data tensor from average pooling across the input tensor.
        Dimensions will vary based on various kernel, stride, and pad sizes.
        Floor value of the dimension is used.

    Attributes (Optional constructor parameters)
    --------------------------------------------
    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    kernel_shape : list of ints
        The size of the kernel along each axis.

    kernel_dim_format : str
        Dimension denotation, which must consists of H', and 'W', where 'H' and 'W' are the
        height and width of input image. The default is 'HW'.

    dimensions : int
        Dimensions. This defaults to 2, which means 2-D image.
        Currently only 2 is available.

    pads : list of ints
        Padding for the beginning and ending along each axis, it can take any value greater
        than or equal to 0. The value represent the number of pixels added to the beginning
        and end part of the corresponding axis. `pads` format should be as follow
        [x1_begin, x2_begin, x1_end, x2_end], where xi_begin the number of pixels added at
        the beginning of axis `i` and xi_end, the number of pixels added at the end of axis
        `i`. If not present, the padding defaults to 0 along start and end of each axis.

    strides : list of ints
        Stride along each axis. If not present, the stride defaults to 1 along each axis.

    """

    @property
    def _dispatch_name(self) -> str:
        return 'average_pool'

    @property
    def is_monotonic(self) -> bool:
        return False


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

    _input_names = ['data']
    _output_names = ['reshaped']

    def __init__(self, name: str, shape: List[int], dtype: DataType, input_ops: Ops) -> None:
        """Init the reshape operator."""
        super().__init__(name, shape, dtype, input_ops)

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


class QTZ_binary_channel_wise_mean_scaling(Quantizer):
    """Quantization operator using binary channel wise scaling.

    Input
    -----
    input
        Input tensor, which must have float falues.

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
        super()._check_consistency()

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

    split : list of interger
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
        return False
                     

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
