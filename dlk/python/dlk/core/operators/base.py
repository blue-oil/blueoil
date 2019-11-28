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
# We need the data_types module from the parent directory.
import functools
from typing import Any, Dict, Optional, cast

from core.view import View
from utils import classproperty

from core.data_types import *

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
        self.update_shape(shape, dimension_format)
        self.view: View = View(self)
        self.__connect_to_outputs()
        self._check_consistency()
        self._rank = len(shape)
        self._available_buffer = ''

    def update_shape(self, shape: List[int], dimension_format: str) -> None:
        self._shape: List[int] = shape
        self._rank = len(shape)
        self._dimension_format = dimension_format
        dimension_format_list = []
        for ch in dimension_format:
            if ch.isupper():
                dimension_format_list.append(ch)
            else:
                dimension_format_list[-1] += ch
        self.index_H = dimension_format_list.index('H') if 'H' in dimension_format_list else None
        self.index_W = dimension_format_list.index('W') if 'W' in dimension_format_list else None
        self.index_N = dimension_format_list.index('N') if 'N' in dimension_format_list \
            else dimension_format_list.index('Oh') if 'Oh' in dimension_format_list \
            else dimension_format_list.index('O') if 'O' in dimension_format_list else None
        self.index_C = dimension_format_list.index('C') if 'C' in dimension_format_list \
            else dimension_format_list.index('Ch') if 'Ch' in dimension_format_list \
            else dimension_format_list.index('I') if 'I' in dimension_format_list \
            else dimension_format_list.index('Ih') if 'Ih' in dimension_format_list else None
        self.index_C_low = dimension_format_list.index('Cl') if 'Cl' in dimension_format_list else None

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

        Args:
            predicate (bool): Assertion to be true
            message (str): Error message in the failure of the assertion

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

        Returns:
            dict: Collection of input operators in a dictionary format.
                The keys are input symbols, which can be taken from `input_names` property.

        """
        return self._input_ops

    @classproperty
    def input_names(cls) -> List[str]:
        """Return the input key names the operator provides.

        For example, `Conv` has two inputs, 'X' for the input data and 'W' for the weight.
        So `Conv.input_names` returns the list `['X', 'W']`.

        Returns:
            list[str]: List of key names

        """
        return cls._input_names

    @property
    def input_nodes(self) -> List['Operator']:
        """Return a list of input operators in proper order (original protobuf argument order).

        Returns:
            list[Operator]: This list is already ordered following the order of the arguments in the original
                protobuf operators (positional order in the list of arguments).

        """
        return [self._input_ops[i] for i in self.input_names if self.input_ops.get(i)]

    @property
    def output_ops(self) -> OutOps:
        """Return a dict of output operators.

        Returns:
            dict: Collection of (list of) output operators in a dictionary format.
                The keys are output symbols, which can be taken from `output_names` property.

        """
        return self._output_ops

    @property
    def output_op_list(self) -> List['Operator']:
        """Return a list of output operators.

        Returns:
            list[Operator]: List of output operators.

        """
        return sum(list(self._output_ops.values()), [])

    @classproperty
    def output_names(cls) -> List[str]:
        """Return the output key names the operator provides.

        For example, `Conv` has one output 'Y'.
        So `Conv.output_names` returns the list `['Y']`.

        Returns:
            list[str]: List of key names

        """
        return cls._output_names

    def add_input(self, ident: str, node: 'Operator') -> None:
        """Add input node.

        Args
            ident (str): key name of the input. This has to be in list `input_names`.
            node (Operator): Node to be registered as the input.

        """
        self._assert(ident in self._input_names, "Illegal input name")
        self._input_ops[ident] = node

    def add_inputs(self, inputs: Ops) -> None:
        """Add input (possibly multiple) nodes at a once.

        Args:
            outputs (dict): Collection of pair of key name and a operator to be registered as the input.
                All the key names have to be in list `input_names`.

        """
        assert set(inputs.keys()).issubset(set(self._input_names)), "Illegal output names included"
        self._input_ops.update(inputs)

    def add_output(self, ident: str, node: 'Operator') -> None:
        """Add output node.

        Args:
            ident (str): key name of the output. This has to be in list `output_names`.
            node (Operator): Node to be registered as the output.

        """
        self._assert(ident in self._output_names, "Illegal output name")
        lst: Optional[List['Operator']] = self._output_ops.get(ident)
        if lst is not None:
            lst.append(node)
        else:
            self._output_ops[ident] = [node]

    def add_outputs(self, outputs: OutOps) -> None:
        """Add output (possibly multiple) nodes at a once.

        Args:
            outputs (Dict of str to list of Operators): Collection of pair of key name
            and a list of operators to be registered as the output.
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

        Args:
            ident (str): Key name of the input node to be removed.
                This key is in `input_names`, not the name of the operator.

        """
        self._input_ops.pop(ident)

    def remove_output(self, ident: str) -> None:
        """Remove an output node.

        Args:
            ident (str): Key name of the output node to be removed.
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
            if self.index_C_low is not None:
                return self.shape[self.index_C] * self.shape[self.index_C_low]
            else:
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
        self._assert(len(set(perm)) == len(self._shape), "Illegal permutation specified.")
        self._assert(max(perm) == len(self._shape) - 1, "Illegal permutation specified.")
        self._assert(min(perm) == 0, "Illegal permutation specified.")

        # change the shape
        new_shape: List[int] = [self._shape[i] for i in perm]

        # change the format
        new_format: str = functools.reduce(
            lambda x, y: x + y, [self._dimension_format[i] for i in perm])

        # update
        self.update_shape(new_shape, new_format)

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

        This is actually an abstract method and should be overridden.
        """
        raise NotImplementedError('run is not implemented yet')

    def run_forward(self) -> np.ndarray:
        """Run the operator, calculate and set the result.

        This is actually an abstract method and should be overridden.
        """
        raise NotImplementedError(
            f'operator {self.op_type} does not have runtime implementation yet.')

    @property
    def _dispatch_name(self) -> str:
        return type(self).__name__.lower()

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        """Infer its output shape from inputs' shapes.

        This is actually an abstract method and should be overridden.
        """
        raise NotImplementedError(f'operator {cls.__name__} cannot infer its shape.')

    @property
    def preserve_quantization(self) -> bool:
        """whether to preserve the operator for quantization"""
        raise NotImplementedError(
            f'Preservation for quantization of operator {self.op_type} is not defined.')


