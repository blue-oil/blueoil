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
"""Graph module."""
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, cast

from blueoil.converter.core.graph_pattern_matching import sort_graph
from blueoil.converter.core.operators import Conv, Operator


class Graph(object):
    """Graph class. This class was formerly named as 'Nodes'."""

    def __init__(self) -> None:
        """Init the graph."""
        self.__ops: OrderedDict = OrderedDict()
        self.__op_type_list: Dict[str, List[Operator]] = defaultdict(lambda: [])
        self.__non_variable_list: List[Operator] = []

    def __eq__(self, other) -> bool:
        """Return the two graphs are equivalent."""
        if other is None or not isinstance(other, Graph):
            name = other.name if other else None
            print(f'{name} is not a Graph object.')
            return False

        def match(op1: Operator, op2: Operator) -> bool:
            if not op1.equals(op2):
                print(f'{op1.name} is different.')
                return False

            # check input nodes and further
            for i1, i2 in zip(op1.input_ops.values(), op2.input_ops.values()):
                if not match(i1, i2):
                    return False
            return True

        for o1, o2 in zip(self.get_outputs(), other.get_outputs()):
            if not match(o1, o2):
                return False
        return True

    def get_op(self, name: str) -> Optional[Operator]:
        return self.__ops.get(name)

    def add_op(self, op: Operator) -> Operator:
        if op.name not in self.__ops.keys():
            self.__ops[op.name] = op
            self.__op_type_list[op.op_type].append(op)

            if not op.is_variable:
                self.__non_variable_list.append(op)

        else:
            ValueError(f'{op.name} is already registered in this graph.')

        return op

    def add_op_and_inputs(self, op: Operator) -> Operator:
        """Add an operator and its inputs recursively.

        Args:
            op (Operator):

        """
        self.add_op(op)
        for i in op.input_ops.values():
            self.add_op_and_inputs(i)

        return op

    def remove_op(self, op: Operator) -> None:
        if self.__ops.get(op.name) is not None:
            del self.__ops[op.name]

        t = type(op).__name__
        to_remove = [i for i, val in
                     enumerate(self.__op_type_list[t])
                     if val.name == op.name]
        assert len(to_remove) <= 1, f'{op.name} was registered more than once.'
        for i in to_remove:
            del self.__op_type_list[t][i]

    @property
    def operators(self) -> List[Operator]:
        """List up all operators in this graph."""
        return list(self.__ops.values())

    def get_inputs(self) -> List[Operator]:
        return list(self.__op_type_list['Input'])

    def get_outputs(self) -> List[Operator]:
        return list(self.__op_type_list['Output'])

    @property
    def consts(self) -> List[Operator]:
        return list(self.__op_type_list['Constant'])

    @property
    def non_variables(self) -> List[Operator]:
        node_list = sort_graph(self)
        node_list = [node for node in node_list if not cast(Operator, node).is_variable]
        return node_list

    def find_node_by_op_type(self, op_type: str) -> List[Operator]:
        """Find nodes which op_type is specified by the argument.

        Args:
            op_type (str): Operator type of the nodes

        Returns:
            list[str]: Operators that have the op_type

        """
        return list(self.__op_type_list[op_type])

    def convs(self, quantized_only: bool = False) -> List[Conv]:
        """Return the list of conv operators in this graph.

        Args:
            quantized_only (bool): Flag that represents if the operators are only
                quantized ones (Default value = False)

        Returns:
            list:

        """

        return list(cast(List['Conv'], self.__op_type_list['Conv'])) \
            if not quantized_only else [x for x in cast(List['Conv'], self.__op_type_list['Conv'])
                                        if cast(Conv, x).is_quantized]

    def check_nodes(self) -> bool:
        """Check whether all operators defined in this graph.

        Currently it checks:
        - for all operators op, for all input in op.inputs, input.outputs includes op.

        Args:

        Returns:
            bool: Whether the graph passes the test.

        """
        # check the input-output consistency
        for op_name in self.__ops:
            op = cast(Operator, self.__ops[op_name])
            inputs: Dict[str, Operator] = op.input_ops
            for i in inputs.values():
                if op not in i.output_op_list:
                    return False

        return True
