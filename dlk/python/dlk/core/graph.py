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
from typing import cast, Any, Dict, List, Optional, Set, TYPE_CHECKING
import functools

from core.operators import Add, AveragePool, BatchNormalization, Constant, Conv, Identity, Input, \
    MaxPool, Operator, Output, Transpose, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half, Reshape, Softmax, \
    Relu, Flatten, Dropout, Gemm, SpaceToDepth, Mul, QTZ_binary_channel_wise_mean_scaling, ConcatOnDepth, Maximum, \
    DepthToSpace, Split, Pad, MatMul


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
        """Add an operator and its inputs recursively."""
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
    def operartors(self) -> List[Operator]:
        """List up all operators in this graph."""
        return list(self.__ops.keys())

    def get_inputs(self) -> List[Operator]:
        return list(self.__op_type_list['Input'])

    def get_outputs(self) -> List[Operator]:
        return list(self.__op_type_list['Output'])

    @property
    def consts(self) -> List[Operator]:
        return list(self.__op_type_list['Constant'])

    @property
    def non_variables(self) -> List[Operator]:
        kwargs: Dict[str, List[Operator]] = {'node_list': []}
        sorter = NodesSorter(self)
        sorter.run(**kwargs)
        return [node for node in kwargs['node_list'] if not cast(Operator, node).is_variable]

    def find_node_by_op_type(self, op_type: str) -> List[Operator]:
        """Find nodes which op_type is specified by the argument.

        Parameters
        ----------
        op_type : str
            Operator type of the nodes

        Returns
        -------
        ops : list of str
            Operators that have the op_type

        """
        return list(self.__op_type_list[op_type])

    def convs(self, quantized_only: bool = False) -> List[Conv]:
        """Return the list of conv operators in this graph.

        Parameters
        ----------
        quantized_only : bool
            Flag that represents if the operators are only quantized ones

        """

        return list(cast(List['Conv'], self.__op_type_list['Conv'])) \
            if not quantized_only else [x for x in cast(List['Conv'], self.__op_type_list['Conv'])
                                        if cast(Conv, x).is_quantized]

    def check_nodes(self) -> bool:
        """Check whether all operators defined in this graph.

        Currently it checks:
        - for all operartors op, forall input in op.inputs, input.outputs includes op.

        Returns
        -------
        result : bool
            Whether the graph passes the test.

        """
        # check the input-output consistency
        for op_name in self.__ops:
            op = cast(Operator, self.__ops[op_name])
            inputs: Dict[str, Operator] = op.input_ops
            for i in inputs.values():
                if op not in i.output_op_list:
                    return False

        return True

    def accept(self, runner: 'GraphRunner', **kwargs: Any) -> None:
        """Accept a graph runner and run it from the output node."""
        if TYPE_CHECKING:
            import core.graph as gp
        runner.initialize(**kwargs)

        if runner.depth_first:  # depth first traversal
            outputs = self.get_outputs()
            for out in outputs:
                out.accept(cast('gp.GraphRunner', runner), **kwargs)

        else:  # breadth first traversal
            # backward 1st
            next = self.get_outputs()
            if runner.is_lazy:
                while next:
                    def get_visit_list(ops: List[Operator]) -> List[bool]:
                        return list(map(lambda n: runner.is_visited(cast(Operator, n)), ops))

                    def and_all(list: List[bool]) -> bool:
                        return functools.reduce(lambda x, y: x and y, list, True)

                    # devide the `next` list into executables and non-executables
                    execs = [op for op in next if and_all(get_visit_list(op.output_op_list))]
                    non_execs = [op for op in next if not and_all(get_visit_list(op.output_op_list))]

                    # if there is no executable operators, terminate this loop
                    if execs == []:
                        names = list(map(lambda x: x.name, non_execs))
                        raise AssertionError(f'dead lock happened. {names} cannot run.')

                    # execute
                    next = non_execs
                    for op in execs:
                        next += op.accept_backward(cast('gp.GraphRunner', runner), **kwargs)
            else:
                for op in next:
                    next += op.accept_backward(cast('gp.GraphRunner', runner), **kwargs)

            # turn
            runner.turn(**kwargs)

            # forward run
            next = self.get_inputs() + self.consts
            if runner.is_lazy:
                while next:
                    def get_inputs(op: Operator) -> List[Operator]:
                        return list(op.input_ops.values())

                    def get_visit_list(ops: List[Operator]) -> List[bool]:
                        return list(map(lambda n: not runner.is_visited(cast(Operator, n)), ops))

                    def and_all(list: List[bool]) -> bool:
                        return functools.reduce(lambda x, y: x and y, list, True)

                    # devide the `next` list into executables and non-executables
                    execs = [op for op in next if and_all(get_visit_list(get_inputs(op)))]
                    non_execs = [op for op in next if not and_all(get_visit_list(get_inputs(op)))]

                    # if there is no executable operators, terminate this loop
                    if execs == []:
                        names = list(map(lambda x: x.name, non_execs))
                        raise AssertionError(f'dead lock happened. {names} cannot run.')

                    # execute
                    next = non_execs
                    for op in execs:
                        next += op.accept_forward(cast('gp.GraphRunner', runner), **kwargs)
            else:
                for op in next:
                    next += op.accept_forward(cast('gp.GraphRunner', runner), **kwargs)

        runner.finalize(**kwargs)


class GraphRunner(object):
    """Visitor class of a graph."""

    def __init__(self, graph: Graph, depth_first: bool = True, lazy: bool = True) -> None:
        """Set up the graph runner.

        Parameters
        ----------
        graph : Graph
            the graph to be traversed.

        depth_first : bool
            a flag that represents if the running is done in a depth first manner.
            Otherwise, this runner runs in a breadth first manner. It defaults to
            True, i.e. a depth first traversal.

        lazy : bool
            True if this runner runs in a lazy mode. This means all operator waits
            for the traversal until the predecessors are traversed.
            This flag is valid only in breadth-first mode. In the depth-first mode,
            this is naturally true.
        """
        self._graph = graph
        self._visited: Set[str] = set()
        self._dfs = depth_first
        self._is_lazy = lazy

    def run(self, **kwargs: Any) -> None:
        """Run this runner on the graph."""
        self._graph.accept(self, **kwargs)

    @property
    def visited(self) -> Set[str]:
        return set(self._visited)

    def visit(self, op: Operator) -> None:
        self._visited.add(op.name)

    def unvisit(self, op: Operator) -> None:
        self._visited.remove(op.name)

    def is_visited(self, node: Operator) -> bool:
        return node.name in self._visited

    @property
    def depth_first(self) -> bool:
        """Returns True if this runs in a depth-first manner.

        Otherwise, this runs in a breadth-first manner.
        """
        return self._dfs

    @property
    def is_lazy(self) -> bool:
        """Returns True if this runs in a lazy mode, i.e. all node waits until all of its predecessors are traversed.

        This flag is valide only in the breadth-first mode.
        """
        return self._is_lazy

    def initialize(self, **kwargs: Any) -> None:
        """Initialize the running.

        This method is called when the run starts.
        """
        pass

    def turn(self, **kwargs: Any) -> None:
        """Turn from backward to forward.

        This method is called only when the run is in a breadth-first manner.
        """
        pass

    def finalize(self, **kwargs: Any) -> None:
        """Finalize the running.

        This method is called when the run finishes.
        """
        pass

    def run_backward_by_default(self, node: Operator, **kwargs: Any) -> None:
        pass

    def run_forward_by_default(self, node: Operator, **kwargs: Any) -> None:
        pass

    def run_backward_input(self, node: Input, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_input(self, node: Input, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_constant(self, node: Constant, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_constant(self, node: Constant, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_output(self, node: Output, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_output(self, node: Output, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_identity(self, node: Identity, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_identity(self, node: Identity, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_transpose(self, node: Transpose, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_transpose(self, node: Transpose, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_conv(self, node: Conv, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_batch_normalization(self, node: BatchNormalization, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_batch_normalization(self, node: BatchNormalization, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_add(self, node: Add, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_add(self, node: Add, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_max_pool(self, node: MaxPool, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_max_pool(self, node: MaxPool, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_average_pool(self, node: AveragePool, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_average_pool(self, node: AveragePool, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_reshape(self, node: Reshape, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_reshape(self, node: Reshape, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_softmax(self, node: Softmax, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_softmax(self, node: Softmax, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_relu(self, node: Relu, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_relu(self, node: Relu, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_flatten(self, node: Flatten, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_flatten(self, node: Flatten, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_dropout(self, node: Dropout, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_dropout(self, node: Dropout, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_gemm(self, node: Gemm, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_gemm(self, node: Gemm, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_SpaceToDepth(self, node: SpaceToDepth, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_SpaceToDepth(self, node: SpaceToDepth, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_mul(self, node: Mul, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_mul(self, node: Mul, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_QTZ_binary_channel_wise_mean_scaling(
            self,
            node: QTZ_binary_channel_wise_mean_scaling,
            **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_QTZ_binary_channel_wise_mean_scaling(
            self,
            node: QTZ_binary_channel_wise_mean_scaling,
            **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_ConcatOnDepth(self, node: ConcatOnDepth, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_ConcatOnDepth(self, node: ConcatOnDepth, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_Maximum(self, node: Maximum, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_Maximum(self, node: Maximum, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_DepthToSpace(self, node: DepthToSpace, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_DepthToSpace(self, node: DepthToSpace, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_Split(self, node: Split, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_Split(self, node: Split, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_Pad(self, node: Pad, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_Pad(self, node: Pad, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)

    def run_backward_MatMul(self, node: MatMul, **kwargs: Any) -> None:
        self.run_backward_by_default(node, **kwargs)

    def run_forward_MatMul(self, node: MatMul, **kwargs: Any) -> None:
        self.run_forward_by_default(node, **kwargs)


class NodesSorter(GraphRunner):
    """Class for sorting the nodes of a graph

    It will sort the nodes of a graph in topological order
    """

    def run_forward_input(self, node: Input, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_constant(self, node: Constant, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_output(self, node: Output, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_identity(self, node: Identity, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_transpose(self, node: Transpose, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_conv(self, node: Conv, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_batch_normalization(self, node: BatchNormalization, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_add(self, node: Add, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_max_pool(self, node: MaxPool, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_average_pool(self, node: AveragePool, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_reshape(self, node: Reshape, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_softmax(self, node: Softmax, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_relu(self, node: Relu, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_flatten(self, node: Flatten, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_dropout(self, node: Dropout, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_gemm(self, node: Gemm, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_SpaceToDepth(self, node: SpaceToDepth, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_mul(self, node: Mul, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_QTZ_binary_channel_wise_mean_scaling(
            self,
            node: QTZ_binary_channel_wise_mean_scaling,
            **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_ConcatOnDepth(self, node: ConcatOnDepth, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_Maximum(self, node: Maximum, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_DepthToSpace(self, node: DepthToSpace, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_Split(self, node: Split, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_Pad(self, node: Pad, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)

    def run_forward_MatMul(self, node: MatMul, **kwargs: Any) -> None:
        kwargs['node_list'].append(node)
