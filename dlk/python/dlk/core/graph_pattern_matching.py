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
"""Graph pattern matching module."""


class Pattern:
    """Pattern is a sub-graph based on the operator types.
       It is a recursive pattern where a Pattern holds a operator type and a list of inputs.
       Each input in this list is also a Pattern.
    """
    def __init__(self, op=str(), inputs=list()):
        self.op = op
        self.inputs = inputs


class NodeMatch:
    """NodeMatch defines a sub-graph that match a given Pattern.
       It is a recursive pattern where a NodeMatch holds a reference to the matched node and a list of inputs.
       Each input in this list is also a NodeMatch.
    """
    def __init__(self):
        self.node = None
        self.inputs = list()


def find_pattern(graph, pattern):
    """Helper function that find a pattern in a graph.

    Parameters
    ----------
    graph : Graph
        The input graph where we will try to find the given pattern.

    pattern : Pattern
        The pattern we want to look for.

    Returns
    -------
    result : [NodeMatch]
        A list of matches. Each element of the list is a NodeMatch.
    """
    gm = GraphMatcher(graph)
    return gm.get_op_type_matches(pattern)


def sort_graph(graph):
    """Helper function to topologically sort a given graph.

    Parameters
    ----------
    graph : Graph
        The input graph to be sorted. It is not modified.

    Returns
    -------
    result : [Operator]
        A list of Operator. Each element of the list is a reference to a Operator object.
    """
    exec_list = list()
    input_nodes = list()
    for node in graph.operators:
        input_nodes += [n.name for n in node.input_nodes]

    output_nodes = list()
    for node in graph.operators:
        if node not in input_nodes:
            output_nodes.append(node)

    visited = {}
    for node in graph.operators:
        visited[node.name] = False

    for node in output_nodes:
        top_order(node, exec_list, visited)

    return exec_list


def top_order(output_node, exec_list, visited):
    """It topologically sorts a given graph.

    Parameters
    ----------
    output_node : Operator
        The starting node. First one in the ordered list.

    exec_list : [Operator]
        The ordered list. Note that this is an output parameter.

    visited : [str]
        List of already visited nodes.
    """
    if visited[output_node.name]:
        return
    for input_node in output_node.input_nodes:
        top_order(input_node, exec_list, visited)

    exec_list.append(output_node)
    visited[output_node.name] = True


def get_nodes_in_branch(starting_node, stop_node, node_list):
    """Helper function that gives us all nodes in a branch defined by a given node.
       The starting node will be the output node of the branch.

       Note that there is an optional stop node. stop_node is allowed to be None.

    Parameters
    ----------
    starting_node : Operator
        The starting node. This node is the output node of the defined branch.

    stop_node : Operator
        The last node in the path. If stop_node is None then this function will give us every node above
        starting_node.

    node_list : [Operator]
        The list of nodes contained in the branch. Note that this is an output parameter.
    """
    if starting_node == stop_node:
        return
    node_list.append(starting_node)

    for node in starting_node.input_nodes:
        get_nodes_in_branch(node, stop_node, node_list)


class GraphMatcher:
    """GraphMatcher is used to find sub-graphs in the computational graph.
    """
    def __init__(self, input_graph):
        self.graph_node_list = list()
        self.graph_node_list = sort_graph(input_graph)

        self._node_map = {node.name: node for node in self.graph_node_list}

    def record_matched_nodes(self, match, matched_nodes):
        matched_nodes.add(match.node.name)
        for input_node in match.inputs:
            self.record_matched_nodes(input_node, matched_nodes)

    def get_op_type_matches(self, pattern):
        matches = list()
        matched_nodes = set()
        for node in self.graph_node_list:
            if node in matched_nodes:
                continue

            match = NodeMatch()
            if self.does_op_type_match(node, pattern, matched_nodes, match):
                self.record_matched_nodes(match, matched_nodes)
                matches.append(match)
        return matches

    def does_op_type_match(self, node, pattern, previously_matched_nodes, match):
        if node.name in previously_matched_nodes:
            return False

        pattern_matched = False
        if pattern.op == '*':
            pattern_matched = True
        else:
            for pattern_op in pattern.op.split('|'):
                if node.op_type == pattern_op:
                    pattern_matched = True
        if not pattern_matched:
            return False

        match.node = node
        if not pattern.inputs:
            return True
        if len(node.input_nodes) != len(pattern.inputs):
            return False

        for i in range(len(pattern.inputs)):
            input_node = self._node_map[node.input_nodes[i].name]
            input_pattern = pattern.inputs[i]
            input_match = NodeMatch()
            match.inputs.append(input_match)

            if not self.does_op_type_match(input_node, input_pattern, previously_matched_nodes, input_match):
                return False

        return True
