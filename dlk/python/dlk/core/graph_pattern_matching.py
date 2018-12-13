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

from core.operators import Operator
from core.graph import Graph


class Pattern:
    def __init__(self, op=str(), inputs=list()):
        self.op = op
        self.inputs = inputs


class NodeMatch:
    def __init__(self):
        self.node = None
        self.inputs = list()


def sort_graph(graph):
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
    if visited[output_node.name]:
        return
    for input_node in output_node.input_nodes:
        top_order(input_node, exec_list, visited)

    exec_list.append(output_node)
    visited[output_node.name] = True


def match_to_execution_list(match, execution_list):
    for input_node in match.inputs:
        match_to_execution_list(input_node, execution_list)
    execution_list.append(match.node)


class GraphMatcher:
    def __init__(self, input_graph=Graph()):
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
