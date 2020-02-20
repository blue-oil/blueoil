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
"""Graph sorting helper functions."""


def sort_graph(graph):
    """Helper function to topologically sort a given graph.

    Args:
        graph (Graph): The input graph to be sorted. It is not modified.

    Returns:
        list(Operator): A list of Operator. Each element of the list is a reference to
            a Operator object.
    
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

    Args:
        output_node (Operator): The starting node. First one in the ordered list.
        exec_list (list[operator]): The ordered list. Note that this is an output
            parameter.
        visited: (list[str]): List of already visited nodes.
    
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

    Args:
        starting_node (Operator): The starting node. This node is the output node of
            the defined branch.
        stop_node (Operator): The last node in the path. If stop_node is None then this
            function will give us every node above starting_node.
        node_list (list[Operator]): The list of nodes contained in the branch. Note
            that this is an output parameter.

    """
    if starting_node == stop_node:
        return
    node_list.append(starting_node)

    for node in starting_node.input_nodes:
        get_nodes_in_branch(node, stop_node, node_list)
