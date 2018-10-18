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
"""Model module. It contains model info, which includes graph inside."""
from core.graph import Graph


class Model(object):
    """Class that represents a model."""

    def __init__(self) -> None:
        """Init the model. Currently, just generates a blank graph inside."""
        self.__graph: Graph = Graph()

    @property
    def graph(self) -> Graph:
        """Return the graph in this model."""
        return self.__graph

    @graph.setter
    def graph(self, val: Graph) -> None:
        del self.__graph
        self.__graph = val

    def is_valid_graph(self) -> bool:
        """Return if the graph is a valid one. This is just for testing."""
        return self.__graph.check_nodes()
