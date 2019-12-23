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
"""Base IO module."""
from abc import ABCMeta, abstractmethod

from core.graph import Graph


class BaseIO(metaclass=ABCMeta):
    """Base class for model IO."""

    @abstractmethod
    def read(self, path: str) -> Graph:
        """Read a model.

        Args:
            path (str): Path to the file to be read
        
        """
        pass

    @abstractmethod
    def write(self, graph: Graph, path: str) -> None:
        """Write the model to a file.

        Args:
            model (Model): Model to be written
        
        """
        pass
