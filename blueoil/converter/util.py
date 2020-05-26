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
"""Utility functions."""
import importlib
import os
import shutil
from os import path
from pathlib import Path
from typing import Any, Generator, List, Mapping, Union

JSON = Union[str, int, float, bool, None, Mapping[str, 'JSON'], List['JSON']]  # type: ignore


def make_dirs(dir_pathes: Union[str, List[str]]) -> None:
    """Create one or more directories."""
    if isinstance(dir_pathes, str):
        dir = Path(dir_pathes)
        Path.mkdir(dir, parents=True, exist_ok=True)

    elif isinstance(dir_pathes, list):
        for dir_path in dir_pathes:
            dir = Path(dir_path)
            Path.mkdir(dir, parents=True, exist_ok=True)


def move_file_or_dir(src_path: str, dest_path: str) -> None:
    """Move a file or a directory."""
    shutil.move(src_path, dest_path)


def get_files(src_dir_path: str, excepts: str = '') -> List[str]:
    """Get a list of file pathes."""
    return [fp for fp in get_files_generator(src_dir_path, excepts)]


def get_files_generator(src_dir_path: str, excepts: str = '') -> Generator[str, None, None]:
    """Get file pathes."""
    for dirpath, dirnames, filenames in os.walk(src_dir_path):
        if excepts not in dirpath:
            for file_name in filenames:
                yield path.join(dirpath, file_name)


def dynamic_class_load(path: str) -> Any:
    """Load a class defined by the argument.

    Args:
        path (str): The full pathname that represents the class to be loaeded.
            (Actually I don't know how to call it in Python.)

    Returns:
        Any: The class to be loaded.

    """
    pathes = path.split('.')
    class_name = pathes.pop()
    module_name = '.'.join(pathes)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class classproperty(object):
    """Decorator for class property."""
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)
