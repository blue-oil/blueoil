#!/usr/bin/env python3
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
import importlib
import importlib.util
import os


def load_module(module_file_path):
    """dynamically load a module from the file path.

    Args:
        module_file_path:

    Returns:
        module: module object

    """

    if not os.path.exists(module_file_path):
        raise ValueError("Module loading error. {} is not exists.".format(module_file_path))

    head, _ = os.path.splitext(module_file_path)
    module_path = ".".join(head.split(os.sep))
    if os.path.isabs(module_file_path):
        spec = importlib.util.spec_from_file_location(module_path, module_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return module


def load_class(module, class_name):
    # this converts the string from snake format into class capital format
    # e.g. example_class_name -> ExampleClassName

    # if class capital format.
    if class_name[0].isupper() and "_" not in class_name:
        class_name = class_name
    else:
        class_name = "".join([str.capitalize() for str in class_name.split("_")])
    cls = module.__dict__[class_name]

    return cls


def _load_class_from_name(name, base_dir):
    """dynamically load a class object from name.

    Args:
        name (string): class name following python module loader rule.
            e.g.
            `name` is dir1.module, load dir1.module.Module .
            `name` is dir1.module.Class, load dir1.module.Class .
        base_dir: base directory of class name.

    Returns:
        class: class object
    """
    names = name.split(".")
    last_name = names[-1]

    # if last_name is class capital format.
    if last_name[0].isupper():
        class_name = last_name
        file_path = os.path.join(base_dir, *names[:-1]) + ".py"

    else:
        class_name = last_name
        file_path = os.path.join(base_dir, *names) + ".py"

    module = load_module(file_path)
    loaded_class = load_class(module, class_name)

    return loaded_class


# TODO(wakisaka): use TASK_TYPE
def load_network_class(name):
    """dynamically load a class object from a network file.

    Args:
        name (str):  Name of network

    Returns:
        lmnet.networks.Base: network class object

    """

    base_dir = os.path.join("lmnet", "networks")
    network_class = _load_class_from_name(name, base_dir)

    return network_class


def load_dataset_class(name):
    """dynamically load a class object from a dataset file.

    Args:
        name (str): Name of dataset

    Returns:
        blueoil.nn.datasets.Base: dataset class object

    """
    base_dir = os.path.join("lmnet", "datasets")
    dataset_class = _load_class_from_name(name, base_dir)

    return dataset_class
