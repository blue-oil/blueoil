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
from __future__ import absolute_import, division, print_function, unicode_literals

from importlib import import_module

import yaml
from easydict import EasyDict

from blueoil.data_processor import Sequence
from blueoil import post_processor, pre_processor


def load_yaml(config_file):
    with open(config_file) as config_file_stream:
        config = yaml.load(config_file_stream, Loader=yaml.Loader)
    # use only upper key.
    return EasyDict({k: v for k, v in config.items() if k.isupper()})


def build_pre_process(pre_processor_config):
    return _build_process(pre_processor, pre_processor_config)


def build_post_process(post_processor_config):
    return _build_process(post_processor, post_processor_config)


def _build_process(module, processor_config=None):
    processors = []
    processor_config = processor_config or []
    for p in processor_config:
        for class_name, kwargs in p.items():
            try:
                cls = getattr(module, class_name)
            except AttributeError:
                cls = import_from_string(class_name)

            processor = cls.__new__(cls)
            processor.__dict__.update(kwargs or {})
            processors.append(processor)

    return Sequence(processors=processors)


def import_from_string(import_string):
    """Import from the import path string.

    Args:
        path (str): Import path string

    Returns:
        Any: Imported object

    Raises:
        ImportError: It occurs when an import path that does not exist is specified.

    Examples:
        >>> join = import_from_string("os.path.join")
    """
    try:
        return import_module(import_string)
    except ImportError:
        pass

    try:
        module_name, attr_name = import_string.rsplit(".", 1)
    except ValueError:
        raise ImportError("Invalid import string '{}'".format(import_string))

    module = import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError:
        raise ImportError("There is no attribute name '{}' in module '{}'".format(
            module_name, attr_name,
        ))
