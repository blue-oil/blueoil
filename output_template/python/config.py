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
    """Load meta.yaml that is output when the model is converted.

    Args:
        config_file (str): Path of the configuration file.

    Returns:
        EasyDict: Dictionary object of loaded configuration file.

    Examples:
        >>> config = load_yaml("/path/of/meta.yaml")
    """
    with open(config_file) as config_file_stream:
        config = yaml.load(config_file_stream, Loader=yaml.Loader)
    # use only upper key.
    return EasyDict({k: v for k, v in config.items() if k.isupper()})


def build_pre_process(pre_processor_config):
    """The pre processor is loaded based on the information passed,
    It is append to a Sequence object and returned in.

    Args:
        pre_processor_config (List[Dict[str, Optional[Dict[str, Any]]]]): List of processors to load.

    Returns:
        Sequence: A Sequence object with a list of processors inside.

    Examples:
        >>> config = [
                {"Resize": {"size": [128, 128], "resample": "NEAREST"}},
                {"PerImageStandardization": None},
            ]
        >>> pre_processors = build_pre_process(config)
    """
    return _build_process(pre_processor, pre_processor_config)


def build_post_process(post_processor_config):
    """The post processor is loaded based on the information passed,
    It is append to a Sequence object and returned in.

    Args:
        post_processor_config (List[Dict[str, Optional[Dict[str, Any]]]]): List of processors to load.

    Returns:
        Sequence: A Sequence object with a list of processors inside.

    Examples:
        >>> config = [
                {"FormatYoloV2": {
                    "anchors": [
                        [1.3221, 1.73145],
                        [3.19275, 4.00944],
                        [5.05587, 8.09892],
                        [9.47112, 4.84053],
                        [11.2364, 10.0071],
                    ],
                    "boxes_per_cell": 5,
                    "data_format": "NHWC",
                    "image_size": [128, 128],
                    "num_classes": 1,
                }},
                {"ExcludeLowScoreBox": {"threshold": 0.3}},
                {"NMS": {
                    "classes": ["person"],
                    "iou_threshold": 0.5,
                    "max_output_size": 100,
                    "per_class": True,
                }},
            ]
        >>> post_processors = build_post_process(config)
    """
    return _build_process(post_processor, post_processor_config)


def _build_process(module, processor_config=None):
    processors = []
    processor_config = processor_config or []
    for p in processor_config:
        for class_name, kwargs in p.items():
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
            else:
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

    if "." not in import_string:
        raise ImportError("Invalid import string '{}'".format(import_string))

    module_name, attr_name = import_string.rsplit(".", 1)
    module = import_module(module_name)
    if not hasattr(module, attr_name):
        raise ImportError("There is no attribute name '{}' in module '{}'".format(
            module_name, attr_name,
        ))

    return getattr(module, attr_name)
