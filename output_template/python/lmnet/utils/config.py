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

import imp

import yaml
from easydict import EasyDict

from nn.data_processor import Sequence


# derived from LeapMind/lmnet/lmnet/utils/config.py
def load_yaml(config_file):
    with open(config_file) as config_file_stream:
        config = yaml.load(config_file_stream, Loader=yaml.Loader)

    # use only upper key.
    keys = [key for key in config.keys() if key.isupper()]
    config_dict = {key: config[key] for key in keys}
    config = EasyDict(config_dict)
    return config


def build_pre_process(pre_processor_config):
    module_name = "lmnet/pre_processor"
    f, pathname, description = imp.find_module(module_name)
    module = imp.load_module(module_name, f, pathname, description)
    processors = []
    if pre_processor_config is None:
        pre_processor_config = {}
    for p in pre_processor_config:
        for class_name in p:
            class_args = p[class_name]
            if class_args is None:
                class_args = {}
            cls = getattr(module, class_name)
            # Create none initialized processor `cls` instance.
            processor = cls.__new__(cls)
            # Fill processor instance member.
            for k in class_args:
                v = class_args[k]
                processor.__dict__[k] = v
            processors.append(processor)
    seq = Sequence(processors=processors)
    return seq


def build_post_process(post_processor_config):
    module_name = "lmnet/post_processor"
    f, pathname, description = imp.find_module(module_name)
    module = imp.load_module(module_name, f, pathname, description)
    processors = []
    if post_processor_config is None:
        post_processor_config = {}
    for p in post_processor_config:
        for class_name in p:
            class_args = p[class_name]
            if class_args is None:
                class_args = {}
            cls = getattr(module, class_name)
            # Create none initialized processor `cls` instance.
            processor = cls.__new__(cls)
            # Fill processor instance member.
            for k in class_args:
                v = class_args[k]
                processor.__dict__[k] = v
            processors.append(processor)
    seq = Sequence(processors=processors)
    return seq
