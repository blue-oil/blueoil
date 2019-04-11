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
from abc import ABCMeta
import os
import pprint

from easydict import EasyDict
import yaml
from yaml.representer import Representer
from tensorflow import gfile

from lmnet.common import Tasks
from lmnet.data_processor import Processor, Sequence
from lmnet.utils import module_loader
from lmnet import environment


PARAMS_FOR_EXPORT = [
    "DATA_FORMAT",
    "TASK",
    "IMAGE_SIZE",
    "CLASSES",
    "PRE_PROCESSOR",
    "POST_PROCESSOR",
]

REQUIEMNT_PARAMS_FOR_INFERENCE = [
    "IS_DEBUG",
    "DATA_FORMAT",
    "TASK",
    "DATASET_CLASS",
    "NETWORK_CLASS",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "CLASSES",
    "PRE_PROCESSOR",
    "POST_PROCESSOR",
]

REQUIEMNT_PARAMS_FOR_TRAINING = REQUIEMNT_PARAMS_FOR_INFERENCE + [
    ("MAX_STEPS", "MAX_EPOCHS"),  # require "MAX_STEPS" or "MAX_EPOCHS".
    "SAVE_STEPS",
    "TEST_STEPS",
    "SUMMARISE_STEPS",
    "IS_PRETRAIN",
    "PRETRAIN_VARS",
    "PRETRAIN_DIR",
    "PRETRAIN_FILE",
    "IS_DISTRIBUTION",
]


def _saved_config_file_path():
    for filename in ('config.py', 'config.yaml'):
        filepath = os.path.join(environment.EXPERIMENT_DIR, filename)
        if os.path.isfile(filepath):
            return filepath


def _config_file_path_to_copy(config_file):
    _, file_extension = os.path.splitext(config_file)

    if file_extension.lower() in '.py':
        filename = 'config.py'
    elif file_extension.lower() in ('.yml', '.yaml'):
        filename = 'config.yaml'
    else:
        raise ValueError('Config file type is not supported.'
                         'Should be .py, .yaml or .yml. Received {}.'.format(file_extension))

    return os.path.join(environment.EXPERIMENT_DIR, filename)


def check_config(config, mode="inference"):
    """Check config dict key. Raise error when requirement keys don't exist in config"""

    if mode == "inference":
        requirements = REQUIEMNT_PARAMS_FOR_INFERENCE
    if mode == "training":
        requirements = REQUIEMNT_PARAMS_FOR_TRAINING

    for key in requirements:
        if isinstance(key, tuple):
            keys = key
            if not any([key in config for key in keys]):
                raise KeyError("config file should be included {} parameter".format(" or ".join(keys)))
        else:
            if key not in config:
                raise KeyError("config file should be included {} parameter".format(key))


def load(config_file):
    """dynamically load a config file as module.

    Return: EasyDict object
    """
    filename, file_extension = os.path.splitext(config_file)
    if file_extension.lower() in '.py':
        loader = _load_py
    elif file_extension.lower() in ('.yml', '.yaml'):
        loader = _load_yaml
    else:
        raise ValueError('Config file type is not supported.'
                         'Should be .py, .yaml or .yml. Received {}.'.format(file_extension))
    config = loader(config_file)

    check_config(config)

    return config


def _load_py(config_file):
    config_module = module_loader.load_module(config_file)

    # use only upper key.
    keys = [key for key in dir(config_module) if key.isupper()]
    config_dict = {key: getattr(config_module, key) for key in keys}
    config = EasyDict(config_dict)
    return config


def _easy_dict_to_dict(config):
    if isinstance(config, EasyDict):
        config = dict(config)

    for key, value in config.items():
        if isinstance(value, EasyDict):
            value = dict(value)
            _easy_dict_to_dict(value)
        config[key] = value
    return config


def _save_meta_yaml(output_dir, config):
    output_file_name = 'meta.yaml'
    config_dict = _easy_dict_to_dict(config)

    meta_dict = {key: value for key, value in config_dict.items() if key in PARAMS_FOR_EXPORT}

    file_path = os.path.join(output_dir, output_file_name)

    class Dumper(yaml.Dumper):
        def ignore_aliases(self, data):
            return True

    def tasks_representer(dumper, data):
        """From Task enum to str"""
        return dumper.represent_str(data.value)

    def sequence_representer(dumper, data):
        """From Sequence to list of Sequence's instance property processors"""
        return dumper.represent_data(data.processors)

    def tuple_representer(dumper, data):
        """From tuple to list"""
        return dumper.represent_list(data)

    def processor_representer(dumper, data):
        """From Processor instance to dictonary of classname and instance property's key and value.

        Acccording to pickle manner, use __reduce_ex__() to get instance property.
        """

        # state is dictonary of Processor instances property's key and value.
        # Ref: https://docs.python.org/3/library/pickle.html#data-stream-format
        _, _, state, _, _ = data.__reduce_ex__(4)

        class_name = data.__class__.__name__
        node = dumper.represent_dict({class_name: state})

        return node

    Dumper.add_representer(tuple, tuple_representer)
    Dumper.add_representer(Tasks, tasks_representer)
    Dumper.add_representer(Sequence, sequence_representer)
    Dumper.add_multi_representer(Processor, processor_representer)

    if type(meta_dict['CLASSES']) != list:
        DatasetClass = config.DATASET_CLASS
        dataset_kwargs = dict((key.lower(), val) for key, val in config.DATASET.items())
        train_dataset = DatasetClass(
            subset="train",
            **dataset_kwargs,
        )
        meta_dict['CLASSES'] = train_dataset.classes

    with gfile.GFile(os.path.join(file_path), 'w') as f:
        yaml.dump(meta_dict, f, default_flow_style=False, Dumper=Dumper)

    return file_path


def _save_config_yaml(output_dir, config):
    file_name = 'config.yaml'
    config_dict = _easy_dict_to_dict(config)
    file_path = os.path.join(output_dir, file_name)

    class Dumper(yaml.Dumper):
        def ignore_aliases(self, data):
            return True
    Dumper.add_representer(ABCMeta, Representer.represent_name)

    if type(config_dict['CLASSES']) != list:
        DatasetClass = config.DATASET_CLASS
        dataset_kwargs = dict((key.lower(), val) for key, val in config.DATASET.items())
        train_dataset = DatasetClass(
            subset="train",
            **dataset_kwargs,
        )
        config_dict['CLASSES'] = train_dataset.classes

    with gfile.GFile(os.path.join(output_dir, file_name), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, Dumper=Dumper)

    return file_path


def save_yaml(output_dir, config):
    """Save two yaml files.

    1. 'config.yaml' is duplication of python config file as yaml.
    2. 'meta.yaml' for application. The yaml's keys defined by `PARAMS_FOR_EXPORT`.
    """

    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)

    config_yaml_path = _save_config_yaml(output_dir, config)
    meta_yaml_path = _save_meta_yaml(output_dir, config)

    return config_yaml_path, meta_yaml_path


def _load_yaml(config_file):
    with gfile.GFile(config_file) as config_file_stream:
        config = yaml.load(config_file_stream, Loader=yaml.Loader)

    # use only upper key.
    keys = [key for key in config.keys() if key.isupper()]
    config_dict = {key: config[key] for key in keys}
    config = EasyDict(config_dict)
    return config


def load_from_experiment():
    """Load saved experiment config as module.

    Return: EasyDict object
    """
    config_file = _saved_config_file_path()
    return load(config_file)


def display(config):
    # print config values
    print("----------------- config value --------------------")
    pprint.pprint(config)
    print("----------------- config value --------------------")


def copy_to_experiment_dir(config_file):
    # copy config file to the experiment directory
    saved_config_file_path = _config_file_path_to_copy(config_file)
    gfile.Copy(config_file, saved_config_file_path)


def init_config(config, training_id, recreate=False):
    """Initialize config.

    Set logging.
    Train id embed to config directories.
    """

    # _init_logging(config)


def restore_saved_image_size(config):
    saved_config_file_path = _saved_config_file_path()
    config = load(saved_config_file_path)

    if hasattr(config, "IMAGE_SIZE"):
        return config.IMAGE_SIZE

    raise Exception("IMAGE_SIZE dont exists in file {}".format(saved_config_file_path))


def merge(base_config, override_config):
    """merge config.

    Return: merged config (EasyDict object).
    """

    result = EasyDict(base_config)
    for k, v in override_config.items():
        if type(v) is EasyDict:
            v = merge(base_config[k], override_config[k])

        result[k] = v

    return result
