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
import os
import pprint
import yaml
from blueoil.utils.smartdict import SmartDict

from blueoil.data_processor import Processor, Sequence
from blueoil import environment
from blueoil.common import Tasks
from blueoil.io import file_io

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
    ("PRE_PROCESSOR", "TFDS_PRE_PROCESSOR"),
    "POST_PROCESSOR",
]

REQUIEMNT_PARAMS_FOR_TRAINING = REQUIEMNT_PARAMS_FOR_INFERENCE + [
    ("MAX_STEPS", "MAX_EPOCHS"),  # require "MAX_STEPS" or "MAX_EPOCHS".
    "SAVE_CHECKPOINT_STEPS",
    "KEEP_CHECKPOINT_MAX",
    "TEST_STEPS",
    "SUMMARISE_STEPS",
    "IS_PRETRAIN",
    "PRETRAIN_VARS",
    "PRETRAIN_DIR",
    "PRETRAIN_FILE",
]


def _saved_config_file_path():
    filepath = os.path.join(environment.EXPERIMENT_DIR, 'config.py')
    if file_io.exists(filepath):
        return filepath

    raise FileNotFoundError("Config file not found: '{}'".format(filepath))


def _config_file_path_to_copy(config_file):
    _, file_extension = os.path.splitext(config_file)

    if file_extension.lower() in '.py':
        filename = 'config.py'
    else:
        raise ValueError('Config file type is not supported.'
                         'Should be .py. Received {}.'.format(file_extension))

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
                raise KeyError("config file should include {} parameter".format(" or ".join(keys)))
        else:
            if key not in config:
                raise KeyError("config file should include {} parameter".format(key))


def load(config_file):
    """dynamically load a config file as module.

    Return: SmartDict object
    """
    filename, file_extension = os.path.splitext(config_file)
    if file_extension.lower() in '.py':
        loader = _load_py
    else:
        raise ValueError('Config file type is not supported.'
                         'Should be .py. Received {}.'.format(file_extension))
    config = loader(config_file)

    check_config(config)

    return config


def _load_py(config_file):
    config = {}
    with file_io.File(config_file) as config_file_stream:
        source = config_file_stream.read()
        exec(source, globals(), config)

    # use only upper key.
    return SmartDict({
        key: value
        for key, value in config.items()
        if key.isupper()
    })


def _smart_dict_to_dict(config):
    if isinstance(config, SmartDict):
        config = dict(config)

    for key, value in config.items():
        if isinstance(value, SmartDict):
            value = dict(value)
            _smart_dict_to_dict(value)
        config[key] = value
    return config


def _save_meta_yaml(output_dir, config):
    output_file_name = 'meta.yaml'
    config_dict = _smart_dict_to_dict(config)

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
        """From Processor instance to dictionary of classname and instance property's key and value.

        According to pickle manner, use __reduce_ex__() to get instance property.
        """

        # state is dictionary of Processor instances property's key and value.
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

    with file_io.File(os.path.join(file_path), mode='w') as f:
        yaml.dump(meta_dict, f, default_flow_style=False, Dumper=Dumper)

    return file_path


def save_yaml(output_dir, config):
    """Save yaml file 'meta.yaml' for application. Keys are defined by `PARAMS_FOR_EXPORT`."""
    if not file_io.exists(output_dir):
        file_io.makedirs(output_dir)

    meta_yaml_path = _save_meta_yaml(output_dir, config)

    return meta_yaml_path


def load_from_experiment():
    """Load saved experiment config as module.

    Return: SmartDict object
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
    file_io.copy(config_file, saved_config_file_path, overwrite=True)


def merge(base_config, override_config):
    """merge config.

    Return: merged config (SmartDict object).
    """

    result = SmartDict(base_config)
    for k, v in override_config.items():
        if type(v) is SmartDict:
            v = merge(base_config[k], override_config[k])

        result[k] = v

    return result
