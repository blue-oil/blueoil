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
import argparse
import os
import re
import importlib
from tempfile import NamedTemporaryFile

import yaml
from jinja2 import Environment, FileSystemLoader

from lmnet.utils.module_loader import load_class
from blueoil.vars import TEMPLATE_DIR


_TASK_TYPE_TEMPLATE_FILE = {
    "classification": "classification.tpl.py",
    "object_detection": "object_detection.tpl.py",
    "semantic_segmentation": "semantic_segmentation.tpl.py",
}

_NETWORK_NAME_NETWORK_MODULE_CLASS = {
    "LmnetV0Quantize": {
        "network_module": "lmnet_v0",
        "network_class": "LmnetV0Quantize",
    },
    "LmnetV1Quantize": {
        "network_module": "lmnet_v1",
        "network_class": "LmnetV1Quantize",
    },
    "ResNetQuantize": {
        "network_module": "lm_resnet",
        "network_class": "LmResnetQuantize",
    },
    "LMFYoloQuantize": {
        "network_module": "lm_fyolo",
        "network_class": "LMFYoloQuantize",
    },
    "LmSegnetV1Quantize": {
        "network_module": "lm_segnet_v1",
        "network_class": "LmSegnetV1Quantize",
    },
}

_DATASET_FORMAT_DATASET_MODULE_CLASS = {
    "Caltech101": {
        "dataset_module": "image_folder",
        "dataset_class": "ImageFolderBase",
    },
    "DeLTA-Mark for Classification": {
        "dataset_module": "delta_mark",
        "dataset_class": "ClassificationBase",
    },
    "OpenImagesV4": {
        "dataset_module": "open_images_v4",
        "dataset_class": "OpenImagesV4BoundingBoxBase",
    },
    "DeLTA-Mark for Object Detection": {
        "dataset_module": "delta_mark",
        "dataset_class": "ObjectDetectionBase",
    },
    "CamvidCustom": {
        "dataset_module": "camvid",
        "dataset_class": "CamvidCustom",
    },
}


def generate(blueoil_config_filename):

    blueoil_config = _load_yaml(blueoil_config_filename)
    lmnet_config = _blueoil_to_lmnet(blueoil_config)
    config_file = _save(lmnet_config)

    return config_file


def _load_yaml(blueoil_config_filename):
    """load blueoil config yaml

    Args:
        blueoil_config_filename(str): File path of blueoil config yaml file.

    Returns:
        blueoil_config(dict): dict of blueoil config.
    """
    if not os.path.exists(blueoil_config_filename):
        FileNotFoundError("File not found: {}".format(blueoil_config_filename))

    with open(blueoil_config_filename, "r") as f:
        blueoil_config = yaml.load(f)

    model_name, _ = os.path.splitext(os.path.basename(blueoil_config_filename))

    blueoil_config["model_name"] = model_name

    return blueoil_config


def _blueoil_to_lmnet(blueoil_config):
    """

    Args:
        blueoil_config(dict):

    Returns:
        lmnet_config(dict):
    """

    # default setting
    default_lmnet_config = {
        "save_steps": 1000,
        "test_steps": 1000,
        "summarise_steps": 100,
    }
    dataset = {

    }

    model_name = blueoil_config["model_name"]

    template_file = _TASK_TYPE_TEMPLATE_FILE[blueoil_config["task_type"]]

    network_module_class = _NETWORK_NAME_NETWORK_MODULE_CLASS[blueoil_config["network_name"]]
    network_module = network_module_class["network_module"]
    network_class = network_module_class["network_class"]

    # dataset
    dataset_module_class = _DATASET_FORMAT_DATASET_MODULE_CLASS[blueoil_config["dataset"]["format"]]
    dataset_module = dataset_module_class["dataset_module"]
    dataset_class = dataset_module_class["dataset_class"]
    dataset_class_extend_dir = blueoil_config["dataset"]["train_path"]
    dataset_class_validation_extend_dir = blueoil_config["dataset"]["test_path"]
    if dataset_class_validation_extend_dir is not None:
        dataset_class_property = {"extend_dir": dataset_class_extend_dir,
                                  "validation_extend_dir": dataset_class_validation_extend_dir}
    else:
        dataset_class_property = {"extend_dir": dataset_class_extend_dir}

    # load dataset python module from string.
    _loaded_dataset_module = importlib.import_module("lmnet.datasets.{}".format(dataset_module))
    # load dataset python module from string.
    _loaded_dataset_class = load_class(_loaded_dataset_module, dataset_class)
    _dataset_class = type('DATASET_CLASS', (_loaded_dataset_class,), dataset_class_property)
    _dataset_obj = _dataset_class(subset="train", batch_size=1)
    classes = _dataset_obj.classes

    # trainer
    batch_size = blueoil_config["trainer"]["batch_size"]
    optimizer  = blueoil_config["trainer"]["optimizer"]
    if optimizer == 'Adam':
        optimizer_class = "tf.train.AdamOptimizer"
    elif optimizer == 'Momentum':
        optimizer_class = "tf.train.MomentumOptimizer"
    else:
        raise ValueError("not supported optimizer.")

    initial_learning_rate = blueoil_config["trainer"]["initial_learning_rate"]
    learning_rate_schedule = blueoil_config["trainer"]["learning_rate_schedule"]
    max_epochs = blueoil_config["trainer"]["epochs"]

    step_per_epoch = float(_dataset_obj.num_per_epoch)/batch_size

    learning_rate_kwargs = None
    if learning_rate_schedule == "constant":
        learning_rate_func = None
    else:
        learning_rate_func = "tf.train.piecewise_constant"

    if learning_rate_schedule == "constant":
        if optimizer == 'Momentum':
            optimizer_kwargs = {"momentum": 0.9, "learning_rate": initial_learning_rate}
        else:
            optimizer_kwargs = {"learning_rate": initial_learning_rate}
    else:
        if optimizer == 'Momentum':
            optimizer_kwargs = {"momentum": 0.9}
        else:
            optimizer_kwargs = {}            
            
    if learning_rate_schedule == "2-step-decay":
        learning_rate_kwargs = {
            "values": [
                initial_learning_rate,
                initial_learning_rate / 10,
                initial_learning_rate / 100
            ],
            "boundaries": [
                int((step_per_epoch * (max_epochs - 1)) / 2),
                int(step_per_epoch * (max_epochs - 1))
            ],
        }

    elif learning_rate_schedule == "3-step-decay":
        learning_rate_kwargs = {
            "values": [
                initial_learning_rate,
                initial_learning_rate / 10,
                initial_learning_rate / 100,
                initial_learning_rate / 1000
            ],
            "boundaries": [
                int((step_per_epoch * (max_epochs - 1)) * 1 / 3),
                int((step_per_epoch * (max_epochs - 1)) * 2 / 3),
                int(step_per_epoch * (max_epochs - 1))
            ],
        }

    elif learning_rate_schedule == "3-step-decay-with-warmup":
        if max_epochs < 4:
            raise ValueError("epoch number must be >= 4, when 3-step-decay-with-warmup is selected.")
        learning_rate_kwargs = {
            "values": [
                initial_learning_rate / 1000,
                initial_learning_rate,
                initial_learning_rate / 10,
                initial_learning_rate / 100,
                initial_learning_rate / 1000
            ],
            "boundaries": [
                int(step_per_epoch * 1),
                int((step_per_epoch * (max_epochs - 1)) * 1 / 3),
                int((step_per_epoch * (max_epochs - 1)) * 2 / 3),
                int(step_per_epoch * (max_epochs - 1))
            ],
        }

    # common
    image_size = blueoil_config["common"]["image_size"]

    data_augmentation = []
    for augmentor in blueoil_config["common"].get("data_augmentation", []):
        key = list(augmentor.keys())[0]
        values = []
        for v in list(list(augmentor.values())[0]):
            v_key, v_value = list(v.keys())[0], list(v.values())[0]
            only_str = isinstance(v_value, str) and re.match('^[\w-]+$', v_value) is not None
            value = (v_key, "'{}'".format(v_value) if only_str else v_value)
            values.append(value)
        data_augmentation.append((key, values))

    # quantize first layer
    quantize_first_convolution = blueoil_config["network"]["quantize_first_convolution"]

    config = {
        "model_name": model_name,
        "template_file": template_file,
        "network_module": network_module,
        "network_class": network_class,

        "dataset_module": dataset_module,
        "dataset_class": dataset_class,
        "dataset_class_property": dataset_class_property,

        "batch_size": batch_size,
        "optimizer_class" : optimizer_class,
        "max_epochs": max_epochs,

        "optimizer_kwargs": optimizer_kwargs,
        "learning_rate_func": learning_rate_func,
        "learning_rate_kwargs": learning_rate_kwargs,

        "image_size": image_size,
        "classes": classes,

        "quantize_first_convolution": quantize_first_convolution,

        "dataset": dataset,
        "data_augmentation": data_augmentation
    }

    # merge dict
    lmnet_config = default_lmnet_config.copy()
    lmnet_config.update(config)

    return lmnet_config


def _save(lmnet_config):
    env = Environment(loader=FileSystemLoader(os.path.join(TEMPLATE_DIR, 'lmnet'), encoding='utf8'))

    template_file = lmnet_config["template_file"]

    tpl = env.get_template(template_file)

    applied = tpl.render(lmnet_config)
    with NamedTemporaryFile(
            prefix="blueoil_config_{}".format(lmnet_config['model_name']),
            suffix=".py", delete=False, mode="w") as fp:
        fp.write(applied)
        return fp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blueoil_config_filename",
                        help="File path of blueoil config yaml.")

    args = parser.parse_args()

    blueoil_config_filename = args.blueoil_config_filename

    lmnet_config_filename = generate(blueoil_config_filename)

    print('Convert configuration file from {} to: {}'.format(
        blueoil_config_filename, lmnet_config_filename))


if __name__ == '__main__':
    main()
