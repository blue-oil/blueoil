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

import yaml
import whaaaaat
from jinja2 import Environment, FileSystemLoader


# TODO(wakisaka): objecte detection, segmentation
_TASK_TYPE_TEMPLATE_FILE = {
    "classification": "classification.tpl.py",
    "object_detection": "object_detection.tpl.py",
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
    "LMFYoloQuantize": {
        "network_module": "lm_fyolo",
        "network_class": "LMFYoloQuantize",
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
        "learning_rate": 1e-3,
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
        dataset_class_property = {"extend_dir": dataset_class_extend_dir, "validation_extend_dir": dataset_class_validation_extend_dir}
    else:
        dataset_class_property = {"extend_dir": dataset_class_extend_dir}

    # trainer
    batch_size = blueoil_config["trainer"]["batch_size"]
    optimizer  = blueoil_config["trainer"]["optimizer"]

    # common
    image_size = blueoil_config["common"]["image_size"]

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
        "max_epochs": "",
        "max_steps": "",
        "optimizer" : optimizer,
        
        "image_size": image_size,

        "quantize_first_convolution": quantize_first_convolution,

        "dataset": dataset,
    }
    
    # max_epochs or max_steps
    if "steps" in blueoil_config["trainer"].keys():
        config["max_steps"] = blueoil_config["trainer"]["steps"]
    elif "epochs" in blueoil_config["trainer"].keys():
        config["max_epochs"] = blueoil_config["trainer"]["epochs"]


    # merge dict
    lmnet_config = default_lmnet_config.copy()
    lmnet_config.update(config)

    return lmnet_config


def _save(lmnet_config):
    env = Environment(loader=FileSystemLoader('./blueoil/templates/lmnet', encoding='utf8'))

    template_file = lmnet_config["template_file"]

    tpl = env.get_template(template_file)

    applied = tpl.render(lmnet_config)
    config_file = "{}.py".format(lmnet_config['model_name'])
    with open(config_file, 'w') as fp:
        fp.write(applied)
    return config_file


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
