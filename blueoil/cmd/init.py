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
from collections import OrderedDict
import inspect
import re
import shutil

import inquirer

import blueoil.data_augmentor as augmentor
from blueoil.generate_lmnet_config import generate
from blueoil.data_processor import Processor


task_type_choices = [
    'classification',
    'object_detection',
    'semantic_segmentation',
    'keypoint_detection'
]

classification_network_definitions = [
    {
        'name': 'LmnetV1Quantize',
        'desc': 'Quantized Lmnet version 1. Accuracy is better than LmnetV0Quantize.',
    },
    {
        'name': 'ResNetQuantize',
        'desc': 'Quantized ResNet 18. Accuracy is better than LmnetV1Quantize.',
    },
]

object_detection_network_definitions = [
    {
        'name': 'LMFYoloQuantize',
        'desc': 'YOLO-like object detection network.',
    },
]

semantic_segmentation_network_definitions = [
    {
        'name': 'LmSegnetV1Quantize',
        'desc': 'Quantized LeapMind original semantic segmentation network, version 1.',
    },
]

keypoint_detection_network_definitions = [
    {
        'name': 'LmSinglePoseV1Quantize',
        'desc': 'Quantized LeapMind original single-person pose estimation network, version 1.',
    },
]

IMAGE_SIZE_VALIDATION = {
    "LmnetV1Quantize": {
        "max_size": 512,
        "divider": 16,
    },
    "ResNetQuantize": {
        "max_size": 512,
        "divider": 16,
    },
    "LMFYoloQuantize": {
        "max_size": 480,
        "divider": 32,
    },
    "LmSegnetV1Quantize": {
        "max_size": 512,
        "divider": 8,
    },
    "LmSinglePoseV1Quantize": {
        "max_size": 512,
        "divider": 8,
    },
}

classification_dataset_formats = [
    {
        'name': 'Caltech101',
        'desc': 'Caletch101 compatible',
    },
    {
        'name': 'DeLTA-Mark for Classification',
        'desc': 'Dataset for classification created by DeLTA-Mark',
    },
]

object_detection_dataset_formats = [
    {
        'name': 'OpenImagesV4',
        'desc': 'OpenImagesV4 compatible',
    },
    {
        'name': 'DeLTA-Mark for Object Detection',
        'desc': 'Dataset for object detection created by DeLTA-Mark',
    },
]

semantic_segmentation_dataset_formats = [
    {
        'name': 'CamvidCustom',
        'desc': 'CamVid base cumstom format',
    },
]

keypoint_detection_dataset_formats = [
    {
        'name': 'Mscoco for Single-Person Pose Estimation',
        'desc': 'Mscoco 2017 for Single-Person Pose Estimation',
    },
]

learning_rate_schedule_map = OrderedDict([
    ("constant", "'constant' -> constant learning rate."),
    ("cosine", "'cosine' -> cosine learning rate."),
    ("2-step-decay", "'2-step-decay' -> learning rate decrease by 1/10 on {epochs}/2 and {epochs}-1."),
    ("3-step-decay", "'3-step-decay' -> learning rate decrease by 1/10 on {epochs}/3 and {epochs}*2/3 and {epochs}-1"),
    (
     "3-step-decay-with-warmup",
     "'3-step-decay-with-warmup' -> "
     "warmup learning rate 1/1000 in first epoch, then train the same way as '3-step-decay'"
    ),
])


def network_name_choices(task_type):
    if task_type == 'classification':
        return [definition['name'] for definition in classification_network_definitions]
    elif task_type == 'object_detection':
        return [definition['name'] for definition in object_detection_network_definitions]
    elif task_type == 'semantic_segmentation':
        return [definition['name'] for definition in semantic_segmentation_network_definitions]
    elif task_type == 'keypoint_detection':
        return [definition['name'] for definition in keypoint_detection_network_definitions]


def dataset_format_choices(task_type):
    if task_type == 'classification':
        return [definition['name'] for definition in classification_dataset_formats]
    elif task_type == 'object_detection':
        return [definition['name'] for definition in object_detection_dataset_formats]
    elif task_type == 'semantic_segmentation':
        return [definition['name'] for definition in semantic_segmentation_dataset_formats]
    elif task_type == 'keypoint_detection':
        return [definition['name'] for definition in keypoint_detection_dataset_formats]


def default_batch_size(task_type):
    default_task_type_batch_sizes = {
        'classification': '10',
        'object_detection': '16',
        'semantic_segmentation': '8',
        'keypoint_detection': '4',
    }
    return default_task_type_batch_sizes[task_type]


def prompt(question):
    """Execute prompt answer

    Args:
        question (list): list of inquirer question

    Returns: string of answer

    """
    answers = inquirer.prompt(question)
    return answers['value']


def generate_image_size_validate(network_name):
    """Generate image_size_validate depending on task_type.

    Args:
        network_name (string): network name.

    Returns: validate function.

    """
    max_size = IMAGE_SIZE_VALIDATION[network_name]["max_size"]
    divider = IMAGE_SIZE_VALIDATION[network_name]["divider"]

    def image_size_validate(answers, current):
        # change to tuple (height, width).
        image_size = image_size_filter(current)
        image_size = (int(size) for size in image_size)

        for size in image_size:
            if not size % divider == 0:
                raise inquirer.errors.ValidationError('',
                                                      reason="Image size should be multiple of {}, but image size is {}"
                                                      .format(divider, current))

            if size > max_size:
                raise inquirer.errors.ValidationError('',
                                                      reason="Image size should be lower than {} but image size is {}"
                                                      .format(max_size, current))

        return True

    return image_size_validate


def integer_validate(answers, current):
    if not current.isdigit():
        raise inquirer.errors.ValidationError('', reason='Input value should be integer')

    return True


def image_size_filter(raw):
    match = re.match(r"([0-9]+)[^0-9]+([0-9]+)", raw)

    # raw: 128x128 -> ('128', '128')
    image_size = match.groups()

    return image_size


def save_config(blueoil_config, output=None):
    if not output:
        output = blueoil_config['model_name'] + ".py"

    tmpfile = generate(blueoil_config)
    shutil.copy(tmpfile, output)

    return output


def ask_questions():
    model_name_question = [
        inquirer.Text(
            name='value',
            message='your model name ()')
    ]
    model_name = prompt(model_name_question)

    task_type_question = [
        inquirer.List(name='value',
                      message='choose task type',
                      choices=task_type_choices)
    ]
    task_type = prompt(task_type_question)

    network_name_question = [
        inquirer.List(name='value',
                      message='choose network',
                      choices=network_name_choices(task_type))
    ]
    network_name = prompt(network_name_question)

    dataset_format_question = [
        inquirer.List(name='value',
                      message='choose dataset format',
                      choices=dataset_format_choices(task_type))
    ]
    dataset_format = prompt(dataset_format_question)

    enable_data_augmentation = [
        inquirer.Confirm(name='value',
                         message='enable data augmentation?',
                         default=True)
    ]

    train_dataset_path_question = [
        inquirer.Text(name='value',
                      message='training dataset path')
    ]
    train_path = prompt(train_dataset_path_question)

    enable_test_dataset_path_question = [
        inquirer.List(name='value',
                      message='set validation dataset?'
                              ' (if answer no, the dataset will be separated for training and validation'
                              ' by 9:1 ratio.)',
                      choices=['yes', 'no'])
    ]
    enable_test_dataset_path = prompt(enable_test_dataset_path_question)

    test_dataset_path_question = [
        inquirer.Text(name='value',
                      message='validation dataset path')
    ]
    if enable_test_dataset_path == 'yes':
        test_path = prompt(test_dataset_path_question)
    else:
        test_path = ''

    batch_size_question = [
        inquirer.Text(name='value',
                      message='batch size (integer)',
                      default=default_batch_size(task_type),
                      validate=integer_validate)
    ]
    batch_size = prompt(batch_size_question)

    image_size_question = [
        inquirer.Text(name='value',
                      message='image size (integer x integer)',
                      default='128x128',
                      validate=generate_image_size_validate(network_name))
    ]
    image_size = image_size_filter(prompt(image_size_question))

    training_epochs_question = [
        inquirer.Text(name='value',
                      message='how many epochs do you run training (integer)',
                      default='100',
                      validate=integer_validate)
    ]
    training_epochs = prompt(training_epochs_question)

    training_optimizer_question = [
        inquirer.List(name='value',
                      message='select optimizer',
                      choices=['Momentum', 'Adam'],
                      default='Momentum')
    ]
    training_optimizer = prompt(training_optimizer_question)

    initial_learning_rate_value_question = [
        inquirer.Text(name='value',
                      message='initial learning rate',
                      default='0.001')
    ]
    initial_learning_rate_value = prompt(initial_learning_rate_value_question)

    # learning rate schedule
    learning_rate_schedule_question = [
        inquirer.List(name='value',
                      message='choose learning rate schedule'
                              ' ({{epochs}} is the number of training epochs you entered before)',
                      choices=list(learning_rate_schedule_map.values()),
                      default=learning_rate_schedule_map["constant"])
    ]
    _tmp_learning_rate_schedule = prompt(learning_rate_schedule_question)
    for key, value in learning_rate_schedule_map.items():
        if value == _tmp_learning_rate_schedule:
            learning_rate_schedule = key

    data_augmentation = {}
    if prompt(enable_data_augmentation):
        all_augmentor = {}
        checkboxes = []
        for name, obj in inspect.getmembers(augmentor):
            if inspect.isclass(obj) and issubclass(obj, Processor):
                argspec = inspect.getfullargspec(obj)
                # ignore self
                args = argspec.args[1:]
                defaults = argspec.defaults
                if len(args) == len(defaults):
                    default_val = [(arg, default) for arg, default in zip(args, defaults)]
                    default_str = " (default: {})".format(", ".join(["{}={}".format(a, d) for a, d in default_val]))
                else:
                    defaults = ("# Please fill a value.",) * (len(args) - len(defaults)) + defaults
                    default_val = [(arg, default) for arg, default in zip(args, defaults)]
                    default_str = " (**caution**: No default value is provided, \
please modify manually after config exported.)"

                all_augmentor[name + default_str] = {"name": name, "defaults": default_val}
                checkboxes.append(name + default_str)
        data_augmentation_question = [
            inquirer.Checkbox(name='value',
                              message='Please choose augmentors',
                              choices=checkboxes)
        ]
        data_augmentation_res = prompt(data_augmentation_question)
        if data_augmentation_res:
            for v in data_augmentation_res:
                data_augmentation[all_augmentor[v]["name"]] = all_augmentor[v]["defaults"]

    quantize_first_convolution_question = [
        inquirer.Confirm(name='value',
                         message='apply quantization at the first layer?',
                         default=True)
    ]
    quantize_first_convolution = prompt(quantize_first_convolution_question)

    return {
        'model_name': model_name,
        'task_type': task_type,
        'network_name': network_name,
        'network': {
            'quantize_first_convolution': quantize_first_convolution,
        },
        'dataset': {
            'format': dataset_format,
            'train_path': train_path,
            'test_path': test_path,
        },
        'trainer': {
            'batch_size': int(batch_size),
            'epochs': int(training_epochs),
            'optimizer': training_optimizer,
            'learning_rate_schedule': learning_rate_schedule,
            'initial_learning_rate': float(initial_learning_rate_value),
            'save_checkpoint_steps': 1000,
            'keep_checkpoint_max': 5,
        },
        'common': {
            'image_size': [int(val) for val in image_size],
            'pretrain_model': False,
            'dataset_prefetch': True,
            'data_augmentation': data_augmentation,
        },
    }
