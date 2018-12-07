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
import re

import whaaaaat
from jinja2 import Environment, FileSystemLoader

task_type_choices = [
    'classification',
    'object_detection',
    # TODO(wakisaka): Remove the commnet-out when support semantic segmentation.
    # 'semantic_segmentation'
]

classification_network_definitions = [
    {
        'name': 'LmnetV1Quantize',
        'desc': 'Quantized Lmnet version 1. Accuracy is better than LmnetV0Quantize.',
    },
]

object_detection_network_definitions = [
    {
        'name': 'LMFYoloQuantize',
        'desc': 'YOLO-like object detection network.',
    },
]

# TODO(wakisaka): Remove the commnet-out when support semantic segmentation.
# semantic_segmentation_network_definitions = [
#     {
#         'name': 'LmSegnetV1Quantize',
#         'desc': 'Quantized LeapMind original semantic segmentation network, version 1.',
#     },
#     {
#         'name': 'LmSegnetV0Quantize',
#         'desc': 'Quantized LeapMind original semantic segmentation network, version 0.',
#     },
# ]

IMAGE_SIZE_VALIDATION = {
    "LmnetV1Quantize": {
        "max_size": 512,
        "divider": 16,
    },
    "LMFYoloQuantize": {
        "max_size": 480,
        "divider": 32,
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

# TODO(wakisaka): Remove the commnet-out when support semantic segmentation.
# semantic_segmentation_dataset_formats = [
#     {
#         'name': "CamVid",
#         'desc': "CamVid compatible",
#     },
#     {
#         'name': "CityScapes",
#         'desc': "CityScapes compatible",
#     },
# ]


def network_name_choices(task_type):
    if task_type == 'classification':
        return [definition['name'] for definition in classification_network_definitions]
    elif task_type == 'object_detection':
        return [definition['name'] for definition in object_detection_network_definitions]
    # TODO(wakisaka): Remove the commnet-out when support semantic segmentation.
    # elif task_type == 'semantic_segmentation':
    #     return [definition['name'] for definition in semantic_segmentation_network_definitions]


def dataset_format_choices(task_type):
    if task_type == 'classification':
        return [definition['name'] for definition in classification_dataset_formats]
    elif task_type == 'object_detection':
        return [definition['name'] for definition in object_detection_dataset_formats]
    # TODO(wakisaka): Remove the commnet-out when support semantic segmentation.
    # elif task_type == 'semantic_segmentation':
    #     return [definition['name'] for definition in semantic_segmentation_dataset_formats]


def default_batch_size(task_type):
    if task_type == 'classification':
        return '10'
    elif task_type == 'object_detection':
        return '16'
    # TODO(wakisaka): Remove the commnet-out when support semantic segmentation.
    # elif task_type == 'semantic_segmentation':
    #     return '8'


def prompt(question):
    input_type = (not ('input_type' in question)) or question.pop('input_type')

    answers = whaaaaat.prompt(question)
    if input_type == 'integer' and not answers['value'].isdigit():
        question['input_type'] = input_type
        return prompt(question)
    return answers['value']


def generate_image_size_validate(network_name):
    """Generate image_size_validate depending on task_type.

    Args:
        network_name(string): network name.

    Return: validate funciton.
    """
    max_size = IMAGE_SIZE_VALIDATION[network_name]["max_size"]
    divider = IMAGE_SIZE_VALIDATION[network_name]["divider"]

    def image_size_validate(raw):
        # change to tuple (height, width).
        image_size = image_size_filter(raw)
        image_size = (int(size) for size in image_size)

        for size in image_size:
            if not size % divider == 0:
                return "Image size should be multiple of {}, but image size is {}".format(divider, raw)

            if size > max_size:
                return "Image size should be lower than {} but image size is {}".format(max_size, raw)

        return True

    return image_size_validate


def image_size_filter(raw):
    match = re.match(r"([0-9]+)[^0-9]+([0-9]+)", raw)

    # raw: 128x128 -> ('128', '128')
    image_size = match.groups()

    return image_size


def save_config(blueoil_config):
    env = Environment(loader=FileSystemLoader('./blueoil/templates', encoding='utf8'))
    tpl = env.get_template('blueoil-config.tpl.yml')

    applied = tpl.render(blueoil_config)
    config_file = blueoil_config['model_name'] + ".yml"
    with open(config_file, 'w') as fp:
        fp.write(applied)
    return config_file


def ask_questions():
    model_name_question = {
        'type': 'input',
        'name': 'value',
        'message': 'your model name ():',
        'input_type': 'name_identifier'
    }
    model_name = prompt(model_name_question)

    task_type_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'choose task type',
        'choices': task_type_choices
    }
    task_type = prompt(task_type_question)

    network_name_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'choose network',
        'choices': network_name_choices(task_type)
    }
    network_name = prompt(network_name_question)

    dataset_format_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'choose dataset format',
        'choices': dataset_format_choices(task_type)
    }
    dataset_format = prompt(dataset_format_question)

    train_dataset_path_question = {
        'type': 'input',
        'name': 'value',
        'message': 'training dataset path:',
    }
    train_path = prompt(train_dataset_path_question)

    enable_test_dataset_path_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'set validataion dataset? \
(if answer no, the dataset will be separated for training and validation by 9:1 ratio.)',
        'choices': ['yes', 'no']
    }
    enable_test_dataset_path = prompt(enable_test_dataset_path_question)

    test_dataset_path_question = {
        'type': 'input',
        'name': 'value',
        'message': 'validataion dataset path:',
    }
    if enable_test_dataset_path == 'yes':
        test_path = prompt(test_dataset_path_question)
    else:
        test_path = ''

    batch_size_question = {
        'type': 'input',
        'name': 'value',
        'message': 'batch size (integer):',
        'input_type': 'integer',
        'default': default_batch_size(task_type),
    }
    batch_size = prompt(batch_size_question)

    image_size_question = {
        'type': 'input',
        'name': 'value',
        'message': 'image size (integer x integer):',
        'default': '128x128',
        "filter": image_size_filter,
        "validate": generate_image_size_validate(network_name),
    }
    image_size = prompt(image_size_question)

    training_epochs_question = {
        'type': 'input',
        'name': 'value',
        'message': 'how many epochs do you run training (integer):',
        'input_type': 'integer',
        'default': '100'
    }
    training_epochs = prompt(training_epochs_question)

    training_optimizer_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'select optimizer:',
        'choices': ['MomentumOptimizer',
                    'GradientDescentOptimizer',
                    'AdamOptimizer',
                    'AdadeltaOptimizer',
                    'AdagradOptimizer',
                    'ProximalAdagradOptimizer',
                    'ProximalGradientDescentOptimizer',
                    'RMSPropOptimizer'],
        'default': 'MomentumOptimizer'
    }
    training_optimizer = prompt(training_optimizer_question)
    
    initial_learning_rate_value_question = {
        'type': 'input',
        'name': 'value',
        'message': 'initial learning rate:',
        'default': '0.001'
    }
    initial_learning_rate_value = prompt(initial_learning_rate_value_question)

    training_learning_rate_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'choose learning rate setting(tune1 / tune2 / tune3 / fixed):',
        'choices': ['tune1 -> "2 times decay"', 'tune2 -> "3 times decay"', 'tune3 -> "warm-up and 3 times decay"', 'fixed'],
        'default': 'tune1 -> "2 times decay"',
    }
    choices_key_map = {
        'tune1 -> "2 times decay"': 'tune1',
        'tune2 -> "3 times decay"': 'tune2',
        'tune3 -> "warm-up and 3 times decay"': 'tune3',
        'fixed': 'fixed',
    }
    tmp_learning_rate_setting = prompt(training_learning_rate_question)
    training_learning_rate_setting = choices_key_map[tmp_learning_rate_setting]

    quantize_first_convolution_question = {
        'type': 'rawlist',
        'name': 'value',
        'message': 'apply quantization at the first layer?',
        'choices': ['yes', 'no']
    }
    quantize_first_convolution= prompt(quantize_first_convolution_question)

    r = {}
    for k, v in locals().items():
        if k != 'r' and not k.endswith("question"):
            r[k] = v
    return r


def main():
    blueoil_config = ask_questions()
    config_filename = save_config(blueoil_config)

    print('')
    print('A new configuration file generated: %s' % (config_filename))
    print('  - Your next step is training.')
    print('  - You can customize some miscellaneous settings according to the comment.')


if __name__ == '__main__':
    main()
