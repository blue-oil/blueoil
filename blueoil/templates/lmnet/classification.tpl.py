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
from easydict import EasyDict
import tensorflow as tf

from blueoil.common import Tasks
from blueoil.networks.classification.{{network_module}} import {{network_class}}
from blueoil.datasets.{{dataset_module}} import {{dataset_class}}
{% if data_augmentation %}from blueoil.data_augmentor import ({% for aug_name in data_augmentation %}
    {{ aug_name }},{% endfor %}
){% endif %}
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    DivideBy255,
    PerImageStandardization
)
from blueoil.nn.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = {{network_class}}

# TODO(wakisaka): should be hidden. generate dataset class on the fly.
DATASET_CLASS = type('DATASET_CLASS', ({{dataset_class}},), {{dataset_class_property}})

IMAGE_SIZE = {{image_size}}
BATCH_SIZE = {{batch_size}}
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = {{classes}}

MAX_EPOCHS = {{max_epochs}}
SAVE_CHECKPOINT_STEPS = {{save_checkpoint_steps}}
KEEP_CHECKPOINT_MAX = {{keep_checkpoint_max}}
TEST_STEPS = {{test_steps}}
SUMMARISE_STEPS = {{summarise_steps}}


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    {% if quantize_first_convolution %}DivideBy255(){% else %}PerImageStandardization(){% endif %}
])
POST_PROCESSOR = None

NETWORK = EasyDict()

NETWORK.OPTIMIZER_CLASS = {{optimizer_class}}
NETWORK.OPTIMIZER_KWARGS = {{optimizer_kwargs}}
NETWORK.LEARNING_RATE_FUNC = {{learning_rate_func}}
NETWORK.LEARNING_RATE_KWARGS = {{learning_rate_kwargs}}

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005

# quantize
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([{% if data_augmentation %}{% for aug_name, aug_val in data_augmentation.items() %}
    {{ aug_name }}({% for param_name, param_value in aug_val %}{{ param_name }}={{ param_value }}, {% endfor %}),{% endfor %}
{% endif %}])
DATASET.ENABLE_PREFETCH = {{ dataset_prefetch }}
