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

from lmnet.common import Tasks
from lmnet.networks.classification.{{network_module}} import {{network_class}}
from lmnet.datasets.{{dataset_module}} import {{dataset_class}}
from lmnet.data_processor import Sequence
from lmnet.data_augmentor import (
    Crop,
    FlipLeftRight,
    Pad,
    Brightness,
    Color,
    Contrast,
    Hue,
)
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
    PerImageStandardization
)
from lmnet.quantizations import (
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
# In order to get instance property `classes`, instantiate DATASET_CLASS.
CLASSES = DATASET_CLASS(subset="train", batch_size=1).classes

{% if max_epochs -%}
MAX_EPOCHS = {{max_epochs}}
{%- elif max_steps -%}
MAX_STEPS = {{max_steps}}
{%- endif %}
SAVE_STEPS = {{save_steps}}
TEST_STEPS = {{test_steps}}
SUMMARISE_STEPS = {{summarise_steps}}

# distributed training
IS_DISTRIBUTION = False

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

if '{{optimizer}}' == 'GradientDescentOptimizer':
    NETWORK.OPTIMIZER_CLASS = tf.train.GradientDescentOptimizer
    NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.01}
elif '{{optimizer}}' == 'MomentumOptimizer':
    NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
    NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9, "learning_rate": {{learning_rate}}}
elif '{{optimizer}}' == 'AdamOptimizer':
    NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
    NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
else:
    raise ValueError

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
DATASET.AUGMENTOR = Sequence([
    Resize(size=IMAGE_SIZE),
    Pad(4),
    Crop(size=IMAGE_SIZE),
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
])

