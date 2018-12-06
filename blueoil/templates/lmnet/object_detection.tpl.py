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
from lmnet.networks.object_detection.{{network_module}} import {{network_class}}
from lmnet.datasets.{{dataset_module}} import {{dataset_class}}
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    ResizeWithGtBoxes,
    DivideBy255,
    PerImageStandardization,
)
from lmnet.post_processor import (
    FormatYoloV2,
    ExcludeLowScoreBox,
    NMS,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
    SSDRandomCrop,
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = {{network_class}}

# TODO(wakisaka): should be hidden. generate dataset class on the fly.
DATASET_CLASS = type('DATASET_CLASS', ({{dataset_class}},), {{dataset_class_property}})

IMAGE_SIZE = {{image_size}}
BATCH_SIZE = {{batch_size}}
DATA_FORMAT = "NHWC"
TASK = Tasks.OBJECT_DETECTION
# In order to get instance property `classes`, instantiate DATASET_CLASS.
dataset_obj = DATASET_CLASS(subset="train", batch_size=1)
CLASSES = dataset_obj.classes
step_per_epoch = float(dataset_obj.num_per_epoch)/BATCH_SIZE

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
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    {% if quantize_first_convolution %}DivideBy255(){% else %}PerImageStandardization(){% endif %}
])
anchors = [
    (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)
]
score_threshold = 0.05
nms_iou_threshold = 0.5
nms_max_output_size = 100
POST_PROCESSOR = Sequence([
    FormatYoloV2(
        image_size=IMAGE_SIZE,
        classes=CLASSES,
        anchors=anchors,
        data_format=DATA_FORMAT,
    ),
    ExcludeLowScoreBox(threshold=score_threshold),
    NMS(iou_threshold=nms_iou_threshold, max_output_size=nms_max_output_size, classes=CLASSES,),
])

NETWORK = EasyDict()

if '{{optimizer}}' == 'GradientDescentOptimizer':
    NETWORK.OPTIMIZER_CLASS = tf.train.GradientDescentOptimizer
elif '{{optimizer}}' == 'MomentumOptimizer':
    NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
    NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
elif '{{optimizer}}' == 'AdamOptimizer':
    NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer

if '{{learning_rate_setting}}' != 'fixed':
    NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
                
if '{{learning_rate_setting}}' == 'tune1':
    NETWORK.LEARNING_RATE_KWARGS = {
        "values": [{{initial_learning_rate}}, {{initial_learning_rate}} / 10, {{initial_learning_rate}} / 100],
        "boundaries": [int((step_per_epoch * (MAX_EPOCHS - 1)) / 2), int(step_per_epoch * (MAX_EPOCHS - 1))],
    }
elif '{{learning_rate_setting}}' == 'tune2':
    NETWORK.LEARNING_RATE_KWARGS = {
        "values": [{{initial_learning_rate}}, {{initial_learning_rate}} / 10, {{initial_learning_rate}} / 100, {{initial_learning_rate}} / 1000],
        "boundaries": [int((step_per_epoch * (MAX_EPOCHS - 1)) * 1 / 3), int((step_per_epoch * (MAX_EPOCHS - 1)) * 2 / 3), int(step_per_epoch * (MAX_EPOCHS - 1))],
    }
elif '{{learning_rate_setting}}' == 'tune3':
    if MAX_EPOCHS < 4:
        raise ValueError("epoch number must be >= 4, when tune3 is selected.")
    NETWORK.LEARNING_RATE_KWARGS = {
        "values": [{{initial_learning_rate}} / 1000, {{initial_learning_rate}}, {{initial_learning_rate}} / 10, {{initial_learning_rate}} / 100, {{initial_learning_rate}} / 1000],
        "boundaries": [int(step_per_epoch * 1), int((step_per_epoch * (MAX_EPOCHS - 1)) * 1 / 3), int((step_per_epoch * (MAX_EPOCHS - 1)) * 2 / 3), int(step_per_epoch * (MAX_EPOCHS - 1))],
    }
elif '{{learning_rate_setting}}' == 'fixed':
    NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9, "learning_rate": {{initial_learning_rate}}}
else:
    raise ValueError

NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ANCHORS = anchors
NETWORK.OBJECT_SCALE = 5.0
NETWORK.NO_OBJECT_SCALE = 1.0
NETWORK.CLASS_SCALE = 1.0
NETWORK.COORDINATE_SCALE = 1.0
NETWORK.LOSS_IOU_THRESHOLD = 0.6
NETWORK.WEIGHT_DECAY_RATE = 0.0005
NETWORK.SCORE_THRESHOLD = score_threshold
NETWORK.NMS_IOU_THRESHOLD = nms_iou_threshold
NETWORK.NMS_MAX_OUTPUT_SIZE = nms_max_output_size
NETWORK.SEEN_THRESHOLD = 8000

# quantize
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = True
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    FlipLeftRight(is_bounding_box=True),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
    SSDRandomCrop(min_crop_ratio=0.7),
])
