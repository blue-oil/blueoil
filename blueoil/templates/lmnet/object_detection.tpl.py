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
{% if data_augmentation %}from lmnet.data_augmentor import ({% for augmentor in data_augmentation %}
    {{ augmentor[0] }},{% endfor %}
){% endif %}
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
CLASSES = {{classes}}

MAX_EPOCHS = {{max_epochs}}
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

NETWORK.OPTIMIZER_CLASS = {{optimizer_class}}

NETWORK.OPTIMIZER_KWARGS = {{optimizer_kwargs}}
NETWORK.LEARNING_RATE_FUNC = {{learning_rate_func}}
NETWORK.LEARNING_RATE_KWARGS = {{learning_rate_kwargs}}

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
NETWORK.LOSS_WARMUP_STEPS = int(8000 / BATCH_SIZE)

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
DATASET.AUGMENTOR = Sequence([{% if data_augmentation %}{% for augmentor in data_augmentation %}
    {{ augmentor[0] }}({% for d_name, d_value in augmentor[1] %}{{ d_name }}={{ d_value }}, {% endfor %}),{% endfor %}
{% endif %}])
