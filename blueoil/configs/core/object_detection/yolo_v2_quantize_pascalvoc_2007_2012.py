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
from blueoil.networks.object_detection.yolo_v2_quantize import YoloV2Quantize
from blueoil.datasets.pascalvoc_2007_2012 import Pascalvoc20072012
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    ResizeWithGtBoxes,
    DivideBy255,
)
from blueoil.post_processor import (
    FormatYoloV2,
    ExcludeLowScoreBox,
    NMS,
)
from blueoil.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
    SSDRandomCrop,
)
from blueoil.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = YoloV2Quantize
DATASET_CLASS = Pascalvoc20072012

IMAGE_SIZE = [416, 416]
BATCH_SIZE = 8
DATA_FORMAT = "NCHW"
TASK = Tasks.OBJECT_DETECTION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 1000000
SAVE_CHECKPOINT_STEPS = 50000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 5000
SUMMARISE_STEPS = 1000


# for debug
# IS_DEBUG = True
# SUMMARISE_STEPS = 100
# TEST_STEPS = 1000
# SAVE_CHECKPOINT_STEPS = 1000


# pretrain
IS_PRETRAIN = True
PRETRAIN_VARS = [
    'block_1/conv/kernel:0',
    'block_1/bn/beta:0',
    'block_1/bn/gamma:0',
    'block_1/bn/moving_mean:0',
    'block_1/bn/moving_variance:0',
    'block_2/conv/kernel:0',
    'block_2/bn/beta:0',
    'block_2/bn/gamma:0',
    'block_2/bn/moving_mean:0',
    'block_2/bn/moving_variance:0',
    'block_3/conv/kernel:0',
    'block_3/bn/beta:0',
    'block_3/bn/gamma:0',
    'block_3/bn/moving_mean:0',
    'block_3/bn/moving_variance:0',
    'block_4/conv/kernel:0',
    'block_4/bn/beta:0',
    'block_4/bn/gamma:0',
    'block_4/bn/moving_mean:0',
    'block_4/bn/moving_variance:0',
    'block_5/conv/kernel:0',
    'block_5/bn/beta:0',
    'block_5/bn/gamma:0',
    'block_5/bn/moving_mean:0',
    'block_5/bn/moving_variance:0',
    'block_6/conv/kernel:0',
    'block_6/bn/beta:0',
    'block_6/bn/gamma:0',
    'block_6/bn/moving_mean:0',
    'block_6/bn/moving_variance:0',
    'block_7/conv/kernel:0',
    'block_7/bn/beta:0',
    'block_7/bn/gamma:0',
    'block_7/bn/moving_mean:0',
    'block_7/bn/moving_variance:0',
    'block_8/conv/kernel:0',
    'block_8/bn/beta:0',
    'block_8/bn/gamma:0',
    'block_8/bn/moving_mean:0',
    'block_8/bn/moving_variance:0',
    'block_9/conv/kernel:0',
    'block_9/bn/beta:0',
    'block_9/bn/gamma:0',
    'block_9/bn/moving_mean:0',
    'block_9/bn/moving_variance:0',
    'block_10/conv/kernel:0',
    'block_10/bn/beta:0',
    'block_10/bn/gamma:0',
    'block_10/bn/moving_mean:0',
    'block_10/bn/moving_variance:0',
    'block_11/conv/kernel:0',
    'block_11/bn/beta:0',
    'block_11/bn/gamma:0',
    'block_11/bn/moving_mean:0',
    'block_11/bn/moving_variance:0',
    'block_12/conv/kernel:0',
    'block_12/bn/beta:0',
    'block_12/bn/gamma:0',
    'block_12/bn/moving_mean:0',
    'block_12/bn/moving_variance:0',
    'block_13/conv/kernel:0',
    'block_13/bn/beta:0',
    'block_13/bn/gamma:0',
    'block_13/bn/moving_mean:0',
    'block_13/bn/moving_variance:0',
    'block_14/conv/kernel:0',
    'block_14/bn/beta:0',
    'block_14/bn/gamma:0',
    'block_14/bn/moving_mean:0',
    'block_14/bn/moving_variance:0',
    'block_15/conv/kernel:0',
    'block_15/bn/beta:0',
    'block_15/bn/gamma:0',
    'block_15/bn/moving_mean:0',
    'block_15/bn/moving_variance:0',
    'block_16/conv/kernel:0',
    'block_16/bn/beta:0',
    'block_16/bn/gamma:0',
    'block_16/bn/moving_mean:0',
    'block_16/bn/moving_variance:0',
    'block_17/conv/kernel:0',
    'block_17/bn/beta:0',
    'block_17/bn/gamma:0',
    'block_17/bn/moving_mean:0',
    'block_17/bn/moving_variance:0',
    'block_18/conv/kernel:0',
    'block_18/bn/beta:0',
    'block_18/bn/gamma:0',
    'block_18/bn/moving_mean:0',
    'block_18/bn/moving_variance:0',
]

PRETRAIN_DIR = "saved/core/classification/quantize_darknet_ilsvrc_2012/checkpoints"
PRETRAIN_FILE = "save.ckpt-1450000"

PRE_PROCESSOR = Sequence([
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    DivideBy255()
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
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
# In the yolov2 paper, with a starting learning rate of 10−3, dividing it by 10 at 60 and 90 epochs.
# Train data num per epoch is 16551
# In first 5000 steps, use small learning rate for warmup.
_epoch_steps = 16551 // BATCH_SIZE
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [1e-6, 1e-4, 1e-5, 1e-6, 1e-7],
    "boundaries": [5000, _epoch_steps * 10, _epoch_steps * 60, _epoch_steps * 90],
}
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
NETWORK.LOSS_WARMUP_STEPS = int(12800 / BATCH_SIZE)

# quantization
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
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
    SSDRandomCrop(min_crop_ratio=0.7),
])
