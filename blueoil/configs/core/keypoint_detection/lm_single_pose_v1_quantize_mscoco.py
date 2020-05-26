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
from blueoil.networks.keypoint_detection.lm_single_pose_v1 import LmSinglePoseV1Quantize
from blueoil.datasets.mscoco_2017 import MscocoSinglePersonKeypoints
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    DivideBy255,
    ResizeWithJoints,
    JointsToGaussianHeatmap
)
from blueoil.post_processor import (
    GaussianHeatmapToJoints
)
from blueoil.data_augmentor import (
    Brightness,
    Color,
    Contrast
)
from blueoil.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LmSinglePoseV1Quantize
DATASET_CLASS = MscocoSinglePersonKeypoints

IMAGE_SIZE = [256, 320]
BATCH_SIZE = 8
DATA_FORMAT = "NHWC"
TASK = Tasks.KEYPOINT_DETECTION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 2000000
SAVE_CHECKPOINT_STEPS = 3000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 10000
SUMMARISE_STEPS = 200

# distributed training
IS_DISTRIBUTION = False

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# for debug
# BATCH_SIZE = 2
# SUMMARISE_STEPS = 1
# IS_DEBUG = True

# stride of output heatmap. the smaller, the slower.
STRIDE = 8

PRE_PROCESSOR = Sequence([
    ResizeWithJoints(image_size=IMAGE_SIZE),
    JointsToGaussianHeatmap(image_size=IMAGE_SIZE,
                            stride=STRIDE, sigma=2),
    DivideBy255()
])
POST_PROCESSOR = Sequence([
    GaussianHeatmapToJoints(num_dimensions=2, stride=STRIDE, confidence_threshold=0.1)
])

step_per_epoch = 149813 // BATCH_SIZE

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {
        "values": [1e-4, 1e-3, 1e-4, 1e-5],
        "boundaries": [5000, step_per_epoch * 5, step_per_epoch * 10],
}
NETWORK.STRIDE = STRIDE
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

DATASET = EasyDict()
DATASET.IMAGE_SIZE = IMAGE_SIZE
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25))
])
DATASET.ENABLE_PREFETCH = True
