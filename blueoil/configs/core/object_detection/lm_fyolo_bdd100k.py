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
import tensorflow as tf
from easydict import EasyDict

from blueoil.common import Tasks
from blueoil.data_augmentor import (Brightness, Color, Contrast, FlipLeftRight,
                                    Hue, SSDRandomCrop)
from blueoil.data_processor import Sequence
from blueoil.networks.object_detection.lm_fyolo import LMFYolo
from blueoil.datasets.bdd100k import BDD100KObjectDetection
from blueoil.pre_processor import DivideBy255, ResizeWithGtBoxes
from blueoil.post_processor import NMS, ExcludeLowScoreBox, FormatYoloV2

IS_DEBUG = False

NETWORK_CLASS = LMFYolo
DATASET_CLASS = BDD100KObjectDetection

IMAGE_SIZE = [320, 320]
BATCH_SIZE = 18
DATA_FORMAT = "NHWC"
TASK = Tasks.OBJECT_DETECTION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 1000000
SAVE_CHECKPOINT_STEPS = 10000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 100


# for debug
# IS_DEBUG = True
# SUMMARISE_STEPS = 1
# SUMMARISE_STEPS = 100
# TEST_STEPS = 10000
# SUMMARISE_STEPS = 100

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    DivideBy255()
])
anchors = [
    (1.3221, 1.73145), (3.19275, 4.00944), (5.05587,
                                            8.09892), (9.47112, 4.84053), (11.2364, 10.0071)
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
    NMS(iou_threshold=nms_iou_threshold,
        max_output_size=nms_max_output_size, classes=CLASSES,),
])

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
# In the origianl yolov2 Paper, with a starting learning rate of 10−3, dividing it by 10 at 60 and 90 epochs.
# Train data num per epoch is 16551
step_per_epoch = 16551 // BATCH_SIZE
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [1e-4, 2e-2, 5e-3, 5e-4],
    "boundaries": [step_per_epoch, step_per_epoch * 80, step_per_epoch * 120],
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
NETWORK.LOSS_WARMUP_STEPS = int(8000 / BATCH_SIZE)

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
DATASET.ENABLE_PREFETCH = True
