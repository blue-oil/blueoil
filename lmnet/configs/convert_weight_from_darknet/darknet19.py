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

from lmnet.common import Tasks
from blueoil.nn.datasets.ilsvrc_2012 import Ilsvrc2012
from lmnet.networks.classification.darknet import Darknet
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)

NETWORK_CLASS = Darknet
DATASET_CLASS = Ilsvrc2012

IS_DEBUG = False

IMAGE_SIZE = [448, 448]
BATCH_SIZE = 4
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

# for debug
# MAX_STEPS = 100
# SAVE_CHECKPOINT_STEPS = 100
# TEST_STEPS = 10
# SUMMARISE_STEPS = 100
# IS_PRETRAIN = False
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
