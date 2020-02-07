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
import pytest
import tensorflow as tf
from easydict import EasyDict

from executor.train import start_training
from lmnet import environment
from lmnet.common import Tasks
from lmnet.datasets.pascalvoc_2007 import Pascalvoc2007
from lmnet.networks.object_detection.yolo_v2_quantize import YoloV2Quantize
from lmnet.pre_processor import ResizeWithGtBoxes
from blueoil.nn.quantizations import binary_channel_wise_mean_scaling_quantizer, linear_mid_tread_half_quantizer
from lmnet.utils.executor import prepare_dirs

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_training():
    """Test only no error raised."""

    config = EasyDict()

    config.NETWORK_CLASS = YoloV2Quantize
    config.DATASET_CLASS = Pascalvoc2007

    config.IS_DEBUG = False
    config.IMAGE_SIZE = [128, 160]
    config.BATCH_SIZE = 2
    config.TEST_STEPS = 1
    config.MAX_STEPS = 2
    config.SAVE_CHECKPOINT_STEPS = 1
    config.KEEP_CHECKPOINT_MAX = 5
    config.SUMMARISE_STEPS = 1
    config.IS_PRETRAIN = False
    config.TASK = Tasks.OBJECT_DETECTION

    # network model config
    config.NETWORK = EasyDict()
    config.NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
    config.NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
    config.NETWORK.IMAGE_SIZE = config.IMAGE_SIZE
    config.NETWORK.BATCH_SIZE = config.BATCH_SIZE
    config.NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
    config.NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
        'bit': 2,
        'max_value': 2.0
    }
    config.NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
    config.NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

    # daasegt config
    config.DATASET = EasyDict()
    config.DATASET.PRE_PROCESSOR = ResizeWithGtBoxes(config.IMAGE_SIZE)
    config.DATASET.BATCH_SIZE = config.BATCH_SIZE

    environment.init("test_yolov_2_quantize")
    prepare_dirs(recreate=True)
    start_training(config)


if __name__ == '__main__':
    test_training()
