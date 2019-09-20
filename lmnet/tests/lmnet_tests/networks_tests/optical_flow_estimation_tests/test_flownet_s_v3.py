# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
import numpy as np
import pytest
import tensorflow as tf

from executor.train import start_training
from lmnet import environment
from lmnet.utils.executor import prepare_dirs
from lmnet.data_processor import Sequence
from lmnet.datasets.optical_flow_estimation import FlyingChairs
from lmnet.networks.optical_flow_estimation.flownet_s_v3 import (
    FlowNetSV3, FlowNetSV3Quantized
)
from lmnet.networks.optical_flow_estimation.data_augmentor import (
    Brightness, Color, Contrast, Gamma, GaussianBlur, GaussianNoise, Hue,
    FlipLeftRight, FlipTopBottom, Scale, Rotate, Translate
)
from lmnet.networks.optical_flow_estimation.pre_processor import (
    DevideBy255
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)
from lmnet.common import Tasks


# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_quantize_training():
    """Test only that no error raised."""
    config = EasyDict()

    config.NETWORK_CLASS = FlowNetSV3Quantized
    config.DATASET_CLASS = FlyingChairs

    config.IS_DEBUG = False
    config.IMAGE_SIZE = [384, 512]
    config.BATCH_SIZE = 8
    config.TEST_STEPS = 200
    config.MAX_STEPS = 5000
    config.SAVE_CHECKPOINT_STEPS = 100
    config.KEEP_CHECKPOINT_MAX = 5
    config.SUMMARISE_STEPS = 100
    config.IS_PRETRAIN = False
    config.IS_DISTRIBUTION = False
    config.TASK = Tasks.OPTICAL_FLOW_ESTIMATION

    # network model config
    config.NETWORK = EasyDict()
    config.NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
    config.NETWORK.OPTIMIZER_KWARGS = {"beta1": 0.9, "beta2": 0.999}
    config.NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
    config.NETWORK.LEARNING_RATE_KWARGS = {
        "values": [0.0000125, 0.00005],
        "boundaries": [5000]
    }

    config.NETWORK.IMAGE_SIZE = config.IMAGE_SIZE
    config.NETWORK.BATCH_SIZE = config.BATCH_SIZE
    config.NETWORK.DATA_FORMAT = "NHWC"
    config.NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
    config.NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
        'bit': 2,
        'max_value': 2.0
    }
    config.NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
    config.NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

    # dataset config
    config.DATASET = EasyDict()
    config.DATASET.PRE_PROCESSOR = None
    config.DATASET.BATCH_SIZE = config.BATCH_SIZE
    config.DATASET.DATA_FORMAT = "NHWC"
    config.DATASET.VALIDATION_RATE = 0.2
    config.DATASET.VALIDATION_SEED = 2019
    config.DATASET.AUGMENTOR = Sequence([
        # Geometric transformation
        FlipLeftRight(0.5),
        FlipTopBottom(0.5),
        Translate(-0.2, 0.2),
        Rotate(-17, +17),
        Scale(1.0, 2.0),
        # Pixel-wise augmentation
        Brightness(0.8, 1.2),
        Contrast(0.2, 1.4),
        Color(0.5, 2.0),
        Gamma(0.7, 1.5),
        # Hue(-128.0, 128.0),
        GaussianNoise(0.0, 10.0)
    ])
    config.DATASET.PRE_PROCESSOR = Sequence([
        DevideBy255(),
    ])
    environment.init("test_flownet_s_v3_quantize")
    prepare_dirs(recreate=True)
    start_training(config)


if __name__ == '__main__':
    test_quantize_training()
