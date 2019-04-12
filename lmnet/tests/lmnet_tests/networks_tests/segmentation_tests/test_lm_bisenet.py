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
from lmnet.datasets.camvid import Camvid
from lmnet.networks.segmentation.lm_bisenet import LMBiSeNet
from lmnet.data_processor import Sequence
from lmnet.post_processor import (
    Bilinear,
    Softmax,
)
from lmnet.pre_processor import Resize
from lmnet.utils.executor import prepare_dirs


# Apply reset_default_graph() and set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


class DummyCamvid(Camvid):
    extend_dir = "camvid"


def test_training():
    """Verify only that no error raised."""
    config = EasyDict()

    config.NETWORK_CLASS = LMBiSeNet
    config.DATASET_CLASS = DummyCamvid

    config.IS_DEBUG = False
    config.IMAGE_SIZE = [128, 160]
    config.BATCH_SIZE = 2
    config.TEST_STEPS = 1
    config.MAX_STEPS = 2
    config.SAVE_STEPS = 1
    config.SUMMARISE_STEPS = 1
    config.IS_PRETRAIN = False
    config.IS_DISTRIBUTION = False

    # network model config
    config.NETWORK = EasyDict()
    config.NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
    config.NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
    config.NETWORK.IMAGE_SIZE = config.IMAGE_SIZE
    config.NETWORK.BATCH_SIZE = config.BATCH_SIZE
    config.NETWORK.DATA_FORMAT = "NHWC"

    # daasegt config
    config.DATASET = EasyDict()
    config.DATASET.PRE_PROCESSOR = Resize(config.IMAGE_SIZE)
    config.DATASET.BATCH_SIZE = config.BATCH_SIZE
    config.DATASET.DATA_FORMAT = "NHWC"

    environment.init("test_lm_bisenet")
    prepare_dirs(recreate=True)
    start_training(config)


def test_lm_bisenet_post_process():
    """Verify LMBiSeNet.post_process() is the same as Bilinear and Softmax post process"""
    tf.InteractiveSession()

    image_size = [96, 64]
    batch_size = 2
    classes = range(5)
    data_format = "NHWC"

    model = LMBiSeNet(
        image_size=image_size,
        batch_size=batch_size,
        classes=classes,
        data_format=data_format,
    )

    post_process = Sequence([
        Bilinear(
            size=image_size,
            data_format=data_format,
            compatible_tensorflow_v1=True,
        ),
        Softmax()
    ])

    shape = (batch_size, image_size[0]//8, image_size[1]//8, len(classes))
    np_output = np.random.uniform(-10., 10., size=shape).astype(np.float32)
    output = tf.constant(np_output)

    output = model.post_process(output)

    expected = post_process(outputs=np_output)["outputs"]

    assert np.allclose(output.eval(), expected, atol=1e-5, rtol=1e-5)
