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
import pytest
import tensorflow as tf

from lmnet import environment
from lmnet.datasets.image_folder import ImageFolderBase
from lmnet.networks.classification.darknet import Darknet
from lmnet.utils.executor import prepare_dirs
from lmnet.pre_processor import Resize
from executor.train import start_training


# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


class Dummy(ImageFolderBase):
    extend_dir = "dummy_classification"


def test_training():
    """Test only no error raised."""
    config = EasyDict()

    config.NETWORK_CLASS = Darknet
    config.DATASET_CLASS = Dummy

    config.IS_DEBUG = False
    config.IMAGE_SIZE = [28, 14]
    config.BATCH_SIZE = 2
    config.TEST_STEPS = 1
    config.MAX_STEPS = 2
    config.SAVE_CHECKPOINT_STEPS = 1
    config.KEEP_CHECKPOINT_MAX = 5
    config.SUMMARISE_STEPS = 1
    config.IS_PRETRAIN = False

    # network model config
    config.NETWORK = EasyDict()
    config.NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
    config.NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
    config.NETWORK.IMAGE_SIZE = config.IMAGE_SIZE
    config.NETWORK.BATCH_SIZE = config.BATCH_SIZE

    # daasegt config
    config.DATASET = EasyDict()
    config.DATASET.PRE_PROCESSOR = Resize(config.IMAGE_SIZE)
    config.DATASET.BATCH_SIZE = config.BATCH_SIZE

    environment.init("test_darknet")
    prepare_dirs(recreate=True)
    start_training(config)


if __name__ == '__main__':
    test_training()
