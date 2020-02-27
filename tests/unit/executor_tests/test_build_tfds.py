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
import numpy as np
import pytest

from blueoil.cmd.build_tfds import run
from blueoil.cmd.train import run as train_run
from blueoil import environment
from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.tfds import TFDSClassification, TFDSObjectDetection
from blueoil.utils import config as config_util

_RUN_AS_A_SCRIPT = False


# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def setup_dataset(dataset_class, subset, **kwargs):
    dataset = dataset_class(subset=subset, **kwargs)
    return DatasetIterator(dataset, seed=0)


def test_build_tfds_classification():
    environment.setup_test_environment()

    # Build TFDS Dataset
    config_file = "unit/fixtures/configs/for_build_tfds_classification.py"
    run(config_file, overwrite=True)

    # Check if the builded dataset can be loaded with the same config file
    expriment_id = "tfds_classification"
    train_run(None, None, config_file, expriment_id, recreate=True)

    # Check if the dataset was build correctly
    train_data_num = 3
    validation_data_num = 2
    config = config_util.load(config_file)

    train_dataset = setup_dataset(TFDSClassification,
                                  subset="train",
                                  batch_size=config.BATCH_SIZE,
                                  **config.DATASET.TFDS_KWARGS)

    validation_dataset = setup_dataset(TFDSClassification,
                                       subset="validation",
                                       batch_size=config.BATCH_SIZE,
                                       **config.DATASET.TFDS_KWARGS)

    assert train_dataset.num_per_epoch == train_data_num
    assert validation_dataset.num_per_epoch == validation_data_num

    for _ in range(train_data_num):
        images, labels = train_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == config.BATCH_SIZE
        assert images.shape[1] == config.IMAGE_SIZE[0]
        assert images.shape[2] == config.IMAGE_SIZE[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == config.BATCH_SIZE
        assert labels.shape[1] == train_dataset.num_classes

    for _ in range(validation_data_num):
        images, labels = validation_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == config.BATCH_SIZE
        assert images.shape[1] == config.IMAGE_SIZE[0]
        assert images.shape[2] == config.IMAGE_SIZE[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == config.BATCH_SIZE
        assert labels.shape[1] == validation_dataset.num_classes


def test_build_tfds_object_detection():
    environment.setup_test_environment()

    # Build TFDS Dataset
    config_file = "unit/fixtures/configs/for_build_tfds_object_detection.py"
    run(config_file, overwrite=True)

    # Check if the builded dataset can be loaded with the same config file
    expriment_id = "tfds_object_detection"
    train_run(None, None, config_file, expriment_id, recreate=True)

    # Check if the dataset was build correctly
    train_data_num = 3
    validation_data_num = 2
    config = config_util.load(config_file)

    train_dataset = setup_dataset(TFDSObjectDetection,
                                  subset="train",
                                  batch_size=config.BATCH_SIZE,
                                  **config.DATASET.TFDS_KWARGS)

    validation_dataset = setup_dataset(TFDSObjectDetection,
                                       subset="validation",
                                       batch_size=config.BATCH_SIZE,
                                       **config.DATASET.TFDS_KWARGS)

    assert train_dataset.num_per_epoch == train_data_num
    assert validation_dataset.num_per_epoch == validation_data_num

    assert train_dataset.num_max_boxes == validation_dataset.num_max_boxes
    num_max_boxes = train_dataset.num_max_boxes

    for _ in range(train_data_num):
        images, labels = train_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == config.BATCH_SIZE
        assert images.shape[1] == config.IMAGE_SIZE[0]
        assert images.shape[2] == config.IMAGE_SIZE[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == config.BATCH_SIZE
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(validation_data_num):
        images, labels = validation_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == config.BATCH_SIZE
        assert images.shape[1] == config.IMAGE_SIZE[0]
        assert images.shape[2] == config.IMAGE_SIZE[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == config.BATCH_SIZE
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5
