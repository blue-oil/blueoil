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
import tensorflow as tf
from easydict import EasyDict

from executor.train import start_training
from lmnet import environment
from lmnet.common import Tasks
from blueoil.networks.object_detection.yolo_v1 import YoloV1
from blueoil.datasets.pascalvoc_2007 import Pascalvoc2007
from lmnet.pre_processor import ResizeWithGtBoxes
from blueoil.utils.executor import prepare_dirs

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_offset_boxes():
    """Test from_real_to_yolo is inverse function of from_yolo_to_real."""
    # shape is [batch_size, cell_size, cell_size, boxes_per_cell]
    expected_x = np.array([
        [
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
        ],
        [
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
        ],
        [
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
            ],
        ],
    ])

    # shape is [batch_size, cell_size, cell_size, boxes_per_cell]
    expected_y = np.array([
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],

            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ],
            [
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
            ]
        ],
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],

            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ],
            [
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
            ]
        ],
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
            ],

            [
                [2, 2],
                [2, 2],
                [2, 2],
                [2, 2],
            ],
            [
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
            ]
        ],
    ])

    model = YoloV1(
        cell_size=4,
        boxes_per_cell=2,
        image_size=[448, 448],
        batch_size=3,
    )

    offset_x, offset_y = model.offset_boxes()

    assert np.all(offset_x == expected_x)
    assert np.all(offset_y == expected_y)


def test_convert_boxes_space_inverse():
    """Test from_real_to_yolo is inverse function of from_yolo_to_real."""
    model = YoloV1(
        cell_size=2,
        boxes_per_cell=1,
        image_size=[400, 400],
        batch_size=1,
    )

    tf.InteractiveSession()
    boxes = np.array([
        [
            [
                [
                    [100, 200, 50, 80, ],
                ],
                [
                    [50, 100, 400, 200, ],
                ],

            ],
            [
                [
                    [10, 20, 300, 380, ],
                ],
                [
                    [200, 400, 40, 44, ],
                ],
            ],
        ],
    ])

    boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.float32)

    yolo_boxes = model.convert_boxes_space_from_real_to_yolo(boxes_tensor)

    # real -> yolo -> real
    reversed_boxes = model.convert_boxes_space_from_yolo_to_real(
        model.convert_boxes_space_from_real_to_yolo(boxes_tensor)
    )

    assert np.allclose(reversed_boxes.eval(), boxes)

    # yolo -> real -> yolo
    yolo_boxes = model.convert_boxes_space_from_real_to_yolo(boxes_tensor)
    reversed_boxes = model.convert_boxes_space_from_real_to_yolo(
        model.convert_boxes_space_from_yolo_to_real(yolo_boxes)
    )
    assert np.allclose(reversed_boxes.eval(), yolo_boxes.eval())


def test_training():
    """Test only that no error raised."""
    config = EasyDict()

    config.NETWORK_CLASS = YoloV1
    config.DATASET_CLASS = Pascalvoc2007

    config.IS_DEBUG = False
    config.IMAGE_SIZE = [70, 70]
    config.BATCH_SIZE = 4
    config.TEST_STEPS = 1
    config.MAX_STEPS = 2
    config.SAVE_CHECKPOINT_STEPS = 1
    config.KEEP_CHECKPOINT_MAX = 5
    config.SUMMARISE_STEPS = 1
    config.IS_PRETRAIN = False
    config.TASK = Tasks.OBJECT_DETECTION

    # network model config
    config.NETWORK = EasyDict()
    config.NETWORK.IMAGE_SIZE = config.IMAGE_SIZE
    config.NETWORK.BATCH_SIZE = config.BATCH_SIZE

    # daasegt config
    config.DATASET = EasyDict()
    config.DATASET.PRE_PROCESSOR = ResizeWithGtBoxes(config.IMAGE_SIZE)
    config.DATASET.BATCH_SIZE = config.BATCH_SIZE

    environment.init("test_yolov_1")
    prepare_dirs(recreate=True)
    start_training(config)


if __name__ == '__main__':
    test_training()
    test_offset_boxes()
    test_convert_boxes_space_inverse()
