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

from blueoil.cmd.train import start_training
from blueoil import environment
from blueoil.common import Tasks
from blueoil.data_processor import Sequence
from blueoil.networks.object_detection.yolo_v2 import YoloV2
from blueoil.datasets.pascalvoc_2007 import Pascalvoc2007
from blueoil.pre_processor import ResizeWithGtBoxes
from blueoil.post_processor import NMS, ExcludeLowScoreBox, FormatYoloV2
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
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
        ],
        [
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
        ],
        [
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ],
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
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
                [0, 0],
            ],
            [
                [1, 1],
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
                [2, 2],
            ],
            [
                [3, 3],
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
                [0, 0],
            ],
            [
                [1, 1],
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
                [2, 2],
            ],
            [
                [3, 3],
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
                [0, 0],
            ],
            [
                [1, 1],
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
                [2, 2],
            ],
            [
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
            ]
        ],
    ])

    anchors = [(4.0, 2.0), (3.5, 2.5)]

    # shape is [batch_size, cell_size, cell_size, boxes_per_cell]
    expected_w = np.broadcast_to(np.array([4.0, 3.5]), (3, 4, 5, 2))

    # shape is [batch_size, cell_size, cell_size, boxes_per_cell]
    expected_h = np.broadcast_to(np.array([2.0, 2.5]), (3, 4, 5, 2))

    model = YoloV2(
        anchors=anchors,
        image_size=[128, 160],
        batch_size=3,
        is_dynamic_image_size=True,
    )

    tf.InteractiveSession()
    offset_x, offset_y, offset_w, offset_h = model.offset_boxes()

    # import ipdb; ipdb.set_trace()
    assert np.all(offset_x.eval() == expected_x)
    assert np.all(offset_x.eval()[:, :, 0, :] == 0)
    assert np.all(offset_x.eval()[:, :, 1, :] == 1)
    assert np.all(offset_x.eval()[:, :, 2, :] == 2)

    assert np.all(offset_y.eval() == expected_y)
    assert np.all(offset_y.eval()[:, 0, :, :] == 0)
    assert np.all(offset_y.eval()[:, 1, :, :] == 1)
    assert np.all(offset_y.eval()[:, 2, :, :] == 2)

    assert np.all(offset_w.eval() == expected_w)
    assert np.all(offset_w.eval()[:, :, :, 0] == anchors[0][0])
    assert np.all(offset_w.eval()[:, :, :, 1] == anchors[1][0])

    assert np.all(offset_h.eval() == expected_h)
    assert np.all(offset_h.eval()[:, :, :, 0] == anchors[0][1])
    assert np.all(offset_h.eval()[:, :, :, 1] == anchors[1][1])


def test_calculate_truth_and_masks():

    model = YoloV2(
        anchors=[(1.0, 1.0), (1.5, 1.5)],
        image_size=[128, 256],
        batch_size=2,
        num_max_boxes=5,
        loss_warmup_steps=0,
    )

    gt_boxes_list = [
        [
            [33, 43, 64, 50, 3],
            [113, 62, 36, 26, 1],
            [173, 53, 30, 32, 2],
            [59, 93, 28, 18, 0],
            [145, 79, 66, 67, 0],
        ],
        [
            [130, 67, 32, 22, 1],
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, -1],
        ],
    ]

    expected_cell_gt_boxes = np.array([
        [
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [33., 43., 64., 50., 3.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[113., 62., 36., 26., 1.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[173., 53., 30., 32., 2.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[59., 93., 28., 18., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [145., 79., 66., 67., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
        ],
        [
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[130., 67., 32., 22., 1.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
            [
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
            ],
        ],
    ])

    expected_object_masks = [
        [
            [
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
            [
                [[0.],
                 [0.]],
                [[0.],
                 [1.]],
                [[0.],
                 [0.]],
                [[1.],
                 [0.]],
                [[0.],
                 [0.]],
                [[1.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
            [
                [[0.],
                 [0.]],
                [[1.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [1.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
            [
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
        ],
        [
            [
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
            [
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
            [
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[1.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
            [
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
                [[0.],
                 [0.]],
            ],
        ]
    ]

    gt_boxes_list = tf.convert_to_tensor(gt_boxes_list, dtype=tf.float32)
    # TODO(wakisaka): prepare numpy predict_boxes.
    predict_boxes = tf.convert_to_tensor(expected_cell_gt_boxes, dtype=tf.float32)
    cell_gt_boxes, truth_confidence, object_masks, coordinate_masks =\
        model.loss_function._calculate_truth_and_masks(gt_boxes_list, predict_boxes, global_step=0)

    tf.InteractiveSession()
    cell_gt_boxes_val = cell_gt_boxes.eval()
    object_masks_val = object_masks.eval()
    coordinate_masks_val = coordinate_masks.eval()

    assert np.all(cell_gt_boxes_val == expected_cell_gt_boxes)
    assert np.all(object_masks_val == expected_object_masks)
    assert np.all(coordinate_masks_val == expected_object_masks)


def test_convert_boxes_space_inverse():
    """Test from_real_to_yolo is inverse function of from_yolo_to_real."""
    model = YoloV2(
        anchors=[(2.0, 4.0), (3.5, 4.5)],
        image_size=[64, 96],
        batch_size=1,
    )

    tf.InteractiveSession()

    # shape is  [batch_size, image_size[0]/32, image_size[1]/32, boxes_per_cell, 4(center_x, center_y, w, h)]
    boxes = np.array([
        [
            [
                [
                    [10, 20, 5, 8, ],
                ],
                [
                    [5, 10, 40, 20, ],
                ],
                [
                    [90, 32, 2, 10],
                ],

            ],
            [
                [
                    [1, 2, 30, 38, ],
                ],
                [
                    [20, 40, 4, 4, ],
                ],
                [
                    [90, 32, 2, 10],
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


def test_reorg():

    inputs_shape = [1, 12, 12, 8]
    inputs_np = np.random.uniform(-10., 10., size=inputs_shape).astype(np.float32)

    inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
    model = YoloV2(
        anchors=[(1.0, 1.0), (1.5, 1.5)],
        image_size=[128, 256],
        batch_size=2,
        num_max_boxes=5,
        loss_warmup_steps=0,
    )

    outputs = model._reorg("reorg", inputs, stride=2, data_format="NHWC", use_space_to_depth=False)

    tf.InteractiveSession()

    outputs_np = outputs.eval()

    assert outputs_np.shape == (inputs_shape[0], inputs_shape[1]/2, inputs_shape[2]/2, inputs_shape[3]*2*2,)

    outputs2 = model._reorg("reorg", inputs, stride=2, data_format="NHWC", use_space_to_depth=True)

    assert np.all(outputs_np == outputs2.eval())


def test_training():
    """Test only that no error raised."""
    config = EasyDict()

    config.NETWORK_CLASS = YoloV2
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
    config.NETWORK.DATA_FORMAT = "NHWC"

    # dataset config
    config.DATASET = EasyDict()
    config.DATASET.PRE_PROCESSOR = ResizeWithGtBoxes(config.IMAGE_SIZE)
    config.DATASET.BATCH_SIZE = config.BATCH_SIZE
    config.DATASET.DATA_FORMAT = "NHWC"

    environment.init("test_yolo_v2")
    prepare_dirs(recreate=True)
    start_training(config)


def test_yolov2_post_process():
    tf.InteractiveSession()

    image_size = [96, 64]
    batch_size = 2
    classes = Pascalvoc2007.classes
    anchors = [(0.1, 0.2), (1.2, 1.1)]
    data_format = "NHWC"
    score_threshold = 0.25
    nms_iou_threshold = 0.5

    model = YoloV2(
        image_size=image_size,
        batch_size=batch_size,
        classes=classes,
        anchors=anchors,
        data_format=data_format,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )

    post_process = Sequence([
        FormatYoloV2(
            image_size=image_size,
            classes=classes,
            anchors=anchors,
            data_format=data_format,
        ),
        ExcludeLowScoreBox(threshold=score_threshold),
        NMS(
            iou_threshold=nms_iou_threshold,
            classes=classes,
        ),
    ])

    shape = (batch_size, len(anchors) * (len(classes) + 5), image_size[0]//32, image_size[1]//32)
    np_output = np.random.uniform(-2., 2., size=shape).astype(np.float32)
    output = tf.constant(np_output)

    ys = model.post_process(output)

    expected_ys = post_process(outputs=np_output)["outputs"]

    for y, expected_y in zip(ys, expected_ys):
        assert np.allclose(y.eval(), expected_y), (y.eval(), expected_y)


if __name__ == '__main__':
    test_offset_boxes()
    test_calculate_truth_and_masks()
    test_convert_boxes_space_inverse()
    test_reorg()
    test_training()
    test_yolov2_post_process()
