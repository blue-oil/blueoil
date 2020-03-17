#!/usr/bin/env python3
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
from blueoil.data_processor import Processor


def tf_resize_with_gt_boxes(image, gt_boxes, size=(256, 256)):
    """Resize an image and gt_boxes.

    Args:
        image    (tf.Tensor): image with 3 dim
        gt_boxes (tf.Tensor): Ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height, class_id)].
        size: [height, width]

    """
    orig_height, orig_width, _ = image.get_shape().as_list()
    height, width = size

    image = tf.image.resize(image, size)
    if gt_boxes is None:
        return image, None

    scale = [height / orig_height, width / orig_width]
    if len(gt_boxes) > 0:
        gt_boxes = tf.stack([
            gt_boxes[:, 0] * scale[1],
            gt_boxes[:, 1] * scale[0],
            gt_boxes[:, 2] * scale[1],
            gt_boxes[:, 3] * scale[0],
            gt_boxes[:, 4]
        ], 1)
        gt_boxes = tf.concat([
            tf.expand_dims(tf.minimum(gt_boxes[:, 0], width - gt_boxes[:, 2]), 1),
            tf.expand_dims(tf.minimum(gt_boxes[:, 1], height - gt_boxes[:, 3]), 1),
            gt_boxes[:, 2:]
        ], 1)
    return image, gt_boxes


class TFResize(Processor):
    """Resize an image"""

    def __init__(self, size=(256, 256)):
        """
        Args:
            size: (height, width)
        """
        self.size = size

    def __call__(self, image, **kwargs):
        """
        Args:
            image (tf.Tensor): an image tensor sized (orig_height, orig_width, channel)
        """
        return dict({'image': tf.image.resize(image, self.size)}, **kwargs)


class TFPerImageStandardization(Processor):
    """Standardization per image."""

    def __call__(self, image, **kwargs):
        return dict({'image': tf.image.per_image_standardization(image)}, **kwargs)


class TFResizeWithGtBoxes(Processor):
    """Resize image with gt boxes.

    Args:
        size: Target size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt_boxes=None, **kwargs):
        image, gt_boxes = tf_resize_with_gt_boxes(image, gt_boxes, self.size)
        return dict({'image': image, 'gt_boxes': gt_boxes}, **kwargs)


class TFDivideBy255(Processor):
    """Divide image by 255"""

    def __call__(self, image, **kwargs):
        image = tf.cast(image, tf.float32) / 255.0
        return dict({'image': image}, **kwargs)
