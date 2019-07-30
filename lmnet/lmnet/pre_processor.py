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
import PIL.Image

from lmnet.data_processor import (
    Processor,
)


def resize(image, size=[256, 256]):
    """Resize an image.

    Args:
        image: an image numpy array.
        size: [height, width]
    """
    width = size[1]
    height = size[0]

    if width == image.shape[1] and height == image.shape[0]:
        return image

    image = PIL.Image.fromarray(image)

    image = image.resize([width, height])

    image = np.array(image)
    assert image.shape[0] == height
    assert image.shape[1] == width

    return image


def square(image, gt_boxes, fill=127.5):
    """Square an image.

    Args:
        image: An image numpy array.
        gt_boxes: Python list ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height)].
        fill: Fill blank by this number.
    """
    origin_width = image.shape[1]
    origin_height = image.shape[0]

    diff = abs(origin_width - origin_height)

    if diff == 0:
        return image, gt_boxes

    if origin_width < origin_height:
        top = bottom = 0
        size = origin_height
        result = np.full((size, size, image.shape[2]), fill, dtype=image.dtype)

        if diff % 2 == 0:
            left = right = int(diff / 2)
        else:
            left = diff // 2
            right = left + 1

        result[:, left:-right, :] = image

    else:
        left = right = 0
        size = origin_width
        result = np.full((size, size, image.shape[2]), fill, dtype=image.dtype)

        if diff % 2 == 0:
            top = bottom = int(diff / 2)
        else:
            top = diff // 2
            bottom = top + 1

        result[top:-bottom, :, :] = image

    if gt_boxes is not None and len(gt_boxes) != 0:
        gt_boxes[:, 0] = gt_boxes[:, 0] + left
        gt_boxes[:, 1] = gt_boxes[:, 1] + top
        gt_boxes[:, 2] = gt_boxes[:, 2]
        gt_boxes[:, 3] = gt_boxes[:, 3]

    return result, gt_boxes


def resize_with_gt_boxes(image, gt_boxes, size=(256, 256)):
    """Resize an image and gt_boxes.

    Args:
        image(np.ndarray): An image numpy array.
        gt_boxes(np.ndarray): Ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height)].
        size: [height, width]
    """

    origin_width = image.shape[1]
    origin_height = image.shape[0]

    width = size[1]
    height = size[0]

    resized_image = resize(image, (height, width))

    if gt_boxes is None:
        return resized_image, None

    resized_gt_boxes = gt_boxes.copy()

    scale = [height / origin_height, width / origin_width]

    if gt_boxes is not None and len(gt_boxes) != 0:
        resized_gt_boxes[:, 0] = gt_boxes[:, 0] * scale[1]
        resized_gt_boxes[:, 1] = gt_boxes[:, 1] * scale[0]
        resized_gt_boxes[:, 2] = gt_boxes[:, 2] * scale[1]
        resized_gt_boxes[:, 3] = gt_boxes[:, 3] * scale[0]

        # move boxes beyond boundary of image for scaling error.
        resized_gt_boxes[:, 0] = np.minimum(resized_gt_boxes[:, 0],
                                            width - resized_gt_boxes[:, 2])
        resized_gt_boxes[:, 1] = np.minimum(resized_gt_boxes[:, 1],
                                            height - resized_gt_boxes[:, 3])

    return resized_image, resized_gt_boxes


def resize_keep_ratio_with_gt_boxes(image, gt_boxes, size=(256, 256)):
    """Resize keeping ratio an image and gt_boxes.

    Args:
        image: An image numpy array.
        gt_boxes: Python list ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height)].
        size: [height, width]
    """
    origin_width = image.shape[1]
    origin_height = image.shape[0]

    if origin_width < origin_height:
        height = size[0]
        width = int(origin_width * height / origin_height)

    else:
        width = size[1]
        height = int(origin_height * width / origin_width)

    resized_image = resize(image, (height, width))

    scale = [height / origin_height, width / origin_width]

    if gt_boxes is not None and len(gt_boxes) != 0:
        gt_boxes[:, 0] = gt_boxes[:, 0] * scale[1]
        gt_boxes[:, 1] = gt_boxes[:, 1] * scale[0]
        gt_boxes[:, 2] = gt_boxes[:, 2] * scale[1]
        gt_boxes[:, 3] = gt_boxes[:, 3] * scale[0]

    return resized_image, gt_boxes


def per_image_standardization(image):
    """Image standardization per image.

    https://www.tensorflow.org/api_docs/python/image/image_adjustments#per_image_standardization

    Args:
        image: An image numpy array.
    """
    image = image.astype(np.float32)
    mean = image.mean()
    stddev = np.std(image)
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(image.size))

    image -= mean
    image /= adjusted_stddev

    return image


def per_image_linear_quantize(image, bit):
    r"""Linear quantize per image.

    .. math::
        \mathbf{Y} =
            \frac{\text{round}\big(\frac{\mathbf{X}}{max\_value} \cdot (2^{bit}-1)\big)}{2^{bit}-1} \cdot max\_value

    Args:
        image: An image numpy array.
        bit: Quantize bit.
    """
    min_value = np.min(image)
    max_value = np.max(image)

    return _liner_quantize(image, bit, min_value, max_value)


def _liner_quantize(x, bit, value_min, value_max):
    x = np.clip(x, value_min, value_max)
    value_range = value_max - value_min
    x = (x - value_min) / value_range

    n = (2 ** bit - 1)
    result = np.round(x * n) / n
    return result * value_range + value_min


class PerImageLinerQuantize(Processor):
    """Linear quantize per image.

    Use :func:`~per_image_linear_quantize` inside.

    Args:
        bit: Quantize bit.
    """

    def __init__(self, bit):
        self.bit = bit

    def __call__(self, image, **kwargs):
        image = per_image_linear_quantize(image, self.bit)
        return dict({'image': image}, **kwargs)


class PerImageStandardization(Processor):
    """Standardization per image.

    Use :func:`~per_image_standardization` inside.
    """

    def __call__(self, image, **kwargs):
        image = per_image_standardization(image)
        return dict({'image': image}, **kwargs)


class Resize(Processor):
    """Resize image.

    Use :func:`~resize` inside.

    Args:
        size: Target size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None, **kwargs):
        image = resize(image, size=self.size)
        if mask is not None:
            mask = resize(mask, size=self.size)
        return dict({'image': image, 'mask': mask}, **kwargs)


class ResizeWithGtBoxes(Processor):
    """Resize image with gt boxes.

    Use :func:`~resize_with_gt_boxes` inside.

    Args:
        size: Target size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt_boxes=None, **kwargs):
        image, gt_boxes = resize_with_gt_boxes(image, gt_boxes, self.size)
        return dict({'image': image, 'gt_boxes': gt_boxes}, **kwargs)


class ResizeWithMask(Processor):
    """Resize image and mask.

    Use :func:`~resize` inside.

    Args:
        size: Target size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None, **kwargs):
        image = resize(image, size=self.size)
        if mask is not None:
            mask = resize(mask, size=self.size)
        return dict({'image': image, 'mask': mask}, **kwargs)


class DivideBy255(Processor):
    """Divide image by 255.
    """

    def __call__(self, image, **kwargs):
        image = image / 255.0
        return dict({'image': image}, **kwargs)


# TODO(wakisaka): test.
class LetterBoxes(Processor):
    """Darknet's letter boxes"""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt_boxes=None, **kwargs):
        image, gt_boxes = resize_keep_ratio_with_gt_boxes(image, gt_boxes, self.size)
        image = image / 255.0
        image, gt_boxes = square(image, gt_boxes, fill=0.5)
        return dict({'image': image, 'gt_boxes': gt_boxes}, **kwargs)
