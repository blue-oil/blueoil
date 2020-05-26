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

from blueoil.data_processor import Processor


def resize(image, size=[256, 256]):
    """Resize an image.

    Args:
        image (np.ndarray): an image numpy array.
        size: [height, width]

    """
    width = size[1]
    height = size[0]

    if width == image.shape[1] and height == image.shape[0]:
        return image

    image = PIL.Image.fromarray(np.uint8(image))

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
        fill: Fill blank by this number. (Default value = 127.5)

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
            left = right = diff // 2
        else:
            left = diff // 2
            right = left + 1

        result[:, left:-right, :] = image

    else:
        left = right = 0
        size = origin_width
        result = np.full((size, size, image.shape[2]), fill, dtype=image.dtype)

        if diff % 2 == 0:
            top = bottom = diff // 2
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
        image (np.ndarray): An image numpy array.
        gt_boxes (np.ndarray): Ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height, class_id)].
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
        image (np.ndarray): An image numpy array.
        gt_boxes (list): Python list ground truth boxes in the image. shape is
            [num_boxes, 5(x, y, width, height)].
        size: [height, width]

    """
    origin_width = image.shape[1]
    origin_height = image.shape[0]

    if origin_width < origin_height:
        height = size[0]
        width = origin_width * height // origin_height

    else:
        width = size[1]
        height = origin_height * width // origin_width

    resized_image = resize(image, (height, width))

    scale = [height / origin_height, width / origin_width]

    if gt_boxes is not None and len(gt_boxes) != 0:
        gt_boxes[:, 0] = gt_boxes[:, 0] * scale[1]
        gt_boxes[:, 1] = gt_boxes[:, 1] * scale[0]
        gt_boxes[:, 2] = gt_boxes[:, 2] * scale[1]
        gt_boxes[:, 3] = gt_boxes[:, 3] * scale[0]

    return resized_image, gt_boxes


def resize_with_joints(image, joints, image_size):
    """Resize image with joints to target image_size.

    Args:
        image: a numpy array of shape (height, width, 3).
        joints: a numpy array of shape (num_joints, 3).
        image_size: a tuple, (new_height, new_width).

    Returns:
        resized_image: a numpy array of shape (new_height, new_width, 3).
        new_joints: a numpy array of shape (num_joints, 3).

    """

    original_height, original_width, _ = image.shape

    scale_height = image_size[0] / original_height
    scale_width = image_size[1] / original_width

    resized_image = resize(image, image_size)

    new_joints = joints.copy()
    new_joints[:, 0] *= scale_width
    new_joints[:, 1] *= scale_height

    return resized_image, new_joints


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

    return _linear_quantize(image, bit, min_value, max_value)


def _linear_quantize(x, bit, value_min, value_max):
    x = np.clip(x, value_min, value_max)
    value_range = value_max - value_min
    x = (x - value_min) / value_range

    n = (2 ** bit - 1)
    result = np.round(x * n) / n
    return result * value_range + value_min


def joints_to_gaussian_heatmap(joints, image_size,
                               num_joints=17, stride=1,
                               sigma=2):
    """Convert joints to gaussian heatmap which can be learned by networks.

    References:
        https://github.com/Microsoft/human-pose-estimation.pytorch

    Args:
        joints (np.ndarray): a numpy array of shape (num_joints).
        image_size (tuple): a tuple, (height, width).
        num_joints (int): int. (Default value = 17)
        stride (int): int, stride = image_height / heatmap_height. (Default value = 1)
        sigma (int): int, used to compute gaussian heatmap. (Default value = 2)

    Returns:
        heatmap: a numpy array of shape (height, width, num_joints).

    """

    assert num_joints == joints.shape[0]

    tmp_size = sigma * 3

    height, width = image_size

    height //= stride
    width //= stride

    heatmap = np.zeros((height, width, num_joints), dtype=np.float32)

    for i in range(num_joints):

        if joints[i, 2] > 0:
            center_x = int(joints[i, 0]) // stride
            center_y = int(joints[i, 1]) // stride

            up_left = [center_x - tmp_size, center_y - tmp_size]
            bottom_right = [center_x + tmp_size + 1, center_y + tmp_size + 1]

            if center_x >= width or center_x < 0:
                continue

            if center_y >= height or center_y < 0:
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2

            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            # Usable gaussian range
            g_x = max(0, -up_left[0]), min(bottom_right[0], width) - up_left[0]
            g_y = max(0, -up_left[1]), min(bottom_right[1], height) - up_left[1]

            # Image range
            img_x = max(0, up_left[0]), min(bottom_right[0], width)
            img_y = max(0, up_left[1]), min(bottom_right[1], height)

            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1], i] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    am = np.amax(heatmap)
    if am > 0:
        heatmap /= am

    # 10 is scaling factor of a ground-truth gaussian heatmap.
    heatmap *= 10

    return heatmap


class PerImageLinearQuantize(Processor):
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


class ResizeWithJoints(Processor):

    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, joints=None, **kwargs):
        if joints is None:
            resized_image = resize(image, size=self.image_size)
            return dict({'image': resized_image}, **kwargs)

        resized_image, new_joints = resize_with_joints(image=image, joints=joints,
                                                       image_size=self.image_size)
        return dict({'image': resized_image, 'joints': new_joints}, **kwargs)


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


class JointsToGaussianHeatmap(Processor):
    """Convert joints to gaussian heatmap which can be learned by networks.

    Use :func:`~joints_to_gaussian_heatmap` inside.

    Args:
        image_size (tuple): a tuple, (height, width).
        num_joints (int): int.
        stride (int): int, stride = image_height / heatmap_height.
        sigma (int): int, used to compute gaussian heatmap.

    Returns:
        dict:

    """

    def __init__(self, image_size, num_joints=17,
                 stride=1, sigma=3):

        self.image_size = image_size
        self.num_joints = num_joints
        self.stride = stride
        self.sigma = sigma

    def __call__(self, joints=None, **kwargs):

        if joints is None:
            return kwargs

        heatmap = joints_to_gaussian_heatmap(joints=joints, image_size=self.image_size,
                                             num_joints=self.num_joints, stride=self.stride,
                                             sigma=self.sigma)
        return dict({'heatmap': heatmap}, **kwargs)
