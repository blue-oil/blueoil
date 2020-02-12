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
# -*- coding:utf-8 -*-
import math
import random
from random import randint

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from lmnet import data_processor, pre_processor
from blueoil.utils.box import fill_dummy_boxes, crop_boxes, iou


class Blur(data_processor.Processor):
    """Gaussian blur filter.

    Reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur

    Args:
        value (int | list | tuple): Blur radius. Default is random number from 0 to 1. References default is 2.
    """

    def __init__(self, value=(0, 1)):

        if type(value) in {int, float}:
            min_value = 0
            max_value = value

        elif len(value) == 2:
            min_value, max_value = value

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(value)))

        assert min_value >= 0
        assert max_value >= 0

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image, **kwargs):
        radius = random.uniform(self.min_value, self.max_value)

        img = Image.fromarray(np.uint8(image))
        img = img.filter(ImageFilter.GaussianBlur(radius))
        image = np.array(img)

        return dict({'image': image}, **kwargs)


class Brightness(data_processor.Processor):
    """Adjust image brightness.

    Reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.PIL.ImageEnhance.Brightness

    Args:
        value (int | list | tuple): An enhancement factor of 0.0 gives a black image.
            A factor of 1.0 gives the original image.
    """

    def __init__(self, value=(0.75, 1.25)):

        if type(value) in {int, float}:
            min_value = value
            max_value = value

        elif len(value) == 2:
            min_value, max_value = value

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(value)))

        assert min_value >= 0
        assert max_value > 0

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image, **kwargs):
        factor = random.uniform(self.min_value, self.max_value)

        img = Image.fromarray(np.uint8(image))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        image = np.array(img)

        return dict({'image': image}, **kwargs)


class Color(data_processor.Processor):
    """Adjust image color.

    Reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.PIL.ImageEnhance.Color

    Args:
        value (int | list | tuple): An enhancement factor of 0.0 gives a black and white image.
            A factor of 1.0 gives the original image.
    """

    def __init__(self, value=(0.75, 1.25)):

        if type(value) in {int, float}:
            min_value = value
            max_value = value

        elif len(value) == 2:
            min_value, max_value = value

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(value)))

        assert min_value >= 0
        assert max_value > 0

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image, **kwargs):
        factor = random.uniform(self.min_value, self.max_value)

        img = Image.fromarray(np.uint8(image))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)
        image = np.array(img)

        return dict({'image': image}, **kwargs)


class Contrast(data_processor.Processor):
    """Adjust image contrast.

    Reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.PIL.ImageEnhance.Contrast

    Args:
        value (int | list | tuple): An enhancement factor of 0.0 gives a solid grey image.
            A factor of 1.0 gives the original image.
    """

    def __init__(self, value=(0.75, 1.25)):

        if type(value) in {int, float}:
            min_value = value
            max_value = value

        elif len(value) == 2:
            min_value, max_value = value

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(value)))

        assert min_value >= 0
        assert max_value > 0

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image, **kwargs):
        factor = random.uniform(self.min_value, self.max_value)

        img = Image.fromarray(np.uint8(image))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        image = np.array(img)

        return dict({'image': image}, **kwargs)


class Crop(data_processor.Processor):
    """Crop image.

    Args:
        size (int | list | tuple): Crop to this size.
        resize (int | list | tuple): If there are resize param, resize and crop.
    """

    def __init__(self, size, resize=None):

        if type(size) in {int, float}:
            height = size
            width = size

        elif len(size) == 2:
            height, width = size

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(size)))

        assert height > 0
        assert width > 0

        self.height = height
        self.width = width

        self.is_resize = False

        if resize:
            self.is_resize = True
            if type(resize) in {int, float}:
                resize_height = resize
                resize_width = resize

            elif len(resize) == 2:
                resize_height, resize_width = resize

            else:
                raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(resize)))

            assert resize_height > 0
            assert resize_width > 0

            self.resize_height = resize_height
            self.resize_width = resize_width

    def __call__(self, image, mask=None, **kwargs):
        if self.is_resize:
            image = pre_processor.resize(image, size=(self.resize_height, self.resize_width))
            if mask is not None:
                mask = pre_processor.resize(mask, size=(self.resize_height, self.resize_width))

        origin_height = image.shape[0]
        origin_width = image.shape[1]

        top = randint(0, origin_height - self.height)
        left = randint(0, origin_width - self.width)

        # crop
        image = image[top:top + self.height, left:left + self.width, :]
        if mask is not None:
            if np.ndim(mask) == 2:
                mask = mask[top:top + self.height, left:left + self.width]
            elif np.ndim(mask) == 3:
                mask = mask[top:top + self.height, left:left + self.width, :]

        return dict({'image': image, 'mask': mask}, **kwargs)


def _flip_left_right_boundingbox(image, boxes):
    """Flip left right only bounding box.

    Args:
        image (np.ndarray): a image. shape is [height, width, channel]
        boxes (np.ndarray): bounding boxes. shape is [num_boxes, 5(x, y, w, h, class_id)]
    """
    width = image.shape[1]

    if len(boxes) > 0:
        boxes[:, 0] = width - boxes[:, 0] - boxes[:, 2]

    return boxes


class FlipLeftRight(data_processor.Processor):
    """Flip left right.

    Args:
        probability (number): Probability for flipping.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, mask=None, gt_boxes=None, **kwargs):
        flg = random.random() > self.probability
        if flg:
            image = image[:, ::-1, :]
            if mask is not None:
                if np.ndim(mask) == 2:
                    mask = mask[:, ::-1]
                elif np.ndim(mask) == 3:
                    mask = mask[:, ::-1, :]
                else:
                    raise RuntimeError('Number of dims in mask should be 2 or 3 but get {}.'.format(np.ndim(mask)))
            if gt_boxes is not None:
                gt_boxes = _flip_left_right_boundingbox(image, gt_boxes)

        return dict({'image': image, 'mask': mask, 'gt_boxes': gt_boxes}, **kwargs)


def _flip_top_bottom_boundingbox(img, boxes):
    """Flip top bottom only bounding box.

    Args:
        img: np array image.
        boxes(np.ndarray): bounding boxes. shape is [num_boxes, 5(x, y, w, h, class_id)]
    """
    height = img.shape[0]
    if len(boxes) > 0:
        boxes[:, 1] = height - boxes[:, 1] - boxes[:, 3]

    return boxes


class FlipTopBottom(data_processor.Processor):
    """Flip top bottom.

    Args:
        probability (number): Probability for flipping.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, mask=None, gt_boxes=None, **kwargs):
        """
        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
            mask (np.ndarray): Annotation data for segmentation.
                shape is [height, width] or [height, width, channel]
            gt_boxes: Ground truth boxes in the image. shape is [num_boxes, 5(x, y, w, h, class)]
        Returns:
            all args (dict): Contains processed image, mask, gt_boxes and etc.
        """
        flg = random.random() > self.probability
        if flg:
            image = image[::-1, :, :]
            if mask is not None:
                if np.ndim(mask) == 2:
                    mask = mask[::-1, :]
                elif np.ndim(mask) == 3:
                    mask = mask[::-1, :, :]
                else:
                    raise RuntimeError('Number of dims in mask should be 2 or 3 but get {}.'.format(np.ndim(mask)))
            if gt_boxes is not None:
                gt_boxes = _flip_top_bottom_boundingbox(image, gt_boxes)

        return dict({'image': image, 'mask': mask, 'gt_boxes': gt_boxes}, **kwargs)


class Hue(data_processor.Processor):
    """Change image hue.

    Args:
        value (int | list | tuple): Assume the value in -255, 255. When the value is 0, nothing to do.
    """

    def __init__(self, value=(-10, 10)):

        if type(value) in {int, float}:
            min_value = - value
            max_value = value

        elif len(value) == 2:
            min_value, max_value = value

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(value)))

        assert min_value > -255
        assert max_value < 255

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image, **kwargs):
        val = random.uniform(self.min_value, self.max_value)

        img = Image.fromarray(np.uint8(image))
        hsv = np.array(img.convert("HSV"))

        hsv[:, :, 0] = hsv[:, :, 0] + val

        image = np.array(Image.fromarray(hsv, "HSV").convert("RGB"))

        return dict({'image': image}, **kwargs)


class Pad(data_processor.Processor):
    """Add padding to images.

    Args:
        value (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int): Pixel fill value. Default is 0.
    """

    def __init__(self, value, fill=0):
        if type(value) is int:
            left = top = right = bottom = value

        elif type(value) is tuple:
            if len(value) == 2:
                left, top = right, bottom = value

            if len(value) == 4:
                left, top, right, bottom = value
        else:
            raise Exception("Expected int, tuple/list with 2 or 4 entries. Got %s." % (type(value)))

        self.left, self.top, self.right, self.bottom = left, top, right, bottom
        self.fill = 0

    def __call__(self, image, mask=None, **kwargs):
        """
        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
            mask (np.ndarray): Annotation data for segmentation.
                shape is [height, width] or [height, width, channel]
        Returns:
            all args (dict): Contains processed image, mask and etc.
        """
        origin_height = image.shape[0]
        origin_width = image.shape[1]

        height = self.top + origin_height + self.bottom
        width = self.left + origin_width + self.right
        result_image = np.full((height, width, image.shape[2]), self.fill, dtype=image.dtype)
        result_image[self.top:-self.bottom, self.left:-self.right, :] = image

        if mask is not None:
            if np.ndim(mask) == 2:
                result_mask = np.full((height, width), self.fill, dtype=mask.dtype)
                result_mask[self.top:-self.bottom, self.left:-self.right] = mask
            elif np.ndim(mask) == 3:
                result_mask = np.full((height, width, mask.shape[2]), self.fill, dtype=mask.dtype)
                result_mask[self.top:-self.bottom, self.left:-self.right, :] = mask
            else:
                raise RuntimeError('Number of dims in mask should be 2 or 3 but get {}.'.format(np.ndim(mask)))
            mask = result_mask

        return dict({'image': result_image, 'mask': mask}, **kwargs)


class RandomPatchCut(data_processor.Processor):
    """Cut out random patches of the image.

    Args:
        num_patch (int): number of random patch cut-outs to generate
        max_size (int): maximum size of the patch edge, in percentages of image size
        square (bool): force square aspect ratio for patch shape
    """

    def __init__(self, num_patch=1, max_size=10, square=True):
        self.num_patch = int(num_patch)
        self.max_size = max_size / 100
        self.square = square

        assert self.num_patch >= 1, "num_patch parameter must be equal or bigger than 1"
        assert 0 < self.max_size < 1, "max_size parameter must be between 0 and 100 (not inclusive)"
        assert type(square) is bool, "square parameter must be bool value"

    def __call__(self, image, **kwargs):
        """Cut random patches.

        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
        Return:
            all args (dict): Contains processed image and etc.
        """
        image_h, image_w, _ = image.shape
        patch_max = int(min(image_w * self.max_size, image_h * self.max_size))

        for _ in range(self.num_patch):
            patch_x = randint(0, image_w - patch_max - 1)
            patch_y = randint(0, image_h - patch_max - 1)
            if self.square:
                # ceil int division: (x + (d-1)) // d to avoid 0 and float for randint arg
                patch_w = patch_h = randint((patch_max + 9) // 10, patch_max)
            else:
                patch_w = randint((patch_max + 9) // 10, patch_max)
                patch_h = randint((patch_max + 9) // 10, patch_max)
            image[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w, :] = 0

        return dict({'image': image}, **kwargs)


class RandomErasing(data_processor.Processor):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    The following Args(hyper parameters) are referred to the paper.

    Args:
        probability (float): The probability that the operation will be performed.
        sl (float): min erasing area
        sh (float): max erasing area
        r1 (float): min aspect ratio
        content_type (string): type of erasing value: {"mean", "random"}
        mean (list): erasing value if you use "mean" mode (mean ImageNet pixel value)
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, content_type="mean", mean=[125, 122, 114]):
        self.hyper_params = {'probability': probability, 'sl': sl,
                             'sh': sh, 'r1': r1, 'content_type': content_type, 'mean': mean}

    def __call__(self, image, **kwargs):
        """Random Erasing in an entire image.

        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
            gt_boxes: Ground truth boxes in the image. shape is [num_boxes, 5(x, y, w, h, class)]
        Return:
            all args (dict): Contains processed image and etc.
        """
        processed_image = image.copy()
        image_h, image_w, _ = image.shape
        entire_box = [0, 0, image_w, image_h]
        processed_image = _random_erasing_in_box(processed_image, entire_box, **self.hyper_params)

        return dict({'image': processed_image}, **kwargs)


class RandomErasingForDetection(data_processor.Processor):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    The following Args(hyper parameters) are referred to the paper.
    This Augmentation can be used when you train object detection task.

    Args:
        probability (float): The probability that the operation will be performed.
        sl (float): min erasing area
        sh (float): max erasing area
        r1 (float): min aspect ratio
        content_type (string): type of erasing value: {"mean", "random"}
        mean (list): erasing value if you use "mean" mode (mean ImageNet pixel value)
        i_a (bool): image-aware, random erase an entire image.
        o_a (bool): object-aware, random erase each object bounding boxes.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3,
                 content_type="mean", mean=[125, 122, 114], i_a=True, o_a=True):
        self.i_a = i_a
        self.o_a = o_a
        self.hyper_params = {'probability': probability, 'sl': sl,
                             'sh': sh, 'r1': r1, 'content_type': content_type, 'mean': mean}

    def __call__(self, image, gt_boxes=None, **kwargs):
        """Random Erasing in boxes (and an entire image).

        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
            gt_boxes: Ground truth boxes in the image. shape is [num_boxes, 5(x, y, w, h, class)]
        Return:
            all args (dict): Contains processed image and etc.
        """
        image_h, image_w, _ = image.shape
        processed_image = image.copy()
        boxes = gt_boxes

        if self.o_a:
            for i in range(boxes.shape[0]):
                processed_image = _random_erasing_in_box(processed_image, boxes[i], **self.hyper_params)

        if self.i_a:
            entire_box = [0, 0, image_w, image_h]
            processed_image = _random_erasing_in_box(processed_image, entire_box, **self.hyper_params)

        return dict({'image': processed_image, 'gt_boxes': boxes}, **kwargs)


def _random_erasing_in_box(image, box, probability, sl, sh, r1, content_type, mean):
    """Random Erasing in a box (util func).

    Args:
        image (np.ndarray): a image. shape is [height, width, channel]
        box: Ground truth boxes in the image. shape is boxes, [5(x, y, w, h, class)]
    """

    if random.uniform(0, 1) > probability:
        return image

    box_x, box_y = box[0], box[1]
    box_w, box_h = box[2], box[3]
    area = box_h * box_w

    for _ in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if h < box_h and w < box_w:
            x1 = random.randint(box_x, box_x + box_w - w)
            y1 = random.randint(box_y, box_y + box_h - h)

            if content_type == "mean":
                image[y1:y1+h, x1:x1+w, :] = mean
            elif content_type == "random":
                image[y1:y1+h, x1:x1+w, :] = np.random.randint(255, size=(h, w, 3))

            return image
    return image


class SSDRandomCrop(data_processor.Processor):
    """SSD random crop.

    References:
        https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py#L208

    Args:
        min_crop_ratio (number): Minimum crop ratio for cropping the
    """

    def __init__(self, min_crop_ratio=0.3):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. min jaccard(iou) with obj in 0.1,0.3,0.7,0.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

        # Crop rectangle aspect ratio constraint between 0.5 & 2
        self.aspect_ratio_min = 0.5
        self.aspect_ratio_max = 2

        # Crop rectangle minimum ratio corresponding to original image.
        self.min_crop_ratio = min_crop_ratio

    def __call__(self, image, gt_boxes, **kwargs):
        """SSDRandomCrop

        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
            gt_boxes: Ground truth boxes in the image. shape is [num_boxes, 5(x, y, w, h, class)]
        Returns:
            all args (dict): Contains processed image, gt_boxes and etc.
        """
        boxes = gt_boxes
        height, width, _ = image.shape
        num_max_boxes = len(gt_boxes)

        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return dict({'image': image, 'gt_boxes': boxes}, **kwargs)

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                crop_w = random.uniform(self.min_crop_ratio * width, width)
                crop_h = random.uniform(self.min_crop_ratio * height, height)

                # aspect ratio constraint between 0.5 & 2
                if crop_h / crop_w < self.aspect_ratio_min or crop_h / crop_w > self.aspect_ratio_max:
                    continue

                crop_x = random.uniform(0, width - crop_w)
                crop_y = random.uniform(0, height - crop_h)

                # convert to integer
                crop_x, crop_y, crop_w, crop_h = int(crop_x), int(crop_y), int(crop_w), int(crop_h)

                crop_rect = np.array([crop_x, crop_y, crop_w, crop_h])

                # cut the crop from the image
                current_image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]

                # if ground truth boxes is zero, return cropped image.
                if boxes.size == 0:
                    return dict({'image': current_image, 'gt_boxes': boxes}, **kwargs)

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = iou(boxes, crop_rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # keep overlap with gt box IF center in sampled patch
                centers = np.stack([
                    boxes[:, 0] + boxes[:, 2] / 2.0,
                    boxes[:, 1] + boxes[:, 3] / 2.0,
                ], axis=1)

                # mask that all gt boxes center be in the crop rectangle.
                mask = (
                        (crop_x < centers[:, 0]) *
                        (crop_y < centers[:, 1]) *
                        ((crop_x + crop_w) > centers[:, 0]) *
                        ((crop_y + crop_h) > centers[:, 1])
                )

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                masked_boxes = boxes[mask, :]
                current_boxes = crop_boxes(masked_boxes, crop_rect)
                current_boxes = fill_dummy_boxes(current_boxes, num_max_boxes)

                return dict({'image': current_image, 'gt_boxes': current_boxes}, **kwargs)


class Rotate(data_processor.Processor):
    """Rotate.

    Args:
        angle_range (int | list | tuple): Angle range.
    """

    def __init__(self, angle_range=(0, 90)):

        if type(angle_range) in {int, float}:
            min_angle = 0
            max_angle = angle_range

        elif len(angle_range) == 2:
            min_angle, max_angle = angle_range

        else:
            raise Exception("Expected float or int, tuple/list with 2 entries. Got %s." % (type(angle_range)))

        assert min_angle >= 0
        assert max_angle <= 90

        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image, mask=None, **kwargs):
        """
        Args:
            image (np.ndarray): a image. shape is [height, width, channel]
            mask (np.ndarray): Annotation data for segmentation.
                shape is [height, width] or [height, width, channel]
        Returns:
            all args (dict): Contains processed image, mask and etc.
        """
        angle = random.uniform(self.min_angle, self.max_angle)

        img_rot = Image.fromarray(np.uint8(image))
        img_rot = img_rot.rotate(angle)
        img_rot = np.array(img_rot)

        if mask is not None:
            mask_rot = Image.fromarray(np.uint8(mask))
            mask_rot = mask_rot.rotate(angle)
            mask_rot = np.array(mask_rot)
            mask = mask_rot

        return dict({'image': img_rot, 'mask': mask}, **kwargs)


# TODO(wakisaka): implement class
def color_filter(img):
    red = randint(0, 255) / 255
    blue = randint(0, 255) / 255
    green = randint(0, 255) / 255
    alpha = randint(0, 127) / 255

    img = img / 255

    new_red = ((1 - alpha) * img[:, :, 0]) + (alpha * red)
    new_blue = ((1 - alpha) * img[:, :, 1]) + (alpha * blue)
    new_green = ((1 - alpha) * img[:, :, 2]) + (alpha * green)

    new_image = np.stack([new_red * 255, new_blue * 255, new_green * 255], axis=2)

    return new_image


# TODO(wakisaka): implement class
def affine_scale(img, scale, fill_color="white"):
    """Resize image to scale size keeping the aspect ratio and place it in center of fill color image."""
    image = Image.fromarray(np.uint8(img))
    original_width = image.size[0]
    original_height = image.size[1]

    outer_width = original_width - (original_width * scale)
    outer_height = original_height - (original_height * scale)
    new_image = Image.new(
        'RGB',
        (original_width, original_height),
        fill_color,
    )

    scaled = image.resize((int(original_width * scale), int(original_height * scale)))
    new_image.paste(scaled, (int(outer_width / 2), int(outer_height / 2)))

    return np.array(new_image)
