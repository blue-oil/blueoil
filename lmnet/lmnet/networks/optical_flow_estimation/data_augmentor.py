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


import math
import numpy as np
from scipy import ndimage

from abc import ABCMeta, abstractmethod
from PIL import Image, ImageEnhance, ImageFilter


class Augmentor(metaclass=ABCMeta):
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dict__)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        NotImplementedError()

    def split_input_tensor(self, input_tensor):
        """
        input_tensor: np.ndarray with shape (H, W, 6)
        return: ndarray(H, W, 3), ndarray(H, W, 3)
        """
        return input_tensor[..., :3], input_tensor[..., 3:]


class ColorConverter(Augmentor):
    """
    Augmentors converting pixel color properties
    """
    def __call__(self, image, *args, **kwargs):
        image_a, image_b = self.split_input_tensor(image)
        factor = np.random.uniform(self.min_value, self.max_value)
        processed_tensor = np.concatenate([
            self.process(image_a, factor),
            self.process(image_b, factor)
        ], axis=2)
        return dict({"image": processed_tensor}, **kwargs)

    def process(self, *args, **kwargs):
        NotImplementedError()


class Brightness(ColorConverter):
    """
    Adjusting image brightness.
    reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.PIL.ImageEnhance.Brightness
    args: min_value, max_value:
        An enhancement factor of 0.0 gives a black image.
        A factor of 1.0 gives the original image.
    """
    def __init__(self, min_value=0.75, max_value=1.25):
        assert min_value >= 0 and max_value >= 0, "Negative value not allowed!"
        self.min_value, self.max_value = min_value, max_value

    def process(self, image, factor):
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        processed_image = enhancer.enhance(factor)
        return np.array(processed_image)


class Color(ColorConverter):
    """
    Adjusting image color.
    reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.PIL.ImageEnhance.Color
    args: min_value, max_value
        An enhancement factor of 0.0 gives a black and white image.
        A factor of 1.0 gives the original image.
    """
    def __init__(self, min_value=0.75, max_value=1.25):
        assert min_value >= 0 and max_value >= 0, "Negative value not allowed!"
        self.min_value, self.max_value = min_value, max_value

    def process(self, image, factor):
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_image)
        processed_image = enhancer.enhance(factor)
        return np.array(processed_image)


class Contrast(ColorConverter):
    """
    Adjusting image contrast.
    reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.PIL.ImageEnhance.Contrast
    args: min_value, max_value
        An enhancement factor of 0.0 gives a solid grey image.
        A factor of 1.0 gives the original image.
    """
    def __init__(self, min_value=0.75, max_value=1.25):
        assert min_value >= 0 and max_value >= 0, "Negative value not allowed!"
        self.min_value, self.max_value = min_value, max_value

    def process(self, image, factor):
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        processed_image = enhancer.enhance(factor)
        return np.array(processed_image)


class Hue(ColorConverter):
    """
    Adjusting image hue.
    args: min_value, max_value
        An enhancement factor of 0.0 gives a solid grey image.
        A factor of 1.0 gives the original image.
    """
    def __init__(self, min_value=-10.0, max_value=10.0):
        assert min_value > -255 and max_value < 255, \
            "Value range should be within (-255, 255)!"
        self.min_value, self.max_value = min_value, max_value

    def process(self, image, factor):
        pil_image = Image.fromarray(image)
        hsv_image = np.array(pil_image.convert("HSV"))
        hsv_image[:, :, 0] = hsv_image[:, :, 0] + factor
        processed_image = Image.fromarray(hsv_image, "HSV").convert("RGB")
        return np.array(processed_image)


class Gamma(ColorConverter):
    """
    Gamma blur filter.
    """
    def __init__(self, min_value=0.0, max_value=1.0):
        assert min_value >= 0 and max_value >= 0, "Negative value not allowed!"
        self.min_value, self.max_value = min_value, max_value

    def process(self, image, gamma):
        processed_image = (((image / 255.0) ** gamma) * 255.0).astype(np.uint8)
        return processed_image


class GaussianBlur(ColorConverter):
    """
    Gaussian blur filter.
    reference:
        https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur
    args: min_value, max_value
        References default is 2.
    """

    def __init__(self, min_value=0.0, max_value=1.0):
        assert min_value >= 0 and max_value >= 0, "Negative value not allowed!"
        self.min_value, self.max_value = min_value, max_value

    def process(self, image, radius):
        pil_image = Image.fromarray(image)
        processed_image = pil_image.filter(ImageFilter.GaussianBlur(radius))
        return np.array(processed_image)


class GaussianNoise(Augmentor):
    """
    Additive Gaussian noise.
    """
    def __init__(self, min_value=0.0, max_value=1.0):
        assert min_value >= 0 and max_value >= 0, "Negative value not allowed!"
        self.min_value, self.max_value = min_value, max_value

    def __call__(self, image, label, **kwargs):
        # print(image.shape)
        noise_amp = np.random.uniform(self.min_value, self.max_value)
        image_noise = noise_amp * np.random.randn(*image.shape)
        processed_image = image + image_noise
        processed_image[processed_image < 0] = 0
        processed_image[processed_image > 255] = 255
        processed_image = processed_image.astype(np.uint8)
        return dict({
            "image": processed_image, "label": label}, **kwargs)


class FlipTopBottom(Augmentor):
    """
    Flip top bottom.
    args: probability
        Probability for flipping.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label, **kwargs):
        if np.random.rand() < self.prob:
            image = image[::-1, ...]
            label = label[::-1, ...]
            label[:, :, 1] *= -1.0
        return dict({
            "image": image, "label": label}, **kwargs)


class FlipLeftRight(Augmentor):
    """
    Flip left right.
    args: probability
        Probability for flipping.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label, **kwargs):
        if np.random.rand() < self.prob:
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]
            label[:, :, 0] *= -1.0
        return dict({
            "image": image, "label": label}, **kwargs)


class Identity(Augmentor):
    """
    create the pair of images with no change
    args: probability
        Probability for applying this process.
    """
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, image, label, **kwargs):
        if np.random.rand() < self.prob:
            image[..., :3] = image[..., 3:]
            label[:] = 0.0
        return dict({
            "image": image, "label": label}, **kwargs)


class Rotate(Augmentor):
    """
    Rotating image
    """
    def __init__(self, min_value=-15, max_value=15):
        self.min_value, self.max_value = min_value, max_value

    def __call__(self, image, label, **kwargs):
        ang = np.random.uniform(self.min_value, self.max_value)
        deg = ang * np.pi / 180
        rot_mat = np.array([
            [np.cos(deg), -np.sin(deg)],
            [np.sin(deg), np.cos(deg)],
        ])
        image_new = ndimage.rotate(image, ang, reshape=False, cval=0.0)
        image_new = image_new.astype(np.uint8)
        flow_new = np.array(label.dot(rot_mat.T))
        flow_new = ndimage.rotate(flow_new, ang, reshape=False, cval=0.0)
        return dict({
            "image": image_new, "label": flow_new}, **kwargs)


class Scale(Augmentor):
    """
    Scaling image
    """
    def __init__(self, min_value=1.0, max_value=2.0):
        assert min_value >= 1.0 or max_value >= 1.0, \
            "scaling parameter should be greater than 1.0"
        self.min_value, self.max_value = min_value, max_value

    def random_crop(self, data, crop_size):
        height, width, _ = data.shape
        if height == crop_size[0] or width == crop_size[1]:
            return data
        top = np.random.randint(0, height - crop_size[0])
        left = np.random.randint(0, width - crop_size[1])
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        return data[top:bottom, left:right, :]

    def __call__(self, image, label, **kwargs):
        image_size = image.shape[:2]
        factor = np.random.uniform(self.min_value, self.max_value)
        data = np.concatenate([image, label * factor], axis=2)
        zoomed_data = ndimage.zoom(data, [factor, factor, 1], order=1)
        data = self.random_crop(zoomed_data, crop_size=image_size)
        image_new = data[..., :-2]
        image_new[image_new < 0] = 0
        image_new[image_new > 255] = 255
        image_new = image_new.astype(np.uint8)
        label_new = data[..., -2:]
        return dict({"image": image_new, "label": label_new}, **kwargs)


class Translate(Augmentor):
    """
    Shifting image
    """
    def __init__(self, min_value=-0.2, max_value=0.2):
        self.min_value, self.max_value = min_value, max_value

    def __call__(self, image, label, **kwargs):
        image_size = image.shape[:2]
        dh = np.random.uniform(self.min_value, self.max_value)
        dw = np.random.uniform(self.min_value, self.max_value)
        shift = [int(image_size[0] * dh), int(image_size[1] * dw), 0]
        shifted_image = ndimage.shift(image, shift, order=1, cval=0)
        shifted_label = ndimage.shift(label, shift, order=1, cval=0)
        return dict({"image": shifted_image, "label": shifted_label}, **kwargs)
