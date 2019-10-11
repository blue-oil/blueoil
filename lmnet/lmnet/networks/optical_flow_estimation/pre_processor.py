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

import math
import numpy as np

from abc import ABCMeta, abstractmethod
from PIL import Image, ImageEnhance, ImageFilter


class Processor(metaclass=ABCMeta):
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


class DevideBy255(Processor):
    def __init__(self, *args, **kwargs):
        self._coef = 1 / 255.0

    def __call__(self, image, **kwargs):
        image = image.astype(np.float32)
        image *= self._coef
        return dict({'image': image}, **kwargs)


class DiscretizeFlow(Processor):
    def __init__(self, radius=10.0, split_num=10, dtype=np.uint16, **kwargs):
        self.radius = radius
        self.split_num = split_num
        self.dtype = dtype

    def __call__(self, image, label, **kwargs):
        class_num = self.split_num + 1
        rad = np.linalg.norm(label, axis=2)
        arg = np.arctan2(-label[..., 1], -label[..., 0]) + np.pi
        arg /= (2 * np.pi)
        discretized_label = (arg * self.split_num).astype(self.dtype)
        discretized_label %= self.split_num
        discretized_label += 1
        discretized_label[rad < self.radius] = 0
        discretized_label = np.eye(
            class_num, dtype=self.dtype)[discretized_label]
        return dict({'image': image, 'label': discretized_label}, **kwargs)
