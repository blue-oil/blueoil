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
from __future__ import division

import math
import numpy as np
from enum import Enum

from blueoil.turbo_color_map import TURBO_CMAP_DATA


class Tasks(Enum):
    CLASSIFICATION = "IMAGE.CLASSIFICATION"
    SEMANTIC_SEGMENTATION = "IMAGE.SEMANTIC_SEGMENTATION"
    OBJECT_DETECTION = "IMAGE.OBJECT_DETECTION"
    KEYPOINT_DETECTION = "IMAGE.KEYPOINT_DETECTION"


def get_color_map(length):
    # Color Palette for General Purpose
    # Sample image is here
    # https://github.com/blue-oil/blueoil/tree/master/docs/_static/color_map.png
    color_map_base = [
        (192, 0, 128),  # COLOR00
        (0, 128, 192),  # COLOR01
        (0, 128, 64),  # COLOR02
        (128, 0, 0),  # COLOR03
        (64, 0, 128),  # COLOR04
        (64, 0, 192),  # COLOR05
        (192, 128, 64),  # COLOR06
        (192, 192, 128),  # COLOR07
        (64, 64, 128),  # COLOR08
        (128, 0, 192),  # COLOR09
        (192, 0, 64),  # COLOR10
        (128, 128, 64),  # COLOR11
        (192, 0, 192),  # COLOR12
        (128, 64, 64),  # COLOR13
        (64, 192, 128),  # COLOR14
        (64, 64, 0),  # COLOR15
        (128, 64, 128),  # COLOR16
        (128, 128, 192),  # COLOR17
        (0, 0, 192),  # COLOR18
        (192, 128, 128),  # COLOR19
    ]

    # This function generate arbitrary length color map.
    color_map = color_map_base * int(math.ceil(length / len(color_map_base)))
    return color_map[:length]


# For replacing the Matplotlib Jet colormap, we use the Turbo color map
# https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
# The colormap allows for a large number of quantization levels:
# https://tinyurl.com/ybm3kpql

# Referred from the following gist:
# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
# Copyright 2019 Google LLC.
# SPDX-License-Identifier: Apache-2.0

# Changes:
# 1. Vectorized the implementation using numpy
# 2. Use numpy.modf to get integer and float parts
# 3. Provided an example in comments

def apply_color_map(image):
    turbo_cmap_data = np.asarray(TURBO_CMAP_DATA)
    x = np.asarray(image)
    x = x.clip(0., 1.)

    # Use numpy.modf to get the integer and decimal parts of feature values
    # in the input feature map (or heatmap) that has to be colored.
    # Example:
    #   >>> import numpy as np
    #   >>> x = np.array([1.2, 2.3, 4.5, 20.45, 6.75, 8.88])
    #   >>> f, i = np.modf(x)       # returns a tuple of length 2
    #   >>> print(i.shape, f.shape)
    #   (6,) (6,)
    #   >>> print(i)
    #   array([ 1.  2.  4. 20.  6.  8.])
    #   >>> print(f)
    #   array([0.2  0.3  0.5  0.45 0.75 0.88])
    f, a = np.modf(x * 255.0)
    a = a.astype(int)
    b = (a + 1).clip(max=255)
    image_colored = (
        turbo_cmap_data[a]
        + (turbo_cmap_data[b] - turbo_cmap_data[a]) * f[..., None]
    )
    return image_colored
