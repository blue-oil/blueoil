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


class Tasks(Enum):
    CLASSIFICATION = "IMAGE.CLASSIFICATION"
    SEMANTIC_SEGMENTATION = "IMAGE.SEMANTIC_SEGMENTATION"
    OBJECT_DETECTION = "IMAGE.OBJECT_DETECTION"
    KEYPOINT_DETECTION = "IMAGE.KEYPOINT_DETECTION"


# Color Palette for all tasks: Turbo color map:
# https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
# The colormap allows for a large number of quantization levels:
# https://github.com/blue-oil/blueoil/tree/master/docs/_static/turbo_cmap.png
# Implementation inspired from the following gist:
# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
TURBO_CMAP_DATA = np.load('turbo_cmap_data.npy')

def interpolate(colormap, x):
    x = max(0.0, min(1.0, x))
    a = int(x*255.0)
    b = min(255, a + 1)
    f = x*255.0 - a
    return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
            colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
            colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]

def interpolate_or_clip(colormap, x):
    if   x < 0.0: return [0.0, 0.0, 0.0]
    elif x > 1.0: return [1.0, 1.0, 1.0]
    else: return interpolate(colormap, x)

def get_color_map(length):
    # This function generate arbitrary length color map.
    # First of all, generate `length` uniformly spaced floats in [0, 1]
    x = np.linspace(0.0, 1.0, num=length, endpoint=True)
    color_map = []
    for x_i in x:
        color = interpolate_or_clip(TURBO_CMAP_DATA, x_i)
        color = [int(c * 255) for c in color]
        color_map.append(color)
    return color_map
