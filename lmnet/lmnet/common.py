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

from enum import Enum
import math


class Tasks(Enum):
    CLASSIFICATION = "IMAGE.CLASSIFICATION"
    SEMANTIC_SEGMENTATION = "IMAGE.SEMANTIC_SEGMENTATION"
    OBJECT_DETECTION = "IMAGE.OBJECT_DETECTION"


# Color Palette for General Purpose
# Sample image is here
# https://github.com/blue-oil/blueoil/tree/master/docs/_static/color_map.png
COLOR_MAP = [
    (192, 0,   128),  # COLOR00
    (0,   128, 192),  # COLOR01
    (0,   128, 64),   # COLOR02
    (128, 0,   0),    # COLOR03
    (64,  0,   128),  # COLOR04
    (64,  0,   192),  # COLOR05
    (192, 128, 64),   # COLOR06
    (192, 192, 128),  # COLOR07
    (64,  64,  128),  # COLOR08
    (128, 0,   192),  # COLOR09
    (192, 0,   64),   # COLOR10
    (128, 128, 64),   # COLOR11
    (192, 0,   192),  # COLOR12
    (128, 64,  64),   # COLOR13
    (64,  192, 128),  # COLOR14
    (64,  64,  0),    # COLOR15
    (128, 64,  128),  # COLOR16
    (128, 128, 192),  # COLOR17
    (0,   0,   192),  # COLOR18
    (192, 128, 128)   # COLOR19
]


def get_color_map(length):
    # This function generate arbitrary length color map.
    color_map = COLOR_MAP * int(math.ceil(length / len(COLOR_MAP)))
    return color_map[:length]
