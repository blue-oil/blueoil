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

__all__ = ["color_function", "flow_to_image", "discretized_flow_to_image"]

import numpy as np
from scipy.interpolate import interp1d


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    color_wheel = np.zeros([ncols + 1, 3])

    col = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    # YG
    color_wheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    color_wheel[col:col + YG, 1] = 255
    col += YG
    # GC
    color_wheel[col:col + GC, 1] = 255
    color_wheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    # CB
    color_wheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    color_wheel[col:col + CB, 2] = 255
    col += CB
    # BM
    color_wheel[col:col + BM, 2] = 255
    color_wheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    # MR
    color_wheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    color_wheel[col:col + MR, 0] = 255
    # loop function
    color_wheel[-1] = color_wheel[0]
    return color_wheel


color_table = make_color_wheel()
color_function = interp1d(
    np.linspace(0, 2 * np.pi, color_table.shape[0]), color_table, axis=0)


def flow_to_image(flow, threshold=10.0):
    # idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    nan_index = np.isnan(flow).any(axis=2)
    rad = np.linalg.norm(flow, axis=2)
    rad[nan_index] = 0.0
    rad *= (1 / max(threshold, np.max(rad)))
    arg = np.arctan2(-flow[..., 1], -flow[..., 0]) + np.pi
    image = color_function(arg)
    height, width, channels = image.shape

    white_image = 255 * (1 - rad)
    for _i in range(channels):
        image[..., _i] *= rad
        image[..., _i] += white_image
    image = image.astype(np.uint8)
    image[image > 255] = 255
    return image


def discretized_flow_to_image(dflow, split_num):
    x = np.arange(0, 1 + split_num) / split_num
    cmap = color_function(2 * np.pi * x)
    cmap[0] = 255
    cmap = cmap.astype(np.uint8)
    return cmap[np.argmax(dflow, axis=2)]
