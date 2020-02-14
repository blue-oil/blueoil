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
import os

import numpy as np
import PIL.Image

OUTPUT_DIR = 'tests'


def horizontal_stripe(size=[240, 120], stripe=2):
    """Create test image
    Args:
        size: [height, width]
    """
    image = np.zeros((size[0], size[1], 3,), dtype=np.uint8)

    for h in range(size[0]):

        if h % stripe == 0:
            image[h, :, :] = 255

    img = PIL.Image.fromarray(image, mode="RGB")

    filename = 'horizontal_stripe_h{}_w{}.png'.format(size[0], size[1])
    filename = os.path.join(OUTPUT_DIR, filename)
    img.save(filename)

    return image


def vertical_stripe(size=[240, 120], stripe=2):
    """Create test image
    Args:
        size: [height, width]
    """
    image = np.zeros((size[0], size[1], 3,), dtype=np.uint8)

    for w in range(size[1]):

        if w % stripe == 0:
            image[:, w, :] = 255

    img = PIL.Image.fromarray(image, mode="RGB")

    filename = 'vertical_stripe_h{}_w{}.png'.format(size[0], size[1])
    filename = os.path.join(OUTPUT_DIR, filename)
    img.save(filename)

    return image


def red(size=[240, 120]):
    """Create test image
    Args:
        size: [height, width]
    """
    image = np.zeros((size[0], size[1], 3,), dtype=np.uint8)

    image[:, :, 0] = 255
    img = PIL.Image.fromarray(image, mode="RGB")

    filename = 'red_h{}_w{}.png'.format(size[0], size[1])
    filename = os.path.join(OUTPUT_DIR, filename)
    img.save(filename)

    return image


def blue(size=[240, 120]):
    """Create test image
    Args:
        size: [height, width]
    """
    image = np.zeros((size[0], size[1], 3,), dtype=np.uint8)

    image[:, :, 2] = 255
    img = PIL.Image.fromarray(image, mode="RGB")

    filename = 'blue_h{}_w{}.png'.format(size[0], size[1])
    filename = os.path.join(OUTPUT_DIR, filename)
    img.save(filename)

    return image


def green(size=[240, 120]):
    """Create test image
    Args:
        size: [height, width]
    """
    image = np.zeros((size[0], size[1], 3,), dtype=np.uint8)

    image[:, :, 1] = 255
    img = PIL.Image.fromarray(image, mode="RGB")

    filename = 'green_h{}_w{}.png'.format(size[0], size[1])
    filename = os.path.join(OUTPUT_DIR, filename)
    img.save(filename)

    return image


if __name__ == '__main__':
    SIZE = [32, 32]
    horizontal_stripe(size=SIZE)
    vertical_stripe(size=SIZE)
    red(size=SIZE)
    blue(size=SIZE)
    green(size=SIZE)
