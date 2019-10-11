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
# ==============================================================

from __future__ import print_function

import os
import sys
import cv2
import time
import math
import socket
import argparse
import threading
import collections
import numpy as np

from io import BytesIO
from scipy import ndimage

from nn_lib import NNLib
from demo_lib import run_demo, run_test

parser = argparse.ArgumentParser()
parser.add_argument('--diff_step', type=int, default=5)
parser.add_argument('--camera_id', type=int, default=0)
parser.add_argument('--movie_path', type=str, default=None)
parser.add_argument('--disable_full_screen', action="store_false")
args = parser.parse_args()


if __name__ == '__main__':

    def _inference(input_data):
        t_begin = time.time()
        pre_image = cv2.cvtColor(input_data[0, ..., :3], cv2.COLOR_BGR2GRAY)
        post_image = cv2.cvtColor(input_data[0, ..., 3:], cv2.COLOR_BGR2GRAY)
        output = cv2.calcOpticalFlowFarneback(
            pre_image, post_image,
            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        calc_time = time.time() - t_begin
        return output[np.newaxis], calc_time

    window_name = "Gunner Farneback"
    # run_demo(
    #     _inference, diff_step=args.diff_step, movie_path=args.movie_path,
    #     window_name=window_name, camera_id=args.camera_id,
    #     input_image_size=(384, 512, 3),
    #     full_screen=args.disable_full_screen
    # )
    run_test(_inference, split_step=1)
