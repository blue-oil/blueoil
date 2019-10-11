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
parser.add_argument('model_path', type=str)
parser.add_argument('--diff_step', type=int, default=5)
parser.add_argument('--camera_id', type=int, default=0)
parser.add_argument('--movie_path', type=str, default=None)
parser.add_argument('--test', action="store_true")
parser.add_argument('--disable_full_screen', action="store_false")
args = parser.parse_args()


if __name__ == '__main__':
    def status_info(t):
        return "success" if t else "fail"

    model = NNLib()
    res = model.load(args.model_path)
    print("load model: {}".format(status_info(res)))
    res = model.init()
    print("init model: {}".format(status_info(res)))

    input_shape = model.get_input_shape()
    output_shape = model.get_output_shape()
    print("input shape: {}".format(input_shape))
    print("output shape: {}".format(output_shape))

    test_flow = model.run(np.random.randn(*input_shape).astype(np.float32))
    print("shape test: {}".format(status_info(test_flow.shape == output_shape)))
    print("value test: {}".format(status_info(np.any(np.isnan(test_flow)))))

    def _inference(input_data):
        _x = (input_data / 255.0).astype(np.float32)
        t_begin = time.time()
        output = model.run(_x)
        calc_time = time.time() - t_begin
        return output, calc_time

    window_name = os.path.basename(args.model_path)

    if args.test:
        run_test(_inference, split_step=4)
    else:
        run_demo(
            _inference, diff_step=args.diff_step, movie_path=args.movie_path,
            window_name=window_name, camera_id=args.camera_id,
            input_image_size=(input_shape[1], input_shape[2], 3),
            full_screen=args.disable_full_screen
        )
