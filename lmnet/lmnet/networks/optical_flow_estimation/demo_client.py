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
import sys
import cv2
import time
import math
import click
import socket
import warnings
import argparse
import threading
import collections
import numpy as np

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from io import BytesIO
from scipy import ndimage

from lmnet.networks.optical_flow_estimation.demo_lib import run_demo, run_test

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True)
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--demo_name', type=str, default="output")
parser.add_argument('--diff_step', type=int, default=5)
parser.add_argument('--movie_path', type=str, default=None)
parser.add_argument('--disable_full_screen', action="store_false")
args = parser.parse_args()


def send_and_receive(input_data, address, verbose=False):
    t_begin = time.time()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(address)
        f = BytesIO()
        np.savez_compressed(f, input=input_data)
        f.seek(0)
        s.sendall(f.read())
        s.sendall(b"__end__")
        data_buffer = b""
        while True:
            received_buffer = s.recv(8192)
            if not received_buffer:
                break
            data_buffer += received_buffer
        output_data = np.load(BytesIO(data_buffer))['output']
    if verbose:
        print("receive {}[{}] ({:.6f} sec)".format(
            output_data.shape, output_data.dtype, time.time() - t_begin),
            end="\r")
    return output_data


if __name__ == '__main__':
    client_info = (args.host, args.port)
    window_name = "{}:{}".format(*client_info)
    run_demo(
        send_and_receive, func_args=[client_info, True],
        diff_step=args.diff_step, window_name=window_name,
        movie_path=args.movie_path, full_screen=args.disable_full_screen,
        demo_name=args.demo_name
    )
    # run_test(send_and_receive, func_args=[client_info, False])
