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

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True)
parser.add_argument('--port', type=int, default=12345)
args = parser.parse_args()


def init_camera(camera_height, camera_width, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def rescale_frame(frame, ratio):
    height = int(frame.shape[0] * ratio)
    width = int(frame.shape[1] * ratio)
    shape = (width, height)
    return cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)


def send_and_receive(address, input_data, verbose=False):
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


def run_client(client_info):
    # initializing worker and variables
    diff_step = 10
    store_num = 150
    cap = init_camera(480, 640, 0)
    frame_list = collections.deque(maxlen=store_num)
    image_size = (384, 512, 3)
    input_image = np.zeros(
        (1, *image_size[:2], image_size[-1] * 2)).astype(np.uint8)
    output_image = np.zeros(image_size).astype(np.uint8)
    for _ in range(store_num):
        frame_list.append(np.zeros(image_size).astype(np.uint8))

    def _get_frame():
        while True:
            begin = time.time()
            res, frame = cap.read()
            assert res, "Something wrong occurs with camera!"
            frame_list.append(rescale_frame(frame[:, ::-1, :], 0.8))
            time.sleep(max(0.0, 1.0 / 30 - (time.time() - begin)))

    def _get_output():
        while True:
            try:
                input_image[0, ..., :3] = frame_list[-diff_step]
                input_image[0, ..., 3:] = frame_list[-1]
                output_image[:] = send_and_receive(
                    client_info, input_image, True)
                time.sleep(0.05)
            except ValueError:
                pass

    t1 = threading.Thread(target=_get_frame)
    t1.setDaemon(True)
    t1.start()
    t2 = threading.Thread(target=_get_output)
    t2.setDaemon(True)
    t2.start()

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        _pre = frame_list[-diff_step].astype(np.float)
        _post = frame_list[-1].astype(np.float)
        diff = np.mean([_pre, _post], axis=0).astype(np.uint8)
        cv2.imshow("diff", diff)
        cv2.imshow("output", output_image)
        key = cv2.waitKey(2)
        if key == 27:
            break


if __name__ == '__main__':
    run_client((args.host, args.port))
