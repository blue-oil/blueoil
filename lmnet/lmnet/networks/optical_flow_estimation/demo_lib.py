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

__all__ = ["run_demo", "run_test"]

import os
import sys
import cv2
import time
import math
import threading
import collections
import numpy as np
from scipy import ndimage

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

from lmnet.networks.optical_flow_estimation.flow_to_image import flow_to_image
from lmnet.datasets.optical_flow_estimation import (
    FlyingChairs, ChairsSDHom
)


def init_camera(camera_height, camera_width, device_id):
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


def run_demo(func_inference, func_args=[], func_kwargs={},
             diff_step=5, window_name="output", full_screen=True,
             movie_path=None, demo_name="output", device_id=0):
    # initializing worker and variables
    store_num = diff_step + 10
    cap = init_camera(480, 640, device_id=device_id)
    time_list = collections.deque(maxlen=5)
    frame_list = collections.deque(maxlen=store_num)
    image_size = (384, 512, 3)
    input_image = np.zeros(
        (1, *image_size[:2], image_size[-1] * 2), dtype=np.uint8)
    output_image = np.zeros(image_size, dtype=np.uint8)
    display_image = np.zeros(
        (9 * 64, 16 * 64, 3), dtype=np.uint8)
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
                begin = time.time()
                input_image[0, ..., :3] = frame_list[-diff_step]
                input_image[0, ..., 3:] = frame_list[-1]
                print(input_image.shape)
                # output_image[:] = func_inference(
                #     input_image, *func_args, **func_kwargs)
                output_image[:] = func_inference(input_image)
                time_list.append(time.time() - begin)
            except ValueError:
                pass

    t1 = threading.Thread(target=_get_frame)
    t1.setDaemon(True)
    t1.start()
    t2 = threading.Thread(target=_get_output)
    t2.setDaemon(True)
    t2.start()

    size_h1 = int(output_image.shape[0] * 1.5)
    size_w1 = int(output_image.shape[1] * 1.5)
    size_w2 = int(size_w1 / 3)
    size_h2 = int(size_h1 / 3)
    color_map = np.dstack(
        np.meshgrid(
            np.linspace(-15, 15, size_w2),
            np.linspace(-10, 10, size_h2)))
    color_map = flow_to_image(-color_map[..., [1, 0]])
    display_image[-size_h2:, -size_w2:] = color_map

    if full_screen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if movie_path is not None:
        from skvideo.io import FFmpegWriter
        movie_shape = display_image.shape[:2]
        out = FFmpegWriter(
            movie_path,
            inputdict={
                '-r': str(10), '-s': '{}x{}'.format(*movie_shape[::-1])},
            outputdict={
                # '-r': str(10),
                # '-c:v': 'libx264',
                # '-crf': str(17),
                # '-preset': 'ultrafast',
                # '-pix_fmt': 'yuv444p'
            }
        )

    def add_text_info():
        # FPS infomation
        fps = 1 / np.mean(time_list)
        fps_text = "{} (FPS: {:.2f})".format(demo_name, fps)
        cv2.putText(
            display_image, fps_text, (10, 25),
            cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 2)

        # title_1
        cv2.putText(
            display_image, "input1", (size_w1 + 10, 25),
            cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)

        cv2.putText(
            display_image, "input2", (size_w1 + 10, size_h2 + 25),
            cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)

        cv2.putText(
            display_image, "color coding", (size_w1 + 10, 2 * size_h2 + 25),
            cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 2)

    while True:
        begin = time.time()
        _pre = frame_list[-diff_step].astype(np.float)
        _post = frame_list[-1].astype(np.float)
        display_image[:size_h1, :size_w1] = ndimage.zoom(
            np.array(output_image), [1.5, 1.5, 1], order=0)
        display_image[:size_h2, size_w1:] = _pre[::2, ::2]
        display_image[size_h2:2 * size_h2, size_w1:] = _post[::2, ::2]
        add_text_info()
        cv2.imshow(window_name, display_image)
        if movie_path is not None:
            out.writeFrame(display_image[..., ::-1])
            time.sleep(max(0.0, 0.1 - (time.time() - begin)))
            print(time.time() - begin)
        key = cv2.waitKey(2)
        if key == 27:
            break


def run_test(func_inference, func_args=[], func_kwargs={}):
    dataset = FlyingChairs(
        subset="validation", validation_rate=0.1, validation_seed=2019)

    def calc_epe(in_flow, out_flow):
        # assert in_flow.shape == out_flow.shape
        return np.mean(np.sqrt(
            np.sum((in_flow - out_flow) ** 2, axis=3))
        )

    epe_list = []
    for _i in range(100):
        input_image, target_flow = dataset[_i]
        output_flow = func_inference(
            input_image[np.newaxis], *func_args, **func_kwargs)
        epe_list.append(calc_epe(output_flow, target_flow))
        print("epe: {:.3f}".format(epe_list[-1]))

    print("mean: {:.3f}, std: {:.3f}".format(
        np.mean(epe_list), np.std(epe_list)))
