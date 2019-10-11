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

__all__ = ["run_demo", "run_test"]

import os
import sys
import cv2
import time
import math
import signal
import threading
import collections
import numpy as np
from scipy import ndimage

from flow_to_image import flow_to_image


def init_camera(camera_height, camera_width, camera_id):
    if hasattr(cv2, 'cv'):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 10)
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def rescale_frame(frame, ratio):
    height = int(frame.shape[0] * ratio)
    width = int(frame.shape[1] * ratio)
    shape = (width, height)
    return cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)


def run_demo(
        func_inference, func_args=[], func_kwargs={},
        input_image_size=(384, 512, 3), diff_step=5, window_name="output",
        full_screen=True, movie_path=None, demo_name="output", camera_id=0,
        video_fps=10.0):

    # initializing worker and variables
    store_num = diff_step + 10
    camera_image_size = (480, 640)
    cap = init_camera(
        camera_image_size[0], camera_image_size[1], camera_id=camera_id)
    calc_time_list = collections.deque(maxlen=5)
    calc_time_list.append(1.0)
    view_time_list = collections.deque(maxlen=5)
    view_time_list.append(1.0)
    frame_list = collections.deque(maxlen=store_num)
    input_image = np.zeros(
        (1, input_image_size[0], input_image_size[1], input_image_size[2] * 2),
        dtype=np.uint8)
    output_flow = np.zeros(
        (1, input_image_size[0], input_image_size[1], 2), dtype=np.float32)
    output_image = np.zeros(input_image_size, dtype=np.uint8)
    camera_to_input_scale = 1.0 * input_image_size[0] / camera_image_size[0]

    # display image (9 * 16 size)
    display_image_size = (9 * 64, 16 * 64, 3)
    display_image = np.zeros(
        display_image_size, dtype=np.uint8)
    input_to_display_scale = (
        (1.0 * display_image_size[0] / input_image_size[0]) / 3,
        (1.0 * display_image_size[0] / input_image_size[0]) / 3,
        1.0
    )
    output_to_display_scale = (
        1.0 * display_image_size[0] / input_image_size[0],
        1.0 * display_image_size[0] / input_image_size[0],
        1.0
    )

    for _ in range(store_num):
        frame_list.append(np.zeros(input_image_size).astype(np.uint8))

    def _get_frame():
        while True:
            begin = time.time()
            res, frame = cap.read()
            assert res, "Something wrong occurs with camera!"
            frame_list.append(
                rescale_frame(frame[:, ::-1, :], camera_to_input_scale))
            time.sleep(max(0.0, 1.0 / 30 - (time.time() - begin)))

    def _get_output():
        while True:
            try:
                t_begin = time.time()
                input_image[0, ..., :3] = frame_list[-diff_step]
                input_image[0, ..., 3:] = frame_list[-1]
                output_flow[:], calc_time = func_inference(
                    input_image, *func_args, **func_kwargs)
                output_image[:] = flow_to_image(
                    -output_flow[0][..., [1, 0]], threshold=10.0)
                view_time = time.time() - t_begin
                calc_time_list.append(calc_time)
                view_time_list.append(view_time)
            except ValueError:
                pass

    t1 = threading.Thread(target=_get_frame)
    t1.setDaemon(True)
    t1.start()
    t2 = threading.Thread(target=_get_output)
    t2.setDaemon(True)
    t2.start()

    size_h1 = display_image_size[0]
    size_w1 = int(display_image_size[0] * 4 / 3)
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
        if hasattr(cv2, 'cv'):
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN,
                cv2.cv.CV_WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)

    if movie_path is not None:
        from skvideo.io import FFmpegWriter
        movie_shape = display_image.shape[:2]
        out = FFmpegWriter(
            movie_path,
            inputdict={
                '-r': str(video_fps),
                '-s': '{}x{}'.format(*movie_shape[::-1])},
            # outputdict={
            #     '-r': str(10),
            #     '-c:v': 'libx264',
            #     '-crf': str(17),
            #     '-preset': 'ultrafast',
            #     '-pix_fmt': 'yuv444p'}
        )

    def add_text_info():
        # FPS infomation
        view_fps = 1 / np.mean(view_time_list)
        calc_fps = 1 / np.mean(calc_time_list)
        fps_text = "{} (FPS: {:>5.2f}/{:>5.2f})".format(
            demo_name, view_fps, calc_fps)
        cv2.putText(
            display_image, fps_text, (10, 25),
            cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 2)

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
        t_begin = time.time()
        _pre = frame_list[-diff_step].astype(np.float)
        _post = frame_list[-1].astype(np.float)
        display_image[:size_h2, size_w1:] = ndimage.zoom(
            _pre, input_to_display_scale, order=0)
        display_image[size_h2:2 * size_h2, size_w1:] = ndimage.zoom(
            _post, input_to_display_scale, order=0)
        display_image[:size_h1, :size_w1] = ndimage.zoom(
            np.array(output_image), output_to_display_scale, order=0)
        add_text_info()
        cv2.imshow(window_name, display_image)
        if movie_path is not None:
            out.writeFrame(display_image[..., ::-1])
            t_sleep = max(0.0, (1.0 / video_fps) - (time.time() - t_begin))
            print(t_sleep)
            time.sleep(t_sleep)
        key = cv2.waitKey(2)
        if key == 27:
            break
    # signal.signal(signal.SIGALRM, _render)
    # signal.setitimer(signal.ITIMER_REAL, 1.0 / video_fps, 1.0 / video_fps)


def run_test(func_inference, split_step=1, func_args=[], func_kwargs={}):
    sys.path.extend(["./lmnet", "/dlk/python/dlk"])
    from lmnet.datasets.optical_flow_estimation import (
        FlyingChairs, ChairsSDHom
    )

    dataset = FlyingChairs(
        subset="validation", validation_rate=0.1, validation_seed=2019)

    def calc_epe(in_flow, out_flow):
        # assert in_flow.shape == out_flow.shape
        return np.mean(np.sqrt(
            np.sum((in_flow - out_flow) ** 2, axis=3))
        )

    epe_list = []
    calc_time_list = []
    for _i in range(1000):
        input_image, target_flow = dataset[_i]
        output_flow, calc_time = func_inference(
            input_image[::split_step, ::split_step, :][np.newaxis],
            *func_args, **func_kwargs)
        epe_list.append(calc_epe(
            output_flow, target_flow[::split_step, ::split_step]))
        calc_time_list.append(calc_time)
        print("epe: {:.3f} | calc_time {:.3f}".format(
            epe_list[-1], calc_time_list[-1]))

    print("mean: {:.3f}, std: {:.3f}".format(
        np.mean(epe_list), np.std(epe_list)))
    print("time: {:.3f}±{:.3f}".format(
        np.mean(calc_time_list), np.std(calc_time_list)))
