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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import os
import sys
from multiprocessing import Process, Queue
from time import sleep
from collections import deque
import queue

import click
import cv2
import numpy as np
from threading import Thread, Lock, Condition

from blueoil.common import get_color_map
from lmnet.nnlib import NNLib
from config import (
    load_yaml,
    build_pre_process,
    build_post_process,
)
from lmnet.utils.demo import (
    add_rectangle,
    add_fps,
    run_inference,
)

from blueoil.visualize import (
    label_to_color_image,
    visualize_keypoint_detection,
)
from blueoil.pre_processor import resize


def _phase_method_decorator(func):
    def wrapper(*args):
        slf = args[0]
        if slf.terminate_program:
            return True
        return func(*args)
    return wrapper


class PhaseManager:
    """
    There are 3 threads in this program

    1. Capture thread: This thread reads the image from camera
    2. Infer thread: This thread runs the NN and predicts
    3. Main thread: This thread draws a image, a fps, a prediction result, etc... on window

    This class manages which thread to run, and holds shared-data
    """
    def __init__(self):
        self._captured_queue = queue.Queue()
        self._infered_queue = queue.Queue()
        self.cv = Condition()
        self.terminate_program = False

    @_phase_method_decorator
    def is_capture_phase(self):
        return (self._captured_queue.empty() and
                self._infered_queue.empty())

    @_phase_method_decorator
    def is_infer_phase(self):
        return not self._captured_queue.empty()

    @_phase_method_decorator
    def is_draw_phase(self):
        return not self._infered_queue.empty()

    def push_captured_result(self, img):
        self._captured_queue.put(img)

    def pop_captured_result(self):
        return self._captured_queue.get()

    def push_infered_result(self, img, pred):
        self._infered_queue.put((img, pred))

    def pop_infered_result(self):
        return self._infered_queue.get()


class FPSCalculator:
    """
    Calculate current FPS
    FPS is calculated by last 10 values which is returned by time.perf_counter
    time.pref_counter is assumed to be called when the main thread is drawing a image
    """
    def __init__(self, count_frame):
        self.count_frame = count_frame
        prev1 = time.perf_counter()
        self._prev = deque([prev1] * self.count_frame)

    def _measure(self, begin, end):
        elapsed = end - begin
        return self.count_frame / elapsed

    def _get_front(self):
        ret = self._prev.popleft()
        self._prev.appendleft(ret)
        return ret

    def _get_back(self):
        ret = self._prev.pop()
        self._prev.pop(ret)
        return ret

    def update(self):
        old = self._prev.popleft()
        now = time.perf_counter()
        self._prev.append(now)
        return self._measure(old, now)

    def get(self):
        begin = self._get_front()
        end = self._get_back()
        return self._measure(begin, end)


def show_object_detection(img, result, fps, window_height, window_width, config):
    window_img = resize(img, size=[window_height, window_width])

    input_width = config.IMAGE_SIZE[1]
    input_height = config.IMAGE_SIZE[0]
    window_img = add_rectangle(
        config.CLASSES,
        window_img,
        result,
        (input_height, input_width),
    )
    img = add_fps(window_img, fps)

    window_name = "Object Detection Demo"
    cv2.imshow(window_name, window_img)


def show_classification(img, result, fps, window_height, window_width, config):
    window_img = resize(img, size=[window_height, window_width])

    result_class = np.argmax(result, axis=1)
    add_class_label(window_img, text=str(result[0, result_class][0]), font_scale=0.52, dl_corner=(230, 230))
    add_class_label(window_img, text=config.CLASSES[result_class[0]], font_scale=0.52, dl_corner=(230, 210))
    window_img = add_fps(window_img, fps)

    window_name = "Classification Demo"
    cv2.imshow(window_name, window_img)


def show_semantic_segmentation(img, result, fps, window_height, window_width, config):
    orig_img = resize(img, size=[window_height, window_width])

    colormap = np.array(get_color_map(len(config.CLASSES)), dtype=np.uint8)
    seg_img = label_to_color_image(result, colormap)
    seg_img = cv2.resize(seg_img, dsize=(window_width, window_height))
    window_img = cv2.addWeighted(orig_img, 1, seg_img, 0.8, 0)
    window_img = add_fps(window_img, fps)

    window_name = "Semantic Segmentation Demo"
    cv2.imshow(window_name, window_img)


def show_keypoint_detection(img, result, fps, window_height, window_width, config):
    window_img = resize(img, size=[window_height, window_width])

    input_width = config.IMAGE_SIZE[1]
    input_height = config.IMAGE_SIZE[0]
    window_img = visualize_keypoint_detection(window_img, result[0], (input_height, input_width))
    window_img = add_fps(window_img, fps)

    window_name = "Keypoint Detection Demo"
    cv2.imshow(window_name, window_img)


def get_show_handle(config):
    show_handles_table = {
        "IMAGE.OBJECT_DETECTION": show_object_detection,
        "IMAGE.CLASSIFICATION": show_classification,
        "IMAGE.SEMANTIC_SEGMENTATION": show_semantic_segmentation,
        "IMAGE.KEYPOINT_DETECTION": show_keypoint_detection
    }
    show_handle = show_handles_table[config.TASK]
    return show_handle


nn = None
pre_process = None
post_process = None
phase_manager = PhaseManager()


def init_camera(camera_width, camera_height):
    if hasattr(cv2, 'cv'):
        vc = cv2.VideoCapture(0)
        vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, camera_width)
        vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, camera_height)
        vc.set(cv2.cv.CV_CAP_PROP_FPS, 60)
    else:
        vc = cv2.VideoCapture(0)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        vc.set(cv2.CAP_PROP_FPS, 60)

    return vc


def add_class_label(canvas,
                    text="Hello",
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.42,
                    font_color=(140, 40, 200),
                    line_type=1,
                    dl_corner=(50, 50)):
    cv2.putText(canvas, text, dl_corner, font, font_scale, font_color, line_type)


def capture_loop():
    global phase_manager

    camera_width = 320
    camera_height = 240

    vc = init_camera(camera_width, camera_height)
    cv = phase_manager.cv

    while True:
        with cv:
            while not phase_manager.is_capture_phase():
                cv.wait()
            if phase_manager.terminate_program:
                return
            cv.acquire()
            while True:
                valid, img = vc.read()
                if valid:
                    phase_manager.push_captured_result(img)
                    break
                else:
                    sleep(0.01)
            cv.release()
            cv.notify_all()


def infer_loop():
    global phase_manager, nn, pre_process, post_process

    nn.init()
    cv = phase_manager.cv

    while True:
        with cv:
            while not phase_manager.is_infer_phase():
                cv.wait()
            if phase_manager.terminate_program:
                return
            cv.acquire()
            img_orig = phase_manager.pop_captured_result()
            img = img_orig
            # img = _adjust_image(img_orig)
            pred, _, _ = run_inference(img, nn, pre_process, post_process)
            phase_manager.push_infered_result(img_orig, pred)
            cv.release()
            cv.notify_all()


def main_loop(config):
    global phase_manager, nn, pre_process, post_process

    window_width = 320
    window_height = 240
    fps_calculator = FPSCalculator(10)
    show_handle = get_show_handle(config)
    cv = phase_manager.cv
    th_capture = Thread(target=capture_loop)
    th_infer = Thread(target=infer_loop)

    th_capture.start()
    th_infer.start()

    while True:
        with cv:
            while not phase_manager.is_draw_phase():
                cv.wait()
            cv.acquire()
            img, pred = phase_manager.pop_infered_result()
            fps = fps_calculator.update()
            show_handle(img, pred, fps, window_height, window_width, config)
            key = cv2.waitKey(1)    # Wait for 1ms
            if key == 27:           # ESC to quit
                phase_manager.terminate_program = True
                cv.release()
                cv.notify_all()     # Notify other threads to terminate
                break
            cv.release()
            cv.notify_all()

    th_capture.join()
    th_infer.join()


def run(model, config_file):
    global nn, pre_process, post_process
    filename, file_extension = os.path.splitext(model)
    supported_files = ['.so', '.pb']

    if file_extension not in supported_files:
        raise Exception("""
            Unknown file type. Got %s%s.
            Please check the model file (-m).
            Only .pb (protocol buffer) or .so (shared object) file is supported.
            """ % (filename, file_extension))

    config = load_yaml(config_file)
    pre_process = build_pre_process(config.PRE_PROCESSOR)
    post_process = build_post_process(config.POST_PROCESSOR)

    if file_extension == '.so':  # Shared library
        nn = NNLib()
        nn.load(model)

    elif file_extension == '.pb':  # Protocol Buffer file
        # only load tensorflow if user wants to use GPU
        from lmnet.tensorflow_graph_runner import TensorflowGraphRunner
        nn = TensorflowGraphRunner(model)

    main_loop(config)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-m",
    "-l",
    "--model",
    type=click.Path(exists=True),
    help=u"""
        Inference Model filename
        (-l is deprecated please use -m instead)
    """,
    default="../models/lib/libdlk_fpga.so",
)
@click.option(
    "-c",
    "--config_file",
    type=click.Path(exists=True),
    help=u"Config file Path",
    default="../models/meta.yaml",
)
def main(model, config_file):
    _check_deprecated_arguments()
    run(model, config_file)


def _check_deprecated_arguments():
    argument_list = sys.argv
    if '-l' in argument_list:
        print("Deprecated warning: -l is deprecated please use -m instead")


if __name__ == "__main__":
    main()
