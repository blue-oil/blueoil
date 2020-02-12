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

import click
import cv2
import numpy as np

from lmnet.common import get_color_map
from lmnet.nnlib import NNLib
from blueoil.utils.config import (
    load_yaml,
    build_pre_process,
    build_post_process,
)
from lmnet.utils.demo import (
    add_rectangle,
    add_fps,
    run_inference,
)

from lmnet.visualize import (
    label_to_color_image,
    visualize_keypoint_detection,
)
from lmnet.pre_processor import resize


nn = None
pre_process = None
post_process = None


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


def infer_loop(q_input, q_output):
    global nn, pre_process, post_process
    nn.init()
    while True:
        img_orig, fps = q_input.get()
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        result, _, _ = run_inference(img, nn, pre_process, post_process)
        q_output.put((result, fps, img_orig))

def show_object_detection(img, result, fps, window_height, window_width, config):
    window_img = resize(img, size=[window_height, window_width])

    input_width = config.IMAGE_SIZE[1]
    input_height = config.IMAGE_SIZE[0]
    window_img = add_rectangle(config.CLASSES,
        window_img, result, (input_height, input_width)
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

def capture_loop(q_input):
    camera_width = 320
    camera_height = 240

    vc = init_camera(camera_width, camera_height)

    count_frames = 10
    prev_1 = time.clock()
    prev = deque([prev_1] * count_frames)

    while True:
        valid, img = vc.read()
        if valid:
            now = time.clock()
            prev.append(now)
            old = prev.popleft()
            fps = count_frames / (now - old)
            q_input.put((img, fps))

def run_impl(config):
    # Set variables
    q_input = Queue(2)
    q_output = Queue(4)

    p_capture = Process(target=capture_loop, args=(q_input,))
    p_capture.start()

    p_infer = Process(target=infer_loop, args=(q_input, q_output))
    p_infer.start()

    window_width = 320
    window_height = 240

    show_handles_table = {
        "IMAGE.OBJECT_DETECTION": show_object_detection,
        "IMAGE.CLASSIFICATION": show_classification,
        "IMAGE.SEMANTIC_SEGMENTATION": show_semantic_segmentation,
        "IMAGE.KEYPOINT_DETECTION": show_keypoint_detection
    }
    show_handle = show_handles_table[config.TASK]

    #  ----------- Beginning of Main Loop ---------------
    while True:
        if not q_output.empty():
            result, fps, img = q_output.get()
            show_handle(img, result, fps, window_height, window_width, config)
            key = cv2.waitKey(1)    # Wait for 1ms
            if key == 27:           # ESC to quit
                sleep(1.0)          # Wait for worker's current task is finished
                p_capture.terminate()
                p_infer.terminate()
                return
    # --------------------- End of main Loop -----------------------


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

    run_impl(config)


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
    default="../models/lib/lib_fpga.so",
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
