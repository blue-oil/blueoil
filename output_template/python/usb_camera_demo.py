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

import click
import cv2
import numpy as np

from lmnet.common import get_color_map
from lmnet.nnlib import NNLib
from lmnet.utils.config import (
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

nn = None
pre_process = None
post_process = None


class MyTime:
    def __init__(self, function_name):
        self.start_time = time.time()
        self.function_name = function_name

    def show(self):
        print("TIME: ", self.function_name, time.time() - self.start_time)


def init_camera(camera_width, camera_height):
    if hasattr(cv2, 'cv'):
        vc = cv2.VideoCapture(0)
        vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, camera_width)
        vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, camera_height)
        vc.set(cv2.cv.CV_CAP_PROP_FPS, 10)

    else:
        vc = cv2.VideoCapture(1)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        vc.set(cv2.CAP_PROP_FPS, 10)

    return vc


def add_class_label(canvas,
                    text="Hello",
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.42,
                    font_color=(140, 40, 200),
                    line_type=1,
                    dl_corner=(50, 50)):
    cv2.putText(canvas, text, dl_corner, font, font_scale, font_color, line_type)


def inference_coroutine():
    global nn, pre_process, post_process
    result = None
    fps = 1.0
    while True:
        input_img = yield result, fps
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        result, fps, _ = run_inference(input_img, nn, pre_process, post_process)


def run_object_detection(config):
    # Init
    global nn
    nn.init()
    camera_width = 320
    camera_height = 240
    window_name = "Object Detection Demo"
    input_width = config.IMAGE_SIZE[1]
    input_height = config.IMAGE_SIZE[0]
    vc = init_camera(camera_width, camera_height)
    inference_coroutine_generator = inference_coroutine()
    next(inference_coroutine_generator)

    grabbed, input_img = vc.read()
    inference_coroutine_generator.send(input_img)
    #  ----------- Beginning of Main Loop ---------------
    while vc.isOpened():
        grabbed, input_img = vc.read()
        result, fps = inference_coroutine_generator.send(input_img)
        if result:
            window_img = add_rectangle(
                config.CLASSES,
                input_img,
                result,
                (input_height, input_width)
            )
            window_img = add_fps(window_img, fps)
        else:
            window_img = input_img

        cv2.imshow(window_name, window_img)
        key = cv2.waitKey(2)  # Wait for 2ms
        if key == 27:  # ESC to quit
            vc.release()
            cv2.destroyAllWindows()
            return
        # --------------------- End of main Loop -----------------------


def run_classification(config):
    # Init
    global nn
    nn.init()
    camera_width = 320
    camera_height = 240
    window_name = "Classification Demo"
    vc = init_camera(camera_width, camera_height)
    inference_coroutine_generator = inference_coroutine()
    next(inference_coroutine_generator)

    grabbed, input_img = vc.read()
    inference_coroutine_generator.send(input_img)
    #  ----------- Beginning of Main Loop ---------------
    while vc.isOpened():
        grabbed, input_img = vc.read()
        result, fps = inference_coroutine_generator.send(input_img)
        if result:
            result_class = np.argmax(result, axis=1)
            window_img = input_img.copy()
            add_class_label(window_img, text=str(result[0, result_class][0]), font_scale=0.52, dl_corner=(230, 230))
            add_class_label(window_img, text=config.CLASSES[result_class[0]], font_scale=0.52, dl_corner=(230, 210))
            window_img = add_fps(window_img, fps)
        else:
            window_img = input_img

        cv2.imshow(window_name, window_img)
        key = cv2.waitKey(2)  # Wait for 2ms
        if key == 27:  # ESC to quit
            vc.release()
            cv2.destroyAllWindows()
            return
        # --------------------- End of main Loop -----------------------


def run_semantic_segmentation(config):
    # Init
    global nn
    nn.init()
    camera_width = 320
    camera_height = 240
    window_name = "Semantic Segmentation Demo"
    vc = init_camera(camera_width, camera_height)
    inference_coroutine_generator = inference_coroutine()
    next(inference_coroutine_generator)
    colormap = np.array(get_color_map(len(config['CLASSES'])), dtype=np.uint8)

    grabbed, input_img = vc.read()
    inference_coroutine_generator.send(input_img)
    #  ----------- Beginning of Main Loop ---------------
    while vc.isOpened():
        grabbed, input_img = vc.read()
        result, fps = inference_coroutine_generator.send(input_img)
        if result:
            seg_img = label_to_color_image(result, colormap)
            seg_img = cv2.resize(seg_img, dsize=(camera_width, camera_height))
            window_img = cv2.addWeighted(input_img, 1, seg_img, 0.8, 0)
            window_img = add_fps(window_img, fps)
        else:
            window_img = input_img

        cv2.imshow(window_name, window_img)
        key = cv2.waitKey(2)  # Wait for 2ms
        if key == 27:  # ESC to quit
            vc.release()
            cv2.destroyAllWindows()
            return
        # --------------------- End of main Loop -----------------------


def run_keypoint_detection(config):
    # Init
    global nn
    nn.init()
    camera_width = 320
    camera_height = 240
    window_name = "Keypoint Detection Demo"
    input_width = config.IMAGE_SIZE[1]
    input_height = config.IMAGE_SIZE[0]
    vc = init_camera(camera_width, camera_height)
    inference_coroutine_generator = inference_coroutine()
    next(inference_coroutine_generator)

    grabbed, input_img = vc.read()
    inference_coroutine_generator.send(input_img)
    #  ----------- Beginning of Main Loop ---------------
    while vc.isOpened():
        grabbed, input_img = vc.read()
        result, fps = inference_coroutine_generator.send(input_img)
        if result:
            window_img = visualize_keypoint_detection(input_img, result[0], (input_height, input_width))
            window_img = add_fps(window_img, fps)
        else:
            window_img = input_img

        cv2.imshow(window_name, window_img)
        key = cv2.waitKey(2)  # Wait for 2ms
        if key == 27:  # ESC to quit
            vc.release()
            cv2.destroyAllWindows()
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

    TASK_HANDLERS = {"IMAGE.CLASSIFICATION": run_classification,
                     "IMAGE.OBJECT_DETECTION": run_object_detection,
                     "IMAGE.SEMANTIC_SEGMENTATION": run_semantic_segmentation,
                     "IMAGE.KEYPOINT_DETECTION": run_keypoint_detection}

    TASK_HANDLERS[config.TASK](config)


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
