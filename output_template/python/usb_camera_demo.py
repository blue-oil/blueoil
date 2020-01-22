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

import os
import sys
import asyncio
import concurrent.futures

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


vcread_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
nn_run_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
imshow_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
post_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)


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


# thread
def io_vcread(vc):
    grabbed, current_img = vc.read()
    return current_img


# thread
def io_nn_run(image):
    global nn, pre_process
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pre_process(image=image)["image"]
    data = np.expand_dims(data, axis=0)
    network_output = nn.run(data)
    return network_output


# process
def cpu_post_process(network_output):
    global post_process
    post_processed = post_process(outputs=network_output)["outputs"]
    return post_processed


# thread
def io_object_detection_imshow(image, network_output, config):
    image_to_show = add_rectangle(config.CLASSES,
                                  image,
                                  network_output,
                                  (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
    cv2.imshow("asyncio_object_detection_demo", image_to_show)
    cv2.waitKey(2)


def io_keypoint_detection_imshow(image, network_output, config):
    image_to_show = visualize_keypoint_detection(image,
                                                 network_output,
                                                 (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
    cv2.imshow("asyncio_keypoint_detection_demo", image_to_show)
    cv2.waitKey(2)


def io_classification_imshow(image, network_output, config):
    image_to_show = visualize_keypoint_detection(image,
                                                 network_output,
                                                 (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
    cv2.imshow("asyncio_keypoint_detection_demo", image_to_show)
    cv2.waitKey(2)


def io_semantic_segmentation_imshow(image, network_output, config):
    seg_img = label_to_color_image(network_output, config.colormap)
    seg_img = cv2.resize(seg_img, dsize=(320, 240))
    image_to_show = cv2.addWeighted(image, 1, seg_img, 0.8, 0)
    cv2.imshow("asyncio_semantic_segmentation_demo", image_to_show)
    cv2.waitKey(2)


@asyncio.coroutine
def run_coro(config, imshow_func):
    vc = init_camera(240, 320)
    colormap = np.array(get_color_map(len(config['CLASSES'])), dtype=np.uint8)
    config.colormap = colormap
    loop = asyncio.get_event_loop()
    while vc.isOpened():
        image = yield from loop.run_in_executor(vcread_thread_pool, io_vcread,
                                                vc)

        network_ouput = yield from loop.run_in_executor(nn_run_thread_pool, io_nn_run,
                                                        image)

        post_processed = yield from loop.run_in_executor(post_process_pool, cpu_post_process,
                                                         network_ouput)

        yield from loop.run_in_executor(imshow_thread_pool, imshow_func,
                                        image, post_processed, config)


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
        nn.init()

    elif file_extension == '.pb':  # Protocol Buffer file
        # only load tensorflow if user wants to use GPU
        from lmnet.tensorflow_graph_runner import TensorflowGraphRunner
        nn = TensorflowGraphRunner(model)

    IMSHOW_FUNCS = {"IMAGE.CLASSIFICATION": io_classification_imshow,
                    "IMAGE.OBJECT_DETECTION": io_object_detection_imshow,
                    "IMAGE.SEMANTIC_SEGMENTATION": io_semantic_segmentation_imshow,
                    "IMAGE.KEYPOINT_DETECTION": io_keypoint_detection_imshow}

    imshow_func = IMSHOW_FUNCS[config.TASK]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_coro(config, imshow_func))


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
