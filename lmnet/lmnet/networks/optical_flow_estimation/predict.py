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
import imghdr

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

import threading
import collections
import numpy as np
import tensorflow as tf

from glob import glob
from multiprocessing import Pool

from lmnet import environment
from lmnet.utils import config as config_util
from lmnet.utils.executor import search_restore_filename
# from lmnet.networks.optical_flow_estimation.flow_to_image import flow_to_image
from lmnet.networks.optical_flow_estimation.flowlib import flow_to_image


'''
references
https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale
https://techoverflow.net/2018/12/18/how-to-set-cv2-videocapture-image-size/
'''


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


def main_process(config, restore_path):
    ModelClass = config.NETWORK_CLASS
    network_kwargs = dict(
        (key.lower(), val) for key, val in config.NETWORK.items())
    network_kwargs["batch_size"] = 1
    image_size = (384, 512, 3)

    # initializing camera
    cap = init_camera(480, 640)

    # initializing tf model
    graph = tf.Graph()
    with graph.as_default():
        model = ModelClass(
            classes=config.CLASSES,
            is_debug=config.IS_DEBUG,
            disable_load_op_library=True,
            **network_kwargs
        )
        is_training = tf.constant(False, name="is_training")
        images_placeholder, _ = model.placeholderes()
        output_op = model.inference(images_placeholder, is_training)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=None)
    session_config = tf.ConfigProto()
    sess = tf.Session(graph=graph, config=session_config)
    sess.run(init_op)
    saver.restore(sess, restore_path)

    # initializing worker and variables
    diff_step = 1
    frame_list = collections.deque(maxlen=300)
    input_image = np.zeros(
        (1, *image_size[:2], image_size[-1] * 2)).astype(np.uint8)
    output_flow = np.zeros((1, *image_size[:2], 2))
    output_image = np.zeros(image_size).astype(np.uint8)
    for _ in range(diff_step):
        frame_list.append(np.zeros(image_size).astype(np.uint8))

    def _get_frame():
        while True:
            begin = time.time()
            res, frame = cap.read()
            assert res, "Something wrong occurs with camera!"
            frame_list.append(rescale_frame(frame[:, ::-1, :], 0.8))
            time.sleep(max(0.0, 1.0 / 30 - (time.time() - begin)))

    def _inference():
        time_list = collections.deque(maxlen=10)
        while True:
            begin = time.time()
            input_image[0, ..., 3:] = frame_list[-diff_step]
            input_image[0, ..., :3] = frame_list[-1]
            feed_dict = {images_placeholder: input_image}
            output_flow[:] = sess.run(output_op, feed_dict=feed_dict)
            time_list.append(time.time() - begin)
            print("\033[KFPS: {:.3f}".format(
                np.mean(1 / np.array(time_list))), end="\r")

    def _post_process():
        while True:
            output_image[:] = flow_to_image(output_flow[0])

    t1 = threading.Thread(target=_get_frame)
    t1.setDaemon(True)
    t2 = threading.Thread(target=_inference)
    t2.setDaemon(True)
    t3 = threading.Thread(target=_post_process)
    t3.setDaemon(True)
    t1.start()
    t2.start()
    t3.start()

    while True:
        # cv2.imshow("raw", input_image[0, ..., :3])
        # cv2.imshow("comp", np.concatenate([
        #     input_image[0, ..., :3], input_image[0, ..., 3:]], axis=1))
        _pre = frame_list[-diff_step].astype(np.float)
        _post = frame_list[-1].astype(np.float)
        # diff = np.abs(_pre - _post).astype(np.uint8)
        diff = np.mean([_pre, _post], axis=0).astype(np.uint8)
        # diff = np.mean([
        #     frame_list[_diff_step * _ - 1] for _ in range(5)],
        #     axis=0).astype(np.uint8)
        cv2.imshow("diff", diff)
        cv2.imshow("output", output_image)
        key = cv2.waitKey(2)
        if key == 27:
            break


def init_process(experiment_id, config_file, restore_path):
    environment.init(experiment_id)
    config = config_util.load_from_experiment()
    if config_file:
        config = config_util.merge(config, config_util.load(config_file))

    if restore_path is None:
        restore_file = search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)
    if not os.path.exists("{}.index".format(restore_path)):
        raise Exception("restore file {} dont exists.".format(restore_path))
    print("Restore from {}".format(restore_path))

    return config, restore_path


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--experiment_id",
    help="Experiment id",
    required=True
)
@click.option(
    "-c",
    "--config_file",
    help="config file path. override saved experiment config.",
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001",
    default=None,
)
def main(experiment_id, config_file, restore_path):
    config, restore_path = init_process(
        experiment_id, config_file, restore_path)
    print("----- start prediction -----")
    main_process(config, restore_path)
    print("----- end prediction -----")


if __name__ == '__main__':
    main()
