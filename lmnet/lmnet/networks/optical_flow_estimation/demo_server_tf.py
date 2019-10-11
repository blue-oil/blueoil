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
import time
import math
import tqdm
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

from flow_to_image import flow_to_image

from lmnet import environment
from lmnet.utils import config as config_util
from lmnet.utils.executor import search_restore_filename

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default=socket.gethostname())
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('-i', '--experiment_id', type=str, required=True)
parser.add_argument('--restore_path', type=str, default=None)
parser.add_argument('--config_file', type=str, default=None)
parser.add_argument('--return_image', action="store_true")
parser.add_argument('--threshold', type=float, default=10.0)
args = parser.parse_args()


class Inference(object):
    def __init__(self, config, restore_path):
        ModelClass = config.NETWORK_CLASS
        network_kwargs = dict(
            (key.lower(), val) for key, val in config.NETWORK.items())
        network_kwargs["batch_size"] = 1
        graph = tf.Graph()
        with graph.as_default():
            self.model = ModelClass(
                classes=config.CLASSES,
                is_debug=config.IS_DEBUG,
                disable_load_op_library=True,
                **network_kwargs
            )
            self.images_placeholder, _ = self.model.placeholders()
            is_training = tf.constant(False, name="is_training")
            self.output_op = self.model.inference(
                self.images_placeholder, is_training)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=session_config)
        self.sess.run(init_op)
        saver.restore(self.sess, restore_path)

    def __call__(self, input_data):
        _x = (input_data / 255.0).astype(np.float32)
        t_begin = time.time()
        output = self.model.run(_x)
        calc_time = time.time() - t_begin
        if args.return_image:
            output = flow_to_image(
                -output[0][..., [1, 0]], threshold=args.threshold)
        return output, calc_time


def receive_and_send(connection, process_func):
    c = connection
    data_buffer = b""
    while True:
        received_buffer = c.recv(8192)
        if not received_buffer:
            break
        data_buffer += received_buffer
        if data_buffer[-7:] == b"__end__":
            break
    try:
        input_data = np.load(BytesIO(data_buffer))['input']
        output_data, calc_time = process_func(input_data)
        f = BytesIO()
        np.savez_compressed(f, output=output_data, calc_time=calc_time)
        f.seek(0)
        c.sendall(f.read())
    except ValueError:
        pass
    c.close()


def run_server(server_info, experiment_id, config_file, restore_path):
    environment.init(experiment_id)
    if config_file is None:
        config = config_util.load_from_experiment()
    else:
        config = config_util.merge(config, config_util.load(config_file))
    if restore_path is None:
        restore_file = search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

    inference_model = Inference(config, restore_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(server_info)
        s.listen(32)
        print("boot: {}:{}".format(*server_info))
        while True:
            client_conn, client_addr = s.accept()
            print("\033[Kfrom: {}:{}".format(*client_addr), end="\r")
            try:
                th = threading.Thread(
                    target=receive_and_send,
                    args=(client_conn, inference_model), daemon=True)
                th.start()
                # th.join()
                # receive_and_send(client_conn, inference_model)
            except BrokenPipeError:
                print("Send data aborted!")
                pass


if __name__ == '__main__':
    run_server(
        (args.host, args.port),
        args.experiment_id, args.config_file, args.restore_path)
