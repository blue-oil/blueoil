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

import os
import sys
import time
import math
import socket
import argparse
import threading
import collections
import numpy as np

from io import BytesIO
from nn_lib import NNLib
from flow_to_image import flow_to_image

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('--host', type=str, default=socket.gethostname())
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--return_image', action="store_true")
parser.add_argument('--threshold', type=float, default=10.0)
args = parser.parse_args()


def status_info(t):
    return "success" if t else "fail"


class Inference(object):
    def __init__(self, model_path):
        self.model = NNLib()
        res = self.model.load(model_path)
        print("load model: {}".format(status_info(res)))
        res = self.model.init()
        print("init model: {}".format(status_info(res)))

        input_shape = self.model.get_input_shape()
        output_shape = self.model.get_output_shape()
        print("input shape: {}".format(input_shape))
        print("output shape: {}".format(output_shape))

        test_flow = self.model.run(np.random.randn(
            *input_shape).astype(np.float32))
        print("shape test: {}".format(
            status_info(test_flow.shape == output_shape)))
        print("value test: {}".format(
            status_info(np.any(np.isnan(test_flow)))))

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


def run_server(server_info):
    inference_model = Inference(args.model_path)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(server_info)
    s.listen(32)
    print("boot: {}:{}".format(*server_info))
    while True:
        client_conn, client_addr = s.accept()
        # print("\033[Kfrom: {}:{}".format(*client_addr), end="\r")
        print("from: {}:{}".format(*client_addr), end="\n")
        try:
            th = threading.Thread(
                target=receive_and_send,
                args=(client_conn, inference_model))
            th.setDaemon(True)
            th.start()
            # th.join()
            # receive_and_send(client_conn, inference_model)
        except socket.error as e:
            print("Send data aborted!")
            pass
    s.close()


if __name__ == '__main__':
    run_server((args.host, args.port))
