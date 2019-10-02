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
import socket
import argparse
import threading
import collections
import numpy as np
import ctypes as ct

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

from io import BytesIO
from scipy import ndimage
from numpy.ctypeslib import ndpointer

from lmnet.networks.optical_flow_estimation.demo_lib import run_demo
from lmnet.networks.optical_flow_estimation.flow_to_image import flow_to_image

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
parser.add_argument('--diff_step', type=int, default=5)
args = parser.parse_args()


class NNLib(object):
    def __init__(self):
        self.lib = None
        self.nnlib = None

    def load(self, libpath):
        self.lib = ct.cdll.LoadLibrary(libpath)

        self.lib.network_create.argtypes = []
        self.lib.network_create.restype = ct.c_void_p

        self.lib.network_init.argtypes = [ct.c_void_p]
        self.lib.network_init.restype = ct.c_bool

        self.lib.network_delete.argtypes = [ct.c_void_p]
        self.lib.network_delete.restype = None

        self.lib.network_get_input_rank.argtypes = [ct.c_void_p]
        self.lib.network_get_input_rank.restype = ct.c_int

        self.lib.network_get_output_rank.argtypes = [ct.c_void_p]
        self.lib.network_get_output_rank.restype = ct.c_int

        self.lib.network_get_input_shape.argtypes = [
            ct.c_void_p, ndpointer(ct.c_int32, flags="C_CONTIGUOUS")]
        self.lib.network_get_input_shape.restype = None

        self.lib.network_get_output_shape.argtypes = [
            ct.c_void_p, ndpointer(ct.c_int32, flags="C_CONTIGUOUS")]
        self.lib.network_get_output_shape.restype = None

        self.lib.network_run.argtypes = [
            ct.c_void_p,
            ndpointer(
                ct.c_float,
                flags="C_CONTIGUOUS"),
            ndpointer(
                ct.c_float,
                flags="C_CONTIGUOUS"),
        ]
        self.lib.network_run.restype = None

        self.nnlib = self.lib.network_create()
        return True

    def init(self):
        return self.lib.network_init(self.nnlib)

    def delete(self):
        if self.nnlib:
            self.lib.network_delete(self.nnlib)
            self.nnlib = None
            self.lib = None

    def __del__(self):
        self.delete()

    def get_input_rank(self):
        return self.lib.network_get_input_rank(self.nnlib)

    def get_output_rank(self):
        return self.lib.network_get_output_rank(self.nnlib)

    def get_input_shape(self):
        r = self.get_input_rank()
        s = np.zeros(r, np.int32)
        self.lib.network_get_input_shape(self.nnlib, s)

        return tuple(s)

    def get_output_shape(self):
        r = self.get_output_rank()
        s = np.zeros(r, np.int32)
        self.lib.network_get_output_shape(self.nnlib, s)

        return tuple(s)

    def run(self, tensor):
        _in_data = tensor.flatten().astype(np.float32)
        _out_data = np.zeros((self.get_output_shape()), np.float32)

        self.lib.network_run(
            self.nnlib,
            _in_data,
            _out_data)

        return _out_data


if __name__ == '__main__':
    def status_info(t):
        return "success" if t else "fail"

    model = NNLib()
    res = model.load(args.model_path)
    print("load model: {}".format(status_info(res)))
    res = model.init()
    print("init model: {}".format(status_info(res)))

    input_shape = model.get_input_shape()
    output_shape = model.get_output_shape()
    print("input shape: {}".format(input_shape))
    print("output shape: {}".format(output_shape))

    test_flow = model.run(np.random.randn(*input_shape).astype(np.float32))
    print("shape test: {}".format(status_info(test_flow.shape == output_shape)))
    print("value test: {}".format(status_info(np.any(np.isnan(test_flow)))))

    def _inference(input_data):
        _x = (input_data / 255.0).astype(np.float32)
        output_flow = model.run(_x)
        return flow_to_image(-output_flow[0][..., [1, 0]])

    window_name = os.path.basename(args.model_path)
    run_demo(
        _inference, diff_step=args.diff_step,
        window_name=window_name,
        input_image_size=(input_shape[1], input_shape[2], 3)
    )
