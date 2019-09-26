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
import imghdr
import argparse
import warnings

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from glob import glob
from multiprocessing import Pool

from lmnet import environment
from lmnet.utils import config as config_util
from lmnet.utils.executor import search_restore_filename
from lmnet.networks.optical_flow_estimation.demo_lib import run_demo
from lmnet.networks.optical_flow_estimation.flow_to_image import flow_to_image


'''
references
https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale
https://techoverflow.net/2018/12/18/how-to-set-cv2-videocapture-image-size/
'''

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--experiment_id', type=str, required=True)
parser.add_argument('-c', '--config_file', type=str, default=None)
parser.add_argument('--restore_path', type=str, default=None)
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--diff_step', type=str, default=5)
args = parser.parse_args()


class Inference(object):
    def __init__(self, config, restore_path):
        _file_name, file_ext = os.path.splitext(restore_path)
        if file_ext == ".pb":
            graph = tf.Graph()
            with graph.as_default():
                with open(restore_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="")
                self.images_placeholder = graph.get_tensor_by_name(
                    'images_placeholder:0')
                self.output_op = graph.get_tensor_by_name('output:0')
                init_op = tf.global_variables_initializer()
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=session_config)
            self.sess.run(init_op)
        else:
            ModelClass = config.NETWORK_CLASS
            network_kwargs = dict(
                (key.lower(), val) for key, val in config.NETWORK.items())
            network_kwargs["batch_size"] = 1
            graph = tf.Graph()
            with graph.as_default():
                model = ModelClass(
                    classes=config.CLASSES,
                    is_debug=config.IS_DEBUG,
                    disable_load_op_library=True,
                    **network_kwargs
                )
                is_training = tf.constant(False, name="is_training")
                init_op = tf.global_variables_initializer()
                self.images_placeholder, _ = model.placeholders()
                self.output_op = model.inference(
                    self.images_placeholder, is_training)
                saver = tf.train.Saver(max_to_keep=None)
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=session_config)
            self.sess.run(init_op)
            saver.restore(self.sess, restore_path)

    def __call__(self, input_data):
        feed_dict = {self.images_placeholder: input_data * (1 / 255.0)}
        output_flow = self.sess.run(
            self.output_op, feed_dict=feed_dict)
        return flow_to_image(-output_flow[0][..., [1, 0]])


if __name__ == '__main__':
    environment.init(args.experiment_id)
    config = config_util.load_from_experiment()
    print(config)
    if args.config_file is not None:
        config = config_util.merge(
            config, config_util.load(args.config_file))

    if args.restore_path is None:
        restore_file = search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)
    else:
        restore_path = args.restore_path
    print("Restore from {}".format(restore_path))
    inference_model = Inference(config, restore_path)
    window_name = os.path.basename(restore_path)
    run_demo(inference_model, diff_step=args.diff_step, window_name=window_name)
