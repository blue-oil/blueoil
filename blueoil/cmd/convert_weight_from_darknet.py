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
import struct

import click
import numpy as np
import tensorflow as tf

from blueoil import environment
from blueoil.networks.classification.darknet import Darknet
from blueoil.networks.object_detection.yolo_v2 import YoloV2
from blueoil.utils import config as config_util
from blueoil.utils import executor


def convert(config, weight_file):
    ModelClass = config.NETWORK_CLASS
    network_kwargs = dict((key.lower(), val) for key, val in config.NETWORK.items())

    graph = tf.Graph()
    with graph.as_default():

        if ModelClass is YoloV2:
            classes = list(range(0, 20))

        elif ModelClass is Darknet:
            classes = list(range(0, 1000))

        model = ModelClass(
            classes=classes,
            is_debug=True,
            **network_kwargs,
        )
        global_step = tf.Variable(0, name="global_step", trainable=False) # NOQA

        is_training = tf.constant(False, name="is_training")

        images_placeholder, labels_placeholder = model.placeholders()

        model.inference(images_placeholder, is_training)

        init_op = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver(max_to_keep=None)

        variables = tf.compat.v1.global_variables()

    session_config = None
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run([init_op, ])
    suffixes = ['bias', 'beta', 'gamma', 'moving_mean', 'moving_variance', 'kernel']
    convert_variables = []
    for var in variables:
        if var.op.name == 'global_step':
            continue
        convert_variables.append(var)

    def sort_key(var):
        block_number = var.op.name.split("/")[0].split("_")[1]
        for i, suffix in enumerate(suffixes):
            if var.op.name.endswith(suffix):
                return int(block_number) * 5 + i

    convert_variables.sort(key=sort_key)
    for var in convert_variables:
        print(var.op.name)

    with open(weight_file, 'rb') as fopen:
        major, minor, revision, seen = struct.unpack('4i', fopen.read(16))
        print("major = %d, minor = %d, revision = %d, seen = %d" % (major, minor, revision, seen))

        # You can only use the version of darknet.
        assert major == 0
        assert minor == 1

        total = 0
        for var in convert_variables:
            remaining = os.fstat(fopen.fileno()).st_size - fopen.tell()
            print("processing layer {}".format(var.op.name))
            print("remaining: {} bytes.".format(remaining))

            shape = var.get_shape().as_list()
            cnt = np.multiply.reduce(shape)
            total += cnt
            print("{}: shape: {}. num elements: {}".format(var.op.name, str(shape), cnt))
            data = struct.unpack('%df' % cnt, fopen.read(4 * cnt))
            data = np.array(data, dtype=np.float32)
            if "kernel" in var.op.name:
                kernel_size_1, kernel_size_2, input_channel, output_channel = shape
                data = data.reshape([output_channel, input_channel, kernel_size_1, kernel_size_2])
                data = np.transpose(data, [2, 3, 1, 0])

            # if yolov2 last layer
            if "conv_23" in var.op.name:
                num_anchors = 5
                if "kernel" in var.op.name:
                    weights = data.reshape([kernel_size_1, kernel_size_2, input_channel, num_anchors, -1])
                    boxes = weights[:, :, :, :, 0:4]
                    conf = np.expand_dims(weights[:, :, :, :, 4], -1)
                    classes = weights[:, :, :, :, 5:]
                    data = np.concatenate([classes, conf, boxes], -1)
                    data = data.reshape([kernel_size_1, kernel_size_2, input_channel, output_channel])

                if "bias" in var.op.name:
                    biases = data.reshape([num_anchors, -1])
                    boxes = biases[:, 0:4]
                    conf = np.expand_dims(biases[:, 4], -1)
                    classes = biases[:, 5:]
                    data = np.concatenate([classes, conf, boxes, ], -1).reshape([-1])

            sess.run(var.assign(data))
            print("total: {} elements".format(total))

        print("")
        print("{} elements assigned".format(total))
        remaining = os.fstat(fopen.fileno()).st_size - fopen.tell()
        print("remaining: {}".format(remaining))
        assert remaining == 0

    checkpoint_file = "save.ckpt"
    saver.save(sess, os.path.join(environment.CHECKPOINTS_DIR,  checkpoint_file))

    print("-------- output --------")
    print("save checkpoint to : {}".format(os.path.join(environment.CHECKPOINTS_DIR,  checkpoint_file)))


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-m",
    "--model",
    help="yolo2 or darknet19",
    type=click.Choice(["yolov2", "darknet19"]),
    required=True,
)
def main(model):
    if model == "yolov2":
        weight_file = 'inputs/yolo-voc.weights'
        experiment_id = "convert_weight_from_darknet/yolo_v2"
        config_file = "configs/convert_weight_from_darknet/yolo_v2.py"

    if model == "darknet19":
        weight_file = 'inputs/darknet19_448.weights'
        experiment_id = "convert_weight_from_darknet/darknet19"
        config_file = "configs/convert_weight_from_darknet/darknet19.py"

    recreate = True
    environment.init(experiment_id)
    executor.prepare_dirs(recreate)

    config = config_util.load(config_file)
    config_util.display(config)

    config_util.copy_to_experiment_dir(config_file)

    convert(config, weight_file)


if __name__ == '__main__':
    main()
