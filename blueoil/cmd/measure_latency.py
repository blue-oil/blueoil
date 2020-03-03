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
import glob
import os
import time

import click
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from blueoil import environment
from blueoil.utils.image import load_image
from blueoil.utils import config as config_util
from blueoil.utils import executor


# TODO(wakisaka): duplicated function with blueoil/cmd/export.py
def _pre_process(raw_image, pre_processor, data_format):
    image = pre_processor(image=raw_image)['image']
    if data_format == 'NCHW':
        image = np.transpose(image, [2, 0, 1])
    return image


def _measure_time(config, restore_path, step_size):
    graph = tf.Graph()

    ModelClass = config.NETWORK_CLASS
    network_kwargs = dict((key.lower(), val) for key, val in config.NETWORK.items())

    with graph.as_default():

        model = ModelClass(
            classes=config.CLASSES,
            is_debug=config.IS_DEBUG,
            **network_kwargs,
        )

        is_training = tf.constant(False, name="is_training")

        images_placeholder, labels_placeholder = model.placeholders()
        output = model.inference(images_placeholder, is_training)

        init_op = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver()

    session_config = None  # tf.ConfigProto(log_device_placement=True)
    # session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run(init_op)

    if restore_path:
        saver.restore(sess, restore_path)

    # Try to inference once before measure time.
    raw_image = np.random.randint(256, size=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3,)).astype('uint8')
    image = _pre_process(raw_image, config.PRE_PROCESSOR, config.DATA_FORMAT)
    images = np.expand_dims(image, axis=0)
    feed_dict = {
        images_placeholder: images,
    }
    output_np = sess.run(output, feed_dict=feed_dict)
    if config.POST_PROCESSOR:
        config.POST_PROCESSOR(**{"outputs": output_np})

    # measure time
    image_files_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "fixtures", "measure_latency_images", "*.jpg")

    image_files = glob.glob(image_files_path)
    overall_times = []
    only_network_times = []

    for test_step in range(step_size):
        index = test_step % len(image_files)
        image_file = image_files[index]
        raw_image = load_image(image_file)

        start_overall = time.time()

        image = _pre_process(raw_image, config.PRE_PROCESSOR, config.DATA_FORMAT)
        images = np.expand_dims(image, axis=0)
        feed_dict = {
            images_placeholder: images,
        }

        start_only_network = time.time()
        output_np = sess.run(output, feed_dict=feed_dict)
        only_network_times.append(time.time() - start_only_network)

        if config.POST_PROCESSOR:
            config.POST_PROCESSOR(**{"outputs": output_np})

        overall_times.append(time.time() - start_overall)

    return overall_times, only_network_times


def _run(config_file, experiment_id, restore_path, image_size, step_size, cpu):

    if experiment_id:
        environment.init(experiment_id)
        config = config_util.load_from_experiment()
        if config_file:
            config = config_util.merge(config, config_util.load(config_file))

        if restore_path is None:
            restore_file = executor.search_restore_filename(environment.CHECKPOINTS_DIR)
            restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

        if not os.path.exists("{}.index".format(restore_path)):
            raise Exception("restore file {} dont exists.".format(restore_path))

    else:
        experiment_id = "measure_latency"
        environment.init(experiment_id)
        config = config_util.load(config_file)

    config.BATCH_SIZE = 1
    config.NETWORK.BATCH_SIZE = 1
    config.DATASET.BATCH_SIZE = 1

    if list(image_size) != [None, None]:
        config.IMAGE_SIZE = list(image_size)
        config.NETWORK.IMAGE_SIZE = list(image_size)

        # override pre processes image size.
        if config.PRE_PROCESSOR:
            config.PRE_PROCESSOR.set_image_size(image_size)

        # override post processes image size.
        if config.POST_PROCESSOR:
            config.POST_PROCESSOR.set_image_size(image_size)

        print("Override IMAGE_SIZE", config.IMAGE_SIZE)

    executor.init_logging(config)
    config_util.display(config)

    overall_times, only_network_times = _measure_time(config, restore_path, step_size)

    overall_times = np.array(overall_times)
    only_network_times = np.array(only_network_times)
    # list of physical_device_desc
    devices = [device.physical_device_desc for device in device_lib.list_local_devices() if device.physical_device_desc]

    message = """
---- measure latency result ----
total number of execution (number of samples): {}
network: {}
use gpu by network: {}
image size: {}
devices: {}

* overall (include pre-post-process which execute on cpu)
total time: {:.4f} msec
latency
   mean (SD=standard deviation): {:.4f} (SD={:.4f}) msec, min: {:.4f} msec, max: {:.4f} msec
FPS
   mean (SD=standard deviation): {:.4f} (SD={:.4f}), min: {:.4f}, max: {:.4f}

* network only (exclude pre-post-process):
total time: {:.4f} msec
latency
   mean (SD=standard deviation): {:.4f} (SD={:.4f}) msec, min: {:.4f} msec, max: {:.4f} msec
FPS
   mean (SD=standard deviation): {:.4f} (SD={:.4f}), min: {:.4f}, max: {:.4f}
---- measure latency result ----
""".format(step_size,
           config.NETWORK_CLASS.__name__,
           not cpu,
           config.IMAGE_SIZE,
           devices,
           # overall
           np.sum(overall_times) * 1000,
           # latency
           np.mean(overall_times) * 1000,
           np.std(overall_times) * 1000,
           np.min(overall_times) * 1000,
           np.max(overall_times) * 1000,
           # FPS
           np.mean(1/overall_times),
           np.std(1/overall_times),
           np.min(1/overall_times),
           np.max(1/overall_times),
           # network only
           np.sum(only_network_times) * 1000,
           # latency
           np.mean(only_network_times) * 1000,
           np.std(only_network_times) * 1000,
           np.min(only_network_times) * 1000,
           np.max(only_network_times) * 1000,
           # FPS
           np.mean(1/only_network_times),
           np.std(1/only_network_times),
           np.min(1/only_network_times),
           np.max(1/only_network_times),)

    print(message)


def run(config_file, experiment_id, restore_path, image_size, step_size, cpu):

    if config_file is None and experiment_id is None:
        raise Exception("config_file or experiment_id are required")
    if not cpu:
        print("use gpu")
        _run(config_file, experiment_id, restore_path, image_size, step_size, cpu)
    else:
        print("use cpu")
        with tf.device("/cpu:0"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            _run(config_file, experiment_id, restore_path, image_size, step_size, cpu)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-c",
    "--config_file",
    help="""config file path.
    When experiment_id is provided, The config override saved experiment config. When experiment_id is provided and the config is not provided, restore from saved experiment config.
    """, # NOQA
)
@click.option(
    "-i",
    "--experiment_id",
    help="id of this experiment.",
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001.",
    default=None,
)
@click.option(
    '--image_size',
    nargs=2,
    type=click.Tuple([int, int]),
    help="input image size height and width. if it is not provided, it restore from saved experiment config.",
    default=(None, None),
)
@click.option(
    "-n",
    "--step_size",
    help="number of execution (number of samples). default is 100.",
    default=100,
)
@click.option(
    "--cpu",
    help="flag use only cpu",
    is_flag=True
)
def main(config_file, experiment_id, restore_path, image_size, step_size, cpu):
    """Measure the average latency of certain model's prediction at runtime.

    The latency is averaged over number of repeated executions -- by default is to run it 100 times.
    Each execution is measured after tensorflow is already initialized and both model and images are loaded.
    Batch size is always 1.

    Measure two types latency,
    First is `overall` (including pre-post-processing which is being executed on CPU), Second is `network-only` (model inference, excluding pre-post-processing).
    """ # NOQA
    run(config_file, experiment_id, restore_path, image_size, step_size, cpu)


if __name__ == '__main__':
    main()
