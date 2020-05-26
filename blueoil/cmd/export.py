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
import shutil

import click
import PIL
import numpy as np
import tensorflow as tf

from blueoil import environment
from blueoil.utils.image import load_image
from blueoil.utils import config as config_util
from blueoil.utils import executor

DEFAULT_INFERENCE_TEST_DATA_IMAGE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "fixtures", "export_inference_test_data_images",
    "5605039097_05baa93bfd_m.jpg")


# TODO(wakisaka): duplicated function with blueoil/cmd/measure_latency.py
def _pre_process(raw_image, pre_processor, data_format):
    image = pre_processor(image=raw_image)['image']
    if data_format == 'NCHW':
        image = np.transpose(image, [2, 0, 1])
    return image


def _save_all_operation_outputs(image_path, output_dir, image, raw_image, all_outputs, image_size):
    shutil.copy(image_path, os.path.join(output_dir))
    tmp_image = PIL.Image.open(image_path)
    tmp_image.save(os.path.join(output_dir, "raw_image.png"))
    np.save(os.path.join(output_dir, "raw_image.npy"), raw_image)

    np.save(os.path.join(output_dir, "preprocessed_image.npy"), image)

    for _output in all_outputs:
        np.save(os.path.join(output_dir, "{}.npy".format(_output['name'])), _output['val'])


def _minimal_operations(sess):
    """Get inference operations."""
    minimal_graph_def = executor.convert_variables_to_constants(sess)
    minimal_graph = tf.Graph()
    with minimal_graph.as_default():
        tf.import_graph_def(minimal_graph_def, name="")
    ops = minimal_graph.get_operations()

    return ops


def _export(config, restore_path, image_path):
    if restore_path is None:
        restore_file = executor.search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

    print("Restore from {}".format(restore_path))

    if not os.path.exists("{}.index".format(restore_path)):
        raise Exception("restore file {} dont exists.".format(restore_path))

    output_root_dir = os.path.join(environment.EXPERIMENT_DIR, "export")
    output_root_dir = os.path.join(output_root_dir, os.path.basename(restore_path))

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

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

        images_placeholder, _ = model.placeholders()
        model.inference(images_placeholder, is_training)
        init_op = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver(max_to_keep=50)

    session_config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run(init_op)

    saver.restore(sess, restore_path)

    main_output_dir = os.path.join(output_root_dir, "{}x{}".format(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    # save inference values as npy files for runtime inference test and debug.
    if image_path:
        all_ops = _minimal_operations(sess)
        inference_values_output_dir = os.path.join(main_output_dir, "inference_test_data")

        if not os.path.exists(inference_values_output_dir):
            os.makedirs(inference_values_output_dir)

        raw_image = load_image(image_path)
        image = _pre_process(raw_image, config.PRE_PROCESSOR, config.DATA_FORMAT)
        images = np.expand_dims(image, axis=0)
        feed_dict = {
            images_placeholder: images,
        }

        all_outputs = []
        index = 0
        for op in all_ops:
            for op_output in op.outputs:
                # HACK: This is for TensorFlow bug workaround.
                # We can remove following 4 lines once it's been resolved in TensorFlow
                # Issue link: https://github.com/tensorflow/tensorflow/issues/36456
                if (not tf.config.experimental.list_physical_devices('GPU')
                        and "FusedBatchNormV3" in op_output.name
                        and int(op_output.name.split(":")[1]) in set(range(1, 6))):
                    continue
                val = sess.run(op_output.name, feed_dict=feed_dict)
                name = '%03d' % index + '_' + op_output.name.replace('/', '_')
                all_outputs.append({'val': val, 'name': name})
                index += 1

        _save_all_operation_outputs(
            image_path, inference_values_output_dir, image, raw_image, all_outputs, config.IMAGE_SIZE)

    yaml_names = config_util.save_yaml(main_output_dir, config)
    pb_name = executor.save_pb_file(sess, main_output_dir)

    message = """
Create pb and yaml files in: {}
pb: {}
yaml: {}, {}
""".format(main_output_dir,
           pb_name,
           *yaml_names)

    if image_path:
        message += "Create npy files in under `inference_test_data` folder \n"
        message += "npy: {}".format([d["name"] for d in all_outputs] + ["raw_image", "preprocessed_image", ])

    print(message)
    print("finish")

    return main_output_dir


def run(experiment_id,
        restore_path=None,
        image_size=(None, None),
        image=DEFAULT_INFERENCE_TEST_DATA_IMAGE,
        config_file=None):
    environment.init(experiment_id)

    config = config_util.load_from_experiment()

    if config_file:
        config = config_util.merge(config, config_util.load(config_file))

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

    return _export(config, restore_path, image)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--experiment_id",
    help="id of this experiment.",
    required=True,
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001",
    default=None,
)
@click.option(
    '--image_size',
    nargs=2,
    type=click.Tuple([int, int]),
    help="input image size height and width. if it is not provided, it restore from saved experiment config. "
         "e.g. --image_size 320 320",
    # NOQA
    default=(None, None),
)
@click.option(
    "--image",
    help="path of target image",
    default=DEFAULT_INFERENCE_TEST_DATA_IMAGE,
)
@click.option(
    "-c",
    "--config_file",
    help="config file path. override saved experiment config.",
)
def main(experiment_id, restore_path, image_size, image, config_file):
    """Exporting a trained model to proto buffer files and meta config yaml.

    In the case with `image` option, create each layer output value npy files into
    `export/{restore_path}/{image_size}/inference_test_data/**.npy` as expected value for inference test and debug.
    """
    run(experiment_id, restore_path, image_size, image, config_file)


if __name__ == '__main__':
    main()
