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
import imghdr
import math
import os
from glob import glob

import numpy as np
import tensorflow as tf

from blueoil import environment
from blueoil.utils.image import load_image
from blueoil.utils import config as config_util
from blueoil.utils.executor import search_restore_filename
from blueoil.utils.predict_output.writer import OutputWriter

DUMMY_FILENAME = "DUMMY_FILE"


def _get_images(filenames, pre_processor, data_format):
    """ """
    images = []
    raw_images = []

    for filename in filenames:
        if filename == DUMMY_FILENAME:
            raw_image = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            raw_image = load_image(filename)

        image = pre_processor(image=raw_image)['image']
        if data_format == 'NCHW':
            image = np.transpose(image, [2, 0, 1])

        images.append(image)
        raw_images.append(raw_image)

    return np.array(images), np.array(raw_images)


def _all_image_files(directory):
    all_image_files = []
    for file_path in glob(os.path.join(directory, "*")):
        if os.path.isfile(file_path) and imghdr.what(file_path) in {"jpeg", "png"}:
            all_image_files.append(os.path.abspath(file_path))

    return all_image_files


def _run(input_dir, output_dir, config, restore_path, save_images):
    ModelClass = config.NETWORK_CLASS
    network_kwargs = dict((key.lower(), val) for key, val in config.NETWORK.items())

    graph = tf.Graph()
    with graph.as_default():
        model = ModelClass(
            classes=config.CLASSES,
            is_debug=config.IS_DEBUG,
            **network_kwargs
        )

        is_training = tf.constant(False, name="is_training")

        images_placeholder, _ = model.placeholders()
        output_op = model.inference(images_placeholder, is_training)

        init_op = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver(max_to_keep=None)

    session_config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run(init_op)
    saver.restore(sess, restore_path)

    all_image_files = _all_image_files(input_dir)

    step_size = int(math.ceil(len(all_image_files) / config.BATCH_SIZE))

    writer = OutputWriter(
        task=config.TASK,
        classes=config.CLASSES,
        image_size=config.IMAGE_SIZE,
        data_format=config.DATA_FORMAT
    )

    results = []
    for step in range(step_size):
        start_index = (step) * config.BATCH_SIZE
        end_index = (step + 1) * config.BATCH_SIZE

        image_files = all_image_files[start_index:end_index]

        while len(image_files) != config.BATCH_SIZE:
            # add dummy image.
            image_files.append(DUMMY_FILENAME)

        images, raw_images = _get_images(
            image_files, config.DATASET.PRE_PROCESSOR, config.DATA_FORMAT)

        feed_dict = {images_placeholder: images}
        outputs = sess.run(output_op, feed_dict=feed_dict)

        if config.POST_PROCESSOR:
            outputs = config.POST_PROCESSOR(outputs=outputs)["outputs"]

        results.append(outputs)

        writer.write(
            output_dir,
            outputs,
            raw_images,
            image_files,
            step,
            save_material=save_images
        )

    return results


def run(input_dir, output_dir, experiment_id, config_file, restore_path, save_images):
    environment.init(experiment_id)
    config = config_util.load_from_experiment()
    if config_file:
        config = config_util.merge(config, config_util.load(config_file))

    if not os.path.isdir(input_dir):
        raise Exception("Input directory {} does not exist.".format(input_dir))

    if restore_path is None:
        restore_file = search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

    print("Restore from {}".format(restore_path))

    if not os.path.exists("{}.index".format(restore_path)):
        raise Exception("restore file {} dont exists.".format(restore_path))

    print("---- start predict ----")

    _run(input_dir, output_dir, config, restore_path, save_images)

    print("---- end predict ----")


def predict(input_dir, output_dir, experiment_id, config_file=None, checkpoint=None, save_images=True):
    """Make predictions from input dir images by using trained model.
        Save the predictions npy, json, images results to output dir.
        npy: `{output_dir}/npy/{batch number}.npy`
        json: `{output_dir}/json/{batch number}.json`
        images: `{output_dir}/images/{some type}/{input image file name}`
    """
    restore_path = None
    if checkpoint:
        saved_dir = os.environ.get("OUTPUT_DIR", "saved")
        restore_path = os.path.join(saved_dir, experiment_id, "checkpoints", checkpoint)

    run(input_dir, output_dir, experiment_id, config_file, restore_path, save_images)
