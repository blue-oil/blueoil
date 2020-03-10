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
import json
import logging
import math
import os

import click
import tensorflow as tf

from blueoil import environment
from blueoil.datasets.base import ObjectDetectionBase
from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.tfds import TFDSClassification, TFDSObjectDetection
from blueoil.utils import config as config_util
from blueoil.utils import executor, module_loader
from blueoil.utils.predict_output.writer import save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataset(config, subset, seed):
    DatasetClass = config.DATASET_CLASS
    dataset_kwargs = {key.lower(): val for key, val in config.DATASET.items()}

    # If there is a settings for TFDS, TFDS dataset class will be used.
    tfds_kwargs = dataset_kwargs.pop("tfds_kwargs", {})
    if tfds_kwargs:
        if issubclass(DatasetClass, ObjectDetectionBase):
            DatasetClass = TFDSObjectDetection
        else:
            DatasetClass = TFDSClassification

    dataset = DatasetClass(subset=subset, **dataset_kwargs, **tfds_kwargs)
    enable_prefetch = dataset_kwargs.pop("enable_prefetch", False)
    return DatasetIterator(dataset, seed=seed, enable_prefetch=enable_prefetch)


def evaluate(config, restore_path, output_dir):
    if restore_path is None:
        restore_file = executor.search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

    if not os.path.exists("{}.index".format(restore_path)):
        raise Exception("restore file {} dont exists.".format(restore_path))

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(restore_path)), "evaluate")

    logger.info(f"restore_path:{restore_path}")

    DatasetClass = config.DATASET_CLASS
    ModelClass = config.NETWORK_CLASS
    network_kwargs = {key.lower(): val for key, val in config.NETWORK.items()}

    if "test" in DatasetClass.available_subsets:
        subset = "test"
    else:
        subset = "validation"

    validation_dataset = setup_dataset(config, subset, seed=0)

    graph = tf.Graph()
    with graph.as_default():

        if ModelClass.__module__.startswith("blueoil.networks.object_detection"):
            model = ModelClass(
                classes=validation_dataset.classes,
                num_max_boxes=validation_dataset.num_max_boxes,
                is_debug=config.IS_DEBUG,
                **network_kwargs,
            )

        else:
            model = ModelClass(
                classes=validation_dataset.classes,
                is_debug=config.IS_DEBUG,
                **network_kwargs,
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        is_training = tf.constant(False, name="is_training")

        images_placeholder, labels_placeholder = model.placeholders()

        output = model.inference(images_placeholder, is_training)

        metrics_ops_dict, metrics_update_op = model.metrics(output, labels_placeholder)
        model.summary(output, labels_placeholder)

        summary_op = tf.compat.v1.summary.merge_all()

        metrics_summary_op, metrics_placeholders = executor.prepare_metrics(metrics_ops_dict)

        init_op = tf.compat.v1.global_variables_initializer()
        reset_metrics_op = tf.compat.v1.local_variables_initializer()
        saver = tf.compat.v1.train.Saver(max_to_keep=None)

    session_config = None  # tf.ConfigProto(log_device_placement=True)
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run([init_op, reset_metrics_op])

    validation_writer = tf.compat.v1.summary.FileWriter(environment.TENSORBOARD_DIR + "/evaluate")

    saver.restore(sess, restore_path)

    last_step = sess.run(global_step)

    # init metrics values
    test_step_size = int(math.ceil(validation_dataset.num_per_epoch / config.BATCH_SIZE))
    logger.info(f"test_step_size{test_step_size}")

    for test_step in range(test_step_size):
        logger.info(f"test_step{test_step}")

        images, labels = validation_dataset.feed()
        feed_dict = {
            images_placeholder: images,
            labels_placeholder: labels,
        }

        # Summarize at only last step.
        if test_step == test_step_size - 1:
            summary, _ = sess.run([summary_op, metrics_update_op], feed_dict=feed_dict)
            validation_writer.add_summary(summary, last_step)
        else:
            sess.run([metrics_update_op], feed_dict=feed_dict)

    metrics_values = sess.run(list(metrics_ops_dict.values()))
    metrics_feed_dict = {
        # TODO: Fix to avoid the implementation depended on the order of dict implicitly
        placeholder: value for placeholder, value in zip(metrics_placeholders, metrics_values)
    }
    metrics_summary, = sess.run(
        [metrics_summary_op], feed_dict=metrics_feed_dict,
    )
    validation_writer.add_summary(metrics_summary, last_step)

    is_tfds = "TFDS_KWARGS" in config.DATASET
    dataset_name = config.DATASET.TFDS_KWARGS["name"] if is_tfds else config.DATASET_CLASS.__name__
    dataset_path = config.DATASET.TFDS_KWARGS["data_dir"] if is_tfds else ""

    metrics_dict = {
        'task_type': config.TASK.value,
        'network_name': config.NETWORK_CLASS.__name__,
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'last_step': int(last_step),
        # TODO: Fix to avoid the implementation depended on the order of dict implicitly
        'metrics': {k: float(v) for k, v in zip(list(metrics_ops_dict.keys()), metrics_values)},
    }
    save_json(output_dir, json.dumps(metrics_dict, indent=4,), metrics_dict["last_step"])
    validation_dataset.close()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-i",
    "--experiment_id",
    help="id of this experiment",
    required=True,
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001",
)
@click.option(
    "-n",
    "--network",
    help="network name. override config.NETWORK_CLASS",
)
@click.option(
    "-d",
    "--dataset",
    help="dataset name. override config.DATASET_CLASS",
)
@click.option(
    "-c",
    "--config_file",
    help="config file path. override(merge) saved experiment config. if it is not provided, it restore from saved experiment config.", # NOQA
)
@click.option(
    "-o",
    "--output_dir",
    help="Output directory to save a evaluated result",
)
def main(network, dataset, config_file, experiment_id, restore_path, output_dir):
    environment.init(experiment_id)

    config = config_util.load_from_experiment()

    if config_file:
        config = config_util.merge(config, config_util.load(config_file))

    if network:
        network_class = module_loader.load_network_class(network)
        config.NETWORK_CLASS = network_class
    if dataset:
        dataset_class = module_loader.load_dataset_class(dataset)
        config.DATASET_CLASS = dataset_class

    executor.init_logging(config)
    config_util.display(config)

    evaluate(config, restore_path, output_dir)


if __name__ == "__main__":
    main()
