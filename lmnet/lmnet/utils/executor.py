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

import tensorflow as tf
from tensorflow import gfile

from lmnet import environment


def init_logging(config):
    """Init tensorflow logging level.

    https://github.com/tensorflow/tensorflow/blob/be52c5c09e39ac2df007fb2d62abe122d5ade6d0/tensorflow/core/platform/default/logging.h#L30
    """
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        if not config.IS_DEBUG:
            # set ERROR log level
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def prepare_dirs(recreate=False):
    """Prepare config dirs

    When recreate is True, if previous execution exists, remove them and recreate.
    When recreate is False, remain previous execution.
    """
    experiment_dir = environment.EXPERIMENT_DIR
    tensorboard_dir = environment.TENSORBOARD_DIR
    checkpoints_dir = environment.CHECKPOINTS_DIR

    if recreate:
        message = """
Delete and recreate these dirs:
experiment_dir: {experiment_dir}
tensorboard_dir: {tensorboard_dir}
checkpoints_dir: {checkpoints_dir}
        """.format(experiment_dir=experiment_dir, tensorboard_dir=tensorboard_dir, checkpoints_dir=checkpoints_dir)
    else:
        message = """
Create these dirs if the dirs dont exist:
experiment_dir: {experiment_dir}
tensorboard_dir: {tensorboard_dir}
checkpoints_dir: {checkpoints_dir}
        """.format(experiment_dir=experiment_dir, tensorboard_dir=tensorboard_dir, checkpoints_dir=checkpoints_dir)

    print(message)

    if recreate:
        if gfile.Exists(experiment_dir):
            gfile.DeleteRecursively(experiment_dir)

        if gfile.Exists(tensorboard_dir):
            gfile.DeleteRecursively(tensorboard_dir)

        if gfile.Exists(checkpoints_dir):
            gfile.DeleteRecursively(checkpoints_dir)

    if not gfile.Exists(experiment_dir):
        gfile.MakeDirs(experiment_dir)

    if not gfile.Exists(tensorboard_dir):
        gfile.MakeDirs(tensorboard_dir)

    if not gfile.Exists(checkpoints_dir):
        gfile.MakeDirs(checkpoints_dir)


def search_restore_filename(checkpoints_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
    if ckpt and ckpt.model_checkpoint_path:
        return os.path.basename(ckpt.model_checkpoint_path)

    raise Exception("no restore file.")


def convert_variables_to_constants(sess, output_node_names=["output"]):
    minimal_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(add_shapes=True),
        output_node_names,
    )

    return minimal_graph_def


def save_pb_file(sess, output_dir, output_node_names=["output"], pb_name="minimal_graph_with_shape.pb"):
    minimal_graph_def = convert_variables_to_constants(sess, output_node_names)
    tf.train.write_graph(minimal_graph_def, output_dir, pb_name, as_text=False)
    return pb_name


def prepare_metrics(metrics_ops_dict):
    """Create summary_op and placeholders for training metrics.

    Params:
        metrics_ops_dict (dict): dict of name and metrics_op.

    Returns:
        metrics_summary_op: summary op of metrics.
        metrics_placeholders: list of metrics placeholder.
    """
    with tf.name_scope("metrics"):
        metrics_placeholders = []
        metrics_summaries = []
        for (metrics_key, metrics_op) in metrics_ops_dict.items():
            metrics_placeholder = tf.placeholder(
                tf.float32, name="{}_placeholder".format(metrics_key)
            )
            summary = tf.summary.scalar(metrics_key, metrics_placeholder)
            metrics_placeholders.append(metrics_placeholder)
            metrics_summaries.append(summary)

        metrics_summary_op = tf.summary.merge(metrics_summaries)

    return metrics_summary_op, metrics_placeholders
