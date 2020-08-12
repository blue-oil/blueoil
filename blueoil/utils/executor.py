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
from tensorflow.io import gfile

from blueoil import environment


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
        if gfile.exists(experiment_dir):
            gfile.rmtree(experiment_dir)

        if gfile.exists(tensorboard_dir):
            gfile.rmtree(tensorboard_dir)

        if gfile.exists(checkpoints_dir):
            gfile.rmtree(checkpoints_dir)

    if not gfile.exists(experiment_dir):
        gfile.makedirs(experiment_dir)

    if not gfile.exists(tensorboard_dir):
        gfile.makedirs(tensorboard_dir)

    if not gfile.exists(checkpoints_dir):
        gfile.makedirs(checkpoints_dir)


def search_restore_filename(checkpoints_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
    if ckpt and ckpt.model_checkpoint_path:
        return os.path.basename(ckpt.model_checkpoint_path)

    raise Exception("no restore file.")


def convert_variables_to_constants(sess, output_node_names=["output"]):
    minimal_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(add_shapes=True),
        output_node_names,
    )

    return minimal_graph_def


def save_pb_file(sess, output_dir, output_node_names=["output"], pb_name="minimal_graph_with_shape.pb"):
    minimal_graph_def = convert_variables_to_constants(sess, output_node_names)
    tf.io.write_graph(minimal_graph_def, output_dir, pb_name, as_text=False)
    return pb_name


def metrics_summary_op(metrics_ops_dict):
    """Create summary_op for training metrics.

    Args:
        metrics_ops_dict (dict): dict of name and metrics_op.

    Returns:
        metrics_summary_op: summary op of metrics.

    """
    with tf.compat.v1.name_scope("metrics"):
        metrics_summaries = [
            tf.compat.v1.summary.scalar(metrics_key, metrics_op)
            for (metrics_key, metrics_op) in metrics_ops_dict.items()
        ]

    return tf.compat.v1.summary.merge(metrics_summaries)


def profile_train_step(step, sess, run_meta):
    profiler = tf.compat.v1.profiler.Profiler(sess.graph)
    profiler.add_step(step, run_meta)
    opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory())
            .with_step(step)
            .select(["bytes"])
            .order_by("bytes")
            .build())
    file_path = os.path.join(environment.EXPERIMENT_DIR, "training_profile_memory")
    opts["output"] = "file:outfile={}".format(file_path)
    profiler.profile_name_scope(options=opts)
    timeline_path = os.path.join(environment.EXPERIMENT_DIR, "training_profile_timeline_step")
    opts["output"] = "timeline:outfile={}".format(timeline_path)
    profiler.profile_name_scope(options=opts)
