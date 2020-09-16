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
from datetime import datetime
import math
import os
import sys

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.keras.utils import Progbar
import yaml

from blueoil import environment
from blueoil.common import Tasks
from blueoil.datasets.base import ObjectDetectionBase, SegmentationBase
from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.tfds import TFDSClassification, TFDSObjectDetection, TFDSSegmentation
from blueoil.utils import config as config_util
from blueoil.utils import executor
from blueoil.utils import horovod as horovod_util


def _save_checkpoint(saver, sess, global_step):
    checkpoint_file = "save.ckpt"
    saver.save(
        sess,
        os.path.join(environment.CHECKPOINTS_DIR, checkpoint_file),
        global_step=global_step,
    )


def setup_dataset(config, subset, rank, local_rank):
    DatasetClass = config.DATASET_CLASS
    dataset_kwargs = {key.lower(): val for key, val in config.DATASET.items()}

    # If there is a settings for TFDS, TFDS dataset class will be used.
    tfds_kwargs = dataset_kwargs.pop("tfds_kwargs", {})
    if tfds_kwargs:
        if issubclass(DatasetClass, ObjectDetectionBase):
            DatasetClass = TFDSObjectDetection
        elif issubclass(DatasetClass, SegmentationBase):
            DatasetClass = TFDSSegmentation
        else:
            DatasetClass = TFDSClassification

    dataset = DatasetClass(subset=subset, **dataset_kwargs, **tfds_kwargs)
    enable_prefetch = dataset_kwargs.pop("enable_prefetch", False)
    return DatasetIterator(dataset, seed=rank, enable_prefetch=enable_prefetch, local_rank=local_rank)


def start_training(config, profile_step):
    use_horovod = horovod_util.is_enabled()
    print("use_horovod:", use_horovod)
    if use_horovod:
        hvd = horovod_util.setup()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
    else:
        rank = 0
        local_rank = -1

    ModelClass = config.NETWORK_CLASS
    network_kwargs = {key.lower(): val for key, val in config.NETWORK.items()}

    train_dataset = setup_dataset(config, "train", rank, local_rank)
    print("train dataset num:", train_dataset.num_per_epoch)

    validation_dataset = setup_dataset(config, "validation", rank, local_rank)
    print("validation dataset num:", validation_dataset.num_per_epoch)

    graph = tf.Graph()
    with graph.as_default():
        if config.TASK == Tasks.OBJECT_DETECTION:
            model = ModelClass(
                classes=train_dataset.classes,
                num_max_boxes=train_dataset.num_max_boxes,
                is_debug=config.IS_DEBUG,
                **network_kwargs,
            )
        else:
            model = ModelClass(
                classes=train_dataset.classes,
                is_debug=config.IS_DEBUG,
                **network_kwargs,
            )

        is_training_placeholder = tf.compat.v1.placeholder(tf.bool, name="is_training_placeholder")

        images_placeholder, labels_placeholder = model.placeholders()

        output = model.inference(images_placeholder, is_training_placeholder)
        loss = model.loss(output, labels_placeholder)
        opt = model.optimizer()
        if use_horovod:
            # add Horovod Distributed Optimizer
            opt = hvd.DistributedOptimizer(opt)
        train_op = model.train(loss, opt)
        metrics_ops_dict, metrics_update_op = model.metrics(output, labels_placeholder)
        # TODO(wakisaka): Deal with many networks.
        model.summary(output, labels_placeholder)

        summary_op = tf.compat.v1.summary.merge_all()
        metrics_summary_op = executor.metrics_summary_op(metrics_ops_dict)

        init_op = tf.compat.v1.global_variables_initializer()
        reset_metrics_op = tf.compat.v1.local_variables_initializer()
        if use_horovod:
            # add Horovod broadcasting variables from rank 0 to all
            bcast_global_variables_op = hvd.broadcast_global_variables(0)

        saver = tf.compat.v1.train.Saver(max_to_keep=config.KEEP_CHECKPOINT_MAX)

        with open(os.path.join(environment.EXPERIMENT_DIR, "pretrain_vars.txt"), 'w') as pretrain_vars_file:
            train_vars = tf.compat.v1.trainable_variables()
            pretrain_vars_file.writelines("[\n")
            pretrain_vars_file.writelines("    '%s',\n" % var.name for var in train_vars)
            pretrain_vars_file.writelines("]\n")

        if config.IS_PRETRAIN:
            all_vars = tf.compat.v1.global_variables()
            pretrain_var_list = [
                var for var in all_vars if var.name.startswith(tuple(config.PRETRAIN_VARS))
            ]
            print("pretrain_vars", [
                var.name for var in pretrain_var_list
            ])
            pretrain_saver = tf.compat.v1.train.Saver(pretrain_var_list, name="pretrain_saver")

    if use_horovod:
        # For distributed training
        session_config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                visible_device_list=str(hvd.local_rank())
            )
        )
    else:
        # TODO(wakisaka): For debug.
        # session_config = tf.ConfigProto(
        #     gpu_options=tf.GPUOptions(
        #         allow_growth=True,
        #         per_process_gpu_memory_fraction=0.1
        #     )
        # )
        session_config = tf.compat.v1.ConfigProto()  # tf.ConfigProto(log_device_placement=True)
    # TODO(wakisaka): XLA JIT
    # session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run([init_op, reset_metrics_op])
    executor.save_pb_file(sess, environment.CHECKPOINTS_DIR)

    if rank == 0:
        train_writer = tf.compat.v1.summary.FileWriter(environment.TENSORBOARD_DIR + "/train", sess.graph)
        val_writer = tf.compat.v1.summary.FileWriter(environment.TENSORBOARD_DIR + "/validation")

        if config.IS_PRETRAIN:
            print("------- Load pretrain data ----------")
            pretrain_saver.restore(sess, os.path.join(config.PRETRAIN_DIR, config.PRETRAIN_FILE))

        # for recovery
        ckpt = tf.train.get_checkpoint_state(environment.CHECKPOINTS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("--------- Restore last checkpoint -------------")
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.recover_last_checkpoints(ckpt.model_checkpoint_path)
            last_step = sess.run(model.global_step)
            # TODO(wakisaka): tensorflow v1.3 remain previous event log in tensorboard.
            # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/supervisor.py#L1072
            train_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=last_step + 1)
            val_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=last_step + 1)
            print("recovered. last step", last_step)

    if use_horovod:
        # broadcast variables from rank 0 to all other processes
        sess.run(bcast_global_variables_op)

    last_step = sess.run(model.global_step)

    # Calculate max steps. The priority of config.MAX_EPOCHS is higher than config.MAX_STEPS.
    if "MAX_EPOCHS" in config:
        max_steps = int(train_dataset.num_per_epoch / config.BATCH_SIZE * config.MAX_EPOCHS)
        if max_steps < 1:
            print("The max_steps is less than 1, consider reduce BATCH_SIZE. exit.", file=sys.stderr)
            sys.exit(1)
    else:
        max_steps = config.MAX_STEPS
        if max_steps < 1:
            print("The max_steps is less than 1, consider set MAX_STEPS greater than 0. exit.", file=sys.stderr)
            sys.exit(1)

    progbar = Progbar(max_steps)
    if rank == 0:
        progbar.update(last_step)
    for step in range(last_step, max_steps):

        images, labels = train_dataset.feed()

        feed_dict = {
            is_training_placeholder: True,
            images_placeholder: images,
            labels_placeholder: labels,
        }

        # Runtime statistics for develop.
        if step == profile_step:
            options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_meta = tf.compat.v1.RunMetadata()
        else:
            options = None
            run_meta = None

        if step * ((step + 1) % config.SUMMARISE_STEPS) == 0 and rank == 0:
            sess.run(reset_metrics_op)
            _, summary, _ = sess.run(
                [train_op, summary_op, metrics_update_op], feed_dict=feed_dict,
                options=options,
                run_metadata=run_meta,
            )
            # train_writer.add_run_metadata(run_metadata, "step: {}".format(step + 1))
            train_writer.add_summary(summary, step + 1)

            metrics_summary = sess.run(metrics_summary_op)
            train_writer.add_summary(metrics_summary, step + 1)
            train_writer.flush()
        else:
            sess.run([train_op], feed_dict=feed_dict, options=options, run_metadata=run_meta)

        if step == profile_step:
            executor.profile_train_step(step, sess, run_meta)

        to_be_saved = step == 0 or (step + 1) == max_steps or (step + 1) % config.SAVE_CHECKPOINT_STEPS == 0

        if to_be_saved and rank == 0:
            _save_checkpoint(saver, sess, model.global_step)

        if step == 0 or (step + 1) % config.TEST_STEPS == 0:
            # init metrics values
            sess.run(reset_metrics_op)
            test_step_size = int(math.ceil(validation_dataset.num_per_epoch / config.BATCH_SIZE))

            for test_step in range(test_step_size):

                images, labels = validation_dataset.feed()
                feed_dict = {
                    is_training_placeholder: False,
                    images_placeholder: images,
                    labels_placeholder: labels,
                }

                if test_step % config.SUMMARISE_STEPS == 0:
                    summary, _ = sess.run([summary_op, metrics_update_op], feed_dict=feed_dict)
                    if rank == 0:
                        val_writer.add_summary(summary, step + 1)
                        val_writer.flush()
                else:
                    sess.run([metrics_update_op], feed_dict=feed_dict)

            metrics_summary = sess.run(metrics_summary_op)
            if rank == 0:
                val_writer.add_summary(metrics_summary, step + 1)
                val_writer.flush()

        if rank == 0:
            progbar.update(step + 1)
    # training loop end.
    train_dataset.close()
    validation_dataset.close()
    print("Done")


def run(config_file, experiment_id, recreate, profile_step):
    environment.init(experiment_id)
    config = config_util.load(config_file)

    if horovod_util.is_enabled():
        horovod_util.setup()

    if horovod_util.is_rank0():
        config_util.display(config)
        executor.init_logging(config)

        executor.prepare_dirs(recreate)
        config_util.copy_to_experiment_dir(config_file)
        config_util.save_yaml(environment.EXPERIMENT_DIR, config)

    start_training(config, profile_step)


def train(config_file, experiment_id=None, recreate=False, profile_step=-1):
    if not experiment_id:
        # Default model_name will be taken from config file: {model_name}.yml.
        model_name = os.path.splitext(os.path.basename(config_file))[0]
        experiment_id = '{}_{:%Y%m%d%H%M%S}'.format(model_name, datetime.now())

    run(config_file, experiment_id, recreate, profile_step)

    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    experiment_dir = os.path.join(output_dir, experiment_id)
    checkpoint = os.path.join(experiment_dir, 'checkpoints', 'checkpoint')

    if not tf.io.gfile.exists(checkpoint):
        raise Exception('Checkpoints are not created in {}'.format(experiment_dir))

    with tf.io.gfile.GFile(checkpoint) as stream:
        data = yaml.load(stream, Loader=yaml.Loader)
    checkpoint_name = os.path.basename(data['model_checkpoint_path'])

    return experiment_id, checkpoint_name
