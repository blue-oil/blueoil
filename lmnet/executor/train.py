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
import math
import os

import click
import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.keras.utils import Progbar

from lmnet import environment
from lmnet.common import Tasks
from lmnet.datasets.base import ObjectDetectionBase
from lmnet.datasets.dataset_iterator import DatasetIterator
from lmnet.datasets.tfds import TFDSClassification, TFDSObjectDetection
from lmnet.utils import config as config_util
from lmnet.utils import executor
from lmnet.utils import horovod as horovod_util
from lmnet.utils import module_loader


def _save_checkpoint(saver, sess, global_step, step):
    checkpoint_file = "save.ckpt"
    saver.save(
        sess,
        os.path.join(environment.CHECKPOINTS_DIR, checkpoint_file),
        global_step=global_step,
    )


def setup_dataset(config, subset, rank):
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
    return DatasetIterator(dataset, seed=rank, enable_prefetch=enable_prefetch)


def start_training(config):
    use_horovod = horovod_util.is_enabled()
    print("use_horovod:", use_horovod)
    if use_horovod:
        hvd = horovod_util.setup()
        rank = hvd.rank()
    else:
        rank = 0

    ModelClass = config.NETWORK_CLASS
    network_kwargs = {key.lower(): val for key, val in config.NETWORK.items()}
    if "train_validation_saving_size".upper() in config.DATASET.keys():
        use_train_validation_saving = config.DATASET.TRAIN_VALIDATION_SAVING_SIZE > 0
    else:
        use_train_validation_saving = False

    if use_train_validation_saving:
        top_train_validation_saving_set_accuracy = 0

    train_dataset = setup_dataset(config, "train", rank)
    print("train dataset num:", train_dataset.num_per_epoch)

    if use_train_validation_saving:
        train_validation_saving_dataset = setup_dataset(config, "train_validation_saving", rank)
        print("train_validation_saving dataset num:", train_validation_saving_dataset.num_per_epoch)

    validation_dataset = setup_dataset(config, "validation", rank)
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

        global_step = tf.Variable(0, name="global_step", trainable=False)
        is_training_placeholder = tf.compat.v1.placeholder(tf.bool, name="is_training_placeholder")

        images_placeholder, labels_placeholder = model.placeholders()

        output = model.inference(images_placeholder, is_training_placeholder)
        if config.TASK == Tasks.OBJECT_DETECTION:
            loss = model.loss(output, labels_placeholder, global_step)
        else:
            loss = model.loss(output, labels_placeholder)
        opt = model.optimizer(global_step)
        if use_horovod:
            # add Horovod Distributed Optimizer
            opt = hvd.DistributedOptimizer(opt)
        train_op = model.train(loss, opt, global_step)
        metrics_ops_dict, metrics_update_op = model.metrics(output, labels_placeholder)
        # TODO(wakisaka): Deal with many networks.
        model.summary(output, labels_placeholder)

        summary_op = tf.compat.v1.summary.merge_all()

        metrics_summary_op, metrics_placeholders = executor.prepare_metrics(metrics_ops_dict)

        init_op = tf.global_variables_initializer()
        reset_metrics_op = tf.local_variables_initializer()
        if use_horovod:
            # add Horovod broadcasting variables from rank 0 to all
            bcast_global_variables_op = hvd.broadcast_global_variables(0)

        if use_train_validation_saving:
            saver = tf.compat.v1.train.Saver(max_to_keep=1)
        else:
            saver = tf.compat.v1.train.Saver(max_to_keep=config.KEEP_CHECKPOINT_MAX)

        if config.IS_PRETRAIN:
            all_vars = tf.global_variables()
            pretrain_var_list = [
                var for var in all_vars if var.name.startswith(tuple(config.PRETRAIN_VARS))
            ]
            print("pretrain_vars", [
                var.name for var in pretrain_var_list
            ])
            pretrain_saver = tf.compat.v1.train.Saver(pretrain_var_list, name="pretrain_saver")

    if use_horovod:
        # For distributed training
        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
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
        session_config = tf.ConfigProto()  # tf.ConfigProto(log_device_placement=True)
    # TODO(wakisaka): XLA JIT
    # session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    sess = tf.Session(graph=graph, config=session_config)
    sess.run([init_op, reset_metrics_op])

    if rank == 0:
        train_writer = tf.summary.FileWriter(environment.TENSORBOARD_DIR + "/train", sess.graph)
        if use_train_validation_saving:
            train_val_saving_writer = tf.summary.FileWriter(environment.TENSORBOARD_DIR + "/train_validation_saving")
        val_writer = tf.summary.FileWriter(environment.TENSORBOARD_DIR + "/validation")

        if config.IS_PRETRAIN:
            print("------- Load pretrain data ----------")
            pretrain_saver.restore(sess, os.path.join(config.PRETRAIN_DIR, config.PRETRAIN_FILE))
            sess.run(tf.assign(global_step, 0))

        last_step = 0

        # for recovery
        ckpt = tf.train.get_checkpoint_state(environment.CHECKPOINTS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("--------- Restore last checkpoint -------------")
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.recover_last_checkpoints(ckpt.model_checkpoint_path)
            last_step = sess.run(global_step)
            # TODO(wakisaka): tensorflow v1.3 remain previous event log in tensorboard.
            # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/supervisor.py#L1072
            train_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=last_step + 1)
            val_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=last_step + 1)
            print("recovered. last step", last_step)

    if use_horovod:
        # broadcast variables from rank 0 to all other processes
        sess.run(bcast_global_variables_op)

    last_step = sess.run(global_step)

    # Calculate max steps. The priority of config.MAX_EPOCHS is higher than config.MAX_STEPS.
    if "MAX_EPOCHS" in config:
        max_steps = int(train_dataset.num_per_epoch / config.BATCH_SIZE * config.MAX_EPOCHS)
    else:
        max_steps = config.MAX_STEPS

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

        if step * ((step + 1) % config.SUMMARISE_STEPS) == 0 and rank == 0:
            # Runtime statistics for develop.
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            sess.run(reset_metrics_op)
            _, summary, _ = sess.run(
                [train_op, summary_op, metrics_update_op], feed_dict=feed_dict,
                # options=run_options,
                # run_metadata=run_metadata,
            )
            # train_writer.add_run_metadata(run_metadata, "step: {}".format(step + 1))
            train_writer.add_summary(summary, step + 1)

            metrics_values = sess.run(list(metrics_ops_dict.values()))
            metrics_feed_dict = {placeholder: value for placeholder, value in zip(metrics_placeholders, metrics_values)}

            metrics_summary, = sess.run(
                [metrics_summary_op], feed_dict=metrics_feed_dict,
            )
            train_writer.add_summary(metrics_summary, step + 1)
            train_writer.flush()
        else:
            sess.run([train_op], feed_dict=feed_dict)

        to_be_saved = step == 0 or (step + 1) == max_steps or (step + 1) % config.SAVE_CHECKPOINT_STEPS == 0

        if to_be_saved and rank == 0:
            if use_train_validation_saving:

                sess.run(reset_metrics_op)
                train_validation_saving_step_size = int(math.ceil(train_validation_saving_dataset.num_per_epoch
                                                                  / config.BATCH_SIZE))
                print("train_validation_saving_step_size", train_validation_saving_step_size)

                current_train_validation_saving_set_accuracy = 0

                for train_validation_saving_step in range(train_validation_saving_step_size):
                    print("train_validation_saving_step", train_validation_saving_step)

                    images, labels = train_validation_saving_dataset.feed()
                    feed_dict = {
                        is_training_placeholder: False,
                        images_placeholder: images,
                        labels_placeholder: labels,
                    }

                    if train_validation_saving_step % config.SUMMARISE_STEPS == 0:
                        summary, _ = sess.run([summary_op, metrics_update_op], feed_dict=feed_dict)
                        train_val_saving_writer.add_summary(summary, step + 1)
                        train_val_saving_writer.flush()
                    else:
                        sess.run([metrics_update_op], feed_dict=feed_dict)

                metrics_values = sess.run(list(metrics_ops_dict.values()))
                metrics_feed_dict = {
                    placeholder: value for placeholder, value in zip(metrics_placeholders, metrics_values)
                }
                metrics_summary, = sess.run(
                    [metrics_summary_op], feed_dict=metrics_feed_dict,
                )
                train_val_saving_writer.add_summary(metrics_summary, step + 1)
                train_val_saving_writer.flush()

                current_train_validation_saving_set_accuracy = sess.run(metrics_ops_dict["accuracy"])

                if current_train_validation_saving_set_accuracy > top_train_validation_saving_set_accuracy:
                    top_train_validation_saving_set_accuracy = current_train_validation_saving_set_accuracy
                    print("New top train_validation_saving accuracy is: ", top_train_validation_saving_set_accuracy)

                    _save_checkpoint(saver, sess, global_step, step)

            else:
                _save_checkpoint(saver, sess, global_step, step)

            if step == 0:
                # check create pb on only first step.
                minimal_graph = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph.as_graph_def(add_shapes=True),
                    ["output"],
                )
                pb_name = "minimal_graph_with_shape_{}.pb".format(step + 1)
                pbtxt_name = "minimal_graph_with_shape_{}.pbtxt".format(step + 1)
                tf.io.write_graph(minimal_graph, environment.CHECKPOINTS_DIR, pb_name, as_text=False)
                tf.io.write_graph(minimal_graph, environment.CHECKPOINTS_DIR, pbtxt_name, as_text=True)

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

            metrics_values = sess.run(list(metrics_ops_dict.values()))
            metrics_feed_dict = {
                placeholder: value for placeholder, value in zip(metrics_placeholders, metrics_values)
            }
            metrics_summary, = sess.run(
                [metrics_summary_op], feed_dict=metrics_feed_dict,
            )
            if rank == 0:
                val_writer.add_summary(metrics_summary, step + 1)
                val_writer.flush()

        if rank == 0:
            progbar.update(step + 1)
    # training loop end.
    train_dataset.close()
    validation_dataset.close()
    if use_train_validation_saving:
        train_validation_saving_dataset.close()
    print("Done")


def run(network, dataset, config_file, experiment_id, recreate):
    environment.init(experiment_id)
    config = config_util.load(config_file)

    if network:
        network_class = module_loader.load_network_class(network)
        config.NETWORK_CLASS = network_class
    if dataset:
        dataset_class = module_loader.load_dataset_class(dataset)
        config.DATASET_CLASS = dataset_class

    if horovod_util.is_enabled():
        horovod_util.setup()

    if horovod_util.is_rank0():
        config_util.display(config)
        executor.init_logging(config)

        executor.prepare_dirs(recreate)
        config_util.copy_to_experiment_dir(config_file)
        config_util.save_yaml(environment.EXPERIMENT_DIR, config)

    start_training(config)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-c",
    "--config_file",
    help="config file path for this training",
    default=os.path.join("configs", "example", "classification.py"),
    required=True,
)
@click.option(
    "-i",
    "--experiment_id",
    help="id of this training",
    default="experiment",
    required=True,
)
@click.option(
    '--recreate',
    is_flag=True,
    help="delete and recreate experiment id dir",
    default=False,
)
@click.option(
    "-n",
    "--network",
    help="network name which you want to use for this training. override config.DATASET_CLASS",
)
@click.option(
    "-d",
    "--dataset",
    help="dataset name which is the source of this training. override config.NETWORK_CLASS",
)
def main(network, dataset, config_file, experiment_id, recreate):
    run(network, dataset, config_file, experiment_id, recreate)


if __name__ == '__main__':
    main()
