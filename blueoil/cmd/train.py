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

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.keras.utils import Progbar
import yaml

from blueoil import environment
from blueoil.common import Tasks
from blueoil.datasets.base import ObjectDetectionBase
from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.tfds import TFDSClassification, TFDSObjectDetection
from blueoil.utils import config as config_util
from blueoil.utils import executor
from blueoil.utils import horovod as horovod_util
from blueoil.utils import module_loader


def _save_checkpoint(saver, sess, global_step, step):
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
        else:
            DatasetClass = TFDSClassification

    dataset = DatasetClass(subset=subset, **dataset_kwargs, **tfds_kwargs)
    enable_prefetch = dataset_kwargs.pop("enable_prefetch", False)
    return DatasetIterator(dataset, seed=rank, enable_prefetch=enable_prefetch, local_rank=local_rank)


class Trainer:
    """
    Trainer class.
    """

    def __init__(self, config):
        "Config stuffs"
        self.config = config
        self.use_horovod = horovod_util.is_enabled()
        print("use_horovod:", self.use_horovod)
        if self.use_horovod:
            self.hvd = horovod_util.setup()
            self.rank = self.hvd.rank()
            self.local_rank = self.hvd.local_rank()
        else:
            self.rank = 0
            self.local_rank = -1

        if "train_validation_saving_size".upper() in self.config.DATASET.keys():
            self.use_train_validation_saving = self.config.DATASET.TRAIN_VALIDATION_SAVING_SIZE > 0
        else:
            self.use_train_validation_saving = False

        if self.use_train_validation_saving:
            self.top_train_validation_saving_set_accuracy = 0

        self.train_dataset = setup_dataset(self.config, "train", self.rank, self.local_rank)
        print("train dataset num:", self.train_dataset.num_per_epoch)

        if self.use_train_validation_saving:
            self.train_validation_saving_dataset = setup_dataset(
                self.config,
                "train_validation_saving",
                self.rank,
                self.local_rank
            )
            print("train_validation_saving dataset num:", self.train_validation_saving_dataset.num_per_epoch)

        self.validation_dataset = setup_dataset(self.config, "validation", self.rank, self.local_rank)
        print("validation dataset num:", self.validation_dataset.num_per_epoch)
        return

    def init_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            ModelClass = self.config.NETWORK_CLASS
            network_kwargs = {key.lower(): val for key, val in self.config.NETWORK.items()}
            if self.config.TASK == Tasks.OBJECT_DETECTION:
                model = ModelClass(
                    classes=self.train_dataset.classes,
                    num_max_boxes=self.train_dataset.num_max_boxes,
                    is_debug=self.config.IS_DEBUG,
                    **network_kwargs,
                )
            else:
                model = ModelClass(
                    classes=self.train_dataset.classes,
                    is_debug=self.config.IS_DEBUG,
                    **network_kwargs,
                )

            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.is_training_placeholder = tf.compat.v1.placeholder(tf.bool, name="is_training_placeholder")

            self.images_placeholder, self.labels_placeholder = model.placeholders()

            output = model.inference(self.images_placeholder, self.is_training_placeholder)
            if self.config.TASK == Tasks.OBJECT_DETECTION:
                loss = model.loss(output, self.labels_placeholder, self.global_step)
            else:
                loss = model.loss(output, self.labels_placeholder)
            opt = model.optimizer(self.global_step)
            if self.use_horovod:
                # add Horovod Distributed Optimizer
                opt = self.hvd.DistributedOptimizer(opt)
            self.train_op = model.train(loss, opt, self.global_step)
            self.metrics_ops_dict, self.metrics_update_op = model.metrics(output, self.labels_placeholder)
            # TODO(wakisaka): Deal with many networks.
            model.summary(output, self.labels_placeholder)

            self.summary_op = tf.compat.v1.summary.merge_all()

            self.metrics_summary_op, self.metrics_placeholders = executor.prepare_metrics(self.metrics_ops_dict)

            init_op = tf.global_variables_initializer()
            self.reset_metrics_op = tf.local_variables_initializer()
            if self.use_horovod:
                # add Horovod broadcasting variables from rank 0 to all
                self.bcast_global_variables_op = self.hvd.broadcast_global_variables(0)

            if self.use_train_validation_saving:
                self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
            else:
                self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.KEEP_CHECKPOINT_MAX)

            if self.config.IS_PRETRAIN:
                all_vars = tf.global_variables()
                pretrain_var_list = [
                    var for var in all_vars if var.name.startswith(tuple(self.config.PRETRAIN_VARS))
                ]
                print("pretrain_vars", [
                    var.name for var in pretrain_var_list
                ])
                self.pretrain_saver = tf.compat.v1.train.Saver(pretrain_var_list, name="pretrain_saver")

        if self.use_horovod:
            # For distributed training
            session_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=str(self.hvd.local_rank())
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

        self.sess = tf.Session(graph=graph, config=session_config)
        self.sess.run([init_op, self.reset_metrics_op])

        return

    def rank_zero(self):
        if self.rank != 0:
            return
        # if rank == 0:
        self.train_writer = tf.summary.FileWriter(environment.TENSORBOARD_DIR + "/train", self.sess.graph)
        if self.use_train_validation_saving:
            self.train_val_saving_writer = tf.summary.FileWriter(
                    environment.TENSORBOARD_DIR + "/train_validation_saving"
            )
        self.val_writer = tf.summary.FileWriter(environment.TENSORBOARD_DIR + "/validation")

        if self.config.IS_PRETRAIN:
            print("------- Load pretrain data ----------")
            self.pretrain_saver.restore(self.sess, os.path.join(self.config.PRETRAIN_DIR, self.config.PRETRAIN_FILE))
            self.sess.run(tf.assign(self.global_step, 0))

        self.last_step = 0

        # for recovery
        ckpt = tf.train.get_checkpoint_state(environment.CHECKPOINTS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("--------- Restore last checkpoint -------------")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            # saver.recover_last_checkpoints(ckpt.model_checkpoint_path)
            self.last_step = self.sess.run(self.global_step)
            # TODO(wakisaka): tensorflow v1.3 remain previous event log in tensorboard.
            # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/supervisor.py#L1072
            self.train_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=self.last_step + 1)
            self.val_writer.add_session_log(SessionLog(status=SessionLog.START), global_step=self.last_step + 1)
            print("recovered. last step", self.last_step)

    def run(self):
        if self.use_horovod:
            # broadcast variables from rank 0 to all other processes
            self.sess.run(self.bcast_global_variables_op)

        self.last_step = self.sess.run(self.global_step)

        # Calculate max steps. The priority of config.MAX_EPOCHS is higher than config.MAX_STEPS.
        if "MAX_EPOCHS" in self.config:
            max_steps = int(self.train_dataset.num_per_epoch / self.config.BATCH_SIZE * self.config.MAX_EPOCHS)
        else:
            max_steps = self.config.MAX_STEPS

        progbar = Progbar(max_steps)
        if self.rank == 0:
            progbar.update(self.last_step)
        for step in range(self.last_step, max_steps):

            images, labels = self.train_dataset.feed()

            feed_dict = {
                self.is_training_placeholder: True,
                self.images_placeholder: images,
                self.labels_placeholder: labels,
            }

            if step * ((step + 1) % self.config.SUMMARISE_STEPS) == 0 and self.rank == 0:
                # Runtime statistics for develop.
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                self.sess.run(self.reset_metrics_op)
                _, summary, _ = self.sess.run(
                    [self.train_op, self.summary_op, self.metrics_update_op], feed_dict=feed_dict,
                    # options=run_options,
                    # run_metadata=run_metadata,
                )
                # train_writer.add_run_metadata(run_metadata, "step: {}".format(step + 1))
                self.train_writer.add_summary(summary, step + 1)

                metrics_values = self.sess.run(list(self.metrics_ops_dict.values()))
                metrics_feed_dict = {
                    placeholder: value for placeholder, value in zip(self.metrics_placeholders, metrics_values)
                }

                metrics_summary, = self.sess.run(
                    [self.metrics_summary_op], feed_dict=metrics_feed_dict,
                )
                self.train_writer.add_summary(metrics_summary, step + 1)
                self.train_writer.flush()
            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)

            to_be_saved = step == 0 or (step + 1) == max_steps or (step + 1) % self.config.SAVE_CHECKPOINT_STEPS == 0

            if to_be_saved and self.rank == 0:
                if self.use_train_validation_saving:

                    self.sess.run(self.reset_metrics_op)
                    train_validation_saving_step_size = int(
                            math.ceil(self.train_validation_saving_dataset.num_per_epoch / self.config.BATCH_SIZE)
                    )
                    print("train_validation_saving_step_size", train_validation_saving_step_size)

                    current_train_validation_saving_set_accuracy = 0

                    for train_validation_saving_step in range(train_validation_saving_step_size):
                        print("train_validation_saving_step", train_validation_saving_step)

                        images, labels = self.train_validation_saving_dataset.feed()
                        feed_dict = {
                            self.is_training_placeholder: False,
                            self.images_placeholder: images,
                            self.labels_placeholder: labels,
                        }

                        if train_validation_saving_step % self.config.SUMMARISE_STEPS == 0:
                            summary, _ = self.sess.run([self.summary_op, self.metrics_update_op], feed_dict=feed_dict)
                            self.train_val_saving_writer.add_summary(summary, step + 1)
                            self.train_val_saving_writer.flush()
                        else:
                            self.sess.run([self.metrics_update_op], feed_dict=feed_dict)

                    metrics_values = self.sess.run(list(self.metrics_ops_dict.values()))
                    metrics_feed_dict = {
                        placeholder: value for placeholder, value in zip(self.metrics_placeholders, metrics_values)
                    }
                    metrics_summary, = self.sess.run(
                        [self.metrics_summary_op], feed_dict=metrics_feed_dict,
                    )
                    self.train_val_saving_writer.add_summary(metrics_summary, step + 1)
                    self.train_val_saving_writer.flush()

                    current_train_validation_saving_set_accuracy = self.sess.run(self.metrics_ops_dict["accuracy"])

                    if current_train_validation_saving_set_accuracy > self.top_train_validation_saving_set_accuracy:
                        self.top_train_validation_saving_set_accuracy = current_train_validation_saving_set_accuracy
                        print(
                            "New top train_validation_saving accuracy is: ",
                            self.top_train_validation_saving_set_accuracy
                        )

                        _save_checkpoint(self.saver, self.sess, self.global_step, step)

                else:
                    _save_checkpoint(self.saver, self.sess, self.global_step, step)

                if step == 0:
                    # check create pb on only first step.
                    minimal_graph = tf.graph_util.convert_variables_to_constants(
                        self.sess,
                        self.sess.graph.as_graph_def(add_shapes=True),
                        ["output"],
                    )
                    pb_name = "minimal_graph_with_shape_{}.pb".format(step + 1)
                    pbtxt_name = "minimal_graph_with_shape_{}.pbtxt".format(step + 1)
                    tf.io.write_graph(minimal_graph, environment.CHECKPOINTS_DIR, pb_name, as_text=False)
                    tf.io.write_graph(minimal_graph, environment.CHECKPOINTS_DIR, pbtxt_name, as_text=True)

            if step == 0 or (step + 1) % self.config.TEST_STEPS == 0:
                # init metrics values
                self.sess.run(self.reset_metrics_op)
                test_step_size = int(math.ceil(self.validation_dataset.num_per_epoch / self.config.BATCH_SIZE))

                for test_step in range(test_step_size):

                    images, labels = self.validation_dataset.feed()
                    feed_dict = {
                        self.is_training_placeholder: False,
                        self.images_placeholder: images,
                        self.labels_placeholder: labels,
                    }

                    if test_step % self.config.SUMMARISE_STEPS == 0:
                        summary, _ = self.sess.run([self.summary_op, self.metrics_update_op], feed_dict=feed_dict)
                        if self.rank == 0:
                            self.val_writer.add_summary(summary, step + 1)
                            self.val_writer.flush()
                    else:
                        self.sess.run([self.metrics_update_op], feed_dict=feed_dict)

                metrics_values = self.sess.run(list(self.metrics_ops_dict.values()))
                metrics_feed_dict = {
                    placeholder: value for placeholder, value in zip(self.metrics_placeholders, metrics_values)
                }
                metrics_summary, = self.sess.run(
                    [self.metrics_summary_op], feed_dict=metrics_feed_dict,
                )
                if self.rank == 0:
                    self.val_writer.add_summary(metrics_summary, step + 1)
                    self.val_writer.flush()

            if self.rank == 0:
                progbar.update(step + 1)

            # training loop end.
        self.train_dataset.close()
        self.validation_dataset.close()
        if self.use_train_validation_saving:
            self.train_validation_saving_dataset.close()


def start_training(config):
    tr = Trainer(config)
    tr.init_graph()
    tr.rank_zero()
    tr.run()
    print("Done.")


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


def train(config_file, experiment_id=None, recreate=False):
    if not experiment_id:
        # Default model_name will be taken from config file: {model_name}.yml.
        model_name = os.path.splitext(os.path.basename(config_file))[0]
        experiment_id = '{}_{:%Y%m%d%H%M%S}'.format(model_name, datetime.now())

    run(None, None, config_file, experiment_id, recreate)

    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    experiment_dir = os.path.join(output_dir, experiment_id)
    checkpoint = os.path.join(experiment_dir, 'checkpoints', 'checkpoint')

    if not tf.io.gfile.exists(checkpoint):
        raise Exception('Checkpoints are not created in {}'.format(experiment_dir))

    with open(checkpoint) as stream:
        data = yaml.load(stream)
    checkpoint_name = os.path.basename(data['model_checkpoint_path'])

    return experiment_id, checkpoint_name
