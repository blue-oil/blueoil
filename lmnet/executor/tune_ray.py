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
import math
import click
import tensorflow as tf

from lmnet.utils import executor, module_loader, config as config_util

import ray
from ray.tune import grid_search, run_experiments, register_trainable, Trainable, function
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))


def get_best_result(trial_list, metric, param):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric).last_result[metric],
            param: get_best_trial(trial_list, metric).last_result[param]}


def update_parameters_for_each_trial(network_kwargs, chosen_kwargs):
    """Update selected parameters for each trial"""
    # print('chosen args', chosen_kwargs)
    # print('before network args', network_kwargs)
    network_kwargs['optimizer_class'] = chosen_kwargs['optimizer_class']['optimizer']
    for key in list(chosen_kwargs['optimizer_class'].keys()):
        if key != 'optimizer':
            network_kwargs['optimizer_kwargs'][key] = chosen_kwargs['optimizer_class'][key]
    network_kwargs['learning_rate_func'] = chosen_kwargs['learning_rate_func']['scheduler']
    base_lr = chosen_kwargs['learning_rate']
    if network_kwargs['learning_rate_func'] is tf.train.piecewise_constant:
        lr_factor = chosen_kwargs['learning_rate_func']['scheduler_factor']
        network_kwargs['learning_rate_kwargs']['values'] = [base_lr,
                                                            base_lr * lr_factor,
                                                            base_lr * lr_factor * lr_factor,
                                                            base_lr * lr_factor * lr_factor * lr_factor]
        network_kwargs['learning_rate_kwargs']['boundaries'] = chosen_kwargs['learning_rate_func']['scheduler_steps']
    elif network_kwargs['learning_rate_func'] is tf.train.polynomial_decay:
        network_kwargs['learning_rate_kwargs']['learning_rate'] = base_lr
        network_kwargs['learning_rate_kwargs']['power'] = chosen_kwargs['learning_rate_func']['scheduler_power']
        network_kwargs['learning_rate_kwargs']['decay_steps'] = chosen_kwargs['learning_rate_func']['scheduler_decay']
    else:
        network_kwargs['learning_rate_kwargs']['learning_rate'] = base_lr

    network_kwargs['weight_decay_rate'] = chosen_kwargs['weight_decay_rate']
    # print('after network args', network_kwargs)
    return network_kwargs


class TrainTunable(Trainable):
    def _setup(self, config):
        self.lm_config = config_util.load(self.config['lm_config'])
        executor.init_logging(self.lm_config)

        dataset_class = self.lm_config.DATASET_CLASS
        model_class = self.lm_config.NETWORK_CLASS
        network_kwargs = dict((key.lower(), val) for key, val in self.lm_config.NETWORK.items())
        dataset_kwargs = dict((key.lower(), val) for key, val in self.lm_config.DATASET.items())

        network_kwargs = update_parameters_for_each_trial(network_kwargs, self.config)

        self.train_dataset = dataset_class(
            subset="train",
            **dataset_kwargs,
        )

        self.validation_dataset = dataset_class(
            subset="validation",
            **dataset_kwargs,
        )

        print("train dataset num:", self.train_dataset.num_per_epoch)

        if model_class.__module__.startswith("lmnet.networks.object_detection"):
            model = model_class(
                classes=self.train_dataset.classes,
                num_max_boxes=self.train_dataset.num_max_boxes,
                is_debug=self.lm_config.IS_DEBUG,
                **network_kwargs,
            )
        elif model_class.__module__.startswith("lmnet.networks.segmentation"):
            model = model_class(
                classes=self.train_dataset.classes,
                label_colors=self.train_dataset.label_colors,
                is_debug=self.lm_config.IS_DEBUG,
                **network_kwargs,
            )
        else:
            model = model_class(
                classes=self.train_dataset.classes,
                is_debug=self.lm_config.IS_DEBUG,
                **network_kwargs,
            )

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")
        self.images_placeholder, self.labels_placeholder = model.placeholderes()

        output = model.inference(self.images_placeholder, self.is_training_placeholder)
        if model_class.__module__.startswith("lmnet.networks.object_detection"):
            loss = model.loss(output, self.labels_placeholder, self.is_training_placeholder)
        else:
            loss = model.loss(output, self.labels_placeholder)
        opt = model.optimizer(self.global_step)

        train_op = model.train(loss, opt, self.global_step)
        metrics_ops_dict, metrics_update_op = model.metrics(output, self.labels_placeholder)

        self.train_op = train_op
        self.metrics_ops_dict = metrics_ops_dict
        self.metrics_update_op = metrics_update_op

        init_op = tf.global_variables_initializer()
        self.reset_metrics_op = tf.local_variables_initializer()

        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=session_config)
        self.sess.run([init_op, self.reset_metrics_op])
        self.iterations = 0
        self.saver = tf.train.Saver()

    def _train(self):
        step_per_epoch = int(self.train_dataset.num_per_epoch / self.lm_config.BATCH_SIZE)

        for _ in range(step_per_epoch):
            images, labels = self.train_dataset.feed()

            feed_dict = {
                self.is_training_placeholder: True,
                self.images_placeholder: images,
                self.labels_placeholder: labels,
            }

            self.sess.run([self.train_op], feed_dict=feed_dict)

        self.sess.run(self.reset_metrics_op)
        test_step_size = int(math.ceil(self.validation_dataset.num_per_epoch / self.lm_config.BATCH_SIZE))
        for _ in range(test_step_size):
            images, labels = self.validation_dataset.feed()
            feed_dict = {
                self.is_training_placeholder: False,
                self.images_placeholder: images,
                self.labels_placeholder: labels,
            }

            self.sess.run([self.metrics_update_op], feed_dict=feed_dict)

        if self.lm_config.NETWORK_CLASS.__module__.startswith("lmnet.networks.segmentation"):
            metric_accuracy = self.sess.run(self.metrics_ops_dict["mean_iou"])
        else:
            metric_accuracy = self.sess.run(self.metrics_ops_dict["accuracy"])

        self.iterations += 1
        return {"mean_accuracy": metric_accuracy}

    def _save(self, checkpoint_dir):
        return self.saver.save(
            self.sess, checkpoint_dir + "/save", global_step=self.iterations)

    def _restore(self, path):
        return self.saver.restore(self.sess, path)


def run(config_file):
    register_trainable("tunable", TrainTunable)
    lm_config = config_util.load(config_file)
    tune_space = lm_config['TUNE_SPACE']
    tune_spec = lm_config['TUNE_SPEC']
    tune_spec['config']['lm_config'] = os.path.join(os.getcwd(), config_file)

    ray.init(num_cpus=8, num_gpus=2)
    algo = HyperOptSearch(tune_space, max_concurrent=4, reward_attr="mean_accuracy")
    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", reward_attr="mean_accuracy", max_t=200)
    trials = run_experiments(experiments={'exp_tune': tune_spec}, search_alg=algo, scheduler=scheduler)
    print("The best result is", get_best_result(trials, metric="mean_accuracy", param='config'))


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-c",
    "--config_file",
    help="config file path for this training",
    default=os.path.join("configs", "example.py"),
    required=True,
)
def main(config_file):
    run(config_file)


if __name__ == '__main__':
    main()
