# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
import six
import math
import click
import tensorflow as tf
import multiprocessing

from easydict import EasyDict
from lmnet.utils import executor, config as config_util
from lmnet.datasets.dataset_iterator import DatasetIterator

import ray
from ray.tune import run_experiments, register_trainable, Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

if six.PY2:
    import subprocess32 as subprocess
else:
    import subprocess


def subproc_call(cmd, timeout=None):
    """
    Execute a command with timeout, and return both STDOUT/STDERR.
    Args:
        cmd(str): the command to execute.
        timeout(float): timeout in seconds.
    Returns:
        output(bytes), retcode(int). If timeout, retcode is -1.
    """
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT,
            shell=True, timeout=timeout)
        return output, 0
    except subprocess.TimeoutExpired as e:
        print("Command '{}' timeout!".format(cmd))
        print(e.output.decode('utf-8'))
        return e.output, -1
    except subprocess.CalledProcessError as e:
        print("Command '{}' failed, return code={}".format(cmd, e.returncode))
        print(e.output.decode('utf-8'))
        return e.output, e.returncode
    except Exception:
        print("Command '{}' failed to run.".format(cmd))
        return "", -2


def get_num_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """

    def warn_return(ret, message):
        built_with_cuda = tf.test.is_built_with_cuda()
        if not built_with_cuda and ret > 0:
            print(message + "But TensorFlow was not built with CUDA support and could not use GPUs!")
        return ret

    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env:
        return warn_return(len(env.split(',')), "Found non-empty CUDA_VISIBLE_DEVICES. ")
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code == 0:
        output = output.decode('utf-8')
        return warn_return(len(output.strip().split('\n')), "Found nvidia-smi. ")
    else:
        print('Not working for this one... But there are other methods you can try...')
        raise NotImplementedError


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))


def trial_str_creator(trial):
    """Rename trial to shorter string"""
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


def get_best_result(trial_list, metric, param):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric).last_result[metric],
            param: get_best_trial(trial_list, metric).last_result[param]}


def update_parameters_for_each_trial(network_kwargs, chosen_kwargs):
    """Update selected parameters to the configuration of each trial"""
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

    if 'weight_decay_rate' in chosen_kwargs:
        network_kwargs['weight_decay_rate'] = chosen_kwargs['weight_decay_rate']

    return network_kwargs


def setup_dataset(config, subset, rank):
    """helper function from lmnet/train.py to setup the data iterator"""
    dataset_class = config.DATASET_CLASS
    dataset_kwargs = dict((key.lower(), val) for key, val in config.DATASET.items())
    dataset = dataset_class(subset=subset, **dataset_kwargs)
    enable_prefetch = dataset_kwargs.pop("enable_prefetch", False)
    return DatasetIterator(dataset, seed=rank, enable_prefetch=enable_prefetch)


class TrainTunable(Trainable):
    """ TrainTunable class interfaces with Ray framework """
    def _setup(self, config):
        self.lm_config = config_util.load(self.config['lm_config'])
        executor.init_logging(self.lm_config)

        model_class = self.lm_config.NETWORK_CLASS
        network_kwargs = dict((key.lower(), val) for key, val in self.lm_config.NETWORK.items())
        network_kwargs = update_parameters_for_each_trial(network_kwargs, self.config)

        # No distributed training was implemented, therefore rank set to 0
        self.train_dataset = setup_dataset(self.lm_config, "train", 0)
        self.validation_dataset = setup_dataset(self.lm_config, "validation", 0)

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


def run(config_file, tunable_id, local_dir):
    register_trainable(tunable_id, TrainTunable)
    lm_config = config_util.load(config_file)

    def easydict_to_dict(config):
        if isinstance(config, EasyDict):
            config = dict(config)

        for key, value in config.items():
            if isinstance(value, EasyDict):
                value = dict(value)
                easydict_to_dict(value)
            config[key] = value
        return config

    tune_space = easydict_to_dict(lm_config['TUNE_SPACE'])
    tune_spec = easydict_to_dict(lm_config['TUNE_SPEC'])
    tune_spec['run'] = tunable_id
    tune_spec['config'] = {'lm_config': os.path.join(os.getcwd(), config_file)}
    tune_spec['local_dir'] = local_dir
    tune_spec['trial_name_creator'] = ray.tune.function(trial_str_creator)

    # Expecting use of gpus to do parameter search
    ray.init(num_cpus=multiprocessing.cpu_count() // 2, num_gpus=max(get_num_gpu(), 1))
    algo = HyperOptSearch(tune_space, max_concurrent=4, reward_attr="mean_accuracy")
    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", reward_attr="mean_accuracy", max_t=200)
    trials = run_experiments(experiments={'exp_tune': tune_spec},
                             search_alg=algo,
                             scheduler=scheduler)
    print("The best result is", get_best_result(trials, metric="mean_accuracy", param='config'))


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '-c',
    '--config_file',
    help="config file path for this training",
    default=os.path.join('configs', 'example.py'),
    required=True,
)
@click.option(
    '-i',
    '--tunable_id',
    help='[optional] id of this tuning',
    default="tunable",
)
@click.option(
    '-s',
    '--local_dir',
    help='[optional] result saving directory of training results, defaults in ~/ray_results',
    default=None,
)
def main(config_file, tunable_id, local_dir):
    run(config_file, tunable_id, local_dir)


if __name__ == '__main__':
    main()
