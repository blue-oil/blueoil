import optuna
import tensorflow as tf

import os
import math
import sys

import numpy as np
import click

from lmnet.utils import executor, module_loader, config as config_util


class OptCIFAR10(object):

    def __init__(self, config_data):
        """Initiate training. It only being called once."""
        print('----------- OptCIFAR10 init -----------')
        self.lm_config = config_util.load(os.path.join(os.getcwd(), config_data))
        config_util.display(self.lm_config)
        executor.init_logging(self.lm_config)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")

        NetworkClass = self.lm_config.NETWORK_CLASS
        network_kwargs = dict((key.lower(), val) for key, val in self.lm_config.NETWORK.items())
        DatasetClass = self.lm_config.DATASET_CLASS
        dataset_kwargs = dict((key.lower(), val) for key, val in self.lm_config.DATASET.items())
        self.train_dataset = DatasetClass(
            subset="train",
            **dataset_kwargs,
        )

        self.validation_dataset = DatasetClass(
            subset="validation",
            **dataset_kwargs,
        )

        if NetworkClass.__module__.startswith("lmnet.networks.object_detection"):
            model = NetworkClass(
                classes=self.train_dataset.classes,
                num_max_boxes=self.train_dataset.num_max_boxes,
                is_debug=self.lm_config.IS_DEBUG,
                **network_kwargs,
            )
        elif NetworkClass.__module__.startswith("lmnet.networks.segmentation"):
            model = NetworkClass(
                classes=self.train_dataset.classes,
                label_colors=self.train_dataset.label_colors,
                is_debug=self.lm_config.IS_DEBUG,
                **network_kwargs,
            )
        else:
            model = NetworkClass(
                classes=self.train_dataset.classes,
                is_debug=self.lm_config.IS_DEBUG,
                **network_kwargs,
            )

        self.images_placeholder, self.labels_placeholder = model.placeholderes()

        output = model.inference(self.images_placeholder, self.is_training_placeholder)
        if NetworkClass.__module__.startswith("lmnet.networks.object_detection"):
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

    def __call__(self, trial):
        """It is being called in every trial."""
        print('----------- OptCIFAR10 call -----------')
        last_step = self.sess.run(self.global_step)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        step_per_epoch = int(self.train_dataset.num_per_epoch / batch_size)
        print('num per epoch:{}, batch size: {}'.format(self.train_dataset.num_per_epoch, batch_size))
        for _ in range(step_per_epoch):
            images, labels = self.train_dataset.feed()

            feed_dict = {
                self.is_training_placeholder: True,
                self.images_placeholder: images,
                self.labels_placeholder: labels,
            }

            self.sess.run([self.train_op], feed_dict=feed_dict)

        self.sess.run(self.reset_metrics_op)
        test_step_size = int(math.ceil(self.validation_dataset.num_per_epoch / batch_size))
        for _ in range(test_step_size):
            images, labels = self.validation_dataset.feed()
            feed_dict = {
                self.is_training_placeholder: False,
                self.images_placeholder: images,
                self.labels_placeholder: labels,
            }

            self.sess.run([self.metrics_update_op], feed_dict=feed_dict)

        metrics_values = self.sess.run(list(self.metrics_ops_dict.values()))
        self.iterations += 1
        print('trial: {0}, metric value: {1}'.format(trial, metrics_values))
        return 1 - metrics_values[0]


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
def run_opt(network, dataset, config_file, experiment_id, recreate):
    study = optuna.create_study()
    study.optimize(OptCIFAR10(config_file),
                   n_trials=4,
                   n_jobs=-1)  # n_jobs is number of parallel jobs. If this argument is set to -1, the number is set to CPU counts.

    duration = time.time() - start_time
    print(study.best_params)


if __name__ == '__main__':
    run_opt()
