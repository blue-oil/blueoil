import os
import math
import click

import optuna
import tensorflow as tf

from lmnet.utils import executor, module_loader, config as config_util


def train_fn(lm_config, tune_space):

    network_fn = lm_config.NETWORK_CLASS
    network_kwargs = dict((key.lower(), val) for key, val in lm_config.NETWORK.items())
    dataset_fn = lm_config.DATASET_CLASS
    dataset_kwargs = dict((key.lower(), val) for key, val in lm_config.DATASET.items())

    network_kwargs['optimizer_class'] = tune_space['optimizer_class']['optimizer']
    for key in list(tune_space['optimizer_class'].keys()):
        if key != 'optimizer':
            network_kwargs['optimizer_kwargs'][key] = tune_space['optimizer_class'][key]
    base_lr = tune_space['learning_rate']
    network_kwargs['learning_rate_kwargs']['values'] = [base_lr,
                                                        base_lr * 0.1,
                                                        base_lr * 0.1 * 0.1,
                                                        base_lr * 0.1 * 0.1 * 0.1]
    print(network_kwargs)

    train_dataset = dataset_fn(
        subset="train",
        **dataset_kwargs,
    )

    validation_dataset = dataset_fn(
        subset="validation",
        **dataset_kwargs,
    )

    # tf.reset_default_graph()

    with tf.Graph().as_default():

        if network_fn.__module__.startswith("lmnet.networks.object_detection"):
            model = network_fn(
                classes=train_dataset.classes,
                num_max_boxes=train_dataset.num_max_boxes,
                is_debug=lm_config.IS_DEBUG,
                **network_kwargs,
            )
        elif network_fn.__module__.startswith("lmnet.networks.segmentation"):
            model = network_fn(
                classes=train_dataset.classes,
                label_colors=train_dataset.label_colors,
                is_debug=lm_config.IS_DEBUG,
                **network_kwargs,
            )
        else:
            model = network_fn(
                classes=train_dataset.classes,
                is_debug=lm_config.IS_DEBUG,
                **network_kwargs,
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")

        images_placeholder, labels_placeholder = model.placeholderes()
        output = model.inference(images_placeholder, is_training_placeholder)
        if network_fn.__module__.startswith("lmnet.networks.object_detection"):
            loss = model.loss(output, labels_placeholder, is_training_placeholder)
        else:
            loss = model.loss(output, labels_placeholder)

        lr_fn = network_kwargs['learning_rate_func']
        network_kwargs['optimizer_kwargs']['learning_rate'] = \
            lr_fn(x=global_step, **network_kwargs['learning_rate_kwargs'])
        opt_fn = network_kwargs['optimizer_class']
        opt = opt_fn(**network_kwargs['optimizer_kwargs'])

        train_op = model.train(loss, opt, global_step)
        metrics_ops_dict, metrics_update_op = model.metrics(output, labels_placeholder)

        init_op = tf.global_variables_initializer()
        reset_metrics_op = tf.local_variables_initializer()
        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True))

        with tf.Session(config=session_config) as sess:
            sess.run([init_op, reset_metrics_op])
            step_per_epoch = int(train_dataset.num_per_epoch / lm_config['BATCH_SIZE'])
            test_step_size = int(math.ceil(validation_dataset.num_per_epoch / lm_config['BATCH_SIZE']))
            metrics_values = sess.run(list(metrics_ops_dict.values()))
            max_epoch = 200
            for _ in range(max_epoch):
                for _ in range(step_per_epoch):
                    images, labels = train_dataset.feed()

                    feed_dict = {
                        is_training_placeholder: True,
                        images_placeholder: images,
                        labels_placeholder: labels,
                    }

                    sess.run([train_op], feed_dict=feed_dict)

                sess.run(reset_metrics_op)
                for _ in range(test_step_size):
                    images, labels = validation_dataset.feed()
                    feed_dict = {
                        is_training_placeholder: False,
                        images_placeholder: images,
                        labels_placeholder: labels,
                    }

                    sess.run([metrics_update_op], feed_dict=feed_dict)

                metrics_values = sess.run(list(metrics_ops_dict.values()))
                # print(metrics_values[0])
                if metrics_values[0] >= 0.82:
                    return 1 - metrics_values[0]

            return 1 - metrics_values[0]


class OptObjective(object):
    def __init__(self, config_file):
        self.config_file = config_file

    def __call__(self, trial):
        """ Examples of searching space
        Categorical parameter
            optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
        Int parameter
            num_layers = trial.suggest_int('num_layers', 1, 3)
        Uniform parameter
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
        Loguniform parameter
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        Discrete-uniform parameter
            drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)
        """
        tune_space = {
            'optimizer_class': trial.suggest_categorical('optimizer_class', [
                {
                    'optimizer': tf.train.AdamOptimizer
                },
                {
                    'optimizer': tf.train.MomentumOptimizer,
                    'momentum': 0.9
                },
                {
                    'optimizer': tf.train.AdagradOptimizer
                }
            ]),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.001),
            'weight_decay_rate': 0.0005
        }
        lm_config = config_util.load(os.path.join(os.getcwd(), self.config_file))
        val_err = train_fn(lm_config, tune_space)
        print('trial: {0}, metric value: {1}'.format(trial, val_err))
        return val_err


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-c",
    "--config_file",
    help="config file path for this training",
    default=os.path.join("configs", "example", "classification.py"),
    required=True,
)
def run_opt(config_file):
    study = optuna.create_study()
    # n_jobs is number of parallel jobs.
    # If this argument is set to -1, the number is set to CPU counts.
    study.optimize(OptObjective(config_file), n_trials=5, n_jobs=5)
    print(study.best_params)


if __name__ == '__main__':
    run_opt()
