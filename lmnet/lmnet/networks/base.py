#!/usr/bin/env python3
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
import tensorflow as tf


class BaseNetwork(object):
    """Base network.

    This base network is for every task, such as classification, object detection and segmentation.
    Every sub task's base network class should extend this class.

    Args:
        is_debug (boolean): Set True to use debug mode. It will summary some histograms,
            use small dataset and step size.
        optimizer_class (class): Optimizer using for training.
        optimizer_kwargs (dict): For init optimizer.
        learning_rate_func (callable): Use for changing learning rate. Such as learning rate decay,
            `tf.train.piecewise_constant`.
        learning_rate_kwargs (dict): For learning rate function. For example of `tf.train.piecewise_constant`,
            `{"values": [5e-5, 1e-5, 5e-6, 1e-6, 5e-7], "boundaries": [20000, 40000, 60000, 80000]}`.
        classes (list | tuple): Classes names list.
        image_size (list | tuple): Image size.
        batch_size (list | tuple): Batch size.
    """

    def __init__(
            self,
            is_debug=False,
            optimizer_class=tf.train.GradientDescentOptimizer,
            optimizer_kwargs=None,
            learning_rate_func=None,
            learning_rate_kwargs=None,
            classes=(),
            image_size=(),  # [height, width]
            batch_size=64,
            data_format='NHWC'
    ):

        if data_format not in ["NCHW", "NHWC"]:
            raise RuntimeError("data format {} should be in ['NCHW', 'NHWC]'.".format(data_format))

        self.is_debug = is_debug
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'learning_rate': 0.01}
        self.learning_rate_func = learning_rate_func
        self.learning_rate_kwargs = learning_rate_kwargs if learning_rate_kwargs is not None else {}
        self.classes = map(lambda _class: _class.replace(' ', '_'), classes)
        self.num_classes = len(classes)
        self.image_size = image_size
        self.batch_size = batch_size
        self.data_format = data_format

    def base(self, images, is_training, *args, **kwargs):
        """Base function contains inference.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """
        raise NotImplemented()

    def placeholders(self):
        """Placeholders.

        Return placeholders.

        Returns:
            tf.placeholder: Placeholders.
        """
        raise NotImplemented()

    def metrics(self, output, labels):
        """Metrics.

        Args:
            output: tensor from inference.
            labels: labels tensor.
        """
        raise NotImplemented()

    # TODO(wakisaka): Deal with many networks.
    def summary(self, output, labels=None):
        """Summary.

        Args:
            output: tensor from inference.
            labels: labels tensor.
        """
        var_list = tf.trainable_variables()
        # Add histograms for all trainable variables like weights in every layer.
        for var in var_list:
            tf.summary.histogram(var.op.name, var)

    def inference(self, images, is_training):
        """Inference.

        Args:
            images: images tensor. shape is (batch_num, height, width, channel)
        """
        raise NotImplemented()

    def loss(self, output, labels):
        """Loss.

        Args:
            output: tensor from inference.
            labels: labels tensor.
        """
        raise NotImplemented()

    def optimizer(self, global_step):
        assert ("learning_rate" in self.optimizer_kwargs.keys()) or \
               (self.learning_rate_func is not None)

        if "learning_rate" in self.optimizer_kwargs.keys():
            learning_rate = self.optimizer_kwargs["learning_rate"]

        else:
            if self.learning_rate_func is tf.train.piecewise_constant:
                learning_rate = self.learning_rate_func(
                    x=global_step,
                    **self.learning_rate_kwargs
                )
            else:
                learning_rate = self.learning_rate_func(
                    global_step=global_step,
                    **self.learning_rate_kwargs
                )

        tf.summary.scalar("learning_rate", learning_rate)
        self.optimizer_kwargs["learning_rate"] = learning_rate

        return self.optimizer_class(**self.optimizer_kwargs)

    def train(self, loss, optimizer, global_step=tf.Variable(0, trainable=False), var_list=[]):
        """Train.

        Args:
           loss: loss function of this network.
           global_step: tensorflow's global_step
        """
        # TODO(wenhao): revert when support `tf.layers.batch_normalization`
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        with tf.name_scope("train"):
            if var_list == []:
                var_list = tf.trainable_variables()

            gradients = optimizer.compute_gradients(loss, var_list=var_list)

            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        # Add histograms for all gradients for every layer.
        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradients", grad)

        return train_op
