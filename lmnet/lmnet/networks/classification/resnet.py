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

from blueoil.layers import average_pooling2d, batch_norm, conv2d, fully_connected
from lmnet.networks.classification.base import Base


class Resnet(Base):
    version = ""

    def __init__(
            self,
            optimizer_class=tf.compat.v1.train.GradientDescentOptimizer,
            optimizer_kwargs={},
            learning_rate_func=None,
            learning_rate_kwargs={},
            classes=[],
            is_debug=False,
            image_size=[448, 448],  # [height, width]
            batch_size=64,
            weight_decay_rate=0.0001,
            num_residual=18,
    ):
        """
        num_residual: all layer number is 2 + (num_residual * 3 * 2).
        """
        super().__init__(
            is_debug=is_debug,
            classes=classes,
            optimizer_kwargs=optimizer_kwargs,
            optimizer_class=optimizer_class,
            learning_rate_func=learning_rate_func,
            learning_rate_kwargs=learning_rate_kwargs,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.num_residual = num_residual
        self.weight_decay_rate = weight_decay_rate

    def _residual(self, inputs, in_filters, out_filters, strides, is_training):
        use_bias = False

        with tf.compat.v1.variable_scope('sub1'):
            bn1 = batch_norm("bn1", inputs, is_training=is_training)

            with tf.compat.v1.variable_scope('relu1'):
                relu1 = tf.nn.relu(bn1)
            conv1 = conv2d(
                "conv1",
                relu1,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=use_bias,
                strides=strides,
                is_debug=self.is_debug,
            )

        with tf.compat.v1.variable_scope('sub2'):
            bn2 = batch_norm("bn2", conv1, is_training=is_training)

            with tf.compat.v1.variable_scope('relu2'):
                relu2 = tf.nn.relu(bn2)

            conv2 = conv2d(
                "conv2",
                relu2,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=use_bias,
                strides=1,
                is_debug=self.is_debug,
            )

        with tf.compat.v1.variable_scope('sub_add'):
            if in_filters != out_filters:
                inputs = tf.nn.avg_pool(
                    inputs,
                    ksize=[1, strides, strides, 1],
                    strides=[1, strides, strides, 1],
                    padding='VALID'
                )
                inputs = tf.pad(
                    inputs,
                    [[0, 0], [0, 0], [0, 0], [(out_filters - in_filters)//2, (out_filters - in_filters)//2]]
                )

        output = conv2 + inputs

        return output

    def base(self, images, is_training):
        use_bias = False

        self.images = self.input = images

        with tf.compat.v1.variable_scope("init"):
            self.conv1 = conv2d(
                "conv1",
                self.images,
                filters=16,
                kernel_size=3,
                activation=None,
                use_bias=use_bias,
                is_debug=self.is_debug,
            )

            self.bn1 = batch_norm("bn1", self.conv1, is_training=is_training)
            with tf.compat.v1.variable_scope("relu1"):
                self.relu1 = tf.nn.relu(self.bn1)

        for i in range(0, self.num_residual):
            with tf.compat.v1.variable_scope("unit1_{}".format(i)):
                if i == 0:
                    out = self._residual(self.relu1, in_filters=16, out_filters=16, strides=1, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=16, out_filters=16, strides=1, is_training=is_training)

        for i in range(0, self.num_residual):
            with tf.compat.v1.variable_scope("unit2_{}".format(i)):
                if i == 0:
                    out = self._residual(out, in_filters=16, out_filters=32, strides=2, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=32, out_filters=32, strides=1, is_training=is_training)

        for i in range(0, self.num_residual):
            with tf.compat.v1.variable_scope("unit3_{}".format(i)):
                if i == 0:
                    out = self._residual(out, in_filters=32, out_filters=64, strides=2, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=64, out_filters=64, strides=1, is_training=is_training)

        with tf.compat.v1.variable_scope("unit4_0"):
            self.bn4 = batch_norm("bn4", out, is_training=is_training, activation=tf.nn.relu)

        # global average pooling
        h = self.bn4.get_shape()[1].value
        w = self.bn4.get_shape()[2].value
        self.global_average_pool = average_pooling2d(
            "global_average_pool", self.bn4, pool_size=[h, w], padding="VALID", is_debug=self.is_debug,)

        self._heatmap_layer = None
        self.fc = fully_connected("fc", self.global_average_pool, filters=self.num_classes, activation=None)

        return self.fc

    def loss(self, softmax, labels):
        """loss.

        Args:
            output: softmaxed tensor from base. shape is (batch_num, num_classes)
            labels: onehot labels tensor. shape is (batch_num, num_classes)

        """
        labels = tf.cast(labels, tf.float32)

        if self.is_debug:
            labels = tf.Print(labels, [tf.shape(labels), tf.argmax(labels, 1)], message="labels:", summarize=200)
            softmax = tf.Print(softmax, [tf.shape(softmax), tf.argmax(softmax, 1)], message="softmax:", summarize=200)

        cross_entropy = -tf.reduce_sum(
            labels * tf.math.log(tf.clip_by_value(softmax, 1e-10, 1.0)),
            axis=[1]
        )

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        loss = cross_entropy_mean + self._decay()
        tf.compat.v1.summary.scalar("loss", loss)
        return loss

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.compat.v1.trainable_variables():
            # exclude batch norm variable
            if not ("bn" in var.name and "beta" in var.name):
                costs.append(tf.nn.l2_loss(var))

        return tf.add_n(costs) * self.weight_decay_rate
