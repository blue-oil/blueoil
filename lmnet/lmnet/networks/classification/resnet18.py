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
import functools

from lmnet.networks.classification.base import Base
from lmnet.networks.base_quantize import BaseQuantize
from lmnet.layers import average_pooling2d, batch_norm, conv2d, fully_connected, max_pooling2d


class Resnet18(Base):
    version = ""

    def __init__(
            self,
            optimizer_class=tf.train.GradientDescentOptimizer,
            optimizer_kwargs={},
            learning_rate_func=None,
            learning_rate_kwargs={},
            classes=[],
            is_debug=False,
            image_size=[224, 224],  # [height, width]
            batch_size=64,
            weight_decay_rate=0.0001,
            num_residual=2,
            *args,
            **kwargs,
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

        with tf.variable_scope('sub1'):
            conv1 = conv2d(
                "conv1",
                inputs,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=use_bias,
                strides=strides,
                is_debug=self.is_debug,
            )
            bn1 = batch_norm("bn1", conv1, is_training=is_training)

            with tf.variable_scope('relu1'):
                relu1 = tf.nn.relu(bn1)
            

        with tf.variable_scope('sub2'):
            conv2 = conv2d(
                "conv2",
                relu1,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=use_bias,
                strides=1,
                is_debug=self.is_debug,
            )

            with tf.variable_scope('sub_add'):
                if in_filters != out_filters:
                    inputs = tf.nn.avg_pool(
                        inputs,
                        ksize=[1, strides, strides, 1],
                        strides=[1, strides, strides, 1],
                        padding='SAME'
                    )
                    inputs = tf.pad(
                        inputs,
                        [[0, 0], [0, 0], [0, 0], [(out_filters - in_filters)//2, (out_filters - in_filters)//2]]
                    )
            output = conv2 + inputs
            bn2 = batch_norm("bn2", output, is_training=is_training)
            with tf.variable_scope('relu2'):
                relu2 = tf.nn.relu(output)

        return relu2

    def base(self, images, is_training):
        use_bias = False

        self.images = self.input = images

        with tf.variable_scope("init"):
            self.conv1 = conv2d(
                "conv1",
                self.images,
                filters=64,
                kernel_size=7,
                activation=None,
                use_bias=use_bias,
                is_debug=self.is_debug,
                strides=2,
            )

            self.bn1 = batch_norm("bn1", self.conv1, is_training=is_training)
            with tf.variable_scope("relu1"):
                self.relu1 = tf.nn.relu(self.bn1)
            self.pool1 = max_pooling2d("init_max_pool", self.relu1, 2, strides=2, padding="SAME")

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit1_{}".format(i)):
                if i == 0:
                    out = self._residual(self.pool1, in_filters=64, out_filters=64, strides=1, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=64, out_filters=64, strides=1, is_training=is_training)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit2_{}".format(i)):
                if i == 0:
                    out = self._residual(out, in_filters=64, out_filters=128, strides=2, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=128, out_filters=128, strides=1, is_training=is_training)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit3_{}".format(i)):
                if i == 0:
                    out = self._residual(out, in_filters=128, out_filters=256, strides=2, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=256, out_filters=256, strides=1, is_training=is_training)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit4_{}".format(i)):
                if i == 0:
                    out = self._residual(out, in_filters=256, out_filters=512, strides=2, is_training=is_training)
                else:
                    out = self._residual(out, in_filters=512, out_filters=512, strides=1, is_training=is_training)

        #with tf.variable_scope("unit4_0"):
            #self.bn4 = batch_norm("bn4", out, is_training=is_training, activation=tf.nn.relu)
        self.bn4=out
        # global average pooling
        h = self.bn4.get_shape()[1].value
        w = self.bn4.get_shape()[2].value
        self.global_average_pool = average_pooling2d(
            "global_average_pool", out, pool_size=[h, w], padding="VALID", is_debug=self.is_debug,)

        self._heatmap_layer = None
        self.fc = fully_connected("fc", self.global_average_pool, filters=self.num_classes, activation=None, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), weights_initializer=tf.contrib.layers.xavier_initializer())

        return self.fc

    def loss(self, softmax, labels):
        """loss.

        Params:
           output: softmaxed tensor from base. shape is (batch_num, num_classes)
           labels: onehot labels tensor. shape is (batch_num, num_classes)
        """
        labels = tf.to_float(labels)

        if self.is_debug:
            labels = tf.Print(labels, [tf.shape(labels), tf.argmax(labels, 1)], message="labels:", summarize=200)
            softmax = tf.Print(softmax, [tf.shape(softmax), tf.argmax(softmax, 1)], message="softmax:", summarize=200)

        cross_entropy = -tf.reduce_sum(
            labels * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)),
            axis=[1]
        )

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        loss = cross_entropy_mean + self._decay()
        tf.summary.scalar("loss", loss)
        return loss

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            # exclude batch norm variable
            if not ("bn" in var.name and "beta" in var.name):
                costs.append(tf.nn.l2_loss(var))

        return tf.add_n(costs) * self.weight_decay_rate



class Resnet18Quantize(Resnet18, BaseQuantize):
    version = 1.0

    def __init__(
            self,        
            quantize_first_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        Resnet18.__init__(
            self,
            *args,
            **kwargs
        )

        BaseQuantize.__init__(
            self,
            activation_quantizer,
            activation_quantizer_kwargs,
            weight_quantizer,
            weight_quantizer_kwargs,
            quantize_first_convolution,
        )
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               quantize_first_convolution=self.quantize_first_convolution,
                                               weight_quantization=self.weight_quantization)


    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)
        
        
    @staticmethod
    def _quantized_variable_getter(getter, name, quantize_first_convolution, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """

        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            #if not quantize_first_convolution:
            #    if var.op.name.startswith("conv1/"):
            #        return var

            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
            if "weights" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
