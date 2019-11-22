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
from lmnet.networks.classification.base import Base


class Resnet(Base):
    version = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _bn(self, name, x, training, data_format):
        return tf.layers.batch_normalization(
            x,
            axis=-1 if data_format in ['NHWC', 'channels_last'] else 1,
            momentum=0.997,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            trainable=True,
            name=name,
            reuse=None,
            renorm=False,
            renorm_clipping=None,
            renorm_momentum=0.9,
            fused=True)

    def _conv(self, name, x, filters, kernel_size, strides, data_format):
        # treat as pooling
        if strides == 2:
            x = tf.space_to_depth(x, block_size=2, data_format=data_format)

        return tf.layers.conv2d(
            x, filters, kernel_size,
            padding="SAME",
            data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False,
            name=name)

    def _res_block(self, x, ch_out, stride, training, data_format, stage, block):
        conv_name_base = 'res{}{}_branch'.format(str(stage), block)
        bn_name_base = 'bn{}{}_branch'.format(str(stage), block)
        shortcut = x

        x = self._conv(conv_name_base + '2a', x, ch_out, 1, 1, data_format)
        x = self._bn(bn_name_base + '2a', x, training, data_format)
        x = tf.nn.relu(x)

        x = self._conv(conv_name_base + '2b', x, ch_out, 3, stride, data_format)
        x = self._bn(bn_name_base + '2b', x, training, data_format)
        x = tf.nn.relu(x)

        x = self._conv(conv_name_base + '2c', x, ch_out, 1, 1, data_format)
        x = self._bn(bn_name_base + '2c', x, training, data_format)

        x = x + shortcut
        x = tf.nn.relu(x)
        return x

    def resnet_backbone(self, image, training, data_format):
        # Stage 1
        x = self._conv('conv1', image, 64, 3, 2, data_format)
        x = self._bn('bn_conv1', x, training, data_format)
        C1 = x = tf.nn.relu(x)
        # Stage 2
        x = self._res_block(x, 64, 2, training, data_format, stage=2, block='a')
        x = self._res_block(x, 64, 1, training, data_format, stage=2, block='b')
        C2 = x = self._res_block(x, 64, 1, training, data_format, stage=2, block='c')
        # Stage 3
        x = self._res_block(x, 128, 2, training, data_format, stage=3, block='a')
        x = self._res_block(x, 128, 1, training, data_format, stage=3, block='b')
        x = self._res_block(x, 128, 1, training, data_format, stage=3, block='c')
        C3 = x = self._res_block(x, 128, 1, training, data_format, stage=3, block='d')
        # Stage 4
        x = self._res_block(x, 256, 2, training, data_format, stage=4, block='a')
        x = self._res_block(x, 256, 1, training, data_format, stage=4, block='b')
        x = self._res_block(x, 256, 1, training, data_format, stage=4, block='c')
        x = self._res_block(x, 256, 1, training, data_format, stage=4, block='d')
        x = self._res_block(x, 256, 1, training, data_format, stage=4, block='e')
        C4 = x = self._res_block(x, 256, 1, training, data_format, stage=4, block='f')
        # Stage 5
        x = self._res_block(x, 512, 2, training, data_format, stage=5, block='a')
        x = self._res_block(x, 512, 1, training, data_format, stage=5, block='b')
        C5 = x = self._res_block(x, 512, 1, training, data_format, stage=5, block='c')

        # Head
        x = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_last' if data_format == 'NHWC' else 'channels_first')(x)
        logits = tf.keras.layers.Dense(1000, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
        # probs = tf.keras.layers.Activation('softmax', name='class_probs')(logits)
        return logits

    def base(self, images, is_training, *args, **kwargs):
        return self.resnet_backbone(images, is_training, self.data_format)

    def loss(self, softmax, labels):
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
            if not ("bn" in var.name or "beta" in var.name):
                costs.append(tf.nn.l2_loss(var))

        return tf.add_n(costs) * self.weight_decay_rate
