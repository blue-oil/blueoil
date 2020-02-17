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

from blueoil.networks.base import BaseNetwork

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16Network(BaseNetwork):
    def __init__(
            self,
            num_classes,
            optimizer_class,
            optimizer_args,
            is_debug=False
    ):

        self.num_classes = num_classes
        super().__init__(
            optimizer_class=optimizer_class,
            optimizer_args=optimizer_args,
            is_debug=is_debug
        )

    def build(self, images, is_training):
        keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))

        self.input = self.convert_rbg_to_bgr(images)

        self.conv1 = self.conv_layer("conv1", self.input, filters=64, kernel_size=3)
        self.conv2 = self.conv_layer("conv2", self.conv1, filters=64, kernel_size=3)

        self.pool1 = self.max_pool("pool1", self.conv2, kernel_size=2, strides=2)

        self.conv3 = self.conv_layer("conv3", self.pool1, filters=128, kernel_size=3)
        self.conv4 = self.conv_layer("conv4", self.conv3, filters=128, kernel_size=3)

        self.pool2 = self.max_pool("pool2", self.conv4, kernel_size=2, strides=2)

        self.conv5 = self.conv_layer("conv5", self.pool2, filters=256, kernel_size=3)
        self.conv6 = self.conv_layer("conv6", self.conv5, filters=256, kernel_size=3)
        self.conv7 = self.conv_layer("conv7", self.conv6, filters=256, kernel_size=3)

        self.pool3 = self.max_pool("pool3", self.conv7, kernel_size=2, strides=2)

        self.conv8 = self.conv_layer("conv8", self.pool3, filters=512, kernel_size=3)
        self.conv9 = self.conv_layer("conv9", self.conv8, filters=512, kernel_size=3)
        self.conv10 = self.conv_layer("conv10", self.conv9, filters=256, kernel_size=3)

        self.pool4 = self.max_pool("pool4", self.conv10, kernel_size=2, strides=2)

        self.conv11 = self.conv_layer("conv11", self.pool4, filters=512, kernel_size=3)
        self.conv12 = self.conv_layer("conv12", self.conv11, filters=512, kernel_size=3)
        self.conv13 = self.conv_layer("conv13", self.conv12, filters=512, kernel_size=3)

        self.pool5 = self.max_pool("pool5", self.conv13, kernel_size=2, strides=2)

        fc14 = self.fc_layer("fc14", self.pool5, filters=4096, activation=tf.nn.relu)
        self.fc14 = tf.nn.dropout(fc14, keep_prob)

        fc15 = self.fc_layer("fc15", self.fc15, filters=4096, activation=tf.nn.relu)
        self.fc15 = tf.nn.dropout(fc15, keep_prob)

        self.fc16 = self.fc_layer("fc16", self.fc15, filters=self.num_classes, activation=None)

        return self.fc16

    def conv_layer(
        self,
        name,
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding="SAME",
        activation=tf.nn.sigmoid,
        *args,
        **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()

        output = super(Vgg16Network, self).conv_layer(
            name=name,
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            biases_initializer=biases_initializer,
            *args,
            **kwargs
        )

        return output

    def fc_layer(
            self,
            name,
            inputs,
            filters,
            *args,
            **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()

        output = super(Vgg16Network, self).fc_layer(
            name=name,
            inputs=inputs,
            filters=filters,
            kernel_initializer=kernel_initializer,
            biases_initializer=biases_initializer,
            *args,
            **kwargs
        )

        return output

    def convert_rbg_to_bgr(self, rgb_images):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_images)

        bgr_images = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        return bgr_images
