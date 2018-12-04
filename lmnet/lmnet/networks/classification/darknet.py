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
from functools import partial

import tensorflow as tf

from lmnet.networks.classification.base import Base
from lmnet.layers import conv2d, max_pooling2d
from lmnet.blocks import darknet as darknet_block
from lmnet.networks.quantize_param_init import QuantizeParamInit


class Darknet(Base):
    """Darknet 19 layer"""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")
        self.before_last_activation = self.activation

        #name of the scope in the first and last layer
        self.first_layer_name="block_1/"
        self.last_layer_name="conv_19/"

    def base(self, images, is_training):
        if self.data_format == "NCHW":
            channel_data_format = "channels_first"
        elif self.data_format == "NHWC":
            channel_data_format = "channels_last"
        else:
            raise RuntimeError("data format {} shodul be in ['NCHW', 'NHWC]'.".format(self.data_format))

        self.inputs = self.images = images

        self.block_1 = darknet_block(
            "block_1",
            self.inputs,
            filters=32,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_1 = max_pooling2d("pool_1", self.block_1, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_2 = darknet_block(
            "block_2",
            self.pool_1,
            filters=64,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_2 = max_pooling2d("pool_2", self.block_2, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_3 = darknet_block(
            "block_3",
            self.pool_2,
            filters=128,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_4 = darknet_block(
            "block_4",
            self.block_3,
            filters=64,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_5 = darknet_block(
            "block_5",
            self.block_4,
            filters=128,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_3 = max_pooling2d("pool_3", self.block_5, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_6 = darknet_block(
            "block_6",
            self.pool_3,
            filters=256,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_7 = darknet_block(
            "block_7",
            self.block_6,
            filters=128,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_8 = darknet_block(
            "block_8",
            self.block_7,
            filters=256,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_4 = max_pooling2d("pool_4", self.block_8, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_9 = darknet_block(
            "block_9",
            self.pool_4,
            filters=512,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_10 = darknet_block(
            "block_10",
            self.block_9,
            filters=256,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_11 = darknet_block(
            "block_11",
            self.block_10,
            filters=512,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_12 = darknet_block(
            "block_12",
            self.block_11,
            filters=256,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_13 = darknet_block(
            "block_13",
            self.block_12,
            filters=512,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_5 = max_pooling2d("pool_5", self.block_13, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_14 = darknet_block(
            "block_14",
            self.pool_5,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_15 = darknet_block(
            "block_15",
            self.block_14,
            filters=512,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_16 = darknet_block(
            "block_16",
            self.block_15,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_17 = darknet_block(
            "block_17",
            self.block_16,
            filters=512,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_18 = darknet_block(
            "block_18",
            self.block_17,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.before_last_activation,
            data_format=self.data_format,
        )

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        self.conv_19 = conv2d(
            "conv_19", self.block_18, filters=self.num_classes, kernel_size=1,
            activation=None, use_bias=True, is_debug=self.is_debug,
            kernel_initializer=kernel_initializer, data_format=channel_data_format,
        )

        if self.is_debug:
            self._heatmap_layer = self.conv_19

        if self.data_format == "NCHW":
            axis = [2, 3]
        if self.data_format == "NHWC":
            axis = [1, 2]
        # TODO(wakisaka): global average pooling should use tf.reduce_mean()

        self.pool_6 = tf.reduce_mean(self.conv_19, axis=axis, name="global_average_pool_6")
        self.base_output = tf.reshape(self.pool_6, [-1, self.num_classes], name="pool6_reshape")

        return self.base_output


class DarknetQuantize(QuantizeParamInit, Darknet):
    """Quantize Darknet Network.
    QuantizeParamInit is a mixin class used to initialize variables for quantization and custom_getter.

    Darknet does not use lmnet_block, need to define a scope for custom_getter in base function.
    """

    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)
