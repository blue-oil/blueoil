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
import functools

import tensorflow as tf

from lmnet.networks.base import BaseNetwork


class FlowNetSV1(BaseNetwork):
    """FlowNetS v1 for optical flow estimation.
    """
    version = 1.00

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")
        self.weight_decay_rate = 0.0004
        self.use_batch_norm = True
        self.custom_getter = None

    # TODO: Import _conv_bn_act from blocks after replacing strides=2 using space to depth.
    def _conv_bn_act(
            self,
            name,
            inputs,
            filters,
            is_training,
            kernel_size=3,
            strides=1,
            enable_detail_summary=False,
    ):
        if self.data_format == "NCHW":
            channel_data_format = "channels_first"
        elif self.data_format == "NHWC":
            channel_data_format = "channels_last"
        else:
            raise ValueError("data format must be 'NCHW' or 'NHWC'. got {}.".format(self.data_format))

        # TODO Think: pytorch used batch_norm but tf did not.
        # pytorch: if batch_norm no bias else use bias.
        with tf.variable_scope(name):
            conved = tf.layers.conv2d(
                inputs,
                filters=filters,
                kernel_size=kernel_size,
                padding='SAME',
                strides=strides,
                use_bias=False,
                data_format=channel_data_format,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay_rate)
            )

            if self.use_batch_norm:
                batch_normed = tf.contrib.layers.batch_norm(
                    conved,
                    is_training=is_training,
                    data_format=self.data_format,
                )
            else:
                batch_normed = conved

            output = self.activation(batch_normed)

            if enable_detail_summary:
                tf.summary.histogram('conv_output', conved)
                tf.summary.histogram('batch_norm_output', batch_normed)
                tf.summary.histogram('output', output)

            return output

    def _deconv(
            self,
            name,
            inputs,
            filters
    ):
        # The paper and pytorch used LeakyReLU(0.1,inplace=True) but tf did not. I decide to still use it.
        with tf.variable_scope(name):
            # tf only allows 'SAME' or 'VALID' padding.
            # In conv2d_transpose, h = h1 * stride if padding == 'Same'
            # https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
            conved =  tf.layers.conv2d_transpose(
                inputs,
                filters,
                kernel_size=4,
                strides=2,
                padding='SAME',
                use_bias=True
            )
            output = self.activation(conved)
            return output


    def _predict_flow(
            self,
            name,
            inputs
    ):
        with tf.variable_scope(name):
            # pytorch uses padding = 1 = (3 -1) // 2. So it is 'SAME'.
            return tf.layers.conv2d(
                inputs,
                2,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bias=True
            )


    def _upsample_flow(self):
        pass

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        # TODO tf version uses padding=VALID and pad to match the original caffe code.
        # Acan DLK handle this?
        # pytorch version uses (kernel_size-1) // 2, which is equal to 'SAME' in tf
        x = self._conv_bn_act('conv1', images, 64, is_training, kernel_size=7, strides=2)
        conv_2 = self._conv_bn_act('conv2', x, 128, is_training, kernel_size=5, strides=2)
        x = self._conv_bn_act('conv3', x, 256, is_training, kernel_size=5, strides=2)
        conv3_1 = self._conv_bn_act('conv3_1', x, 256, is_training)
        x = self._conv_bn_act('conv4', conv3_1, 512, is_training, strides=2)
        conv4_1 = self._conv_bn_act('conv4_1', x, 512, is_training)
        x = self._conv_bn_act('conv5', conv4_1, 512, is_training, strides=2)
        conv5_1 = self._conv_bn_act('conv5_1', x, 512, is_training) # 12x16
        x = self._conv_bn_act('conv6', conv5_1, 1024, is_training, strides=2) # 12x16
        conv6_1 = self._conv_bn_act('conv6_1', x, 1024, is_training) # 6x8

        deconv5 = self._deconv('deconv5', conv6_1, 512)
        predict_flow6 = self._predict_flow('predict_flow6', conv6_1)
        upsample_flow6 = self._upsample_flow()

        # Same order as pytorch and tf
        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6], axis=3)


        return {
            'predict_flow6': predict_flow6,
            'predict_flow5': predict_flow5,
            'predict_flow4': predict_flow4,
            'predict_flow3': predict_flow3,
            'predict_flow2': predict_flow2,
            'flow': flow,
        }







