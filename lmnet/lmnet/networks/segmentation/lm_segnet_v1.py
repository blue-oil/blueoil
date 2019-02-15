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

from lmnet.blocks import lmnet_block
from lmnet.networks.segmentation.base import Base
from lmnet.networks.quantize_param_init import QuantizeParamInit


class LmSegnetV1(Base):
    """LM original semantic segmentation network.
    This network is composed of 11 convolution layers with space_to_depth and depth_to_space."""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

        # name of the scope in the first and last layer
        self.first_layer_name = "conv1/"
        self.last_layer_name = "conv11/"

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(lmnet_block,
                                 activation=self.activation,
                                 custom_getter=self.custom_getter,
                                 is_training=is_training,
                                 is_debug=self.is_debug,
                                 use_bias=False,
                                 data_format=channels_data_format)

    def _space_to_depth(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])
        output = tf.space_to_depth(inputs, block_size=block_size, name=name)
        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def _depth_to_space(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])
        output = tf.depth_to_space(inputs, block_size=block_size, name=name)
        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def base(self, images, is_training, *args, **kwargs):
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images

        x = self._space_to_depth(name='space2depth1', inputs=images)
        x = lmnet_block('conv1', x, 16, 1)
        x = self._space_to_depth(name='space2depth2', inputs=x)
        x = lmnet_block('conv2', x, 64, 3)
        x = lmnet_block('conv3', x, 64, 3)
        x = self._space_to_depth(name='space2depth3', inputs=x)
        x = lmnet_block('conv4', x, 256, 3)
        x = lmnet_block('conv5', x, 256, 3)
        x = lmnet_block('conv6', x, 256, 3)
        x = lmnet_block('conv7', x, 256, 3)
        x = self._depth_to_space(name='depth2space1', inputs=x)
        x = lmnet_block('conv8', x, 128, 3)
        x = self._depth_to_space(name='depth2space2', inputs=x)
        x = lmnet_block('conv9', x, 128, 3)
        x = self._depth_to_space(name='depth2space3', inputs=x)
        x = lmnet_block('conv10', x, 32, 3)
        x = lmnet_block('conv11', x, self.num_classes, 3)

        return x


class LmSegnetV1Quantize(QuantizeParamInit, LmSegnetV1):
    """LM original quantize semantic segmentation network.
    QuantizeParamInit is a mixin class used to initialize variables for quantization and custom_getter.

    Scope of custom_getter is defined in lmnet_block so there is no need to define base function.
    """

    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)
