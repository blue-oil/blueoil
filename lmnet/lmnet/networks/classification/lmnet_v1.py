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
from lmnet.networks.classification.base import Base
from lmnet.networks.quantize_param_init import QuantizeParamInit


class LmnetV1(Base):
    """Lmnet v1 for classification.
    """
    version = 1.0

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
        self.last_layer_name = "conv7/"

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

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images

        x = _lmnet_block('conv1', images, 32, 3)
        x = _lmnet_block('conv2', x, 64, 3)
        x = self._space_to_depth(name='pool2', inputs=x)
        x = _lmnet_block('conv3', x, 128, 3)
        x = _lmnet_block('conv4', x, 64, 3)
        x = self._space_to_depth(name='pool4', inputs=x)
        x = _lmnet_block('conv5', x, 128, 3)
        x = self._space_to_depth(name='pool5', inputs=x)
        x = _lmnet_block('conv6', x, 64, 1, activation=tf.nn.relu)

        x = tf.layers.dropout(x, training=is_training)

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        x = tf.layers.conv2d(name='conv7',
                             inputs=x,
                             filters=self.num_classes,
                             kernel_size=1,
                             kernel_initializer=kernel_initializer,
                             activation=None,
                             use_bias=True,
                             data_format=channels_data_format)

        self._heatmap_layer = x

        h = x.get_shape()[1].value if self.data_format == 'NHWC' else x.get_shape()[2].value
        w = x.get_shape()[2].value if self.data_format == 'NHWC' else x.get_shape()[3].value
        x = tf.layers.average_pooling2d(name='pool7',
                                        inputs=x,
                                        pool_size=[h, w],
                                        padding='VALID',
                                        strides=1,
                                        data_format=channels_data_format)

        self.base_output = tf.reshape(x, [-1, self.num_classes], name='pool7_reshape')

        return self.base_output, tf.get_default_graph()


class LmnetV1Quantize(QuantizeParamInit, LmnetV1):
    """Lmnet quantize network for classification, version 1.0
    QuantizeParamInit is a mixin class used to initialize variables for quantization and custom_getter.

    Scope of custom_getter is defined in lmnet_block so there is no need to define base function.
    """
    pass
