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
from lmnet.networks.keypoint_detection.base import Base


class LmSinglePoseV1(Base):
    """LM original single-person pose estimation network."""

    def __init__(self, stride=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = tf.nn.relu
        self.custom_getter = None
        self.num_joints = 17

        if stride not in {2, 4, 8, 16}:
            raise ValueError("Only stride 2, 4, 8, 16 are supported.")

        self.stride = stride

    def loss(self, output, labels):
        """Loss function for single-person pose estimation.

        Args:
           output: Tensor of network output. shape is (batch_size, height, width, num_joints).
           labels: Tensor of ground-truth labels. shape is (batch_size, height, width, num_joints).

        """
        if self.data_format == 'NCHW':
            output = tf.transpose(output, perm=[0, 2, 3, 1])
        with tf.name_scope("loss"):
            global_loss = tf.reduce_mean((output - labels)**2)

            refine_loss = tf.reduce_mean((output - labels)**2, axis=(1, 2))
            top_k = 8
            top_k_values, top_k_idx = tf.nn.top_k(refine_loss, k=top_k, sorted=False)
            refine_loss = tf.reduce_sum(top_k_values) / top_k

            loss = refine_loss + global_loss

            tf.summary.scalar("global_loss", global_loss)
            tf.summary.scalar("refine_loss", refine_loss)
            tf.summary.scalar("loss", loss)
            return loss

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
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        def connect_block(lateral, up, name="connect_block"):
            with tf.variable_scope(name_or_scope=name):
                shape = lateral.get_shape().as_list()
                up = tf.image.resize_nearest_neighbor(up, size=shape[1:3])
                out = tf.concat([up, lateral], axis=-1)
                out = _lmnet_block("block", out, filters=32, kernel_size=3)
            return out

        self.images = images

        x = _lmnet_block('conv1', images, 32, 1)
        x = self._space_to_depth(name='space2depth1', inputs=x)
        x = _lmnet_block('conv2', x, 32, 3)
        x = _lmnet_block('conv3', x, 32, 3)
        c1 = _lmnet_block('c1', x, 32, 1)

        x = self._space_to_depth(name='space2depth2', inputs=x)
        x = _lmnet_block('conv4', x, 32, 3)
        x = _lmnet_block('conv5', x, 32, 3)
        x = _lmnet_block('conv6', x, 32, 3)
        c2 = _lmnet_block('c2', x, 32, 1)

        x = self._space_to_depth(name='space2depth3', inputs=x)
        x = _lmnet_block('conv7', x, 64, 3)
        x = _lmnet_block('conv8', x, 64, 3)
        x = _lmnet_block('conv9', x, 64, 3)
        c3 = _lmnet_block('c3', x, 32, 1)

        x = self._space_to_depth(name='space2depth4', inputs=x)
        x = _lmnet_block('conv10', x, 128, 3)
        x = _lmnet_block('conv11', x, 128, 3)
        x = _lmnet_block('conv12', x, 128, 3)
        c4 = _lmnet_block('c4', x, 32, 1)

        x = self._space_to_depth(name='space2depth5', inputs=x)
        x = _lmnet_block('conv13', x, 256, 3)
        x = _lmnet_block('conv14', x, 256, 3)
        x = _lmnet_block('conv15', x, 256, 3)
        x = _lmnet_block('conv16', x, 256, 3)
        x = _lmnet_block('conv17', x, 256, 3)
        c5 = _lmnet_block('c5', x, 32, 1)

        x = connect_block(lateral=c4, up=c5, name="connect_block1")

        if self.stride < 16:
            x = connect_block(lateral=c3, up=x, name="connect_block2")
        if self.stride < 8:
            x = connect_block(lateral=c2, up=x, name="connect_block3")
        if self.stride < 4:
            x = connect_block(lateral=c1, up=x, name="connect_block4")

        x = _lmnet_block('conv_final', x, self.num_joints, 3, activation=None)
        return x


class LmSinglePoseV1Quantize(LmSinglePoseV1):
    """LM original quantized single-person pose estimation network.

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `blueoil.nn.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `blueoil.nn.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.

    """

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.

        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                if var.op.name.startswith("conv1/"):
                    return var
                if var.op.name.startswith("conv_final/"):
                    return var
                return weight_quantization(var)
        return var
