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
import sys

import tensorflow as tf

from lmnet.layers import conv2d
from lmnet.networks.object_detection.yolo_v2 import YoloV2


def _batch_norm(inputs, is_training):
    return tf.contrib.layers.batch_norm(
        inputs,
        decay=0.997,
        updates_collections=None,
        is_training=is_training,
        activation_fn=None,
        center=True,
        scale=True)


def _glaze_block(
        name,
        inputs,
        filters,
        kernel_size=5,
        strides=1,
        custom_getter=None,
        is_training=tf.constant(True),
        activation=None,
        use_bias=True,
        use_batch_norm=True,
        is_debug=False,
        data_format='channels_last',
        num_groups=1,
        channel_coeff=1):

    filters *= channel_coeff
    num_groups *= channel_coeff

    hs = []
    first_conv_filters = filters // num_groups
    xs = tf.split(inputs, num_groups, axis=-1)
    for i in range(num_groups):
        k = kernel_size

        if i == 0 or (num_groups >= 4 and i < 2):
            k = 3

        h = tf.layers.conv2d(
            xs[i], first_conv_filters, k, strides=strides, padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False,
            name=name+"/group_"+str(i)+"_1stconv")
        hs.append(h)

    h = tf.concat(hs, axis=-1)
    h = _batch_norm(h, is_training)
    h = activation(h)

    if num_groups > 1:
        h = tf.layers.conv2d(
            h, filters, 1, padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False,
            name=name+"_1x1conv")
        h = _batch_norm(h, is_training)
        h = activation(h)

        input_channel_num = inputs.get_shape()[-1]
        output_channel_num = filters
        if input_channel_num == output_channel_num and strides == 1:
            return inputs + h
        else:
            return h
    else:
        return h


class GlazedYolo(YoloV2):
    """YOLOv2 + BlazeFace + MixConv + Group Convolution

        YoloV2 https://arxiv.org/abs/1612.08242
        BlazeFace https://arxiv.org/abs/1907.05047
        MixConv https://arxiv.org/abs/1907.09595
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.downsampling_rate = 16

        super().__init__(
            *args,
            **kwargs,
        )

    def train(self, loss, optimizer, global_step=tf.Variable(0, trainable=False), var_list=[]):
        with tf.name_scope("train"):
            if var_list == []:
                var_list = tf.trainable_variables()

            gradients = optimizer.compute_gradients(loss, var_list=var_list)

            # Add gradient clipping
            gradients = [
                (tf.clip_by_value(gradient, -10.0, 10.0), var)
                for gradient, var in gradients
            ]
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradients", grad)

        return train_op

    def base(self, images, is_training):
        """Base network.
        Args:
            images: input image tensor,expects NHWC format
            is_training: boolean flag which indicates whether it's on training
        """

        self.inputs = self.images = images

        if self.data_format == "NCHW":
            channel_data_format = "channels_first"
        elif self.data_format == "NHWC":
            channel_data_format = "channels_last"
        else:
            raise RuntimeError(
                "data format {} should be in ['NCHW', 'NHWC]'.".format(
                    self.data_format))

        if self.data_format == "NCHW":
            # currently, only NHWC format is supported
            sys.exit(-1)

        glaze_block = partial(_glaze_block, is_training=is_training,
                              activation=self.activation, data_format=self.data_format, channel_coeff=1)

        h = tf.layers.conv2d(
            self.inputs, 32, 3, strides=2, padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False, name="conv_first")

        h = glaze_block("block_1", h, filters=32, num_groups=1)
        h = glaze_block("block_2", h, filters=64, num_groups=1, strides=2)
        h = glaze_block("block_3", h, filters=64, num_groups=2)
        h = glaze_block("block_4", h, filters=64, num_groups=2)
        h = glaze_block("block_5", h, filters=128, num_groups=2, strides=2)
        h = glaze_block("block_6", h, filters=128, num_groups=4)
        h = glaze_block("block_7", h, filters=128, num_groups=4)
        h = glaze_block("block_8", h, filters=128, num_groups=4, strides=2)
        h = glaze_block("block_9", h, filters=128, num_groups=4)
        h = glaze_block("block_10", h, filters=128, num_groups=4)

        output_filters = (self.num_classes + 5) * self.boxes_per_cell
        self.block_last = conv2d("conv_last", h, filters=output_filters, kernel_size=1,
                                 activation=None, use_bias=True, is_debug=self.is_debug,
                                 data_format=channel_data_format)

        if self.change_base_output:
            predict_classes, predict_confidence, predict_boxes = self._predictions(self.block_last)

            with tf.name_scope("convert_boxes_space_from_yolo_to_real"):
                predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

            output = self._concat_predictions(predict_classes, predict_confidence, predict_boxes)

        else:
            output = self.block_last

        return output


class GlazedYoloQuantize(GlazedYolo):

    def __init__(
            self,
            quantize_first_convolution=True,
            quantize_last_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        """
        Args:
            quantize_first_convolution(bool): use quantization in first conv.
            quantize_last_convolution(bool): use quantization in last conv.
            weight_quantizer (callable): weight quantizer.
            weight_quantize_kwargs(dict): Initialize kwargs for weight quantizer.
            activation_quantizer (callable): activation quantizer
            activation_quantize_kwargs(dict): Initialize kwargs for activation quantizer.
        """

        super().__init__(
            *args,
            **kwargs,
        )

        self.quantize_first_convolution = quantize_first_convolution
        self.quantize_last_convolution = quantize_last_convolution

        activation_quantizer_kwargs = activation_quantizer_kwargs if not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if not None else {}

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        self.weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.activation = activation_quantizer(**activation_quantizer_kwargs)

        if self.quantize_last_convolution:
            self.before_last_activation = self.activation
        else:
            self.before_last_activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

    @staticmethod
    def _quantized_variable_getter(
            weight_quantization,
            quantize_first_convolution,
            quantize_last_convolution,
            getter,
            name,
            *args,
            **kwargs):
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
            if "kernel" == var.op.name.split("/")[-1]:

                if not quantize_first_convolution:
                    if var.op.name.startswith("conv_first/"):
                        return var

                if not quantize_last_convolution:
                    if var.op.name.startswith("conv_last/"):
                        return var

                # Apply weight quantize to variable whose last word of name is "kernel".
                quantized_kernel = weight_quantization(var)
                tf.summary.histogram("quantized_kernel", quantized_kernel)
                return quantized_kernel

        return var

    def base(self, images, is_training):
        custom_getter = partial(
            self._quantized_variable_getter,
            self.weight_quantization,
            self.quantize_first_convolution,
            self.quantize_last_convolution,
        )
        with tf.variable_scope("", custom_getter=custom_getter):
            return super().base(images, is_training)
