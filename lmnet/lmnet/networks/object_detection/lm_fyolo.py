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

from lmnet.blocks import darknet
from lmnet.layers import conv2d
from lmnet.networks.object_detection.yolo_v2 import YoloV2


class LMFYolo(YoloV2):
    """LM original objecte detection network based on Yolov2 and F-Yolo.

    Ref:
        F-Yolo https://arxiv.org/abs/1805.06361
        YoloV2 https://arxiv.org/abs/1612.08242
    """

    def train(self, loss, optimizer, global_step=tf.Variable(0, trainable=False), var_list=[]):
        with tf.name_scope("train"):
            if var_list == []:
                var_list = tf.trainable_variables()

            gradients = optimizer.compute_gradients(loss, var_list=var_list)

            # Add clip grad
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
        Returns: Output. output shape depends on parameter.
            When `data_format` is `NHWC`
            shape is [
                batch_size,
                num_cell[0],
                num_cell[1],
                (num_classes + 5(x, y ,w, h, confidence)) * boxes_per_cell(length of anchors),
            ]
            When `data_format` is `NCHW`
            shape is [
                batch_size,
                (num_classes + 5(x, y ,w, h, confidence)) * boxes_per_cell(length of anchors),
                num_cell[0],
                num_cell[1],
            ]
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

        darknet_block = partial(darknet, is_training=is_training,
                                activation=self.activation, data_format=self.data_format)

        x = darknet_block("block_1", self.inputs, filters=32, kernel_size=1)
        x = darknet_block("block_2", x, filters=8, kernel_size=3)
        x = self._reorg("pool_1", x, stride=2, data_format=self.data_format)

        x = darknet_block("block_3", x, filters=16, kernel_size=3)
        x = self._reorg("pool_2", x, stride=2, data_format=self.data_format)

        x4 = darknet_block("block_4", x, filters=32, kernel_size=3)
        x = self._reorg("pool_3", x4, stride=2, data_format=self.data_format)

        x5 = darknet_block("block_5", x, filters=64, kernel_size=3)
        x = self._reorg("pool_4", x5, stride=2, data_format=self.data_format)

        x6 = darknet_block("block_6", x, filters=128, kernel_size=3)
        x = self._reorg("pool_5", x6, stride=2, data_format=self.data_format)

        x4_1 = darknet_block("block_4_1x1", x4, filters=4, kernel_size=1)
        x5_1 = darknet_block("block_5_1x1", x5, filters=16, kernel_size=1)
        x6_1 = darknet_block("block_6_1x1", x6, filters=32, kernel_size=1)

        x4_s2d = self._reorg("block_4_s2d", x4_1, stride=8, data_format=self.data_format)
        x5_s2d = self._reorg("block_5_s2d", x5_1, stride=4, data_format=self.data_format)
        x6_s2d = self._reorg("block_6_s2d", x6_1, stride=2, data_format=self.data_format)

        x7 = darknet_block("block_7", x, filters=128, kernel_size=3)
        x8 = darknet_block("block_8", x7, filters=256, kernel_size=3)

        if self.data_format == "NHWC":
            concat_axis = -1
        if self.data_format == "NCHW":
            concat_axis = 1

        merged = tf.concat([x8, x7, x6_s2d, x5_s2d, x4_s2d],
                           axis=concat_axis, name="block_merged")

        x = darknet_block("block_9", merged, filters=256, kernel_size=3)
        x = darknet_block("block_10", x, filters=128, kernel_size=3)

        x = darknet_block("block_11", x, filters=256, kernel_size=3)
        x = darknet_block("block_12", x, filters=128, kernel_size=3)
        x = darknet_block("block_13", x, filters=256, kernel_size=3, activation=self.before_last_activation)

        output_filters = (self.num_classes + 5) * self.boxes_per_cell
        self.block_last = conv2d("block_last", x, filters=output_filters, kernel_size=1,
                                 activation=None, use_bias=True, is_debug=self.is_debug,
                                 data_format=channel_data_format)

        if self.change_base_output:

            predict_classes, predict_confidence, predict_boxes = self._predictions(self.block_last)

            with tf.name_scope("convert_boxes_space_from_yolo_to_real"):
                predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

            output = self._concat_predictions(predict_classes, predict_confidence, predict_boxes)

        else:
            # with tf.control_dependencies([assert_num_cell_x, assert_num_cell_y]):
            output = self.block_last

        return output


class LMFYoloQuantize(LMFYolo):

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
                    if var.op.name.startswith("block_1/"):
                        return var

                if not quantize_last_convolution:
                    if var.op.name.startswith("block_last/"):
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
