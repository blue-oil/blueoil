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


class LmSegnetV1(Base):
    """LM original semantic segmentation network.
    This network is composed of 11 convolution layers with space_to_depth and depth_to_space."""

    def __init__(
            self,
            auxiliary_weight=0.5,
            use_feature_fusion=True,
            use_attention_refinement=True,
            use_losss_weight=False,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

        self.auxiliary_weight = auxiliary_weight
        # I want to usesigmoid.
        self.attention_act = tf.nn.relu
        self.batch_norm_decay = 0.1
        self.use_feature_fusion = use_feature_fusion  # use Feature fusion module
        self.use_attention_refinement = use_attention_refinement  # use attention refinement module
        self.use_losss_weight = use_losss_weight

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(
            lmnet_block,
            activation=self.activation,
            custom_getter=self.custom_getter,
            is_training=is_training,
            is_debug=self.is_debug,
            use_bias=False,
            data_format=channels_data_format,
            batch_norm_decay=self.batch_norm_decay,
        )

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

    def spatial(self, x):
        with tf.variable_scope("spatial"):
            x = self._space_to_depth(name='s2d_1', inputs=x)
            x = self.lmnet_block('conv_1', x, 32, 3)
            x = self._space_to_depth(name='s2d_2', inputs=x)
            x = self.lmnet_block('conv_2', x, 96, 3)
            x = self._space_to_depth(name='s2d_3', inputs=x)
            x = self.lmnet_block('conv_3', x, 160, 3)

            return x

    def batch_norm(self, inputs, training):
        return tf.contrib.layers.batch_norm(
            inputs,
            decay=self.batch_norm_decay,
            updates_collections=None,
            is_training=training,
            activation_fn=None,
            center=True,
            scale=True)

    def concat_block(self, name, inputs, channel, kernel, rate=4):
        in_ch = inputs.get_shape()[3].value
        diff_channel = int(channel - in_ch)
        with tf.variable_scope(name):
            x = self.lmnet_block('c_1', inputs, diff_channel * rate, 1)
            x = self.lmnet_block('c_2', x, diff_channel, kernel)

            x = tf.concat([inputs, x], axis=3)

            return x

    def context(self, x):
        with tf.variable_scope("context"):
            with tf.variable_scope("block_1"):
                x = self._space_to_depth(name='s2d_1', inputs=x, block_size=8)
                x = self.lmnet_block('conv_1', x, 128, 1)
                for i in range(1, 4):
                    x = self.concat_block('concat_{}'.format(i), x, 128 + i*32, 3, rate=2)

                # x = self.lmnet_block('conv_2', x, 128, 1)

            with tf.variable_scope("block_2"):
                x = self._space_to_depth(name='s2d_2', inputs=x)
                x = self.lmnet_block('conv_1', x, 256, 1)
                for i in range(1, 4):
                    x = self.concat_block('concat_{}'.format(i), x, 256 + i*32, 3)

                # x = self.lmnet_block('conv_2', x, 256, 1)

            x_down_16 = x

            with tf.variable_scope("block_3"):
                x = self._space_to_depth(name='s2d_3', inputs=x)
                x = self.lmnet_block('conv_1', x, 512, 1)
                for i in range(1, 4):
                    x = self.concat_block('concat_{}'.format(i), x, 512 + i*256, 3, rate=1)

                x = self.lmnet_block('conv_2', x, 1024, 1, activation=tf.nn.relu)

            x_down_32 = x
            return x_down_32, x_down_16

    def attention(self, name, x):
        with tf.variable_scope(name):
            stock = x
            # global average pooling
            h = x.get_shape()[1].value
            w = x.get_shape()[2].value
            in_ch = x.get_shape()[3].value
            x = tf.layers.average_pooling2d(name="gap", inputs=x, pool_size=[h, w], padding="VALID", strides=1)

            x = self.batch_norm(x, self.is_training)
            x = self.activation(x)
            x = self.lmnet_block("conv", x, in_ch, 1, activation=self.attention_act)

            x = stock * x

            return x

    def conv_bias(
        self,
        name,
        inputs,
        filters,
        kernel_size,
    ):

        with tf.variable_scope(name):
            conv = tf.layers.conv2d(
                inputs, filters=filters, kernel_size=kernel_size, padding='SAME',
                use_bias=True,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            )
            return conv

    def fusion(self, sp, cx):
        with tf.variable_scope("fusion"):
            fusion_channel = 32
            x = tf.concat([cx, sp], axis=3)
            if self.use_feature_fusion:
                x = self.lmnet_block('conv_base', x, fusion_channel, 1, activation=tf.nn.relu)
                stock = x
                h = x.get_shape()[1].value
                w = x.get_shape()[2].value
                x = tf.layers.average_pooling2d(name="gap", inputs=x, pool_size=[h, w], padding="VALID", strides=1)
                # conv_1, conv_2 need to float convolution because support only 32x channle on our FPGA IP.
                x = self.lmnet_block('float_conv_1', x, fusion_channel, 1, activation=tf.nn.relu)
                x = self.lmnet_block('float_conv_2', x, fusion_channel, 1, activation=self.attention_act)

                x = stock * x
                x = stock + x
                return x

            else:
                x = self.lmnet_block('conv_base', x, fusion_channel, 1, activation=tf.nn.relu)
                return x

    def base(self, images, is_training, *args, **kwargs):
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.is_training = is_training
        self.images = images
        self.lmnet_block = lmnet_block

        x = lmnet_block('block_1', images, 32, 1)
        sp = self.spatial(x)

        cx_32, cx_16 = self.context(x)
        tail = cx_32
        h = tail.get_shape()[1].value
        w = tail.get_shape()[2].value
        tail = tf.layers.average_pooling2d(name="gap", inputs=tail, pool_size=[h, w], padding="VALID", strides=1)

        if self.use_attention_refinement:
            # cx_16 = self.attention("attention_16", cx_16)
            cx_32 = self.attention("attention_32", cx_32)

        cx_32 = cx_32 * tail

        cx_16 = self._depth_to_space(name="d2s_1", inputs=cx_16, block_size=2)
        cx_1 = self.lmnet_block('cx_16_conv_1', cx_16, 64, 1)

        cx_32 = self.activation(cx_32)
        cx_32 = self._depth_to_space(name="d2s_2", inputs=cx_32, block_size=2)
        cx_32 = self.lmnet_block('cx_32_conv_1', cx_32, 128, 1)
        cx_32 = self._depth_to_space(name="d2s_3", inputs=cx_32, block_size=2)
        cx_2 = self.lmnet_block('cx_32_conv_2', cx_32, 32, 1)

        cx = tf.concat([cx_1, cx_2], axis=3)

        x = self.fusion(sp, cx)

        x = self.conv_bias('last', x, self.num_classes, 1)

        # only for train
        self.cx_1 = self.conv_bias("block_cx1", cx_1, self.num_classes, 1)
        self.cx_2 = self.conv_bias("block_cx2", cx_2, self.num_classes, 1)

        return x

    def _cross_entropy(self, x, labels, loss_weight):
        reshape_output = tf.reshape(x, (-1, self.num_classes))
        softmax = tf.nn.softmax(reshape_output)
        cross_entropy = -tf.reduce_sum(
            (labels * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0))) * loss_weight,
            axis=[1]
        )
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        return cross_entropy_mean

    def loss(self, output, labels):
        x = self.post_process(output)
        cx_1 = self.post_process(self.cx_1)
        cx_2 = self.post_process(self.cx_2)

        with tf.name_scope("loss"):
            if self.use_losss_weight:
                loss_weight = []
                all_size = tf.to_float(tf.reduce_prod(tf.shape(labels)))
                for class_index in range(self.num_classes):
                    num_label = tf.reduce_sum(tf.to_float(tf.equal(labels, class_index)))
                    weight = (all_size - num_label) / all_size
                    loss_weight.append(weight)
            else:
                loss_weight = 1.0

            label_flat = tf.reshape(labels, (-1, 1))
            labels = tf.reshape(tf.one_hot(label_flat, depth=self.num_classes), (-1, self.num_classes))

            loss_main = self._cross_entropy(x, labels, loss_weight)
            loss_cx_1 = self._cross_entropy(cx_1, labels, loss_weight) * self.auxiliary_weight
            loss_cx_2 = self._cross_entropy(cx_2, labels, loss_weight) * self.auxiliary_weight

            loss = loss_main + loss_cx_1 + loss_cx_2

            if self.weight_decay_rate:
                weight_decay_loss = self._weight_decay_loss()
                tf.summary.scalar("weight_decay", weight_decay_loss)
                loss = loss + weight_decay_loss

            tf.summary.scalar("loss", loss)
            return loss

    def summary(self, output, labels=None):
        x = self.post_process(output)
        return super().summary(x, labels)

    def metrics(self, output, labels):
        x = self.post_process(output)
        return super().metrics(x, labels)

    def post_process(self, output):
        with tf.name_scope("post_process"):
            h = output.get_shape()[1].value
            w = output.get_shape()[2].value
            x = tf.image.resize_bilinear(output, [h*8, w*8])
            return x


class LmSegnetV1Quantize(LmSegnetV1):
    """LM original quantize semantic segmentation network.

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
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
        super().__init__(
            *args,
            **kwargs
        )

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None,
                                   quantize_first_convolution=False, *args, **kwargs):
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

                if "float" in var.op.name:
                    print("not quantized", var.op.name)
                    return var

                if not quantize_first_convolution:
                    if var.op.name.startswith("block_1/"):
                        print("not quantized", var.op.name)
                        return var

                print("quantized", var.op.name)
                return weight_quantization(var)
        return var
