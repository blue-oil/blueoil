# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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

from lmnet.networks.segmentation.base import Base
from lmnet.blocks import conv_bn_act, densenet_group


class LMBiSeNet(Base):
    """LM original semantic segmentation network reference to [BiSeNet](https://arxiv.org/abs/1808.00897)

    Major difference from BiSeNet:
    * Apply first convolution then branch into contextual and spatial path.
    * Use space to depth (s2d) and depth to space (d2s) as upsample and downsample.
    * Use only 1 stride 1x1 and 3x3 convolution (not use multi stride and dilated convolution)
        for be limited by our convolution IP.
    * Use DensNet block in contextual part.
    * All of convolution out channels are less than BiSeNet for inference time.
    * Use attention refinement module reference to BiSeNet after last layer of 1/32
        (BiSeNet: both 1/16 and 1/32) in context path.
    * Use relu activation instead of sigmoid in attention refinement and feature fusion module.
    * In up-sampling followed by context path, alternate d2s and 1x1 conv for reducing channel size.
    """

    def __init__(
            self,
            weight_decay_rate=0.0,
            auxiliary_loss_weight=0.5,
            use_feature_fusion=True,
            use_attention_refinement=True,
            use_tail_gap=True,
            *args,
            **kwargs
    ):
        """

        Args:
            weight_decay_rate (float): Rate of weight (convolution kernel) decay.
            auxiliary_loss_weight (float): Rate of auxiliary loss.
            use_feature_fusion (bool): Flag of using feature fusion module.
            use_attention_refinement (bool): Flag of using attention refinement module.
            use_tail_gap (bool): Flag of using GAP (global average pooling) followed by context path.
        """
        super().__init__(
            *args,
            **kwargs
        )

        assert self.data_format == 'NHWC'

        self.activation = tf.nn.relu

        # I want to use sigmoid.
        self.attention_act = tf.nn.relu
        self.batch_norm_decay = 0.1
        self.enable_detail_summary = False
        # Always don't use attention refinement module for 1/16 size feature map.
        self.use_attention_refinement_16 = False

        self.weight_decay_rate = weight_decay_rate
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.use_feature_fusion = use_feature_fusion
        self.use_attention_refinement = use_attention_refinement
        self.use_tail_gap = use_tail_gap

    def _space_to_depth(self, name, inputs=None, block_size=2):
        output = tf.space_to_depth(inputs, block_size=block_size, name=name)
        return output

    def _depth_to_space(self, name, inputs=None, block_size=2):
        output = tf.depth_to_space(inputs, block_size=block_size, name=name)
        return output

    def _batch_norm(self, inputs, training):
        return tf.contrib.layers.batch_norm(
            inputs,
            decay=self.batch_norm_decay,
            updates_collections=None,
            is_training=training,
            center=True,
            scale=True)

    def _conv_bias(
        self,
        name,
        inputs,
        filters,
        kernel_size,
    ):
        with tf.variable_scope(name):
            output = tf.layers.conv2d(
                inputs,
                filters=filters,
                kernel_size=kernel_size,
                padding='SAME',
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  # he initializer
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay_rate),
            )

            if self.enable_detail_summary:
                tf.summary.histogram("conv_bias_output", output)
            return output

    def _spatial(self, x):
        with tf.variable_scope("spatial"):
            x = self._space_to_depth(name='s2d_1', inputs=x)
            x = self._block('conv_1', x, 32, 3)
            x = self._space_to_depth(name='s2d_2', inputs=x)
            x = self._block('conv_2', x, 96, 3)
            x = self._space_to_depth(name='s2d_3', inputs=x)
            x = self._block('conv_3', x, 160, 3)

            return x

    def _context(self, x):
        with tf.variable_scope("context"):
            with tf.variable_scope("block_1"):
                x = self._space_to_depth(name='s2d_1', inputs=x, block_size=8)
                x = self._block('conv_1', x, 128, 1)
                growth_rate = 32
                bottleneck_rate = 2
                x = densenet_group(
                    "dense",
                    inputs=x,
                    num_blocks=3,
                    growth_rate=growth_rate,
                    bottleneck_rate=bottleneck_rate,
                    weight_decay_rate=self.weight_decay_rate,
                    is_training=self.is_training,
                    activation=self.activation,
                    batch_norm_decay=self.batch_norm_decay,
                )

            with tf.variable_scope("block_2"):
                x = self._space_to_depth(name='s2d_2', inputs=x)
                x = self._block('conv_1', x, 256, 1)
                growth_rate = 32
                bottleneck_rate = 4
                x = densenet_group(
                    "dense",
                    inputs=x,
                    num_blocks=3,
                    growth_rate=growth_rate,
                    bottleneck_rate=bottleneck_rate,
                    weight_decay_rate=self.weight_decay_rate,
                    is_training=self.is_training,
                    activation=self.activation,
                    batch_norm_decay=self.batch_norm_decay,
                )
                if self.use_attention_refinement and self.use_attention_refinement_16:
                    # attention module needs float inputs.
                    x_down_16 = self._block('conv_2', x, 256, 1, activation=tf.nn.relu)
                    x = self.activation(x_down_16)
                else:
                    x_down_16 = self._block('conv_2', x, 256, 1)
                    x = x_down_16

            with tf.variable_scope("block_3"):
                x = self._space_to_depth(name='s2d_3', inputs=x)
                x = self._block('conv_1', x, 512, 1)
                growth_rate = 256
                bottleneck_rate = 1
                x = densenet_group(
                    "dense",
                    inputs=x,
                    num_blocks=3,
                    growth_rate=growth_rate,
                    bottleneck_rate=bottleneck_rate,
                    weight_decay_rate=self.weight_decay_rate,
                    is_training=self.is_training,
                    activation=self.activation,
                    batch_norm_decay=self.batch_norm_decay,
                )
                if self.use_attention_refinement or self.use_tail_gap:
                    # attention module and tail gap needs float inputs.
                    x_down_32 = self._block('conv_2', x, 1024, 1, activation=tf.nn.relu)
                else:
                    x_down_32 = self._block('conv_2', x, 1024, 1)

            return x_down_32, x_down_16

    def _attention(self, name, x):
        with tf.variable_scope(name):
            stock = x
            # global average pooling
            h = x.get_shape()[1].value
            w = x.get_shape()[2].value
            in_ch = x.get_shape()[3].value
            x = tf.layers.average_pooling2d(name="gap", inputs=x, pool_size=[h, w], padding="VALID", strides=1)

            x = self._batch_norm(x, self.is_training)
            x = self.activation(x)
            x = self._block("conv", x, in_ch, 1, activation=self.attention_act)

            x = stock * x

            return x

    def _fusion(self, sp, cx):
        """Feature fusion module"""
        with tf.variable_scope("fusion"):
            fusion_channel = 64
            x = tf.concat([cx, sp], axis=3)
            if self.use_feature_fusion:
                x = self._block('conv_base', x, fusion_channel, 1, activation=tf.nn.relu)
                stock = x
                h = x.get_shape()[1].value
                w = x.get_shape()[2].value
                x = tf.layers.average_pooling2d(name="gap", inputs=x, pool_size=[h, w], padding="VALID", strides=1)
                # conv_1, conv_2 need to float convolution because our FPGA IP support only 32x channles.
                x = self._block('float_conv_1', x, fusion_channel, 1, activation=tf.nn.relu)
                x = self._block('float_conv_2', x, fusion_channel, 1, activation=self.attention_act)

                x = stock * x
                x = stock + x
                return x

            else:
                x = self._block('conv_base', x, fusion_channel, 1, activation=tf.nn.relu)
                return x

    def base(self, images, is_training, *args, **kwargs):
        self.is_training = is_training
        self.images = images

        self._block = functools.partial(
            conv_bn_act,
            weight_decay_rate=self.weight_decay_rate,
            is_training=is_training,
            activation=self.activation,
            batch_norm_decay=self.batch_norm_decay,
            data_format=self.data_format,
            enable_detail_summary=self.enable_detail_summary,
        )

        x = self._block("block_first", images, 32, 1)
        spatial = self._spatial(x)

        context_32, context_16 = self._context(x)

        with tf.variable_scope("context_merge"):
            if self.use_tail_gap:
                tail = context_32
                h = tail.get_shape()[1].value
                w = tail.get_shape()[2].value
                tail = tf.layers.average_pooling2d(
                    name="gap", inputs=tail, pool_size=[h, w], padding="VALID", strides=1)

            if self.use_attention_refinement:
                if self.use_attention_refinement_16:
                    context_16 = self._attention("attention_16", context_16)
                    context_16 = self.activation(context_16)
                context_32 = self._attention("attention_32", context_32)

            context_16 = self._depth_to_space(name="d2s_1", inputs=context_16, block_size=2)
            context_1 = self._block("context_16_conv_1", context_16, 64, 1)

            if self.use_tail_gap:
                context_32 = context_32 * tail

            if self.use_attention_refinement or self.use_tail_gap:
                context_32 = self.activation(context_32)

            context_32 = self._depth_to_space(name="d2s_2", inputs=context_32, block_size=2)
            context_32 = self._block("context_32_conv_1", context_32, 128, 1)
            context_32 = self._depth_to_space(name="d2s_3", inputs=context_32, block_size=2)
            context_2 = self._block("context_32_conv_2", context_32, 64, 1)

            context = tf.concat([context_1, context_2], axis=3)

        x = self._fusion(spatial, context)

        x = self._conv_bias("block_last", x, self.num_classes, 1)

        # only for training
        self.context_1 = self._conv_bias("float_block_context_1", context_1, self.num_classes, 1)
        self.context_2 = self._conv_bias("float_block_context_2", context_2, self.num_classes, 1)

        return x

    def _cross_entropy(self, x, labels):
        x = tf.reshape(x, [-1, self.num_classes])
        cross_entropy = -tf.reduce_sum(
            (labels * tf.log(tf.clip_by_value(x, 1e-10, 1.0))),
            axis=[1]
        )
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        return cross_entropy_mean

    def _weight_decay_loss(self):
        """L2 weight decay (regularization) loss."""
        losses = []
        print("apply l2 loss these variables")
        for var in tf.trainable_variables():

            # exclude batch norm variable
            if "kernel" in var.name:
                print(var.name)
                losses.append(tf.nn.l2_loss(var))

        return tf.add_n(losses) * self.weight_decay_rate

    def loss(self, output, labels):
        x = self.post_process(output)
        context_1 = self.post_process(self.context_1)
        context_2 = self.post_process(self.context_2)

        with tf.name_scope("loss"):
            labels = tf.reshape(labels, (-1, 1))
            labels = tf.reshape(tf.one_hot(labels, depth=self.num_classes), (-1, self.num_classes))

            loss_main = self._cross_entropy(x, labels)
            loss_context_1 = self._cross_entropy(context_1, labels) * self.auxiliary_loss_weight
            loss_context_2 = self._cross_entropy(context_2, labels) * self.auxiliary_loss_weight

            loss = loss_main + loss_context_1 + loss_context_2

            weight_decay_loss = tf.losses.get_regularization_loss()
            loss = loss + weight_decay_loss
            tf.summary.scalar("weight_decay", weight_decay_loss)

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
            # resize bilinear
            output = tf.image.resize_bilinear(output, self.image_size, align_corners=True)

            # softmax
            output = tf.nn.softmax(output)
            return output


class LMBiSeNetQuantize(LMBiSeNet):
    """
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
                    if var.op.name.startswith("block_first/"):
                        print("not quantized", var.op.name)
                        return var

                if var.op.name.startswith("block_last/"):
                    print("not quantized", var.op.name)
                    return var

                print("quantized", var.op.name)
                return weight_quantization(var)
        return var

    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)
