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

from lmnet.networks.base import BaseNetwork
from .flow_to_image import flow_to_image


class LmFlowNet(BaseNetwork):
    """
    LmFlowNet for optical flow estimation.
    """
    version = 1.00

    def __init__(
            self, *args, weight_decay_rate=0.0004,
            div_flow=20.0, conv_depth=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.images = None
        self.base_dict = None
        self.activation_first_layer = lambda x: tf.nn.leaky_relu(
            x, alpha=0.1, name="leaky_relu")
        self.activation = lambda x: tf.nn.leaky_relu(
            x, alpha=0.1, name="leaky_relu")
        self.activation_before_last_layer = self.activation
        self.weight_decay_rate = weight_decay_rate
        self.div_flow = div_flow
        self.conv_depth = conv_depth
        self.use_batch_norm = True
        self.custom_getter = None

    def _space_to_depth(self, name, inputs, block_size):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(
                inputs, perm=[self.data_format.find(d) for d in 'NHWC'])

        output = tf.space_to_depth(
            inputs, block_size=block_size, name=name + "_space_to_depth")

        if self.data_format != 'NHWC':
            output = tf.transpose(
                output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def _conv_bn_act(
            self, name, inputs, filters, is_training,
            kernel_size=3, strides=1, enable_detail_summary=False,
            activation=None):
        if self.data_format == "NCHW":
            channel_data_format = "channels_first"
        elif self.data_format == "NHWC":
            channel_data_format = "channels_last"
        else:
            raise ValueError(
                "data format must be 'NCHW' or 'NHWC'. got {}.".format(
                    self.data_format))

        if strides > 1:
            _, _, _, channels = inputs.get_shape().as_list()
            q, r = divmod(channels, 8)
            if r != 0:
                inputs = tf.layers.conv2d(
                    inputs,
                    filters=8 * (q + 1),
                    kernel_size=1,
                    padding='SAME',
                )
            inputs = self._space_to_depth(name, inputs, strides)
            strides = 1

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
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    self.weight_decay_rate)
            )

            if self.use_batch_norm:
                batch_normed = tf.contrib.layers.batch_norm(
                    conved,
                    decay=0.99,
                    scale=True,
                    center=True,
                    updates_collections=None,
                    is_training=is_training,
                    data_format=self.data_format,
                )
            else:
                batch_normed = conved

            if activation is None:
                output = self.activation(batch_normed)
            else:
                output = activation(batch_normed)

            if enable_detail_summary:
                tf.summary.histogram('conv_output', conved)
                tf.summary.histogram('batch_norm_output', batch_normed)
                tf.summary.histogram('output', output)

            return output

    def _deconv(self, name, inputs, filters, is_training, activation=None):
        # The paper and pytorch used LeakyReLU(0.1,inplace=True) but tf did not.
        # I decide to still use it.
        with tf.variable_scope(name):
            # tf only allows 'SAME' or 'VALID' padding.
            # In conv2d_transpose, h = h1 * stride if padding == 'Same'
            # https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
            # TODO in flownet2-tf, he typed 'biases_initializer'=None. I don't know if it worked.

            _, height, width, _ = inputs.get_shape().as_list()

            inputs = tf.image.resize_nearest_neighbor(
                inputs, (height * 2, width * 2), align_corners=True, name=name)

            conved = tf.layers.conv2d(
                inputs,
                filters,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bias=True,
                bias_initializer=None,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    self.weight_decay_rate)
            )

            if self.use_batch_norm:
                batch_normed = tf.contrib.layers.batch_norm(
                    conved,
                    decay=0.99,
                    scale=True,
                    center=True,
                    updates_collections=None,
                    is_training=is_training,
                    data_format=self.data_format,
                )
            else:
                batch_normed = conved

            if activation is None:
                output = self.activation(batch_normed)
            else:
                output = activation(batch_normed)
            return output

    def _predict_flow(self, name, inputs, is_training, activation=None):
        with tf.variable_scope(name):
            # pytorch uses padding = 1 = (3 -1) // 2. So it is 'SAME'.
            conved = tf.layers.conv2d(
                inputs,
                2,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bias=True
            )
            return conved

    def _upsample_flow(self, name, inputs, is_training, activation=None):
        # TODO Think: tf uses bias but pytorch did not
        with tf.variable_scope(name):

            _, height, width, _ = inputs.get_shape().as_list()

            inputs = tf.image.resize_nearest_neighbor(
                inputs, (height * 2, width * 2), align_corners=True, name=name)

            conved = tf.layers.conv2d(
                inputs,
                2,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bias=False
            )

            batch_normed = tf.contrib.layers.batch_norm(
                conved,
                decay=0.99,
                scale=True,
                center=True,
                updates_collections=None,
                is_training=is_training,
                data_format=self.data_format,
            )

            if activation is None:
                output = self.activation(batch_normed)
            else:
                output = activation(batch_normed)
            return output

    def _downsample(self, name, inputs, size):
        with tf.variable_scope(name):
            return tf.image.resize_images(
                inputs, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _average_endpoint_error(self, output, labels):
        """
        Given labels and outputs of size (batch_size, height, width, 2),
            calculates average endpoint error:
            sqrt{sum_across_the_2_channels[(X - Y)^2]}
        """
        # batch_size, height, width, _ = output.get_shape().as_list()
        with tf.name_scope(None, "average_endpoint_error", (output, labels)):
            output.get_shape().assert_is_compatible_with(labels.get_shape())

            squared_difference = tf.square(tf.subtract(output, labels))
            loss = tf.reduce_sum(squared_difference, axis=3, keepdims=True)
            loss = tf.sqrt(loss)
            return tf.reduce_mean(loss)

    def _contractive_block(self, images, is_training):
        # NOTE: tf version uses padding=VALID and pad
        # to match the original caffe code.
        contractive_dict = {}
        default_conv_channel = [64, 128, 256, 512, 512, 1024, 1024, 1024]
        conv1 = self._conv_bn_act(
            'conv1', images, default_conv_channel[0], is_training, strides=2,
            activation=self.activation_first_layer)
        conv2 = self._conv_bn_act(
            'conv2', conv1, default_conv_channel[1], is_training, strides=2,
            activation=self.activation_before_last_layer)
        contractive_dict['conv2'] = conv2

        conv_pre = conv2
        for _ in range(2, self.conv_depth):
            name1 = 'conv{}'.format(_ + 1)
            name2 = 'conv{}_1'.format(_ + 1)
            num_channel = default_conv_channel[_]
            convx = self._conv_bn_act(
                name1, conv_pre, num_channel, is_training, strides=2)
            contractive_dict[name2] = self._conv_bn_act(
                name2, convx, num_channel, is_training)
            conv_pre = contractive_dict[name2]
        return contractive_dict

    def _refinement_block(self, images, conv_dict, is_training):
        refinement_dict = {}
        concat = None
        default_deconv_channel = [0, 0, 64, 128, 256, 512, 512, 512]
        for _ in reversed(range(2, self.conv_depth)):
            conv_name = 'conv{}_1'.format(_ + 1)
            predict_name = 'predict_flow{}'.format(_ + 1)
            upsample_name = 'upsample_flow{}'.format(_ + 1)
            deconv_name = 'deconv{}'.format(_)
            num_channel = default_deconv_channel[_]
            if concat is None:
                concat = conv_dict[conv_name]
            else:
                concat = tf.concat(
                    [conv_dict[conv_name], deconv, upsample_flow], axis=3)
            refinement_dict[predict_name] = self._predict_flow(
                predict_name, concat, is_training)
            upsample_flow = self._upsample_flow(
                upsample_name, refinement_dict[predict_name], is_training)
            deconv = self._deconv(
                deconv_name, concat, num_channel, is_training)

        concat = tf.concat(
            [conv_dict['conv2'], deconv, upsample_flow], axis=3)
        _, _, _, channels = concat.get_shape().as_list()
        cast_conv = tf.layers.conv2d(
            concat,
            channels,
            kernel_size=1,
            strides=1,
            padding='SAME',
            name='cast_conv',
        )

        predict_flow2 = self._predict_flow(
            'predict_flow2', cast_conv, is_training)
        predict_flow2 = predict_flow2 * self.div_flow
        refinement_dict['predict_flow2'] = predict_flow2

        # TODO should we move upsampling to post-process?
        # Reason not to move: we need variable flow for both training (for tf.summary) and not training.
        _, height, width, _ = images.get_shape().as_list()

        # Reasons to use align_corners=True:
        # https://stackoverflow.com/questions/51077930/tf-image-resize-bilinear-when-align-corners-false
        # https://github.com/tensorflow/tensorflow/issues/6720#issuecomment-298190596
        flow = tf.image.resize_nearest_neighbor(
            predict_flow2, (height // 2, width // 2), align_corners=True)
        flow = tf.image.resize_nearest_neighbor(
            flow, (height, width), align_corners=True)
        refinement_dict['flow'] = flow
        return refinement_dict

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images. shape is (batch_size, height, width, 6)
            is_training: A flag for if is training.
        Returns:
            A dictionary of tf.Tensors.
        """
        self.images = images
        conv_dict = self._contractive_block(images, is_training)
        self.base_dict = self._refinement_block(images, conv_dict, is_training)
        return self.base_dict["flow"]

    def metrics(self, output, labels):
        """Metrics.

        Args:
            output: dict of tensors from inference.
            labels: labels tensor.
        """
        # TODO Check if this is okay
        metrics_ops_dict = {
            "no_op": tf.no_op(name=None)
        }
        metrics_update_op = tf.no_op(name=None)
        return metrics_ops_dict, metrics_update_op

    def summary(self, output, labels=None):
        """Summary.

        Args:
            output: dict of tensors from inference.
            labels: labels tensor.
        """
        super().summary(output, labels)

        images = self.images if self.data_format == 'NHWC' else tf.transpose(
            self.images, perm=[0, 2, 3, 1])

        # Visualize input images in TensorBoard.
        # Split a batch of two stacked images into two batch of unstacked, separate images.
        # We visualize the first and second image in each batch, so max_outputs is set to 2.
        images_a, images_b = tf.split(images, [3, 3], 3)
        tf.summary.image("input_images_a", images_a, max_outputs=2)
        tf.summary.image("input_images_b", images_b, max_outputs=2)

        # Visualize output flow in TensorBoard with color encoding.
        # We visualize the first (0) and second (1) flow in each batch.
        output_flow_0 = output[0, :, :, :]
        output_flow_0 = tf.py_func(flow_to_image, [output_flow_0], tf.uint8)
        output_flow_1 = output[1, :, :, :]
        output_flow_1 = tf.py_func(flow_to_image, [output_flow_1], tf.uint8)
        output_flow_img = tf.stack([output_flow_0, output_flow_1], 0)
        tf.summary.image('output_flow', output_flow_img, max_outputs=2)

        # Visualize labels flow in TensorBoard with color encoding.
        # We visualize the first (0) and second (1) flow in each batch.
        labels_flow_0 = labels[0, :, :, :]
        labels_flow_0 = tf.py_func(flow_to_image, [labels_flow_0], tf.uint8)
        labels_flow_1 = labels[1, :, :, :]
        labels_flow_1 = tf.py_func(flow_to_image, [labels_flow_1], tf.uint8)
        labels_flow_img = tf.stack([labels_flow_0, labels_flow_1], 0)
        tf.summary.image('labels_flow', labels_flow_img, max_outputs=2)

    def placeholders(self):
        """Placeholders.

        Return placeholders.

        Returns:
            tf.placeholder: Placeholders.
        """
        shape = (self.batch_size, self.image_size[0], self.image_size[1], 6) \
            if self.data_format == 'NHWC' else (self.batch_size, 6, self.image_size[0], self.image_size[1])
        images_placeholder = tf.placeholder(
            tf.float32,
            shape=shape,
            name="images_placeholder")

        labels_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.image_size[0], self.image_size[1], 2),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        flow = self.base(images, is_training)
        # TODO why do we need tf.identity? Perhaps for adding a name.
        return tf.identity(flow, name="output")

    # TODO the _output is not used because we need dictionary from self.base_dict
    # _output can only be a tensor, which is the flow
    def loss(self, _output, labels):
        """loss.

        Params:
           output: A dictionary of tensors.
           Each tensor is a network output. shape is (batch_size, output_height, output_width, num_classes).
           labels: Tensor of optical flow labels. shape is (batch_size, height, width, 2).
        """

        losses = []
        base_dict = self.base_dict

        default_weight_list = [0, 0.005, 0.01, 0.02, 0.08, 0.32, 0.32, 0.32]
        for _ in range(1, self.conv_depth):
            ave_epe_name = "ave_epe_predict_flow{}".format(_ + 1)
            predict_name = "predict_flow{}".format(_ + 1)
            downsampled_name = "downsampled_flow{}".format(_ + 1)

            predict_flow = base_dict[predict_name]
            size = [predict_flow.shape[1], predict_flow.shape[2]]
            downsampled_flow = self._downsample(
                downsampled_name, labels, size)
            losses.append(self._average_endpoint_error(
                downsampled_flow, predict_flow))
            tf.summary.scalar(ave_epe_name, losses[-1])

        weighted_epe = tf.losses.compute_weighted_loss(
            losses, default_weight_list[1:self.conv_depth])

        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar("total_loss", total_loss)
        return total_loss


class LmFlowNetQuantized(LmFlowNet):
    """ Quantized LmFlowNet V1.
    """

    def __init__(
            self,
            quantize_first_convolution=False,
            quantize_last_convolution=False,
            quantize_activation_before_last_layer=False,
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
        self.quantize_activation_before_last_layer = quantize_activation_before_last_layer

        activation_quantizer_kwargs = activation_quantizer_kwargs if not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if not None else {}

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        self.weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.activation = activation_quantizer(**activation_quantizer_kwargs)

        if quantize_activation_before_last_layer:
            self.activation_before_last_layer = self.activation
        else:
            self.activation_before_last_layer = lambda x: tf.nn.leaky_relu(
                x, alpha=0.1, name="leaky_relu")

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
                    if var.op.name.startswith("conv1/"):
                        return var

                if not quantize_last_convolution:
                    if var.op.name.startswith("predict_flow2/"):
                        return var

                # Apply weight quantize to variable whose last word of name is "kernel".
                quantized_kernel = weight_quantization(var)
                tf.summary.histogram("quantized_kernel", quantized_kernel)
                return quantized_kernel

        return var

    def base(self, images, is_training, *args, **kwargs):
        custom_getter = functools.partial(
            self._quantized_variable_getter,
            self.weight_quantization,
            self.quantize_first_convolution,
            self.quantize_last_convolution,
        )
        with tf.variable_scope("", custom_getter=custom_getter):
            return super().base(images, is_training)
