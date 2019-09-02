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
from .flowlib import flow_to_image


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

        # TODO PyCharm warning. I think we should define self.images first here. Check other networks.
        self.images = None
        self.activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")
        self.weight_decay_rate = 0.0004
        self.use_batch_norm = True
        self.custom_getter = None
        # TODO Where should I put the c files and where do we compile custom ops?
        self.downsample_so = tf.load_op_library(
            tf.resource_loader.get_path_to_datafile("downsample.so")
        )

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
                use_bias=True,
                biases_initializer=None,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay_rate)
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

    def _upsample_flow(
            self,
            name,
            inputs
    ):
        # TODO Think: tf uses bias but pytorch did not
        with tf.variable_scope(name):
            return tf.layers.conv2d_transpose(
                inputs,
                2,
                kernel_size=4,
                strides=2,
                padding='SAME',
                use_bias=False
            )

    def _downsample(
            self,
            name,
            inputs,
            size
    ):
        with tf.variable_scope(name):
            return self.downsample_so.downsample(inputs, size)

    def _average_endpoint_error(
            self,
            output,
            labels
    ):
        """
        Given labels and outputs of size (batch_size, height, width, 2), calculates average endpoint error:
            sqrt{sum_across_the_2_channels[(X - Y)^2]}
        """
        batch_size = output.get_shape().as_list()[0]
        with tf.name_scope(None, "average_endpoint_error", (output, labels)):
            # TODO I don't think the two lines below is necessary.
            # output = tf.to_float(output)
            # labels = tf.to_float(labels)
            output.get_shape().assert_is_compatible_with(labels.get_shape())

            squared_difference = tf.square(tf.subtract(output, labels))
            # sum across the 2 channels: sum[(X - Y)^2] -> N, H, W, 1
            loss = tf.reduce_sum(squared_difference, axis=3, keep_dims=True)
            loss = tf.sqrt(loss)
            return tf.reduce_sum(loss) / batch_size

    def _contractive_block(self, images, is_training):
        # TODO tf version uses padding=VALID and pad to match the original caffe code.
        # Can DLK handle this?
        # pytorch version uses (kernel_size-1) // 2, which is equal to 'SAME' in tf
        x = self._conv_bn_act('conv1', images, 64, is_training, kernel_size=7, strides=2)
        conv2 = self._conv_bn_act('conv2', x, 128, is_training, kernel_size=5, strides=2)
        x = self._conv_bn_act('conv3', x, 256, is_training, kernel_size=5, strides=2)
        conv3_1 = self._conv_bn_act('conv3_1', x, 256, is_training)
        x = self._conv_bn_act('conv4', conv3_1, 512, is_training, strides=2)
        conv4_1 = self._conv_bn_act('conv4_1', x, 512, is_training)
        x = self._conv_bn_act('conv5', conv4_1, 512, is_training, strides=2)
        conv5_1 = self._conv_bn_act('conv5_1', x, 512, is_training) # 12x16
        x = self._conv_bn_act('conv6', conv5_1, 1024, is_training, strides=2) # 12x16
        conv6_1 = self._conv_bn_act('conv6_1', x, 1024, is_training) # 6x8

        return {
            'conv2': conv2,
            'conv3_1': conv3_1,
            'conv4_1': conv4_1,
            'conv5_1': conv5_1,
            'conv6_1': conv6_1,
        }

    def _refinement_block(self, images, conv_dict, is_training):
        predict_flow6 = self._predict_flow('predict_flow6', conv_dict['conv6_1'])
        upsample_flow6 = self._upsample_flow('upsample_flow6', predict_flow6)
        deconv5 = self._deconv('deconv5', conv_dict['conv6_1'], 512)

        # Same order as pytorch and tf
        concat5 = tf.concat([conv_dict['conv5_1'], deconv5, upsample_flow6], axis=3)
        predict_flow5 = self._predict_flow('predict_flow5', concat5)
        upsample_flow5 = self._upsample_flow('upsample_flow5', predict_flow5)
        deconv4 = self._deconv('deconv4', concat5, 256)

        concat4 = tf.concat([conv_dict['conv4_1'], deconv4, upsample_flow5], axis=3)
        predict_flow4 = self._predict_flow('predict_flow4', concat4)
        upsample_flow4 = self._upsample_flow('upsample_flow4', predict_flow4)
        deconv3 = self._deconv('deconv3', concat4, 256)

        concat3 = tf.concat([conv_dict['conv3_1'], deconv3, upsample_flow4], axis=3)
        predict_flow3 = self._predict_flow('predict_flow3', concat3)
        upsample_flow3 = self._upsample_flow('upsample_flow3', predict_flow3)
        deconv2 = self._deconv('deconv2', concat3, 256)

        concat2 = tf.concat([conv_dict['conv_2'], deconv2, upsample_flow3], axis=3)
        predict_flow2 = self._predict_flow('predict_flow2', concat2)

        # TODO why * 20.0? Wait for issue or email reply
        predict_flow2 = predict_flow2 * 20.0

        # TODO should we move upsampling to post-process?
        # TODO Reason not to move: we need variable flow for both training (for tf.summary) and not training.
        _, height, width, _ = images.get_shape().as_list()
        # Reasons to use align_corners=True:
        # https://stackoverflow.com/questions/51077930/tf-image-resize-bilinear-when-align-corners-false
        # https://github.com/tensorflow/tensorflow/issues/6720#issuecomment-298190596
        flow = tf.image.resize_bilinear(predict_flow2, tf.stack[height, width], align_corners=True)

        # TODO Check if returning dict causes memory error. Maybe we can return a tensor when not training?
        if is_training:
            return {
                'predict_flow6': predict_flow6,
                'predict_flow5': predict_flow5,
                'predict_flow4': predict_flow4,
                'predict_flow3': predict_flow3,
                'predict_flow2': predict_flow2,
                'flow': flow
            }
        else:
            return {
                'flow': flow
            }

    # TODO Perhaps we don't need this.
    # def _post_process(self, predict_flow2, height, width):
    #     pass

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
        return self._refinement_block(images, conv_dict, is_training)

    def metrics(self, output, labels):
        """Metrics.

        Args:
            output: dict of tensors from inference.
            labels: labels tensor.
        """
        # TODO check if pass breaks the code
        # TODO would be nice to return the EPE of all flow predictions. Think how to do so with loss function.
        pass

    def summary(self, output, labels=None):
        """Summary.

        Args:
            output: dict of tensors from inference.
            labels: labels tensor.
        """
        super().summary(output, labels)

        images = self.images if self.data_format == 'NHWC' else tf.transpose(self.images, perm=[0, 2, 3, 1])

        # Visualize input images in TensorBoard.
        # Split a batch of two stacked images into two batch of unstacked, separate images.
        images_a, images_b = tf.split(images, [3, 3], 3)
        tf.summary.image("input_images_a", images_a, max_outputs=2)
        tf.summary.image("input_images_b", images_b, max_outputs=2)

        # Visualize output flow in TensorBoard with color encoding.
        output_flow_0 = output['flow'][0, :, :, :]
        output_flow_0 = tf.py_func(flow_to_image, [output_flow_0], tf.uint8)
        output_flow_1 = output['flow'][1, :, :, :]
        output_flow_1 = tf.py_func(flow_to_image, [output_flow_1], tf.uint8)
        output_flow_img = tf.stack([output_flow_0, output_flow_1], 0)
        tf.summary.image('output_flow', output_flow_img, max_outputs=2)

        # Visualize labels flow in TensorBoard with color encoding.
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

        shape = (self.batch_size, self.image_size[0], self.image_size[1], 3) \
            if self.data_format == 'NHWC' else (self.batch_size, 3, self.image_size[0], self.image_size[1])
        images_placeholder = tf.placeholder(
            tf.float32,
            shape=shape,
            name="images_placeholder")

        labels_placeholder = tf.placeholder(
            # TODO check dataloader.py. I think it should be float32
            tf.float32,
            shape=(self.batch_size, self.image_size[0], self.image_size[1], 2),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        base_dict = self.base(images, is_training)
        # TODO why do we need tf.identity? Perhaps for adding a name.
        # TODO train.py and predict.py both call inference => Is returning dict (train) and tensor (predict) okay?
        if is_training:
            # TODO I don't think we need tf.identity here because the tensors are already named.
            # return {k: tf.identity(v, name=k) for k, v in base_dict.items()}
            return base_dict
        else:
            flow = base_dict["flow"]
            return tf.identity(flow, name="output")

    # TODO make it a function.
    def loss(self, output, labels):
        """loss.

        Params:
           output: A dictionary of tensors.
           Each tensor is a network output. shape is (batch_size, output_height, output_width, num_classes).
           labels: Tensor of optical flow labels. shape is (batch_size, height, width, 2).
        """

        losses = []

        # L2 loss between predict_flow6 (weighted w/ 0.32)
        predict_flow6 = output['predict_flow6']
        size = [predict_flow6.shape[1], predict_flow6.shape[2]]
        downsampled_flow6 = self._downsample(labels, size)
        avg_epe_predict_flow6 = self._average_endpoint_error(downsampled_flow6, predict_flow6)
        tf.summary.scalar("avg_epe_predict_flow6", avg_epe_predict_flow6)
        losses.append(avg_epe_predict_flow6)

        # L2 loss between predict_flow5 (weighted w/ 0.08)
        predict_flow5 = output['predict_flow5']
        size = [predict_flow5.shape[1], predict_flow5.shape[2]]
        downsampled_flow5 = self._downsample(labels, size)
        avg_epe_predict_flow5 = self._average_endpoint_error(downsampled_flow5, predict_flow5)
        tf.summary.scalar("avg_epe_predict_flow5", avg_epe_predict_flow5)
        losses.append(avg_epe_predict_flow5)

        # L2 loss between predict_flow4 (weighted w/ 0.02)
        predict_flow4 = output['predict_flow4']
        size = [predict_flow4.shape[1], predict_flow4.shape[2]]
        downsampled_flow4 = self._downsample(labels, size)
        avg_epe_predict_flow4 = self._average_endpoint_error(downsampled_flow4, predict_flow4)
        tf.summary.scalar("avg_epe_predict_flow4", avg_epe_predict_flow4)
        losses.append(avg_epe_predict_flow4)

        # L2 loss between predict_flow3 (weighted w/ 0.01)
        predict_flow3 = output['predict_flow3']
        size = [predict_flow3.shape[1], predict_flow3.shape[2]]
        downsampled_flow3 = self._downsample(labels, size)
        avg_epe_predict_flow3 = self._average_endpoint_error(downsampled_flow3, predict_flow3)
        tf.summary.scalar("avg_epe_predict_flow3", avg_epe_predict_flow3)
        losses.append(avg_epe_predict_flow3)

        # L2 loss between predict_flow2 (weighted w/ 0.005)
        predict_flow2 = output['predict_flow2']
        size = [predict_flow2.shape[1], predict_flow2.shape[2]]
        downsampled_flow2 = self._downsample(labels, size)
        avg_epe_predict_flow2 = self._average_endpoint_error(downsampled_flow2, predict_flow2)
        tf.summary.scalar("avg_epe_predict_flow2", avg_epe_predict_flow2)
        losses.append(avg_epe_predict_flow2)

        # TODO put weight in config file
        # This adds the weighted loss to the loss collection
        weighted_epe = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])
        tf.summary.scalar("weighted_epe", weighted_epe)

        # Return the total loss: weighted epe + regularization terms defined in the base function
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar("total_loss", total_loss)
        return total_loss
