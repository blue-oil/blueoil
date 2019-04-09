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
import tensorflow as tf
from matplotlib import cm

from lmnet.networks.base import BaseNetwork


class Base(BaseNetwork):
    """base network for classification

    This base network is for classification.
    Every classification's network class should extend this class.

    """

    def __init__(
            self,
            weight_decay_rate=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.weight_decay_rate = weight_decay_rate

    def placeholderes(self):
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
            tf.bool,
            shape=(self.batch_size, self.num_classes),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        """inference.

        Params:
           images: images tensor. shape is (batch_num, height, width, channel)
        """
        base = self.base(images, is_training)
        softmax = tf.nn.softmax(base)

        self.output = tf.identity(softmax, name="output")
        return self.output

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

    def loss(self, softmax, labels):
        """loss.

        Params:
           output: softmaxed tensor from base. shape is (batch_num, num_classes)
           labels: onehot labels tensor. shape is (batch_num, num_classes)
        """

        with tf.name_scope("loss"):
            labels = tf.to_float(labels)
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)),
                axis=[1]
            )

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
            tf.summary.scalar("cross_entropy", cross_entropy_mean)
            loss = cross_entropy_mean

            if self.weight_decay_rate:
                weight_decay_loss = self._weight_decay_loss()
                tf.summary.scalar("weight_decay", weight_decay_loss)
                loss = loss + weight_decay_loss

            tf.summary.scalar("loss", loss)

            return loss

    def _heatmaps(self, target_feature_map):
        """Generate heatmap from target feature map.

        Args:
            target_feature_map (Tensor): Tensor to be generate heatmap. shape is [batch_size, h, w, num_classes].
        """
        assert target_feature_map.get_shape()[3].value == self.num_classes

        results = []

        # shpae: [batch_size, height, width, num_classes]
        heatmap = tf.image.resize_images(
            target_feature_map, [self.image_size[0], self.image_size[1]],
            method=tf.image.ResizeMethod.BICUBIC,
        )
        epsilon = 1e-10
        # standrization. all element are in the interval [0, 1].
        heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + epsilon)

        for i, class_name in enumerate(self.classes):
            class_heatmap = heatmap[:, :, :, i]
            indices = tf.to_int32(tf.round(class_heatmap * 255))
            color_map = cm.jet
            # Init color map for useing color lookup table(_lut).
            color_map._init()
            colors = tf.constant(color_map._lut[:, :3], dtype=tf.float32)
            # gather
            colored_class_heatmap = tf.gather(colors, indices)
            results.append(colored_class_heatmap)

        return results

    def summary(self, output, labels=None):
        super().summary(output, labels)

        images = self.images if self.data_format == 'NHWC' else tf.transpose(self.images, perm=[0, 2, 3, 1])

        tf.summary.image("input_images", images)

        if hasattr(self, "_heatmap_layer") and isinstance(self._heatmap_layer, tf.Tensor):
            heatmap_layer = self._heatmap_layer if self.data_format == 'NHWC' else tf.transpose(self._heatmap_layer,
                                                                                                perm=[0, 2, 3, 1])
            with tf.variable_scope('heatmap'):
                colored_class_heatmaps = self._heatmaps(heatmap_layer)
                for class_name, colored_class_heatmap in zip(self.classes, colored_class_heatmaps):
                    alpha = 0.1
                    overlap = alpha * images + colored_class_heatmap
                    tf.summary.image(class_name, overlap, max_outputs=1)

    def _calc_top_k(self, softmax, labels, k):
        """Calculate the mean top k accuracy.
        In the case that multiple classes are on the top k boundary, the order of the class indices is used
        to break the tie - lower indices given preference - so that only k predictions are included in the top k.

        Args:
            softmax (Tensor): class predictions from the softmax. Shape is [batch_size, num_classes].
            labels (Tensor): onehot ground truth labels. Shape is [batch_size, num_classes].
            k (Int): number of top predictions to use.
        """

        argmax_labels = tf.cast(tf.argmax(labels, 1), tf.int32)
        argmax_labels = tf.expand_dims(argmax_labels, 1)
        _, top_predicted_indices = tf.nn.top_k(softmax, k)
        accuracy_topk, accuracy_topk_update = tf.metrics.mean(
            tf.cast(tf.reduce_any(tf.equal(top_predicted_indices, argmax_labels), axis=1), tf.float32)
        )
        return accuracy_topk, accuracy_topk_update

    def metrics(self, softmax, labels):
        """metrics.

        Params:
           softmax: probabilities applied softmax. shape is (batch_num, num_classes)
           labels: onehot labels tensor. shape is (batch_num, num_classes)
        """
        with tf.name_scope("metrics_calc"):
            labels = tf.to_float(labels)

            if self.is_debug:
                labels = tf.Print(labels, [tf.shape(labels), tf.argmax(labels, 1)], message="labels:", summarize=200)
                softmax = tf.Print(softmax,
                                   [tf.shape(softmax), tf.argmax(softmax, 1)], message="softmax:", summarize=200)

            accuracy, accuracy_update = self._calc_top_k(softmax, labels, k=1)

            if(self.num_classes > 3):
                accuracy_top3, accuracy_top3_update = self._calc_top_k(softmax, labels, k=3)
            else:
                accuracy_top3, accuracy_top3_update = tf.metrics.mean(tf.ones(self.batch_size))

            if(self.num_classes > 5):
                accuracy_top5, accuracy_top5_update = self._calc_top_k(softmax, labels, k=5)
            else:
                accuracy_top5, accuracy_top5_update = tf.metrics.mean(tf.ones(self.batch_size))

            updates = tf.group(accuracy_update, accuracy_top3_update, accuracy_top5_update)

        metrics_dict = {"accuracy": accuracy, "accuracy_top3": accuracy_top3, "accuracy_top5": accuracy_top5}
        return metrics_dict, updates
