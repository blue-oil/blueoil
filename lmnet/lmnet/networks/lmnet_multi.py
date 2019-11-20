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

from lmnet.metrics.metrics import tp_tn_fp_fn, tp_tn_fp_fn_for_each


def safe_log(arg):
    return tf.math.log(tf.clip_by_value(arg, 1e-10, 1.0))


# TODO(wakisaka): WIP
class LmnetMulti:
    """Multi label prediction"""
    version = 0.01

    @property
    def placeholders(self):
        """placeholders"""

        images_placeholder = tf.compat.v1.placeholder(
            tf.float32,
            shape=(self.batch_size, self.image_size[0], self.image_size[1], 3),
            name="images_placeholder")

        labels_placeholder = tf.compat.v1.placeholder(
            tf.bool,
            shape=(self.batch_size, self.num_classes),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        """inference.

        Args:
            images: images tensor. shape is (batch_num, height, width, channel)
            is_training:

        """
        base = self.base(images, is_training)
        softmax = tf.sigmoid(base)

        self.output = tf.identity(softmax, name="output")
        return self.output

    def loss(self, output, labels):
        """loss.

        Args:
            output: network output sigmoided tensor.
            labels:  multi label encoded tensor. shape is (batch_num, num_classes)

        """

        with tf.name_scope("loss"):
            labels = tf.cast(labels, tf.float32)
            if self.is_debug:
                labels = tf.Print(labels, [tf.shape(labels), tf.argmax(labels, 1)], message="labels:", summarize=200)
                output = tf.Print(output, [tf.shape(output), tf.argmax(output, 1)], message="output:", summarize=200)

            loss = tf.reduce_mean(
                - tf.reduce_sum(
                    (labels * safe_log(output)) + ((1 - labels) * safe_log(1 - output)),
                    axis=[1],
                )
            )

            tf.compat.v1.summary.scalar("loss", loss)

        return loss

    def metrics(self, output, labels, thresholds=[0.3, 0.5, 0.7]):
        self.metrics_for_each_class(output, labels, thresholds)
        with tf.name_scope("metrics"):
            for threshold in thresholds:
                tp, tn, fp, fn = tp_tn_fp_fn(output, labels, threshold=threshold)
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                tf.compat.v1.summary.scalar("accuracy/prob_{}".format(threshold), accuracy)

                recall = (tp) / (tp + fn)

                tf.compat.v1.summary.scalar("recall/prob_{}".format(threshold), recall)

                precision = (tp) / (tp + fp)

                tf.compat.v1.summary.scalar("precision/prob_{}".format(threshold), precision)
        return accuracy

    def metrics_for_each_class(self, output, labels, thresholds=[0.3, 0.5, 0.7]):
        with tf.name_scope("metrics"):
            for threshold in thresholds:
                tp_tn_fp_fn = tp_tn_fp_fn_for_each(output, labels, threshold=threshold)
                for label_i in range(len(self.classes)):
                    tp = tf.gather(tf.gather(tp_tn_fp_fn, 0), label_i)
                    tn = tf.gather(tf.gather(tp_tn_fp_fn, 1), label_i)
                    fp = tf.gather(tf.gather(tp_tn_fp_fn, 2), label_i)
                    fn = tf.gather(tf.gather(tp_tn_fp_fn, 3), label_i)
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                    tf.compat.v1.summary.scalar(
                        "accuracy/prob_{}/{}".format(threshold, self.classes[label_i]),
                        accuracy
                    )

                    recall = (tp) / (tp + fn)

                    tf.compat.v1.summary.scalar("recall/prob_{}/{}".format(threshold, self.classes[label_i]), recall)

                    precision = (tp) / (tp + fp)
                    tf.compat.v1.summary.scalar(
                        "precision/prob_{}/{}".format(threshold, self.classes[label_i]),
                        precision
                    )
        return accuracy
