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
import numpy as np

from lmnet.networks.base import BaseNetwork
from lmnet.metrics.object_keypoint_similarity import compute_oks_batch
from lmnet.post_processor import gaussian_heatmap_to_joints
from lmnet.visualize import visualize_pose_estimation


class Base(BaseNetwork):
    """base network for keypoints detection

    This base network is for keypoints detection.
    Each keypoints detection network class should extend this class.

    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def placeholderes(self):
        shape = (self.batch_size, self.image_size[0], self.image_size[1], 3) \
            if self.data_format == 'NHWC' else (self.batch_size, 3, self.image_size[0], self.image_size[1])
        images_placeholder = tf.placeholder(
            tf.float32,
            shape=shape,
            name="images_placeholder")
        labels_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.batch_size,
                   self.image_size[0] // self.stride, self.image_size[1] // self.stride,
                   self.num_joints),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        base = self.base(images, is_training)
        return tf.identity(base, name="output")

    def _colored_heatmaps(self, heatmaps, name=""):

        heatmaps_colored = tf.expand_dims(heatmaps, axis=-1)
        heatmaps_colored *= self.color
        heatmaps_colored = tf.reduce_sum(heatmaps_colored, axis=3)

        tf.summary.image(name, heatmaps_colored)

    @staticmethod
    def py_post_process(heatmaps, num_dimensions=2):
        batch_size = heatmaps.shape[0]
        list_joints = []

        for i in range(batch_size):
            joints = gaussian_heatmap_to_joints(heatmaps[i], num_dimensions, stride=2)
            list_joints.append(joints)

        batch_joints = np.stack(list_joints)

        return batch_joints

    def post_process(self, output):

        joints = tf.py_func(self.py_post_process,
                            [output],
                            tf.float32)

        return joints

    def _visualize_output(self, images, output, name="visualize_output"):

        drawed_images = tf.py_func(visualize_pose_estimation,
                                   [images, output],
                                   tf.uint8)

        tf.summary.image(name, drawed_images)

    def _compute_oks(self, output, labels):

        joints_gt = self.post_process(labels)
        joints_pred = self.post_process(output)

        oks = tf.py_func(compute_oks_batch,
                         [joints_gt, joints_pred],
                         tf.float32)

        return oks

    def summary(self, output, labels=None):
        images = self.images if self.data_format == 'NHWC' else tf.transpose(self.images, perm=[0, 2, 3, 1])
        tf.summary.image("input", images)

        self.color = np.random.randn(1, 1, 1, 17, 3)

        self._colored_heatmaps(labels, name="labels_heatmap")
        self._colored_heatmaps(output, name="output_heatmap")

        self._visualize_output(images, labels, "visualize_labels")
        self._visualize_output(images, output, "visualize_output")

    def metrics(self, output, labels):

        output = output if self.data_format == 'NHWC' else tf.transpose(output, perm=[0, 2, 3, 1])

        oks = self._compute_oks(output, labels)

        results = {}
        updates = []

        mean_oks, update_oks = tf.metrics.mean(oks)

        updates.append(update_oks)

        updates_op = tf.group(*updates)

        results["mean_oks"] = mean_oks

        return results, updates_op
