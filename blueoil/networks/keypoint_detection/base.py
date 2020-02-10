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

from blueoil.networks.base import BaseNetwork
from lmnet.metrics.object_keypoint_similarity import compute_object_keypoint_similarity
from lmnet.post_processor import gaussian_heatmap_to_joints
from lmnet.visualize import visualize_keypoint_detection


class Base(BaseNetwork):
    """base network for keypoint detection

    This base network is for keypoint detection.
    Each keypoint detection network class should extend this class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def placeholders(self):
        if self.data_format == 'NHWC':
            shape = (self.batch_size, self.image_size[0], self.image_size[1], 3)
        else:
            shape = (self.batch_size, 3, self.image_size[0], self.image_size[1])

        images_placeholder = tf.placeholder(tf.float32, shape=shape, name="images_placeholder")
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

    def _colored_heatmaps(self, heatmaps, color, name=""):
        """Visualize heatmaps with given color.

        Args:
            heatmaps: a Tensor of shape (batch_size, height, width, num_joints).
            color: a numpy array of shape (batch_size, 1, 1, num_joints, 3).
            name: str, name to display on tensorboard.

        """
        heatmaps_colored = tf.expand_dims(heatmaps, axis=-1)
        heatmaps_colored *= color
        heatmaps_colored = tf.reduce_sum(heatmaps_colored, axis=3)

        tf.summary.image(name, heatmaps_colored)

    @staticmethod
    def py_post_process(heatmaps, num_dimensions=2, stride=2):
        """Convert from heatmaps to joints, it is mainly used for visualization and metrics in training time.

        Args:
            heatmaps: a numpy array of shape (batch_size, height, width, num_joints).
            num_dimensions: int.
            stride: int, stride = image_height / heatmap_height.

        Returns:
            batch_joints: a numpy array of shape (batch_size, num_joints, 3).

        """
        batch_size = heatmaps.shape[0]
        list_joints = [gaussian_heatmap_to_joints(heatmaps[i], num_dimensions, stride=stride)
                       for i in range(batch_size)]
        return np.stack(list_joints)

    def post_process(self, output):
        """Tensorflow mirror method for py_post_process(),
        it is mainly used for visualization and metrics in training time.

        Args:
            output: a Tensor of shape (batch_size, height, width, num_joints).

        Returns:
            joints: a Tensor of shape (batch_size, num_joints, 3).

        """
        return tf.py_func(self.py_post_process,
                          [output, 2, self.stride],
                          tf.float32)

    @staticmethod
    def py_visualize_output(images, heatmaps, stride=2):
        """Visualize pose estimation, it is mainly used for visualization in training time.

        Args:
            images: a numpy array of shape (batch_size, height, width, 3).
            heatmaps: a numpy array of shape (batch_size, height, width, num_joints).
            stride: int, stride = image_height / heatmap_height.

        Returns:
            drawed_images: a numpy array of shape (batch_size, height, width, 3).

        """
        drawed_images = np.uint8(images * 255.0)

        for i in range(images.shape[0]):
            joints = gaussian_heatmap_to_joints(heatmaps[i], stride=stride)
            drawed_images[i] = visualize_keypoint_detection(drawed_images[i], joints)
        return drawed_images

    def _visualize_output(self, images, output, name="visualize_output"):
        """A tensorflow mirror method for py_visualize_output().

        Args:
            images: a Tensor of shape (batch_size, height, width, 3).
            output: a Tensor of shape (batch_size, height, width, num_joints).
            name: str, name to display on tensorboard.

        """
        drawed_images = tf.py_func(self.py_visualize_output,
                                   [images, output, self.stride],
                                   tf.uint8)
        tf.summary.image(name, drawed_images)

    def _compute_oks(self, output, labels):
        """Compute object keypoint similarity between output and labels.

        Args:
            output: a Tensor of shape (batch_size, height, width, num_joints).
            labels: a Tensor of shape (batch_size, height, width, num_joints).

        Returns:
            oks: a Tensor represents object keypoint similarity.
        """
        joints_gt = self.post_process(labels)
        joints_pred = self.post_process(output)

        return tf.py_func(compute_object_keypoint_similarity,
                          [joints_gt, joints_pred, self.image_size],
                          tf.float32)

    def summary(self, output, labels=None):
        """Summary for tensorboard.

        Args:
            output: a Tensor of shape (batch_size, height, width, num_joints).
            labels: a Tensor of shape (batch_size, height, width, num_joints).

        """
        images = self.images if self.data_format == 'NHWC' else tf.transpose(self.images, perm=[0, 2, 3, 1])
        tf.summary.image("input", images)

        color = np.random.randn(1, 1, 1, self.num_joints, 3)

        self._colored_heatmaps(labels, color, name="labels_heatmap")
        self._colored_heatmaps(output, color, name="output_heatmap")

        self._visualize_output(images, labels, "visualize_labels")
        self._visualize_output(images, output, "visualize_output")

    def metrics(self, output, labels):
        """Compute metrics for single-person pose estimation task.

        Args:
            output: a Tensor of shape (batch_size, height, width, num_joints).
            labels: a Tensor of shape (batch_size, height, width, num_joints).

        Returns:
            results: a dict, {metric_name: metric_tensor}.
            updates_op: an operation that increments the total and count variables appropriately.

        """
        output = output if self.data_format == 'NHWC' else tf.transpose(output, perm=[0, 2, 3, 1])
        oks = self._compute_oks(output, labels)

        results = {}
        mean_oks, update_oks = tf.metrics.mean(oks)
        updates = [update_oks]
        updates_op = tf.group(*updates)
        results["mean_object_keypoint_similarity"] = mean_oks
        return results, updates_op
