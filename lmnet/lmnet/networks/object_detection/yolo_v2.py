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
import math
import numpy as np
import tensorflow as tf

from lmnet.blocks import darknet as darknet_block
from lmnet.layers import conv2d, max_pooling2d
from lmnet.metrics.mean_average_precision import tp_fp_in_the_image
from lmnet.metrics.mean_average_precision import average_precision
from lmnet.networks.base import BaseNetwork


# TODO(wakisaka): When update tensorflow version, remove it.
# For tensorflow v1.4 bug, I have to override space to depth grad func.
# The bug fixed at v1.5. When update tensorflow version, remove ovrride map.
# Ref:
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/array_grad.py#L625
# https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/python/ops/array_grad.py#L654
@tf.RegisterGradient("CustomSpaceToDepth")
def _SpaceToDepthGrad(op, grad):
    block_size = op.get_attr("block_size")
    data_format = op.get_attr("data_format")
    return tf.depth_to_space(grad, block_size, data_format=data_format)


# TODO(wakisaka): there are so many duplicates with yolo_v1.py .
# TODO(wakisaka): dynamic image size change.
class YoloV2(BaseNetwork):
    """YOLO version2.

    YOLO v2.
    paper: https://arxiv.org/abs/1612.08242
    """

    def __init__(
            self,
            num_max_boxes=5,
            anchors=[(0.25, 0.25), (0.5, 0.5), (1.0, 1.0)],
            leaky_relu_scale=0.1,
            object_scale=5.0,
            no_object_scale=1.0,
            class_scale=1.0,
            coordinate_scale=1.0,
            loss_iou_threshold=0.6,
            weight_decay_rate=0.0005,
            score_threshold=0.05,
            nms_iou_threshold=0.5,
            nms_max_output_size=100,
            nms_per_class=True,
            seen_threshold=12800,
            is_dynamic_image_size=False,
            use_cross_entropy_loss=True,
            change_base_output=False,
            *args,
            **kwargs
    ):
        """
        Args:
            num_max_boxes: Number of input ground truth boxes size.
            anchors: list of (anchor_w, anchor_h). Anchors are assumed to be parcentage of cell size.
                cell size (image size/32) is the spatial size of the final convolutional features.
            leaky_relu_scale: Scale of leaky relu.
            object_scale: Scale of object loss.
            no_object_scale: Scale of no object loss.
            class_scale: Scale of class loss.
            coordinate_scale: Scale of coordinate loss.
            loss_iou_threshold: Loss iou threshold.
            weight_decay_rate: Decay rate of weight.
            score_threshold: Exculde lower socre boxes than this threshold in post process.
            nms_iou_threshold: Non max suppression IOU threshold in post process.
            nms_max_output_size: Non max suppression's max output boxes number per class in post process.
            seen_threshold: Threshold for encourage predictions to match anchor on training.
            is_dynamic_image_size: Be able to dynamic change image size.
            use_cross_entropy_loss(bool): Use cross entropy loss instead of mean square error of class loss.
            change_base_output(bool): If it is ture, the output of network be activated with softmax and sigmoid.
        """
        super().__init__(
            *args,
            **kwargs
        )

        self.anchors = anchors
        self.boxes_per_cell = len(anchors)
        self.leaky_relu_scale = leaky_relu_scale
        self.num_max_boxes = num_max_boxes
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_max_output_size = nms_max_output_size
        self.nms_per_class = nms_per_class

        self.is_dynamic_image_size = is_dynamic_image_size
        self.change_base_output = change_base_output

        # Assert image size can mod `32`.
        # TODO(wakisaka): Be enable to cnahge `32`. it depends on pooling times.
        assert self.image_size[0] % 32 == 0
        assert self.image_size[1] % 32 == 0

        if self.is_dynamic_image_size:
            self.image_size = tf.tuple([
                tf.constant(self.image_size[0], dtype=tf.int32), tf.constant(self.image_size[1], dtype=tf.int32)
            ])

            # TODO(wakisaka): Be enable to cnahge `32`. it depends on pooling times.
            # Number of cell is the spatial dimension of the final convolutional features.
            self.num_cell = tf.tuple([tf.to_int32(self.image_size[0] / 32), tf.to_int32(self.image_size[1] / 32)])
        else:
            self.num_cell = self.image_size[0] // 32, self.image_size[1] // 32

        self.loss_function = YoloV2Loss(
            is_debug=self.is_debug,
            anchors=self.anchors,
            num_cell=self.num_cell,
            boxes_per_cell=self.boxes_per_cell,
            object_scale=object_scale,
            no_object_scale=no_object_scale,
            class_scale=class_scale,
            coordinate_scale=coordinate_scale,
            loss_iou_threshold=loss_iou_threshold,
            weight_decay_rate=weight_decay_rate,
            image_size=self.image_size,
            batch_size=self.batch_size,
            classes=self.classes,
            yolo=self,
            seen_threshold=seen_threshold,
            use_cross_entropy_loss=use_cross_entropy_loss,
        )

        self.activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")
        self.before_last_activation = self.activation

    def placeholderes(self):
        """placeholders"""

        if self.is_dynamic_image_size:
            # shape is [batch_size, height, width, 3]
            if self.data_format == "NHWC":
                images_placeholder = tf.placeholder(
                    tf.float32,
                    shape=(self.batch_size, None, None, 3),
                    name="images_placeholder")

            if self.data_format == "NCHW":
                images_placeholder = tf.placeholder(
                    tf.float32,
                    shape=(self.batch_size, 3, None, None),
                    name="images_placeholder")

        else:

            if self.data_format == "NHWC":
                images_placeholder = tf.placeholder(
                    tf.float32,
                    shape=(self.batch_size, self.image_size[0], self.image_size[1], 3),
                    name="images_placeholder")

            if self.data_format == "NCHW":
                images_placeholder = tf.placeholder(
                    tf.float32,
                    shape=(self.batch_size, 3, self.image_size[0], self.image_size[1]),
                    name="images_placeholder")

        labels_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.num_max_boxes, 5),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def summary(self, output, labels=None):
        super().summary(output, labels)

        with tf.name_scope("post_process"):
            tf.summary.scalar("score_threshold", self.score_threshold)
            tf.summary.scalar("nms_iou_threshold", self.nms_iou_threshold)
            tf.summary.scalar("nms_max_output_size", self.nms_max_output_size)
            tf.summary.scalar("nms_per_class", tf.to_int32(self.nms_per_class))

        if self.change_base_output:
            predict_classes, predict_confidence, predict_boxes = self._split_predictions(output)
        else:
            predict_classes, predict_confidence, predict_boxes = self._predictions(output)

        tf.summary.histogram("predict_classes", predict_classes)
        tf.summary.histogram("predict_confidence", predict_confidence)

        predict_score = predict_confidence * predict_classes
        tf.summary.histogram("predict_score", predict_score)

        if labels is not None:
            with tf.name_scope("gt_boxes"):
                gt_boxes = self.convert_gt_boxes_xywh_to_cxcywh(labels)

                summary_boxes(
                    "gt_boxes",
                    self.images,
                    format_CXCYWH_to_YX(gt_boxes[:, :, :4], axis=2),
                    self.image_size,
                    max_outputs=3,
                    data_format=self.data_format,
                )

        with tf.name_scope("raw_predict_boxes"):
            if not self.change_base_output:
                predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

            resized_predict_boxes_for_summary = tf.reshape(predict_boxes, [self.batch_size, -1, 4])

            tf.summary.histogram("predict_boxes_center_x", resized_predict_boxes_for_summary[:, :, 0])
            tf.summary.histogram("predict_boxes_center_y", resized_predict_boxes_for_summary[:, :, 1])
            # tf.summary.histogram("predict_boxes_w", resized_predict_boxes_for_summary[:, :, 2])
            # tf.summary.scalar("predict_boxes_max_w", tf.reduce_max(resized_predict_boxes_for_summary[:, :, 2]))
            # tf.summary.histogram("predict_boxes_h", resized_predict_boxes_for_summary[:, :, 3])
            # tf.summary.scalar("predict_boxes_max_h", tf.reduce_max(resized_predict_boxes_for_summary[:, :, 3]))

            summary_boxes(
                "boxes",
                self.images,
                format_CXCYWH_to_YX(resized_predict_boxes_for_summary, axis=2),
                self.image_size,
                data_format=self.data_format,
            )

        with tf.name_scope("final_detect_boxes"):
            detect_boxes_list = self.post_process(output)

            for i in range(self.batch_size):
                detect_boxes = detect_boxes_list[i]
                image = tf.expand_dims(self.images[i, :, :, :], 0)
                boxes = tf.expand_dims(detect_boxes[:, :4], 0)
                summary_boxes(
                    "boxes",
                    image,
                    format_XYWH_to_YX(boxes, axis=2),
                    self.image_size,
                    data_format=self.data_format,
                )

    def metrics(self, output, labels, thresholds=[0.3, 0.5, 0.7]):
        with tf.variable_scope("calc_metrics"):
            detect_boxes = self.post_process(output)
            metrics_ops_dict = {}
            updates = []

            for overlap_thresh in thresholds:
                average_precisions = []
                for class_id, class_name in enumerate(self.classes):
                    tps = []
                    fps = []
                    scores = []
                    num_gt_boxes_list = []

                    for image_index, detect_boxes_in_the_image in enumerate(detect_boxes):
                        mask = tf.equal(detect_boxes_in_the_image[:, 4], class_id)
                        per_class_detect_boxes_in_the_image = tf.boolean_mask(detect_boxes_in_the_image, mask)

                        labels_in_the_image = labels[image_index, :, :]
                        mask = tf.equal(labels_in_the_image[:, 4], class_id)
                        labels_in_the_image = tf.boolean_mask(labels_in_the_image, mask)

                        num_gt_boxes = tf.shape(labels_in_the_image)[0]

                        tp, fp, score = tf.py_func(
                            tp_fp_in_the_image,
                            [per_class_detect_boxes_in_the_image, labels_in_the_image, overlap_thresh],
                            [tf.float32, tf.float32, tf.float32],
                            stateful=False,
                        )

                        tps.append(tp)
                        fps.append(fp)
                        scores.append(score)
                        num_gt_boxes_list.append(num_gt_boxes)

                    tp = tf.concat(tps, axis=0)
                    fp = tf.concat(fps, axis=0)
                    score = tf.concat(scores, axis=0)
                    num_gt_boxes = tf.add_n(num_gt_boxes_list)

                    (average_precision_value, precision_array, recall_array, precision, recall), update_op =\
                        average_precision(num_gt_boxes, tp, fp, score, class_name)

                    updates.append(update_op)

                    average_precisions.append(average_precision_value)

                    if overlap_thresh == 0.5:
                        metrics_key = 'AveragePrecision_{}/{}'.format(overlap_thresh, class_name)
                        metrics_value = average_precision_value
                        metrics_ops_dict[metrics_key] = metrics_value

                        metrics_key = 'Recall_{}/{}'.format(overlap_thresh, class_name)
                        metrics_value = recall
                        metrics_ops_dict[metrics_key] = metrics_value

                        metrics_key = 'Precision_{}/{}'.format(overlap_thresh, class_name)
                        metrics_value = precision
                        metrics_ops_dict[metrics_key] = metrics_value

                metrics_key = 'MeanAveragePrecision_{}'.format(overlap_thresh)
                metrics_value = tf.add_n(average_precisions) / len(self.classes)

                metrics_ops_dict[metrics_key] = metrics_value

        return metrics_ops_dict, tf.group(*updates)

    # TODO(wakisaka): duplication with yolo_v1
    def convert_gt_boxes_xywh_to_cxcywh(self, gt_boxes):
        """Convert gt_boxes format form (x, y, w, h) to (center_x, center_y, w, h).

        Args:
            gt_boxes :3D tensor [batch_size, max_num_boxes, 5(x, y, w, h, class_id)]
        """
        axis = 2
        gt_boxes = tf.to_float(gt_boxes)
        gt_boxes_without_label = gt_boxes[:, :, :4]
        gt_boxes_only_label = tf.reshape(gt_boxes[:, :, 4], [self.batch_size, -1, 1])
        gt_boxes_without_label = format_XYWH_to_CXCYWH(gt_boxes_without_label, axis=axis)

        result = tf.concat(axis=axis, values=[gt_boxes_without_label, gt_boxes_only_label], name="concat_gt_boxes")

        return result

    @staticmethod
    def py_offset_boxes(num_cell_y, num_cell_x, batch_size, boxes_per_cell, anchors):
        """Numpy implementing of offset_boxes.
        Return yolo space offset of x and y and w and h.

        Args:
            num_cell_y: Number of cell y. The spatial dimension of the final convolutional features.
            num_cell_x: Number of cell x. The spatial dimension of the final convolutional features.
            batch_size: int, Batch size.
            boxes_per_cell: int, number of boxes per cell.
            anchors: list of tuples.
        """

        offset_y = np.arange(num_cell_y)
        offset_y = np.reshape(offset_y, (1, num_cell_y, 1, 1))
        offset_y = np.broadcast_to(offset_y, [batch_size, num_cell_y, num_cell_x, boxes_per_cell])

        offset_x = np.arange(num_cell_x)
        offset_x = np.reshape(offset_x, (1, 1, num_cell_x, 1))
        offset_x = np.broadcast_to(offset_x, [batch_size, num_cell_y, num_cell_x, boxes_per_cell])

        w_anchors = [anchor_w for anchor_w, anchor_h in anchors]
        offset_w = np.broadcast_to(w_anchors, (batch_size, num_cell_y, num_cell_x, boxes_per_cell))
        offset_w = offset_w.astype(np.float32)

        h_anchors = [anchor_h for anchor_w, anchor_h in anchors]
        offset_h = np.broadcast_to(h_anchors, (batch_size, num_cell_y, num_cell_x, boxes_per_cell))
        offset_h = offset_h.astype(np.float32)

        return offset_x, offset_y, offset_w, offset_h

    def offset_boxes(self):
        """Return yolo space offset of x and y and w and h.

        Return:
            offset_x: shape is [batch_size, num_cell[0], num_cell[1], boxes_per_cell]
            offset_y: shape is [batch_size, num_cell[0], num_cell[1], boxes_per_cell]
            offset_w: shape is [batch_size, num_cell[0], num_cell[1], boxes_per_cell]
            offset_h: shape is [batch_size, num_cell[0], num_cell[1], boxes_per_cell]
        """

        if self.is_dynamic_image_size:
            result = tf.py_func(self.py_offset_boxes,
                                (self.num_cell[0], self.num_cell[1],
                                 self.batch_size, self.boxes_per_cell, self.anchors),
                                [tf.int64, tf.int64, tf.float32, tf.float32])
            offset_x, offset_y, offset_w, offset_h = result

        else:
            offset_x, offset_y, offset_w, offset_h = self.py_offset_boxes(self.num_cell[0], self.num_cell[1],
                                                                          self.batch_size, self.boxes_per_cell,
                                                                          self.anchors)
        return offset_x, offset_y, offset_w, offset_h

    def convert_boxes_space_from_real_to_yolo(self, boxes):
        """Convert boxes space size from real to yolo.

        Real space boxes coodinates are in the interval [0, image_size].
        Yolo space boxes x,y are in the interval [-1, 1]. w,h are in the interval [-inf, +inf].

        Args:
            boxes: 5D Tensor, shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].

        Returns:
            resized_boxes: 5D Tensor,
                           shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].
        """
        image_size_h, image_size_w = tf.to_float(self.image_size[0]), tf.to_float(self.image_size[1])
        num_cell_y, num_cell_x = tf.to_float(self.num_cell[0]), tf.to_float(self.num_cell[1])

        offset_x, offset_y, offset_w, offset_h = self.offset_boxes()
        offset_x, offset_y = tf.to_float(offset_x), tf.to_float(offset_y)

        resized_boxes = boxes / [
            image_size_w,
            image_size_h,
            image_size_w,
            image_size_h,
        ]

        epsilon = 1e-10

        resized_boxes = tf.stack([
            (resized_boxes[:, :, :, :, 0] * num_cell_x - offset_x),
            (resized_boxes[:, :, :, :, 1] * num_cell_y - offset_y),
            tf.log(resized_boxes[:, :, :, :, 2] * num_cell_x / offset_w + epsilon),
            tf.log(resized_boxes[:, :, :, :, 3] * num_cell_y / offset_h + epsilon),
        ], axis=4)

        return resized_boxes

    def convert_boxes_space_from_yolo_to_real(self, predict_boxes):
        """Convert predict boxes space size from yolo to real.

        Real space boxes coodinates are in the interval [0, image_size].
        Yolo space boxes x,y are in the interval [-1, 1]. w,h are in the interval [-inf, +inf].

        Args:
            boxes: 5D Tensor, shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].

        Returns:
            resized_boxes: 5D Tensor,
                           shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].
        """
        image_size_h, image_size_w = tf.to_float(self.image_size[0]), tf.to_float(self.image_size[1])
        num_cell_y, num_cell_x = tf.to_float(self.num_cell[0]), tf.to_float(self.num_cell[1])

        offset_x, offset_y, offset_w, offset_h = self.offset_boxes()
        offset_x, offset_y = tf.to_float(offset_x), tf.to_float(offset_y)

        resized_predict_boxes = tf.stack([
            (predict_boxes[:, :, :, :, 0] + offset_x) / num_cell_x,
            (predict_boxes[:, :, :, :, 1] + offset_y) / num_cell_y,
            tf.exp(predict_boxes[:, :, :, :, 2]) * offset_w / num_cell_x,
            tf.exp(predict_boxes[:, :, :, :, 3]) * offset_h / num_cell_y,
        ], axis=4)

        resized_predict_boxes = resized_predict_boxes * [
            image_size_w,
            image_size_h,
            image_size_w,
            image_size_h,
        ]

        return resized_predict_boxes

    def _predictions(self, output):
        with tf.name_scope("predictions"):
            if self.data_format == "NCHW":
                output = tf.transpose(output, [0, 2, 3, 1])

            predict_classes, predict_confidence, predict_boxes = self._split_predictions(output)

            # apply activations
            with tf.name_scope("output_activation"):
                predict_classes = tf.nn.softmax(predict_classes)
                predict_confidence = tf.sigmoid(predict_confidence)
                predict_boxes = tf.stack([
                    tf.sigmoid(predict_boxes[:, :, :, :, 0]),
                    tf.sigmoid(predict_boxes[:, :, :, :, 1]),
                    predict_boxes[:, :, :, :, 2],
                    predict_boxes[:, :, :, :, 3],
                ], axis=4)

            return predict_classes, predict_confidence, predict_boxes

    def _split_predictions(self, output):
        """Separate combined final convolution outputs to predictions.

        Args:
            output: combined final convolution outputs 4D Tensor.
                shape is [batch_size, num_cell[0], num_cell[1],  (num_classes + 5) * boxes_per_cell]

        Returns:
            predict_classes(Tensor): [batch_size, num_cell[0], num_cell[1], boxes_per_cell, num_classes]
            predict_confidence(Tensor): [batch_size, num_cell[0], num_cell[1], boxes_per_cell, 1]
            predict_boxes(Tensor): [batch_size, num_cell[0], num_cell[1], boxes_per_cell, 4(center_x, center_y, w, h)]
        """
        with tf.name_scope("split_predictions"):
            num_cell_y, num_cell_x = self.num_cell[0], self.num_cell[1]

            output = tf.reshape(
                output,
                [self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell, self.num_classes + 5]
            )

            predict_classes, predict_confidence, predict_boxes = tf.split(output, [self.num_classes, 1, 4], axis=4)

            return predict_classes, predict_confidence, predict_boxes

    def _concat_predictions(self, predict_classes, predict_confidence, predict_boxes):
        """Concat predictions to inference output.

        """
        with tf.name_scope("concat_predictions"):
            num_cell_y, num_cell_x = self.num_cell[0], self.num_cell[1]

            output = tf.concat([predict_classes, predict_confidence, predict_boxes], axis=4, name="concat_predictions")

            output = tf.reshape(
                output,
                [self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell * (self.num_classes + 5)]
            )

            return output

    def post_process(self, output):
        with tf.name_scope("post_process"):

            output = self._format_output(output)
            output = self._exclude_low_score_box(output, threshold=self.score_threshold)
            output = self._nms(
                output,
                iou_threshold=self.nms_iou_threshold,
                max_output_size=self.nms_max_output_size,
                per_class=self.nms_per_class,
            )
            return output

    def _format_output(self, output):
        """Format yolov2 inference output to predict boxes list.

        Args:
            output: Tensor of inference() outputs.

        Retrun:
            List of predict_boxes Tensor.
            The Shape is [batch_size, num_predicted_boxes, 6(x, y, w, h, class_id, score)].
            The score be calculated by for each class probability and confidence.
        """

        if self.change_base_output:
            predict_classes, predict_confidence, predict_boxes = self._split_predictions(output)

        else:
            predict_classes, predict_confidence, predict_boxes = self._predictions(output)
            predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

        predict_boxes = format_CXCYWH_to_XYWH(predict_boxes, axis=4)
        results = []

        for class_id in range(len(self.classes)):
            predict_prob = predict_classes[:, :, :, :, class_id]
            predict_score = predict_prob * predict_confidence[:, :, :, :, 0]

            result = tf.stack([
                predict_boxes[:, :, :, :, 0],
                predict_boxes[:, :, :, :, 1],
                predict_boxes[:, :, :, :, 2],
                predict_boxes[:, :, :, :, 3],
                tf.fill(predict_score.shape, float(class_id)),
                predict_score,
            ], axis=4)

            results.append(result)
        results = tf.concat(results, axis=1)

        results = tf.reshape(
            results,
            [self.batch_size, self.num_cell[0] * self.num_cell[1] * self.boxes_per_cell * self.num_classes, 6]
        )
        return results

    def _exclude_low_score_box(self, formatted_output, threshold=0.05):
        """Exclude low score boxes.
        The score be calculated by for each class probability and confidence.

        Args:
            formatted_output: Formatted predict_boxes Tensor.
                The Shape is [batch_size, num_predicted_boxes, 6(x, y, w, h, class_id, score)].
            threshold: low threshold of predict score.

        Returns:
            python list of predict_boxes Tensor.
            predict_boxes shape is [num_predicted_boxes, 6(x, y, w, h, class_id, probability)].
            python list lenght is batch size.
        """

        results = []
        for i in range(self.batch_size):

            predicted_per_batch = formatted_output[i]
            score_mask = predicted_per_batch[:, 5] > threshold
            result = tf.boolean_mask(predicted_per_batch, score_mask)

            results.append(result)

        return results

    def _nms(self, formatted_output, iou_threshold, max_output_size, per_class):
        """Non Maximum Suppression.

        Args:
            formatted_output: python list of predict_boxes Tensor.
                predict_boxes shape is [num_predicted_boxes, 6(x, y, w, h, class_id, probability)].

            iou_threshold(float): The threshold for deciding whether boxes overlap with respect to IOU.
            max_output_size(int): The maximum number of boxes to be selected
            per_class(boolean): Whether or not, NMS respect to per class.
        Returns:
            python list of predict_boxes Tensor.
            predict_boxes shape is [num_predicted_boxes, 6(x, y, w, h, class_id, probability)].
            python list lenght is batch size.
        """

        results = []
        for i in range(self.batch_size):
            predicted_per_batch = formatted_output[i]

            if per_class:

                nms_per_batch = []
                for class_id in range(len(self.classes)):
                    class_mask = tf.equal(predicted_per_batch[:, 4], class_id)
                    class_masked = tf.boolean_mask(predicted_per_batch, class_mask)

                    nms_indices = tf.image.non_max_suppression(
                        format_XYWH_to_YX(class_masked[:, 0:4], axis=1),
                        class_masked[:, 5],
                        max_output_size,
                        iou_threshold=iou_threshold,
                        name="nms",
                    )
                    nmsed_per_class = tf.gather(class_masked, nms_indices)
                    nms_per_batch.append(nmsed_per_class)
                nmsed = tf.concat(nms_per_batch, axis=0)

            else:
                nms_indices = tf.image.non_max_suppression(
                    format_XYWH_to_YX(predicted_per_batch[:, 0:4], axis=1),
                    predicted_per_batch[:, 5],
                    max_output_size,
                    iou_threshold=iou_threshold,
                    name="nms",
                )

                nmsed = tf.gather(predicted_per_batch, nms_indices)

            results.append(nmsed)

        return results

    def loss(self, output, gt_boxes, is_training):
        """Loss.

        Args:
            output: 2D tensor.
                shape is [batch_size, self.num_cell * self.num_cell * (self.num_classes + self.boxes_per_cell * 5)]
            gt_boxes: ground truth boxes 3D tensor. [batch_size, max_num_boxes, 4(x, y, w, h, class_id)].
        """
        if self.change_base_output:
            predict_classes, predict_confidence, predict_boxes = self._split_predictions(output)
        else:
            predict_classes, predict_confidence, predict_boxes = self._predictions(output)
            predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

        gt_boxes = self.convert_gt_boxes_xywh_to_cxcywh(gt_boxes)
        return self.loss_function(predict_classes, predict_confidence, predict_boxes, gt_boxes, is_training)

    def inference(self, images, is_training):
        tf.summary.histogram("images", images)
        base = self.base(images, is_training)
        self.output = tf.identity(base, name="output")
        return self.output

    def _reorg(self, name, inputs, stride, data_format, use_space_to_depth=True, darknet_original=False):
        with tf.name_scope(name):
            # darknet original reorg layer is weird.
            # As default we don't use darknet original reorg.
            # See detail: https://github.com/LeapMind/lmnet/issues/16
            if darknet_original:
                # TODO(wakisaka): You can use only data_format == "NHWC", yet.
                assert data_format == "NHWC"
                input_shape = tf.shape(inputs)
                # channel shape use static.
                b, h, w, c = input_shape[0], input_shape[1], input_shape[2], inputs.get_shape().as_list()[3]
                output_h = h // stride
                output_w = w // stride
                output_c = c * stride * stride
                transpose_tensor = tf.transpose(inputs, [0, 3, 1, 2])
                reshape_tensor = tf.reshape(transpose_tensor, [b, (c // (stride * stride)), h, stride, w, stride])
                transpose_tensor = tf.transpose(reshape_tensor, [0, 3, 5, 1, 2, 4])
                reshape_tensor = tf.reshape(transpose_tensor, [b, output_c, output_h, output_w])
                transpose_tensor = tf.transpose(reshape_tensor, [0, 2, 3, 1])
                outputs = tf.reshape(transpose_tensor, [b, output_h, output_w, output_c])

                return outputs
            else:
                # tf.extract_image_patches() raise error with images_placehodler `None` shape as dynamic image.
                # Github issue: https://github.com/leapmindadmin/lmnet/issues/17
                # Currently, I didn't try to space_to_depth with images_placehodler `None` shape as dynamic image.
                if use_space_to_depth:

                    # TODO(wakisaka): When update tensorflow version, remove it.
                    g = tf.get_default_graph()
                    with g.gradient_override_map({"SpaceToDepth": "CustomSpaceToDepth"}):
                        outputs = tf.space_to_depth(inputs, stride, data_format=data_format)
                    return outputs

                else:
                    # TODO(wakisaka): You can use only data_format == "NHWC", yet.
                    assert data_format == "NHWC"
                    ksize = [1, stride, stride, 1]
                    strides = [1, stride, stride, 1]
                    rates = [1, 1, 1, 1]
                    outputs = tf.extract_image_patches(inputs, ksize, strides, rates, "VALID")

                    return outputs

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
            raise RuntimeError("data format {} shodul be in ['NCHW', 'NHWC]'.".format(self.data_format))

        self.block_1 = darknet_block(
            "block_1",
            self.inputs,
            filters=32,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_1 = max_pooling2d("pool_1", self.block_1, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_2 = darknet_block(
            "block_2",
            self.pool_1,
            filters=64,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_2 = max_pooling2d("pool_2", self.block_2, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_3 = darknet_block(
            "block_3",
            self.pool_2,
            filters=128,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_4 = darknet_block(
            "block_4",
            self.block_3,
            filters=64,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_5 = darknet_block(
            "block_5",
            self.block_4,
            filters=128,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_3 = max_pooling2d("pool_3", self.block_5, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_6 = darknet_block(
            "block_6",
            self.pool_3,
            filters=256,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_7 = darknet_block(
            "block_7",
            self.block_6,
            filters=128,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_8 = darknet_block(
            "block_8",
            self.block_7,
            filters=256,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.pool_4 = max_pooling2d("pool_4", self.block_8, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_9 = darknet_block(
            "block_9",
            self.pool_4,
            filters=512,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_10 = darknet_block(
            "block_10",
            self.block_9,
            filters=256,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_11 = darknet_block(
            "block_11",
            self.block_10,
            filters=512,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_12 = darknet_block(
            "block_12",
            self.block_11,
            filters=256,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_13 = darknet_block(
            "block_13",
            self.block_12,
            filters=512,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_21 = darknet_block(
            "block_21",
            self.block_13,
            filters=64,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.reorged = self._reorg("reorg", self.block_21, stride=2, data_format=self.data_format)

        self.pool_5 = max_pooling2d("pool_5", self.block_13, pool_size=2, strides=2, data_format=channel_data_format)

        self.block_14 = darknet_block(
            "block_14",
            self.pool_5,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_15 = darknet_block(
            "block_15",
            self.block_14,
            filters=512,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_16 = darknet_block(
            "block_16",
            self.block_15,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_17 = darknet_block(
            "block_17",
            self.block_16,
            filters=512,
            kernel_size=1,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_18 = darknet_block(
            "block_18",
            self.block_17,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        # End of darknet pretrain used. Start yolo .

        self.block_19 = darknet_block(
            "block_19",
            self.block_18,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        self.block_20 = darknet_block(
            "block_20",
            self.block_19,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.activation,
            data_format=self.data_format,
        )

        if self.data_format == "NHWC":
            self.concated = tf.concat([self.reorged, self.block_20], 3, name="concat_reorg_block_20")
        if self.data_format == "NCHW":
            self.concated = tf.concat([self.reorged, self.block_20], 1, name="concat_reorg_block_20")

        self.block_22 = darknet_block(
            "block_22",
            self.concated,
            filters=1024,
            kernel_size=3,
            is_training=is_training,
            activation=self.before_last_activation,
            data_format=self.data_format,
        )

        output_filters = (self.num_classes + 5) * self.boxes_per_cell
        self.conv_23 = conv2d(
            "conv_23", self.block_22, filters=output_filters, kernel_size=1,
            activation=None, use_bias=True, is_debug=self.is_debug,
            data_format=channel_data_format,
        )

        # assert_num_cell_y = tf.assert_equal(self.num_cell[0], tf.shape(self.conv_23)[1])
        # assert_num_cell_x = tf.assert_equal(self.num_cell[1], tf.shape(self.conv_23)[2])

        if self.change_base_output:

            # with tf.control_dependencies([assert_num_cell_x, assert_num_cell_y]):

            predict_classes, predict_confidence, predict_boxes = self._predictions(self.conv_23)

            with tf.name_scope("convert_boxes_space_from_yolo_to_real"):
                predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

            output = self._concat_predictions(predict_classes, predict_confidence, predict_boxes)

        else:
            # with tf.control_dependencies([assert_num_cell_x, assert_num_cell_y]):
            output = self.conv_23

        return output


class YoloV2Loss:
    """YOLO v2 loss function."""
    def __init__(
            self,
            is_debug=False,
            anchors=[(1.0, 1.0), (2.0, 2.0)],
            num_cell=[4, 4],
            boxes_per_cell=2,
            object_scale=5.0,
            no_object_scale=1.0,
            class_scale=1.0,
            coordinate_scale=1.0,
            loss_iou_threshold=0.6,
            weight_decay_rate=0.0005,
            image_size=[448, 448],
            batch_size=64,
            classes=[],
            yolo=None,
            seen_threshold=12800,
            use_cross_entropy_loss=True,
    ):
        self.is_debug = is_debug
        self.anchors = anchors
        self.num_cell = num_cell
        self.boxes_per_cell = boxes_per_cell
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.class_scale = class_scale
        self.coordinate_scale = coordinate_scale
        self.loss_iou_threshold = loss_iou_threshold
        self.weight_decay_rate = weight_decay_rate
        self.image_size = image_size
        self.batch_size = batch_size
        self.classes = classes
        self.num_classes = len(classes)
        # extra coordinate loss for early training steps to encourage predictions to match anchor priors.
        # https://github.com/pjreddie/darknet/blob/2f212a47425b2e1002c7c8a20e139fe0da7489b5/src/region_layer.c#L248
        self.seen_threshold = seen_threshold
        self.seen = 0

        self.use_cross_entropy_loss = use_cross_entropy_loss

        self.convert_boxes_space_from_yolo_to_real = yolo.convert_boxes_space_from_yolo_to_real
        self.convert_boxes_space_from_real_to_yolo = yolo.convert_boxes_space_from_real_to_yolo

    def _iou_per_gtbox(self, boxes, box):
        """Calculate ious.

        Args:
            boxes: 4-D np.ndarray [num_cell, num_cell, boxes_per_cell, 4(x_center, y_center, w, h)]
            box: 1-D np.ndarray [4(x_center, y_center, w, h)]

        Return:
            iou: 3-D np.ndarray [num_cell, num_cell, boxes_per_cell]
        """
        # left, top, right, bottom
        # format left, top, right, bottom
        boxes = np.stack([
            boxes[:, :, :, 0] - boxes[:, :, :, 2] / 2,
            boxes[:, :, :, 1] - boxes[:, :, :, 3] / 2,
            boxes[:, :, :, 0] + boxes[:, :, :, 2] / 2,
            boxes[:, :, :, 1] + boxes[:, :, :, 3] / 2,
        ], axis=3)

        box = np.array([
            box[0] - box[2] / 2,
            box[1] - box[3] / 2,
            box[0] + box[2] / 2,
            box[1] + box[3] / 2,
        ])

        # calculate the left up point
        left_top = np.maximum(boxes[:, :, :, 0:2], box[0:2])
        right_bottom = np.minimum(boxes[:, :, :, 2:], box[2:])

        # calculate intersection. [num_cell, num_cell, boxes_per_cell, 2]
        inter = right_bottom - left_top
        inter_square = inter[:, :, :, 0] * inter[:, :, :, 1]
        mask = (inter[:, :, :, 0] > 0) * (inter[:, :, :, 1] > 0)

        intersection = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes[:, :, :, 2] - boxes[:, :, :, 0]) * (boxes[:, :, :, 3] - boxes[:, :, :, 1])
        square2 = (box[2] - box[0]) * (box[3] - box[1])

        epsilon = 1e-10
        union = square1 + square2 - intersection

        iou = intersection / (union + epsilon)

        iou[np.isnan(iou)] = 0.0
        iou = np.clip(iou, 0.0, 1.0)

        return iou

    def __iou_gt_boxes(self, boxes, gt_boxes_list, num_cell):
        num_cell_y = num_cell[0]
        num_cell_x = num_cell[1]

        best_iou = np.zeros((self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell), dtype=np.float32)

        for batch_index in range(self.batch_size):
            current_boxes = boxes[batch_index, :, :, :, :]
            gt_boxes = gt_boxes_list[batch_index, :, :]

            best_iou_per_batch = np.zeros((num_cell_y, num_cell_x, self.boxes_per_cell), dtype=np.float32)

            for box_index in range(gt_boxes.shape[0]):

                gt_box = gt_boxes[box_index, :]

                # exclude dummy gt_box. class id `-1` is dummy.
                if gt_box[4] == -1:
                    continue

                iou_per_gtbox = self._iou_per_gtbox(current_boxes, gt_box[0:4])
                filter_iou = iou_per_gtbox > best_iou_per_batch
                best_iou_per_batch[filter_iou] =\
                    iou_per_gtbox[filter_iou]

            best_iou[batch_index, :, :, :] = best_iou_per_batch

        return best_iou

    def _iou_gt_boxes(self, boxes, gt_boxes_list):
        """Calculate ious between predict box and gt box.
        And choice best iou for each gt box.

        Args:
            boxes: Predicted boxes in real space coordinate.
                 5-D tensor [batch_size, num_cell, num_cell, boxes_per_cell, 4(x_center, y_center, w, h)].
            gt_boxes_list: 5-D tensor [batch_size, max_num_boxes, 5(center_x, center_y, w, h, class_id)]

        Return:
            iou: 4-D tensor [batch_size, num_cell, num_cell, boxes_per_cell]
        """

        ious = tf.py_func(
            self.__iou_gt_boxes,
            [boxes, gt_boxes_list, self.num_cell],
            tf.float32
        )

        return ious

    def _one_iou(self, box1, box2):
        # format left, top, right, bottom
        box1 = np.array([
            box1[0] - box1[2] / 2,
            box1[1] - box1[3] / 2,
            box1[0] + box1[2] / 2,
            box1[1] + box1[3] / 2,
        ])

        box2 = np.array([
            box2[0] - box2[2] / 2,
            box2[1] - box2[3] / 2,
            box2[0] + box2[2] / 2,
            box2[1] + box2[3] / 2,
        ])

        left_top = np.maximum(box1[0:2], box2[0:2])
        right_bottom = np.minimum(box1[2:], box2[2:])

        inter = right_bottom - left_top

        if inter[0] <= 0 and inter[1] <= 0:
            return 0.0

        inter_square = inter[0] * inter[1]

        # calculate the box1 square and box2 square
        square1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        square2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        epsilon = 1e-10

        union = square1 + square2 - inter_square

        iou = inter_square / (union + epsilon)

        if np.isnan(iou):
            return 0.0
        iou = np.clip(iou, 0.0, 1.0)

        return iou

    def __calculate_truth_and_maskes(
            self, gt_boxes_list, predict_boxes, num_cell, image_size, is_training,
            predict_classes=None, predict_confidence=None
    ):
        """Calculate truth and maskes for loss function from gt boxes and predict boxes.

        1. When trained images is less than seen_threshold, set cell_gt_boxes and coordinate_maskes
        to manage coordinate loss for early training steps to encourage predictions to match anchor.

        2. About not dummy gt_boxes, calculate between gt boxes and anchor iou, and select best ahcnor.

        3. In the best anchor, create cell_gt_boxes from the gt_boxes
        and calculate truth_confidence and assign maskes ture.

        Args:
            gt_boxes_list(np.ndarray): The ground truth boxes. Shape is
                [batch_size, max_num_boxes, 5(center_x, center_y, w, h, class_id)].
            predict_boxes(np.ndarray): Predicted boxes.
                Shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].
            num_cell: Number of cell. num_cell[0] is y axis, num_cell[1] is x axis.
            image_size: Image size(px). image_size[0] is height, image_size[1] is width.
            is_training(np.ndarray): Boolean.
            predict_classes: Use only in debug. Shape is [batch_size, num_cell, num_cell, boxes_per_cell, num_classes]
            predict_confidence: Use only in debug. Shape is [batch_size, num_cell, num_cell, boxes_per_cell, 1]

        Return:
            cell_gt_boxes: The cell anchor corresponding gt_boxes from gt_boxes_list. Dummy cell gt boxes are zeros.
                shape is [batch_size, num_cell, num_cell, box_per_cell, 5(center_x, center_y, w, h, class_id)].
            truth_confidence: The confidence values each for cell anchos.
                Tensor shape is [batch_size, num_cell, num_cell, box_per_cell, 1].
            object_maskes: The cell anchor that has gt boxes is 1, none is 0.
                Tensor shape is [batch_size, num_cell, num_cell, box_per_cell, 1].
            coordinate_maskes: the cell anchor that has gt boxes is 1, none is 0.
                Tensor [batch_size, num_cell, num_cell, box_per_cell, 1].
        """
        if is_training:
            self.seen += self.batch_size

        num_cell_y = num_cell[0]
        num_cell_x = num_cell[1]

        cell_gt_boxes = np.zeros((self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell, 5), dtype=np.float32)
        truth_confidence = np.zeros((self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell, 1), dtype=np.float32)
        object_maskes = np.zeros((self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell, 1), dtype=np.int64)
        coordinate_maskes = np.zeros((self.batch_size, num_cell_y, num_cell_x, self.boxes_per_cell, 1), dtype=np.int64)

        # for debug:
        num_gt_boxes = 0
        num_correct_prediction_conf = 0
        sum_iou = 0.0
        sum_conf = 0.0
        sum_diff_conf = 0.0

        # extra coordinate loss for early training steps to encourage predictions to match anchor.
        # https://github.com/pjreddie/darknet/blob/2f212a47425b2e1002c7c8a20e139fe0da7489b5/src/region_layer.c#L248
        if self.seen < self.seen_threshold:
            if self.is_debug:
                print("current seen: {}. To calculate coordinate loss for early training step.".format(self.seen))

            offset_x, offset_y, offset_w, offset_h = YoloV2.py_offset_boxes(num_cell_y, num_cell_x,
                                                                            self.batch_size, self.boxes_per_cell,
                                                                            self.anchors)

            stride_x = image_size[1] / num_cell_x
            stride_y = image_size[0] / num_cell_y

            cell_gt_boxes[:, :, :, :, 0] = (offset_x + 0.5) * stride_x
            cell_gt_boxes[:, :, :, :, 1] = (offset_y + 0.5) * stride_y
            cell_gt_boxes[:, :, :, :, 2] = stride_x * offset_w
            cell_gt_boxes[:, :, :, :, 3] = stride_y * offset_h
            cell_gt_boxes[:, :, :, :, 4] = -1

            coordinate_maskes[:] = 1  # True

        for batch_index in range(self.batch_size):

            # calculate iou anchor and gt box
            gt_boxes = gt_boxes_list[batch_index, :, :]

            for box_index in range(gt_boxes.shape[0]):

                gt_box = gt_boxes[box_index, :]

                # exclude dummy gt_box. class id `-1` is dummy.
                if gt_box[4] == -1:
                    continue

                num_gt_boxes += 1
                center_y = gt_box[1]
                center_x = gt_box[0]

                cell_y_index = math.floor(center_y / image_size[0] * num_cell_y)
                cell_x_index = math.floor(center_x / image_size[1] * num_cell_x)

                resized_h = gt_box[3] / image_size[0]
                resized_w = gt_box[2] / image_size[1]

                best_iou = 0.0
                best_anchor_index = -1

                for anchor_index, (anchor_w, anchor_h) in enumerate(self.anchors):
                    iou = self._one_iou(
                        (0, 0, resized_w, resized_h),
                        (0, 0, anchor_w / num_cell_x, anchor_h / num_cell_y)
                    )

                    if (iou > best_iou):
                        best_iou = iou
                        best_anchor_index = anchor_index

                if best_anchor_index == -1:
                    print("---- Can't find best anchor ---" * 100)
                else:
                    predict_box = predict_boxes[batch_index, cell_y_index, cell_x_index, best_anchor_index, :]

                    iou = self._one_iou(predict_box, gt_box)

                    # the cell_gt_boxes is assigned gt_box coordinate
                    cell_gt_boxes[batch_index, cell_y_index, cell_x_index, best_anchor_index, :] = gt_box

                    truth_confidence[batch_index, cell_y_index, cell_x_index, best_anchor_index, :] = iou

                    # the box of cell object_mask is assigned 1.0(True),
                    object_maskes[batch_index, cell_y_index, cell_x_index, best_anchor_index, :] = 1  # True

                    coordinate_maskes[batch_index, cell_y_index, cell_x_index, best_anchor_index, :] = 1  # True

                    if self.is_debug:
                        print("best_anchor_index", best_anchor_index)

                        pred_conf = predict_confidence[batch_index, cell_y_index, cell_x_index, best_anchor_index, :]
                        predict_probabilities =\
                            predict_classes[batch_index, cell_y_index, cell_x_index, best_anchor_index, :]
                        argmax = np.argmax(predict_probabilities)
                        message = "truth_class: {}. pred_class: {}. pred_prob: {}".format(
                            gt_box[4], argmax, 100 * predict_probabilities[argmax])
                        print(message)

                        sum_conf += float(pred_conf)
                        sum_iou += iou
                        diff_conf = abs(float(pred_conf) - iou)
                        sum_diff_conf += diff_conf
                        if iou > 0.5:
                            num_correct_prediction_conf += 1

        if self.is_debug:
            message = "num_gt_boxes: {}. num_correct_prediction_conf: {}. avg_iou: {}. avg_conf: {}. avg_diff_conf: {}"
            message = message.format(
                num_gt_boxes, num_correct_prediction_conf, sum_iou/num_gt_boxes,
                sum_conf/num_gt_boxes, sum_diff_conf/num_gt_boxes
            )

            print(message)

        return cell_gt_boxes, truth_confidence, object_maskes, coordinate_maskes

    def _calculate_truth_and_maskes(self, gt_boxes_list, predict_boxes, is_training,
                                    predict_classes=None, predict_confidence=None):
        """Calculate truth and maskes for loss function from gt boxes and predict boxes.

        Args:
            gt_boxes_list: The ground truth boxes. Tensor shape is
                [batch_size, max_num_boxes, 5(center_x, center_y, w, h, class_id)].
            predict_boxes: Predicted boxes.
                Tensor shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].
            is_training: Boolean tensor.
            predict_classes: Use only in debug. Shape is [batch_size, num_cell, num_cell, boxes_per_cell, num_classes]
            predict_confidence: Use only in debug. Shape is [batch_size, num_cell, num_cell, boxes_per_cell, 1]

        Return:
            cell_gt_boxes: The cell anchor corresponding gt_boxes from gt_boxes_list. Dummy cell gt boxes are zeros.
                shape is [batch_size, num_cell, num_cell, box_per_cell, 5(center_x, center_y, w, h, class_id)].
            truth_confidence: The confidence values each for cell anchos.
                Tensor shape is [batch_size, num_cell, num_cell, box_per_cell, 1].
            object_maskes: The cell anchor that has gt boxes is 1, none is 0.
                Tensor shape is [batch_size, num_cell, num_cell, box_per_cell, 1].
            coordinate_maskes: the cell anchor that has gt boxes is 1, none is 0.
                Tensor [batch_size, num_cell, num_cell, box_per_cell, 1].
        """
        if self.is_debug:
            cell_gt_boxes, truth_confidence, object_maskes, coordinate_maskes = tf.py_func(
                self.__calculate_truth_and_maskes,
                [gt_boxes_list, predict_boxes, self.num_cell, self.image_size, is_training,
                 predict_classes, predict_confidence],
                [tf.float32, tf.float32, tf.int64, tf.int64]
            )
        else:
            cell_gt_boxes, truth_confidence, object_maskes, coordinate_maskes = tf.py_func(
                self.__calculate_truth_and_maskes,
                [gt_boxes_list, predict_boxes, self.num_cell, self.image_size, is_training],
                [tf.float32, tf.float32, tf.int64, tf.int64]
            )

        object_maskes = tf.to_float(object_maskes)
        coordinate_maskes = tf.to_float(coordinate_maskes)
        return cell_gt_boxes, truth_confidence, object_maskes, coordinate_maskes

    def _weight_decay_loss(self):
        """L2 weight decay (regularization) loss."""
        losses = []
        for var in tf.trainable_variables():

            # exclude batch norm variable
            if "conv/kernel" in var.name:
                losses.append(tf.nn.l2_loss(var))

        return tf.add_n(losses) * self.weight_decay_rate

    def __call__(self, predict_classes, predict_confidence, predict_boxes, gt_boxes, is_training):
        """Loss function.

        Args:
            predict_classes: [batch_size, num_cell, num_cell, boxes_per_cell, num_classes]
            predict_confidence: [batch_size, num_cell, num_cell, boxes_per_cell, 1]
            predict_boxes: [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)]
            gt_boxes: ground truth boxes 3D tensor. [batch_size, max_num_boxes, 5(center_x, center_y, w, h, class_id)].
            is_training: boolean tensor.

        Returns:
            loss: loss value scalr tensor.
        """
        with tf.name_scope("loss"):
            if self.is_debug:
                predict_boxes = tf.Print(
                    predict_boxes,
                    [tf.shape(predict_boxes), predict_boxes],
                    message="predict_boxes:",
                    summarize=20000)

            # best_iou: [batch_size, num_cell, num_cell, boxes_per_cell]
            best_iou = self._iou_gt_boxes(predict_boxes, gt_boxes)

            if self.is_debug:
                best_iou = tf.Print(
                    best_iou,
                    [tf.shape(best_iou), best_iou],
                    message="iou:",
                    summarize=20000)

            # iou_mask: [batch_size, num_cell, num_cell, boxes_per_cell, 1]
            iou_mask = best_iou > self.loss_iou_threshold
            iou_mask = tf.expand_dims(iou_mask, 4)
            iou_mask = tf.to_float(iou_mask)

            if self.is_debug:
                iou_mask = tf.Print(
                    iou_mask,
                    [tf.shape(iou_mask), iou_mask],
                    message="iou_mask:",
                    summarize=20000)

            # cell_gt_boxes: [batch_size, num_cell, num_cell, box_per_cell, 5]
            # truth_confidence: [batch_size, num_cell, num_cell, box_per_cell, 1]
            # object_mask: [batch_size, num_cell, num_cell, box_per_cell, 1]
            # coordinate_mask: [batch_size, num_cell, num_cell, box_per_cell, 1]
            cell_gt_boxes, truth_confidence, object_mask, coordinate_mask =\
                self._calculate_truth_and_maskes(gt_boxes, predict_boxes, is_training,
                                                 predict_classes, predict_confidence)

            # for class loss
            truth_classes = tf.to_int32(cell_gt_boxes[:, :, :, :, 4])
            truth_classes = tf.one_hot(truth_classes, self.num_classes)

            cell_gt_boxes = cell_gt_boxes[:, :, :, :, :4]

            # no_object_mask: [batch_size, num_cell, num_cell, boxes_per_cell, 1]
            no_object_mask = (1.0 - object_mask) * (1.0 - iou_mask)

            if self.is_debug:
                no_object_mask = tf.Print(
                    no_object_mask,
                    [tf.shape(no_object_mask), no_object_mask],
                    message="no_object_mask:",
                    summarize=20000)

            # resize to yolo space.
            truth_boxes = self.convert_boxes_space_from_real_to_yolo(cell_gt_boxes)
            resized_predict_boxes = self.convert_boxes_space_from_real_to_yolo(predict_boxes)

            if self.is_debug:
                truth_boxes = tf.Print(
                    truth_boxes,
                    [tf.shape(truth_boxes), truth_boxes],
                    message="truth_boxes:",
                    summarize=20000)

            # class_loss
            if self.use_cross_entropy_loss:
                class_loss = tf.reduce_mean(
                    - tf.reduce_sum(
                        object_mask * (truth_classes * tf.log(tf.clip_by_value(predict_classes, 1e-10, 1.0))),
                        axis=[1, 2, 3, 4]
                    ),
                    name='class_loss'
                ) * self.class_scale

            else:
                class_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        object_mask * tf.square(predict_classes - truth_classes),
                        axis=[1, 2, 3, 4]
                    ),
                    name='class_loss'
                ) * self.class_scale

            # object_loss
            object_loss = tf.reduce_mean(
                tf.reduce_sum(
                    object_mask * tf.square(predict_confidence - truth_confidence),
                    axis=[1, 2, 3, 4]),
                name='object_loss') * self.object_scale

            # no_object_response_mask_loss
            no_object_loss = tf.reduce_mean(
                tf.reduce_sum(
                    no_object_mask * tf.square(0 - predict_confidence),
                    axis=[1, 2, 3, 4]),
                name='no_object_loss') * self.no_object_scale

            # coordinate_loss
            coordinate_loss = tf.reduce_mean(
                tf.reduce_sum(
                    coordinate_mask * tf.square(resized_predict_boxes - truth_boxes),
                    axis=[1, 2, 3, 4]),
                name='coordinate_loss') * self.coordinate_scale

            weight_decay_loss = self._weight_decay_loss()
            loss = class_loss + object_loss + no_object_loss + coordinate_loss + weight_decay_loss

            if self.is_debug:
                loss = tf.Print(
                    loss,
                    [class_loss, object_loss, no_object_loss, coordinate_loss],
                    message="loss: class_loss, object_loss, no_object_loss, coordinate_loss:",
                    summarize=20000)

            tf.summary.scalar("class", class_loss)
            tf.summary.scalar("object", object_loss)
            tf.summary.scalar("no_object", no_object_loss)
            tf.summary.scalar("coordinate", coordinate_loss)
            tf.summary.scalar("weight_decay", weight_decay_loss)

            tf.summary.scalar("loss", loss)

        return loss


def summary_boxes(tag, images, boxes, image_size, max_outputs=3, data_format="NHWC"):
    """Draw bounding boxes images on Tensroboard.

    Args:
    tag: name of summary tag.
    images: Tesnsor of images [batch_size, height, widths, 3].
    boxes: Tensor of boxes. assumed shape is [batch_size, num_boxes, 4(y1, x1, y2, x2)].
    image_size: python list image size [height, width].
    """
    with tf.name_scope("summary_boxes"):

        if data_format == "NCHW":
            images = tf.transpose(images, [0, 2, 3, 1])

        boxes = tf.stack([
            boxes[:, :, 0] / tf.to_float(image_size[0]),
            boxes[:, :, 1] / tf.to_float(image_size[1]),
            boxes[:, :, 2] / tf.to_float(image_size[0]),
            boxes[:, :, 3] / tf.to_float(image_size[1]),
        ], axis=2)

        bb_images = tf.image.draw_bounding_boxes(images, boxes)
        summary = tf.summary.image(tag, bb_images, max_outputs=max_outputs)

        return summary


def format_XYWH_to_CXCYWH(boxes, axis=1):
    """Format form (x, y, w, h) to (center_x, center_y, w, h) along specific dimention.

    Args:
    boxes :a Tensor include boxes. [:, 4(x, y, w, h)]
    axis: which dimension of the inputs Tensor is boxes.
    """
    with tf.name_scope('format_xywh_to_cxcywh'):
        x, y, w, h = tf.split(axis=axis, num_or_size_splits=4, value=boxes)

        center_x = x + (w / 2)
        center_y = y + (h / 2)

        return tf.concat([center_x, center_y, w, h], axis=axis, name="concat_format_xywh_to_cxcywh")


def format_CXCYWH_to_XYWH(boxes, axis=1):
    """Format form (center_x, center_y, w, h) to (x, y, w, h) along specific dimention.

    Args:
    boxes: A tensor include boxes. [:, 4(x, y, w, h)]
    axis: Which dimension of the inputs Tensor is boxes.
    """
    with tf.name_scope('format_xywh_to_cxcywh'):
        center_x, center_y, w, h = tf.split(axis=axis, num_or_size_splits=4, value=boxes)

        x = center_x - (w / 2)
        y = center_y - (h / 2)

        return tf.concat([x, y, w, h], axis=axis, name="concat_format_xywh_to_cxcywh")


def format_CXCYWH_to_YX(inputs, axis=1):
    """Format from (x, y, w, h) to (y1, x1, y2, x2) boxes along specific dimention.

    Args:
      inputs: a Tensor include boxes.
      axis: which dimension of the inputs Tensor is boxes.
    """
    with tf.name_scope('format_xywh_to_yx'):
        center_x, center_y, w, h = tf.split(axis=axis, num_or_size_splits=4, value=inputs)

        x1 = center_x - (w / 2)
        x2 = center_x + (w / 2)
        y1 = center_y - (h / 2)
        y2 = center_y + (h / 2)

        return tf.concat([y1, x1, y2, x2], axis=axis, name="concat_format_xywh_to_yx")


def format_XYWH_to_YX(inputs, axis=1):
    """Format from (x, y, w, h) to (y1, x1, y2, x2) boxes along specific dimention.

    Args:
      inputs: a Tensor include boxes.
      axis: which dimension of the inputs Tensor is boxes.
    """
    with tf.name_scope('format_xywh_to_yx'):
        x, y, w, h = tf.split(axis=axis, num_or_size_splits=4, value=inputs)

        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h

        return tf.concat([y1, x1, y2, x2], axis=axis, name="concat_format_xywh_to_yx")
