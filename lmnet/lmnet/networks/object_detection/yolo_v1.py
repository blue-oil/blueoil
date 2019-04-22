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
import numpy as np
import tensorflow as tf


from lmnet.layers import conv2d, fully_connected, max_pooling2d
from lmnet.metrics.mean_average_precision import tp_fp_in_the_image
from lmnet.metrics.mean_average_precision import average_precision
from lmnet.networks.base import BaseNetwork


class YoloV1(BaseNetwork):
    """YOLO version1.

    YOLO v1.
    paper: https://arxiv.org/abs/1506.02640
    """

    def __init__(
            self,
            cell_size=7,
            boxes_per_cell=2,
            leaky_relu_scale=0.1,
            object_scale=1.0,
            no_object_scale=1.0,
            class_scale=1.0,
            coordinate_scale=5.0,
            num_max_boxes=1,
            *args,
            **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.cell_size = cell_size
        self.boxes_per_cell = boxes_per_cell
        self.leaky_relu_scale = leaky_relu_scale
        self.num_max_boxes = num_max_boxes

        self.loss_function = YoloV1Loss(
            is_debug=self.is_debug,
            cell_size=cell_size,
            boxes_per_cell=boxes_per_cell,
            object_scale=object_scale,
            no_object_scale=no_object_scale,
            class_scale=class_scale,
            coordinate_scale=coordinate_scale,
            image_size=self.image_size,
            batch_size=self.batch_size,
            classes=self.classes,
            yolo=self,
        )

    def placeholderes(self):
        """placeholders"""

        images_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.image_size[0], self.image_size[1], 3),
            name="images_placeholder")

        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(self.batch_size, self.num_max_boxes, 5),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def summary(self, output, labels=None):
        predict_classes, predict_confidence, predict_boxes = self._predictions(output)

        tf.summary.histogram("predict_classes", predict_classes)
        tf.summary.histogram("predict_confidence", predict_confidence)

        self._summary_predict_boxes(predict_classes, predict_confidence, predict_boxes, threshold=0.05)

    def metrics(self, output, labels, thresholds=[0.3, 0.5, 0.7]):
        predict_boxes = self.predict_boxes(output)

        metrics_ops_dict = {}
        updates = []

        for overlap_thresh in thresholds:
            average_precisions = []
            for class_id, class_name in enumerate(self.classes):
                tps = []
                fps = []
                scores = []
                num_gt_boxes_list = []

                for image_index, predict_boxes_in_the_image in enumerate(predict_boxes):
                    mask = tf.equal(predict_boxes_in_the_image[:, 4], class_id)
                    predict_boxes_in_the_image = tf.boolean_mask(predict_boxes_in_the_image, mask)

                    labels_in_the_image = labels[image_index, :, :]
                    mask = tf.equal(labels_in_the_image[:, 4], class_id)
                    labels_in_the_image = tf.boolean_mask(labels_in_the_image, mask)

                    num_gt_boxes = tf.shape(labels_in_the_image)[0]

                    tp, fp, score = tf.py_func(
                        tp_fp_in_the_image,
                        [predict_boxes_in_the_image, labels_in_the_image, overlap_thresh],
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

            metrics_key = 'MeanAveragePrecision_{}'.format(overlap_thresh)
            metrics_value = tf.add_n(average_precisions) / len(self.classes)

            metrics_ops_dict[metrics_key] = metrics_value

        return metrics_ops_dict, tf.group(*updates)

    def leaky_relu(self, inputs):
        return tf.maximum(self.leaky_relu_scale * inputs, inputs, name="leaky_relu")

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

        result = tf.concat(axis=axis, values=[gt_boxes_without_label, gt_boxes_only_label])

        return result

    def offset_boxes(self):
        """Return yolo space offset of x and y.

        Return:
        offset_x: shape is [batch_size, cell_size, cell_size, boxes_per_cell]
        offset_y: shape is [batch_size, cell_size, cell_size, boxes_per_cell]
        """
        offset_y = np.arange(self.cell_size)
        offset_y = np.reshape(offset_y, (1, self.cell_size, 1, 1))
        offset_y = np.tile(offset_y, [self.batch_size, 1, self.cell_size, self.boxes_per_cell])

        offset_x = np.transpose(offset_y, (0, 2, 1, 3))
        return offset_x, offset_y

    def convert_boxes_space_from_real_to_yolo(self, boxes):
        """Convert boxes space size from real to yolo.

        Real space boxes coodinates are in the interval [0, image_size].
        Yolo space boxes coodinates are in the interval [-1, 1].

        Args:
        boxes: 5D Tensor, shape is [batch_size, cell_size, cell_size, boxes_per_cell, 4(center_x, center_y, w, h)].
        """
        offset_x, offset_y = self.offset_boxes()

        resized_boxes = boxes / [
            self.image_size[1],
            self.image_size[0],
            self.image_size[1],
            self.image_size[0],
        ]

        resized_boxes = tf.stack([
            (resized_boxes[:, :, :, :, 0] * self.cell_size - offset_x),
            (resized_boxes[:, :, :, :, 1] * self.cell_size - offset_y),
            tf.sqrt(resized_boxes[:, :, :, :, 2]),
            tf.sqrt(resized_boxes[:, :, :, :, 3]),
        ], axis=4)

        return resized_boxes

    def convert_boxes_space_from_yolo_to_real(self, predict_boxes):
        """Convert predict boxes space size from yolo to real.

        Real space boxes coodinates are in the interval [0, image_size].
        Yolo space boxes coodinates are in the interval [-1, 1].

        Args:
        boxes: 5D Tensor, shape is [batch_size, cell_size, cell_size, boxes_per_cell, 4(center_x, center_y, w, h)].
        """
        offset_x, offset_y = self.offset_boxes()

        resized_predict_boxes = tf.stack([
            (predict_boxes[:, :, :, :, 0] + offset_x) / self.cell_size,
            (predict_boxes[:, :, :, :, 1] + offset_y) / self.cell_size,
            tf.square(predict_boxes[:, :, :, :, 2]),
            tf.square(predict_boxes[:, :, :, :, 3]),
        ], axis=4)

        resized_predict_boxes = resized_predict_boxes * [
            self.image_size[1],
            self.image_size[0],
            self.image_size[1],
            self.image_size[0],
        ]

        return resized_predict_boxes

    def _predictions(self, output):
        """Separate combined inference outputs to predictions.

        Args:
        output: combined fc outputs 2D Tensor.
            shape is [batch_size, self.cell_size * self.cell_size * (self.num_classes + self.boxes_per_cell * 5)]

        Returns:
        predict_classes: 4D Tensor [batch_size, cell_size, cell_size, num_classes]
        predict_confidence: 4D Tensor [batch_size, cell_size, cell_size, boxes_per_cell]
        predict_boxes: 5D Tensor [batch_size, cell_size, cell_size, boxes_per_cell, 4(center_x, center_y, w, h)]
        """
        boundary_classes = self.cell_size * self.cell_size * self.num_classes
        boundary_boxes = boundary_classes + self.cell_size * self.cell_size * self.boxes_per_cell

        predict_classes = tf.reshape(
            output[:, :boundary_classes],
            [self.batch_size, self.cell_size, self.cell_size, self.num_classes])

        predict_confidence = tf.reshape(
            output[:, boundary_classes:boundary_boxes],
            [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])

        predict_boxes = tf.reshape(
            output[:, boundary_boxes:],
            [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        return predict_classes, predict_confidence, predict_boxes

    def predict_boxes(self, output, threshold=0.05):
        """Predict boxes with probabilty threshold.

        Args:
        output: Tensor of inference() outputs.
        threshold: threshold of predict score.

        Retrun:
            python list of predict_boxes Tensor.
            predict_boxes shape is [num_predicted_boxes, 6(x, y, w, h, class_id, probability)].
            python list lenght is batch size.
        """
        predict_classes, predict_confidence, predict_boxes = self._predictions(output)
        return self._post_process(predict_classes, predict_confidence, predict_boxes, threshold)

    def _post_process(self, predict_classes, predict_confidence, predict_boxes, threshold=0.05):
        """Predict boxes with probabilty threshold.

        Args:
        predict_classes: [batch_size, cell_size, cell_size, num_classes]
        predict_confidence: [batch_size, cell_size, cell_size, boxes_per_cell]
        predict_boxes: [batch_size, cell_size, cell_size, boxes_per_cell, 4(center_x, center_y, w, h)]
        threshold: threshold of predict score.

        Return: python list of predict_boxes Tensor.
        predict_boxes shape is [num_predicted_boxes, 6(x, y, w, h, class_id, probability)].
        python list lenght is batch size.
        """
        with tf.name_scope("post_process"):
            predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

            if self.is_debug:
                predict_classes = tf.Print(
                    predict_classes,
                    [tf.shape(predict_classes), predict_classes],
                    message="predict_classes:",
                    summarize=2000)

            # [batch_size, cell_size, cell_size, boxes_per_cell, num_classes]
            predict_probability = tf.expand_dims(predict_confidence, 4) * tf.expand_dims(predict_classes, 3)

            mask = predict_probability > threshold

            if self.is_debug:
                mask = tf.Print(
                    mask,
                    [tf.shape(mask), mask],
                    message="mask:",
                    summarize=2000)

            # box_mask: shape is [batch_size, cell_size, cell_size, boxes_per_cell]
            box_mask = tf.reduce_any(mask, axis=4)

            max_predict_classes = tf.reduce_max(predict_classes, axis=3)
            max_predict_classes = tf.reshape(
                max_predict_classes, [self.batch_size, self.cell_size, self.cell_size, 1]
            )
            max_predict_classes = tf.tile(max_predict_classes, [1, 1, 1, self.boxes_per_cell])

            # argmax_predict_classes be:  [batch_size, cell_size, cell_size, self.boxes_per_cell]
            argmax_predict_classes = tf.argmax(predict_classes, axis=3)
            argmax_predict_classes = tf.reshape(
                argmax_predict_classes, [self.batch_size, self.cell_size, self.cell_size, 1]
            )
            argmax_predict_classes = tf.tile(argmax_predict_classes, [1, 1, 1, self.boxes_per_cell])

            if self.is_debug:
                argmax_predict_classes = tf.Print(
                    argmax_predict_classes,
                    [tf.shape(argmax_predict_classes), argmax_predict_classes],
                    message="argmax_predict_classes:",
                    summarize=2000)

            boxes_list = []
            for i in range(self.batch_size):
                box_mask_by_batch = box_mask[i, :, :, :]
                predict_boxes_by_batch = predict_boxes[i, :, :, :, :]

                # masked_boxes: [?, 4]
                masked_boxes = tf.boolean_mask(predict_boxes_by_batch, box_mask_by_batch)

                if self.is_debug:
                    masked_boxes = tf.Print(
                        masked_boxes,
                        [i, tf.shape(masked_boxes), masked_boxes],
                        message="predicted_masked_boxes:",
                        summarize=2000)

                predict_classes_by_batch = argmax_predict_classes[i, :, :, :]
                masked_classes = tf.boolean_mask(predict_classes_by_batch, box_mask_by_batch)
                if self.is_debug:
                    masked_classes = tf.Print(
                        masked_classes,
                        [i, tf.shape(masked_classes), masked_classes],
                        message="masked_classes:",
                        summarize=2000)

                predict_classes_probability_by_batch = max_predict_classes[i, :, :, :]
                masked_class_probability = tf.boolean_mask(
                    predict_classes_probability_by_batch, box_mask_by_batch
                )

                if self.is_debug:
                    masked_class_probability = tf.Print(
                        masked_class_probability,
                        [i, tf.shape(masked_class_probability), masked_class_probability],
                        message="masked_class_probability:",
                        summarize=2000)

                masked_boxes = format_CXCYWH_to_XYWH(masked_boxes, axis=1)
                # boxes: shape is [num_predicted_boxes, 6(x, y, w, h, class_id, probability)]
                boxes = tf.stack([
                    masked_boxes[:, 0],
                    masked_boxes[:, 1],
                    masked_boxes[:, 2],
                    masked_boxes[:, 3],
                    tf.to_float(masked_classes),
                    masked_class_probability,
                ], axis=1)

                boxes_list.append(boxes)

            return boxes_list

    # TODO(wakisaka): The main process is the same as _post_process()
    def _summary_predict_boxes(self, predict_classes, predict_confidence, predict_boxes, threshold=0.05):
        """Summary predict boxes on tensorboard.

        Args:
        predict_classes: [batch_size, cell_size, cell_size, num_classes]
        predict_confidence: [batch_size, cell_size, cell_size, boxes_per_cell]
        predict_boxes: [batch_size, cell_size, cell_size, boxes_per_cell, 4(center_x, center_y, w, h)]
        threshold: threshold of predict score.
        """
        predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

        with tf.name_scope("raw_predict_boxes"):

            resized_predict_boxes_for_summary = tf.reshape(predict_boxes, [self.batch_size, -1, 4])
            summary_boxes(
                "predicted_boxes",
                self.images, format_CXCYWH_to_YX(resized_predict_boxes_for_summary, axis=2),
                self.image_size,
            )

        with tf.name_scope("predict_boxes"):
            # [batch_size, cell_size, cell_size, boxes_per_cell, num_classes]
            predict_probability = tf.expand_dims(predict_confidence, 4) * tf.expand_dims(predict_classes, 3)

            tf.summary.histogram("predict_probability", predict_probability)

            mask = predict_probability > threshold

            # shape is [batch_size, cell_size, cell_size, boxes_per_cell]
            box_mask = tf.reduce_any(mask, axis=4)

            for i in range(self.batch_size):
                box_mask_by_batch = box_mask[i, :, :, :]
                predict_boxes_by_batch = predict_boxes[i, :, :, :, :]
                masked_boxes = tf.boolean_mask(predict_boxes_by_batch, box_mask_by_batch)

                image = tf.expand_dims(self.images[i, :, :, :], 0)
                masked_boxes = tf.expand_dims(masked_boxes, 0)
                summary_boxes(
                    "summary_predict_boxes",
                    image,
                    format_CXCYWH_to_YX(masked_boxes, axis=2),
                    self.image_size,
                )

    def loss(self, output, gt_boxes, *args):
        """Loss.

        Args:
            output: 2D tensor.
                shape is [batch_size, self.cell_size * self.cell_size * (self.num_classes + self.boxes_per_cell * 5)]
            gt_boxes: ground truth boxes 3D tensor. [batch_size, max_num_boxes, 4(x, y, w, h)].
        """
        gt_boxes = self.convert_gt_boxes_xywh_to_cxcywh(gt_boxes)

        with tf.name_scope("gt_boxes"):
            summary_boxes(
                "gt_boxes",
                self.images,
                format_CXCYWH_to_YX(gt_boxes[:, :, :4], axis=2),
                self.image_size,
                max_outputs=3,
            )

        predict_classes, predict_confidence, predict_boxes = self._predictions(output)

        return self.loss_function(predict_classes, predict_confidence, predict_boxes, gt_boxes)

    def inference(self, images, is_training):
        base = self.base(images, is_training)
        self.output = tf.identity(base, name="output")
        return self.output

    def base(self, images, is_training):
        self.images = images
        keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))

        self.conv_1 = conv2d("conv_1", images, filters=64, kernel_size=7, strides=2,
                             activation=self.leaky_relu)
        self.pool_2 = max_pooling2d("pool_2", self.conv_1, pool_size=2, strides=2)
        self.conv_3 = conv2d("conv_3", self.pool_2, filters=192, kernel_size=3,
                             activation=self.leaky_relu)
        self.pool_4 = max_pooling2d("pool_4", self.conv_3, pool_size=2, strides=2)
        self.conv_5 = conv2d("conv_5", self.pool_4, filters=128, kernel_size=1,
                             activation=self.leaky_relu)
        self.conv_6 = conv2d("conv_6", self.conv_5, filters=256, kernel_size=3,
                             activation=self.leaky_relu)
        self.conv_7 = conv2d("conv_7", self.conv_6, filters=256, kernel_size=1,
                             activation=self.leaky_relu)
        self.conv_8 = conv2d("conv_8", self.conv_7, filters=512, kernel_size=3,
                             activation=self.leaky_relu)
        self.pool_9 = max_pooling2d("pool_9", self.conv_8, pool_size=2, strides=2)
        self.conv_10 = conv2d("conv_10", self.pool_9, filters=256, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_11 = conv2d("conv_11", self.conv_10, filters=512, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_12 = conv2d("conv_12", self.conv_11, filters=256, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_13 = conv2d("conv_13", self.conv_12, filters=512, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_14 = conv2d("conv_14", self.conv_13, filters=256, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_15 = conv2d("conv_15", self.conv_14, filters=512, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_16 = conv2d("conv_16", self.conv_15, filters=256, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_17 = conv2d("conv_17", self.conv_16, filters=512, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_18 = conv2d("conv_18", self.conv_17, filters=512, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_19 = conv2d("conv_19", self.conv_18, filters=1024, kernel_size=3,
                              activation=self.leaky_relu)
        self.pool_20 = max_pooling2d("pool_20", self.conv_19, pool_size=2, strides=2)
        self.conv_21 = conv2d("conv_21", self.pool_20, filters=512, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_22 = conv2d("conv_22", self.conv_21, filters=1024, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_23 = conv2d("conv_23", self.conv_22, filters=512, kernel_size=1,
                              activation=self.leaky_relu)
        self.conv_24 = conv2d("conv_24", self.conv_23, filters=1024, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_25 = conv2d("conv_25", self.conv_24, filters=1024, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_26 = conv2d("conv_26", self.conv_25, filters=1024, kernel_size=3, strides=2,
                              activation=self.leaky_relu)
        self.conv_27 = conv2d("conv_27", self.conv_26, filters=1024, kernel_size=3,
                              activation=self.leaky_relu)
        self.conv_28 = conv2d("conv_28", self.conv_27, filters=1024, kernel_size=3,
                              activation=self.leaky_relu)
        self.fc_29 = fully_connected("fc_29", self.conv_28, filters=512,
                                     activation=self.leaky_relu)
        self.fc_30 = fully_connected("fc_30", self.fc_29, filters=4096,
                                     activation=self.leaky_relu)

        self.dropout_31 = tf.nn.dropout(self.fc_30, keep_prob)

        output_size = (self.cell_size * self.cell_size) * (self.num_classes + self.boxes_per_cell * 5)
        self.fc_32 = fully_connected("fc_32", self.dropout_31, filters=output_size, activation=None)

        return self.fc_32


class YoloV1Loss:
    """YOLO v1 loss function."""
    def __init__(
            self,
            is_debug=False,
            cell_size=7,
            boxes_per_cell=2,
            object_scale=1.0,
            no_object_scale=1.0,
            class_scale=1.0,
            coordinate_scale=5.0,
            image_size=[448, 448],
            batch_size=64,
            classes=[],
            yolo=None,
    ):
        self.is_debug = is_debug
        self.cell_size = cell_size
        self.boxes_per_cell = boxes_per_cell
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.class_scale = class_scale
        self.coordinate_scale = coordinate_scale
        self.image_size = image_size
        self.batch_size = batch_size
        self.classes = classes
        self.num_classes = len(classes)
        self.yolo = yolo

        self.convert_boxes_space_from_yolo_to_real = yolo.convert_boxes_space_from_yolo_to_real
        self.convert_boxes_space_from_real_to_yolo = yolo.convert_boxes_space_from_real_to_yolo

    def _iou(self, boxes1, boxes2):
        """Calculate ious.

        Args:
        boxes1: 5-D tensor [batch_size, cell_size, cell_size, boxes_per_cell, 4(x_center, y_center, w, h)]
        boxes2: 5-D tensor [batch_size, cell_size, cell_size, boxes_per_cell, 4(x_center, y_center, w, h)]
        Return:

        iou: 4-D tensor [batch_size, cell_size, cell_size, boxes_per_cell]
        """
        # left, top, right, bottom
        boxes1 = tf.stack([
            boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2,
            boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2,
            boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2,
            boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2,
        ], axis=4)

        # left, top, right, bottom
        boxes2 = tf.stack([
            boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2,
            boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2,
            boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2,
            boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2,
        ], axis=4)

        # calculate the left up point
        left_top = tf.maximum(boxes1[:, :, :, :, 0:2], boxes2[:, :, :, :, 0:2])

        right_bottom = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # calculate intersection. [batch_size, cell_size, cell_size, boxes_per_cell, 2]
        inter = right_bottom - left_top
        inter_square = inter[:, :, :, :, 0] * inter[:, :, :, :, 1]
        mask = tf.to_float(inter[:, :, :, :, 0] > 0) * tf.to_float(inter[:, :, :, :, 1] > 0)

        intersection = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        epsilon = 1e-10
        union = square1 + square2 - mask * inter_square

        return intersection / (union + epsilon)

    def _gt_boxes_to_cell_loop_cond(self, i, gt_boxes, cell_gt_box, object_mask):
        """Return True when gt_boxes is not dummy.

        Args:
        i: scalr Tensor. while loop counter
        gt_boxes: 2D Tensor [max_num_boxes, 5(center_x, center_y, w, h, class_id)]
        """
        # exclude dummy. class id `-1` is dummy.
        gt_mask = tf.not_equal(gt_boxes[:, 4], -1)

        result = i < tf.reduce_sum(tf.to_int32(gt_mask))

        return result

    def _gt_boxes_to_cell_loop_body(self, i, gt_boxes, cell_gt_box, object_mask):
        """Calculate the gt_boxes corresponding cell.

        the cell`s object_mask is assigned 1.0 and the cell gt boxes assign gt_boxes coordinate.

        Args:
        i: scalr Tensor. while loop counter
        gt_boxes: 2D Tensor [max_num_boxes, 5(center_x, center_y, w, h, class_id)]
        cell_gt_box: 3D Tensor [max_num_boxes, 5(center_x, center_y, w, h, class_id)]
        """
        center_y = gt_boxes[i, 1]
        center_x = gt_boxes[i, 0]

        cell_y_index = tf.to_int32(tf.floor((center_y / self.image_size[1]) * self.cell_size))
        cell_x_index = tf.to_int32(tf.floor((center_x / self.image_size[0]) * self.cell_size))

        boxes = []
        mask_list = []
        for y in range(self.cell_size):
            for x in range(self.cell_size):

                # Set True, when gt_boxes is in this [y, x] cell.
                condition = tf.logical_and(tf.equal(y, cell_y_index), tf.equal(x, cell_x_index))

                old_box = cell_gt_box[y, x, :]
                # If gt_boxes is in this [y, x] cell, the cell_gt_boxes is assigned gt_boxes coordinate,
                # else is assigned old box coordinate.
                box = tf.where(condition, gt_boxes[i, :], old_box)

                # If gt_boxes is in this [y, x] cell, the cell`s object_mask is assigned 1.0(True),
                # else is assigned old object_mask.
                mask = tf.where(condition, 1.0, object_mask[y, x, 0])

                boxes.append(box)
                mask_list.append(mask)

        updated_cell_gt_box = tf.reshape(tf.stack(boxes), (self.cell_size, self.cell_size, 5))
        updated_object_mask = tf.reshape(tf.stack(mask_list), (self.cell_size, self.cell_size, 1))

        return i + 1, gt_boxes, updated_cell_gt_box, updated_object_mask

    def _gt_boxes_to_cell(self, gt_boxes_list):
        """Check gt_boxes are not dummy, create cell_gt_boxes and object_mask from the gt_boxes.

        Args:
        gt_boxes_list: Tensor [batch_size, max_num_boxes, 4(center_x, center_y, w, h)]

        Return:
        cell_gt_boxes: Tensor [batch_size, cell_size, cell_size, 4(center_x, center_y, w, h)].
            copy from non dummy gt boxes coodinate to corresponding cell.
        object_maskes: Tensor [batch_size, cell_size, cell_size, 1]. the cell that has gt boxes is 1, none is 0.
        """
        cell_gt_boxes = []
        object_maskes = []
        for batch_index in range(self.batch_size):
            i = tf.constant(0)
            gt_boxes = gt_boxes_list[batch_index, :, :]
            cell_gt_box = tf.zeros([self.cell_size, self.cell_size, 5])
            object_mask = tf.zeros([self.cell_size, self.cell_size, 1])

            _, _, result_cell_gt_box, result_object_mask = tf.while_loop(
                self._gt_boxes_to_cell_loop_cond,
                self._gt_boxes_to_cell_loop_body,
                [i, gt_boxes, cell_gt_box, object_mask]
            )

            cell_gt_boxes.append(result_cell_gt_box)
            object_maskes.append(result_object_mask)

        cell_gt_boxes = tf.stack(cell_gt_boxes)
        object_maskes = tf.stack(object_maskes)

        return cell_gt_boxes, object_maskes

    def __call__(self, predict_classes, predict_confidence, predict_boxes, gt_boxes):
        """Loss function.

        Args:
        predict_classes: [batch_size, cell_size, cell_size, num_classes]
        predict_confidence: [batch_size, cell_size, cell_size, boxes_per_cell]
        predict_boxes: [batch_size, cell_size, cell_size, boxes_per_cell, 4(center_x, center_y, w, h)]
        gt_boxes: ground truth boxes 3D tensor. [batch_size, max_num_boxes, 4(center_x, center_y, w, h)].
        """
        with tf.name_scope("loss"):
            cell_gt_boxes, object_mask = self._gt_boxes_to_cell(gt_boxes)

            # for class loss
            truth_classes = tf.to_int32(cell_gt_boxes[:, :, :, 4])
            truth_classes = tf.one_hot(truth_classes, self.num_classes)

            # resize to real space.
            resized_predict_boxes = self.convert_boxes_space_from_yolo_to_real(predict_boxes)

            reshaped_gt_boxes = tf.reshape(
                cell_gt_boxes[:, :, :, :4],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])

            # reshaped_gt_boxes: [batch_size, cell_size, cell_size, boxes_per_cell, 4]
            reshaped_gt_boxes = tf.tile(reshaped_gt_boxes, [1, 1, 1, self.boxes_per_cell, 1])

            # iou: [batch_size, cell_size, cell_size, boxes_per_cell]
            iou = self._iou(resized_predict_boxes, reshaped_gt_boxes)

            # object_response_mask: [batch_size, cell_size, cell_size, boxes_per_cell]
            object_response_mask = tf.reduce_max(iou, 3, keep_dims=True)
            object_response_mask = tf.cast((iou >= object_response_mask), tf.float32) * object_mask

            if self.is_debug:
                object_response_mask = tf.Print(
                    object_response_mask,
                    [tf.shape(object_response_mask), object_response_mask],
                    message="object_response_mask:",
                    summarize=20000)

            # no_object_response_mask_mask: [batch_size, cell_size, cell_size, boxes_per_cell]
            no_object_response_mask = 1.0 - object_response_mask

            if self.is_debug:
                no_object_response_mask = tf.Print(
                    no_object_response_mask,
                    [tf.shape(no_object_response_mask), no_object_response_mask],
                    message="no_object_response_mask:",
                    summarize=20000)

            truth_boxes = self.convert_boxes_space_from_real_to_yolo(reshaped_gt_boxes)

            if self.is_debug:
                truth_boxes = tf.Print(
                    truth_boxes,
                    [tf.shape(truth_boxes), truth_boxes],
                    message="truth_boxes:",
                    summarize=20000)

            # coordinate_mask: [batch_size, cell_size, cell_size, boxes_per_cell, 1]
            coordinate_mask = tf.expand_dims(object_response_mask, 4)

            if self.is_debug:
                coordinate_mask = tf.Print(
                    coordinate_mask,
                    [tf.shape(coordinate_mask), coordinate_mask],
                    message="coordinate_mask:",
                    summarize=20000)

            # class_loss
            class_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(object_mask * (predict_classes - truth_classes)),
                    axis=[1, 2, 3]
                ),
                name='class_loss'
            ) * self.class_scale

            # object_loss
            object_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(object_response_mask * (predict_confidence - iou)),
                    axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # no_object_response_mask_loss
            no_object_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(no_object_response_mask * predict_confidence),
                    axis=[1, 2, 3]),
                name='no_object_loss') * self.no_object_scale

            # coordinate_loss
            coordinate_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(coordinate_mask * (predict_boxes - truth_boxes)),
                    axis=[1, 2, 3, 4]),
                name='coordinate_loss') * self.coordinate_scale

            loss = class_loss + object_loss + no_object_loss + coordinate_loss

            tf.summary.scalar("class", class_loss)
            tf.summary.scalar("object", object_loss)
            tf.summary.scalar("no_object", no_object_loss)
            tf.summary.scalar("coordinate", coordinate_loss)
            tf.summary.scalar("loss", loss)
        return loss


def summary_boxes(tag, images, boxes, image_size, max_outputs=3):
    """Draw bounding boxes images on Tensroboard.

    Args:
    tag: name of summary tag.
    images: Tesnsor of images [batch_size, height, widths, 3].
    boxes: Tensor of boxes. assumed shape is [batch_size, num_boxes, 4(y1, x1, y2, x2)].
    image_size: python list image size [height, width].
    """
    with tf.name_scope("summary_boxes"):

        boxes = tf.stack([
            boxes[:, :, 0] / image_size[0],
            boxes[:, :, 1] / image_size[1],
            boxes[:, :, 2] / image_size[0],
            boxes[:, :, 3] / image_size[1],
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

        center_x = x + w / 2
        center_y = y + h / 2

        return tf.concat([center_x, center_y, w, h], axis=axis)


def format_CXCYWH_to_XYWH(boxes, axis=1):
    """Format form (center_x, center_y, w, h) to (x, y, w, h) along specific dimention.

    Args:
    boxes: A tensor include boxes. [:, 4(x, y, w, h)]
    axis: Which dimension of the inputs Tensor is boxes.
    """
    with tf.name_scope('format_xywh_to_cxcywh'):
        center_x, center_y, w, h = tf.split(axis=axis, num_or_size_splits=4, value=boxes)

        x = center_x - w / 2
        y = center_y - h / 2

        return tf.concat([x, y, w, h], axis=axis)


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

        return tf.concat([y1, x1, y2, x2], axis=axis)


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

        return tf.concat([y1, x1, y2, x2], axis=axis)
