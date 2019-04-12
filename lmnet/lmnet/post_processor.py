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

from lmnet.data_processor import (
    Processor,
)
from lmnet.data_augmentor import iou


def _softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.expand_dims(exp.sum(axis=-1), -1)


def format_cxcywh_to_xywh(boxes, axis=1):
    """Format form (center_x, center_y, w, h) to (x, y, w, h) along specific dimention.

    Args:
    boxes: A tensor include boxes. [:, 4(x, y, w, h)]
    axis: Which dimension of the inputs Tensor is boxes.
    """
    results = np.split(boxes, [1, 2, 3, 4], axis=axis)
    center_x, center_y, w, h = results[0], results[1], results[2], results[3]
    x = center_x - (w / 2)
    y = center_y - (h / 2)

    return np.concatenate([x, y, w, h], axis=axis)


class FormatYoloV2(Processor):
    """Yolov2 postprocess.
    Format outputs of yolov2 last convolution to object detection style.
    """

    def __init__(self, image_size, classes, anchors, data_format):
        self.image_size = image_size
        self.num_classes = len(classes)
        self.anchors = anchors
        self.boxes_per_cell = len(anchors)
        self.data_format = data_format

    @property
    def num_cell(self):
        return self.image_size[0] // 32, self.image_size[1] // 32

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _sprit_prediction(self, outputs):
        """Separate combined final convolution outputs to predictions.

        Args:
            outputs: combined final convolution outputs 4D Tensor.
                shape is [batch_size, num_cell[0], num_cell[1],  (num_classes + 5) * boxes_per_cell]

        Returns:
            predict_classes(Tensor): [batch_size, num_cell[0], num_cell[1], boxes_per_cell, num_classes]
            predict_confidence(Tensor): [batch_size, num_cell[0], num_cell[1], boxes_per_cell, 1]
            predict_boxes(Tensor): [batch_size, num_cell[0], num_cell[1], boxes_per_cell, 4(center_x, center_y, w, h)]
        """
        batch_size = len(outputs)
        num_cell_y, num_cell_x = self.num_cell[0], self.num_cell[1]

        outputs = np.reshape(
            outputs,
            [batch_size, num_cell_y, num_cell_x, self.boxes_per_cell, self.num_classes + 5]
        )

        outputs = np.split(outputs, [self.num_classes, self.num_classes+1, self.num_classes+1+4], axis=4)

        predict_classes, predict_confidence, predict_boxes = outputs[0], outputs[1], outputs[2]
        return predict_classes, predict_confidence, predict_boxes

    def _offset_boxes(self, batch_size, num_cell_y, num_cell_x):
        """Numpy implementing of offset_boxes.
        Return yolo space offset of x and y and w and h.

        Args:
            batch_size(int): batch size
            num_cell_y: Number of cell y. The spatial dimension of the final convolutional features.
            num_cell_x: Number of cell x. The spatial dimension of the final convolutional features.
        """

        offset_y = np.arange(num_cell_y)
        offset_y = np.reshape(offset_y, (1, num_cell_y, 1, 1))
        offset_y = np.broadcast_to(offset_y, [batch_size, num_cell_y, num_cell_x, self.boxes_per_cell])

        offset_x = np.arange(num_cell_x)
        offset_x = np.reshape(offset_x, (1, 1, num_cell_x, 1))
        offset_x = np.broadcast_to(offset_x, [batch_size, num_cell_y, num_cell_x, self.boxes_per_cell])

        w_anchors = [anchor_w for anchor_w, anchor_h in self.anchors]
        offset_w = np.broadcast_to(w_anchors, (batch_size, num_cell_y, num_cell_x, self.boxes_per_cell))
        offset_w = offset_w.astype(np.float32)

        h_anchors = [anchor_h for anchor_w, anchor_h in self.anchors]
        offset_h = np.broadcast_to(h_anchors, (batch_size, num_cell_y, num_cell_x, self.boxes_per_cell))
        offset_h = offset_h.astype(np.float32)

        return offset_x, offset_y, offset_w, offset_h

    def _convert_boxes_space_from_yolo_to_real(self, predict_boxes):
        """Convert predict boxes space size from yolo to real.

        Real space boxes coodinates are in the interval [0, image_size].
        Yolo space boxes x,y are in the interval [-1, 1]. w,h are in the interval [-inf, +inf].

        Args:
            predict_boxes: 5D np.ndarray.
                           shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].

        Returns:
            resized_boxes: 5D np.ndarray,
                           shape is [batch_size, num_cell, num_cell, boxes_per_cell, 4(center_x, center_y, w, h)].
        """
        batch_size = len(predict_boxes)
        image_size_h, image_size_w = self.image_size[0], self.image_size[1]
        num_cell_y, num_cell_x = self.num_cell[0], self.num_cell[1]

        offset_x, offset_y, offset_w, offset_h = self._offset_boxes(batch_size, num_cell_y, num_cell_x)

        resized_predict_boxes = np.stack([
            (predict_boxes[:, :, :, :, 0] + offset_x) / num_cell_x,
            (predict_boxes[:, :, :, :, 1] + offset_y) / num_cell_y,
            np.exp(predict_boxes[:, :, :, :, 2]) * offset_w / num_cell_x,
            np.exp(predict_boxes[:, :, :, :, 3]) * offset_h / num_cell_y,
        ], axis=4)

        resized_predict_boxes = resized_predict_boxes * [
            image_size_w,
            image_size_h,
            image_size_w,
            image_size_h,
        ]

        return resized_predict_boxes

    def __call__(self, outputs, **kwargs):
        """
        Args:
            outputs (np.ndarray): Outputs of yolov2 last convolution.
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
        Returns:
            all args (dict): Contains processed outputs.
                outputs: Object detection formatted list op np.ndarray.
                List is [predict_boxes(np.ndarray), predict_boxes(np.ndarray), ...] which length is batch size.
                Each predict_boxes shape is [num_predict_boxes, 6(x(left), y(top), w, h, class_id, score)]
        """

        num_cell_y, num_cell_x = self.num_cell[0], self.num_cell[1]
        batch_size = len(outputs)

        if self.data_format == "NCHW":
            outputs = np.transpose(outputs, [0, 2, 3, 1])

        predict_classes, predict_confidence, predict_boxes = self._sprit_prediction(outputs)

        predict_classes = _softmax(predict_classes)
        predict_confidence = self.sigmoid(predict_confidence)
        predict_boxes = np.stack([
            self.sigmoid(predict_boxes[:, :, :, :, 0]),
            self.sigmoid(predict_boxes[:, :, :, :, 1]),
            predict_boxes[:, :, :, :, 2],
            predict_boxes[:, :, :, :, 3],
        ], axis=4)

        predict_boxes = self._convert_boxes_space_from_yolo_to_real(predict_boxes)

        predict_boxes = format_cxcywh_to_xywh(predict_boxes, axis=4)

        results = []
        for class_id in range(self.num_classes):
            precit_prob = predict_classes[:, :, :, :, class_id]
            predict_score = precit_prob * predict_confidence[:, :, :, :, 0]

            result = np.stack([
                predict_boxes[:, :, :, :, 0],
                predict_boxes[:, :, :, :, 1],
                predict_boxes[:, :, :, :, 2],
                predict_boxes[:, :, :, :, 3],
                np.full(predict_score.shape, class_id),
                predict_score,
            ], axis=4)

            results.append(result)

        results = np.stack(results, axis=1)

        results = np.reshape(
            results,
            [batch_size, num_cell_y * num_cell_x * self.boxes_per_cell * self.num_classes, 6]
        )

        return dict({"outputs": results}, **kwargs)


class ExcludeLowScoreBox(Processor):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, outputs, **kwargs):
        results = []
        batch_size = len(outputs)
        for i in range(batch_size):
            boxes_per_batch = outputs[i]
            result = boxes_per_batch[boxes_per_batch[:, 5] > self.threshold, :]
            results.append(result)

        return dict({"outputs": results}, **kwargs)


class NMS(Processor):
    """Non Maximum Suppression"""

    def __init__(self, classes, iou_threshold, max_output_size=100, per_class=True):
        """
        Args:
            classes(list): List of class names.
            iou_threshold(float): The threshold for deciding whether boxes overlap with respect to IOU.
            max_output_size(int): The maximum number of boxes to be selected
            per_class(boolean): Whether or not, NMS respect to per class.
        """

        self.classes = classes
        self.iou_threshold = iou_threshold
        self.max_output_size = max_output_size
        self.per_class = per_class

    def _nms(self, boxes):
        scores = boxes[:, 5]

        order_indices = np.argsort(-scores)

        keep_indices = []

        while order_indices.size > 0:
            i = order_indices[0]
            keep_indices.append(i)
            if order_indices.size == 1:
                break
            ious = iou(boxes[order_indices[1:], :], boxes[i, :])
            remain_boxes = np.where(ious < self.iou_threshold)[0]
            remain_boxes = remain_boxes + 1

            order_indices = order_indices[remain_boxes]

        nms_boxes = boxes[keep_indices, :]
        if len(nms_boxes) > self.max_output_size:
            nms_boxes = nms_boxes[:self.max_output_size, :]
        return nms_boxes

    def __call__(self, outputs, **kwargs):
        """
        Args:
            outputs: The boxes list of predict boxes for each image.
                The format is [boxes, boxes, boxes, ...]. len(boxes) == batch_size.
                boxes[image_id] is np.array, the shape is (num_boxes, 6[x(left), y(top), h, w, class_id, sore])

        Returns:
            all args (dict): Contains boxes list (list).
                outputs: The boxes list of predict boxes for each image.
                    The format is [boxes, boxes, boxes, ...]. len(boxes) == batch_size.
                    boxes[image_id] is np.array, the shape is (num_boxes, 6[x(left), y(top), h, w, class_id, sore])
        """
        results = []
        batch_size = len(outputs)
        for i in range(batch_size):
            boxes_per_batch = outputs[i]
            result_per_batch = []

            if self.per_class:
                for class_id in range(len(self.classes)):
                    class_masked = boxes_per_batch[boxes_per_batch[:, 4] == class_id]
                    nms_boxes = self._nms(class_masked)
                    result_per_batch.append(nms_boxes)
            else:
                nms_boxes = self._nms(boxes_per_batch)
                result_per_batch.append(nms_boxes)

            result_per_batch = np.concatenate(result_per_batch)
            results.append(result_per_batch)

        return dict({"outputs": results}, **kwargs)


class Bilinear(Processor):
    """Bilinear

    Change feature map spatial size with bilinear method, currenly support only up-sampling.
    """

    def __init__(self, size, data_format="NHWC", compatible_tensorflow_v1=True):
        """
        Args:
            size (list): Target size [height, width].
            data_format (string): currently support only "NHWC".
            compatible_tensorflow_v1 (bool): When the flag is True, it is compatible with tensorflow v1 resize function. Otherwise tensorflow v2 and Pillow resize function.
                Tensorflow v1 image resize function `tf.image.resize_bilinear()` which has the bug of calculataion of center pixel. The bug is fixed in tensorflow v2 `tf.image.resize()` which given the same result as Pillow's resize.
                See also https://github.com/tensorflow/tensorflow/issues/6720 and https://github.com/tensorflow/tensorflow/commit/3ae2c6691b7c6e0986d97b150c9283e5cc52c15f
        """ # NOQA

        self.size = size
        self.data_format = data_format
        # TODO(wakisaka): support "NCHW" format.
        assert data_format == "NHWC", "data_format only support NHWC but given {}".format(data_format)
        self.compatible_tensorflow_v1 = compatible_tensorflow_v1

    def __call__(self, outputs, **kwargs):
        """
        Args:
            outputs (numpy.ndarray): 4-D ndarray of network outputs to be resized cahnnel-wise.

        Returns:
            all args (dict):
                outputs (numpy.ndarray): resized outputs. 4-D ndarray.
        """
        output_height = self.size[0]
        output_width = self.size[1]

        input_height = outputs.shape[0]
        input_width = outputs.shape[1]

        # TODO(wakisaka): should support downsample.
        assert output_height >= input_height
        assert output_width >= input_width

        batch_size = len(outputs)
        results = []
        for i in range(batch_size):
            image = outputs[i, :, :, :]
            image = self._bilinear(image, size=self.size, compatible_tensorflow_v1=self.compatible_tensorflow_v1)
            results.append(image)
        results = np.array(results)
        return dict({'outputs': results}, **kwargs)

    @staticmethod
    def _bilinear(inputs, size, compatible_tensorflow_v1=True):
        output_height = size[0]
        output_width = size[1]

        input_height = inputs.shape[0]
        input_width = inputs.shape[1]
        channel = inputs.shape[2]

        if compatible_tensorflow_v1:
            scale = [(input_height - 1)/(output_height - 1), (input_width - 1)/(output_width - 1)]
        else:
            scale = [(input_height - 0)/(output_height - 0), (input_width - 0)/(output_width - 0)]

        output = np.zeros((output_height, output_width, channel)).astype(inputs.dtype)

        for h in range(output_height):
            for w in range(output_width):
                if compatible_tensorflow_v1:
                    center_y = h * scale[0]
                    center_x = w * scale[1]
                    int_y = int(center_y)
                    int_x = int(center_x)
                else:
                    center_y = (h + 0.5) * (scale[0]) - 0.5
                    center_x = (w + 0.5) * (scale[1]) - 0.5
                    int_y = int(math.floor(center_y))
                    int_x = int(math.floor(center_x))

                dx = center_x - int_x
                dy = center_y - int_y

                top = max(int_y, 0)
                bottom = min(int_y + 1, input_height-1)
                left = max(int_x, 0)
                right = min(int_x + 1, input_width-1)

                t_l = inputs[top, left, :]
                t_r = inputs[top, right, :]
                b_l = inputs[bottom, left, :]
                b_r = inputs[bottom, right, :]

                t = t_l + (t_r - t_l) * dx
                b = b_l + (b_r - b_l) * dx
                out = t + (b - t) * dy

                output[h, w, :] = out
        return output


class Softmax(Processor):

    def __call__(self, outputs, **kwargs):
        results = _softmax(outputs)
        return dict({'outputs': results}, **kwargs)
