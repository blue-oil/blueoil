#!/usr/bin/env python3
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
import re

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops


# ===========================================================================
# Average precision computations on tensorflow
# Ref: https://github.com/balancap/SSD-Tensorflow/blob/master/tf_extended/metrics.py
# ===========================================================================
def average_precision(
        num_gt_boxes,
        tp,
        fp,
        score,
        class_name,
        metrics_collections=None,
        updates_collections=None,
):
    """Compute average precision.

    Args:
        num_gt_boxes(tf.Tensor): a scalar tensor. number of gt boxes.
        tp(tf.Tensor): tp vector. elements are int or bool.
        fp(tf.Tensor): fp vector. elements are int or bool.
        score(tf.Tensor): score vector.
        class_name(str): class_name

    Return:
        average precision(tf.Tensor): scalar
        presicion_array(tf.Tensor): vector of presicion.
        recall_array(tf.Tensor): vector of recall.
        presicion(tf.Tensor): scalr of presicion.
        recall(tf.Tensor): scalar of recall.
    """
    # replace non-alpha-num string.
    class_name = re.sub('[^0-9a-zA-Z]+', '_', class_name)

    (tp_value, fp_value, scores_value, num_gt_boxes_value), update_op = \
        _streaming_tp_fp_array(num_gt_boxes, tp, fp, score, class_name,
                               metrics_collections=metrics_collections, updates_collections=updates_collections)

    precision_array, recall_array, precision, recall = \
        _precision_recall(tp_value, fp_value, scores_value, num_gt_boxes_value, class_name)

    average_precision = _average_precision(precision_array, recall_array)

    return tf.tuple([average_precision, precision_array, recall_array, precision, recall]), update_op


def _streaming_tp_fp_array(
        num_gt_boxes,
        tp,
        fp,
        scores,
        class_name,
        remove_zero_scores=True,
        metrics_collections=None,
        updates_collections=None,
        name=None):
    """Streaming computation of True Positive and False Positive arrays. This metrics
    also keeps track of scores and number of grountruth objects.
    """
    default_name = 'streaming_tp_fp_{}'.format(class_name)
    # Input Tensors...
    with variable_scope.variable_scope(name, default_name,
                                       [num_gt_boxes, tp, fp, scores]):
        tp = tf.cast(tp, tf.bool)
        fp = tf.cast(fp, tf.bool)
        scores = tf.to_float(scores)
        num_gt_boxes = tf.to_int64(num_gt_boxes)

        # Reshape TP and FP tensors and clean away 0 class values.
        tp = tf.reshape(tp, [-1])
        fp = tf.reshape(fp, [-1])
        scores = tf.reshape(scores, [-1])

        # Remove TP and FP both false.
        if remove_zero_scores:
            mask = tf.logical_or(tp, fp)
            rm_threshold = 1e-4
            mask = tf.logical_and(mask, tf.greater(scores, rm_threshold))
            tp = tf.boolean_mask(tp, mask)
            fp = tf.boolean_mask(fp, mask)
            scores = tf.boolean_mask(scores, mask)

        # Local variables accumlating information over batches.
        tp_value = metrics_impl.metric_variable(
            shape=[0, ], dtype=tf.bool, name="tp_value", validate_shape=False)
        fp_value = metrics_impl.metric_variable(
            shape=[0, ], dtype=tf.bool, name="fp_value", validate_shape=False)
        scores_value = metrics_impl.metric_variable(
            shape=[0, ], dtype=tf.float32, name="scores_value", validate_shape=False)
        num_gt_boxes_value = metrics_impl.metric_variable(
            shape=[], dtype=tf.int64, name="num_gt_boxes_value")

        # Update operations.
        tp_op = tf.assign(tp_value, tf.concat([tp_value, tp], axis=0), validate_shape=False)
        fp_op = tf.assign(fp_value, tf.concat([fp_value, fp], axis=0), validate_shape=False)
        scores_op = tf.assign(scores_value, tf.concat([scores_value, scores], axis=0), validate_shape=False)
        num_gt_boxes_op = tf.assign_add(num_gt_boxes_value, num_gt_boxes)

        # Value and update ops.
        values = (tp_value, fp_value, scores_value, num_gt_boxes_value)
        update_ops = (tp_op, fp_op, scores_op, num_gt_boxes_op)

        if metrics_collections:
            ops.add_to_collections(metrics_collections, values)
        if updates_collections:
            ops.add_to_collections(updates_collections, update_ops)

        update_op = tf.group(*update_ops)
        return values, update_op


def _average_precision(precision, recall, name=None):
    """Compute (interpolated) average precision from precision and recall array Tensors.
    The implementation follows Pascal 2012 and ILSVRC guidelines.
    See also: https://sanchom.wordpress.com/tag/average-precision/
    """
    with tf.name_scope(name, 'average_precision', [precision, recall]):
        # Convert to float64 to decrease error on Riemann sums.
        precision = tf.cast(precision, dtype=tf.float64)
        recall = tf.cast(recall, dtype=tf.float64)

        # Add bounds values to precision and recall.
        precision = tf.concat([[0.], precision, [0.]], axis=0)
        recall = tf.concat([[0.], recall, [1.]], axis=0)
        # Ensures precision is increasing in reverse order.
        precision = _cummax(precision, reverse=True)

        # Riemann sums for estimating the integral.
        # mean_pre = (precision[1:] + precision[:-1]) / 2.
        mean_pre = precision[1:]
        diff_rec = recall[1:] - recall[:-1]
        ap = tf.reduce_sum(mean_pre * diff_rec)
        return ap


def _safe_div_zeros(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        tf.greater(denominator, 0),
        tf.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def _safe_div_ones(numerator, denominator, name):
    """Divides two values, returning 1 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      1 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        tf.greater(denominator, 0),
        tf.divide(numerator, denominator),
        tf.ones_like(numerator),
        name=name)


def _precision_recall(
        tp,
        fp,
        scores,
        num_gt_boxes,
        class_name,
        dtype=tf.float64,
        scope=None):
    """Compute precision and recall from scores, true positives and false
    positives booleans arrays
    """
    default_name = 'precision_recall_{}'.format(class_name)
    # Sort by score.
    with tf.name_scope(scope, default_name, [num_gt_boxes, tp, fp, scores]):
        num_detections = tf.size(scores)
        # Sort detections by score.
        scores, idxes = tf.nn.top_k(scores, k=num_detections, sorted=True)
        tp = tf.gather(tp, idxes)
        fp = tf.gather(fp, idxes)
        # Computer recall and precision.
        tp = tf.cumsum(tf.cast(tp, dtype), axis=0)
        fp = tf.cumsum(tf.cast(fp, dtype), axis=0)

        recall = _safe_div_ones(tp, tf.cast(num_gt_boxes, dtype), 'recall')
        precision = _safe_div_zeros(tp, tp + fp, 'precision')

        scalar_precision = tf.cond(
            tf.equal(tf.size(precision), 0),
            lambda: tf.constant(0, dtype=dtype),
            lambda: precision[-1],
            name="scalar_precision"
        )

        scalar_recall = tf.cond(
            tf.equal(tf.size(recall), 0),
            lambda: tf.constant(0, dtype=dtype),
            lambda: recall[-1],
            name="scalar_recall"
        )

        return tf.tuple([precision, recall, scalar_precision, scalar_recall])


def _cummax(x, reverse=False, name=None):
    """Compute the cumulative maximum of the tensor `x` along `axis`. This
    operation is similar to the more classic `cumsum`. Only support 1D Tensor
    for now.
    Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
       axis: A `Tensor` of type `int32` (default: 0).
       reverse: A `bool` (default: False).
       name: A name for the operation (optional).
    Returns:
    A `Tensor`. Has the same type as `x`.
    """
    with tf.name_scope(name, "Cummax", [x]) as name:
        # x = tf.convert_to_tensor(x, name="x")
        # Not very optimal: should directly integrate reverse into tf.scan.
        if reverse:
            x = tf.reverse(x, axis=[0])
        # 'Accumlating' maximum: ensure it is always increasing.
        cmax = tf.scan(lambda a, y: tf.maximum(a, y), x,
                       initializer=None, parallel_iterations=1,
                       back_prop=False, swap_memory=False)
        if reverse:
            cmax = tf.reverse(cmax, axis=[0])
    return cmax


# ===========================================================================
# Mean average precision computations on numpy
# ===========================================================================
def _mean_average_precision(all_predict_boxes, all_gt_boxes, classes, overlap_thresh=0.5):
    """Calcurate mean average precision.
    Args:
        all_predict_boxes(list): python list of numpy.ndarray. all images predicted boxes.
            all_pred_boxes[image_index] shape is [num_pred_boxes, 6(x, y, w, h, class, scores)]
        all_gt_boxes(numpy.ndarray): ground truth boxes.
            shape is [num_images, num_max_gt_boxes, 5(x, y, w, h, class)]
        classes: classes list.
        overlap_thresh: threshold of overlap.

    Return:
       dictionary include 'MeanAveragePrecision' 'AveragePrecision', 'Precision', 'Recall', 'OrderedPrecision',
       'OrderedRecall', 'OrderedMaxedPrecision'
    """

    result = {
        'MeanAveragePrecision': None,
        'AveragePrecision': [],
        'Precision': [],
        'Recall': [],
        'OrderedPrecision': [],
        'OrderedRecall': [],
    }

    for class_index, class_name in enumerate(classes):
        pred_boxes, gt_boxes = \
            _boxes_in_the_class(all_predict_boxes, all_gt_boxes, class_index)

        tp, fp, num_gt_boxes = \
            _tp_and_fp(pred_boxes, gt_boxes, overlap_thresh)

        ordered_precision, ordered_recall, precision, recall = _calc_precision_recall(tp, fp, num_gt_boxes)
        average_precision = _calc_average_precision(ordered_precision, ordered_recall)

        result['AveragePrecision'].append(average_precision)
        result['Precision'].append(precision)
        result['Recall'].append(recall)
        result['OrderedPrecision'].append(ordered_precision)
        result['OrderedRecall'].append(ordered_recall)

    MAP = sum(result['AveragePrecision'])/len(classes)
    result['MeanAveragePrecision'] = MAP

    return result


def _boxes_in_the_class(all_predict_boxes, all_gt_boxes, class_index):
    """Get pred_boxes and gt_boxes in a class to make it easy to be calculated precision, rec all and so on.

    Args:
        all_predict_boxes(list): python list of numpy.ndarray. all images predicted boxes.
            all_pred_boxes[image_index] shape is [num_pred_boxes, 6(x, y, w, h, class, scores)]
        all_gt_boxes(numpy.ndarray): ground truth boxes.
            shape is [num_images, num_max_gt_boxes, 5(x, y, w, h, class)]
        class_index(int): index of classes.

    Return:
        pred_boxes_in_the_class_list(list): predicted boxes in the class.
            pred_boxes_in_the_class_list[image_index] shape is [num_pred_boxes, 6(x, y, w, h, class, scores)]
        gt_boxes_in_the_class_list(list): ground truth boxes in the class.
            gt_boxes_in_the_class_list[image_index] shape is [num_gt_boxes, 5(x, y, w, h, class)]
   """
    num_images = len(all_predict_boxes)
    gt_boxes_in_the_class_list = []
    pred_boxes_in_the_class_list = []

    assert len(all_predict_boxes) == len(all_gt_boxes)

    for image_index in range(num_images):
        gt_boxes = all_gt_boxes[image_index]
        pred_boxes = all_predict_boxes[image_index]
        if len(gt_boxes) == 0:
            gt_boxes_in_the_class = np.empty(shape=(0, 5))
        else:
            gt_boxes_in_the_class = gt_boxes[gt_boxes[:, 4] == class_index]

        pred_boxes_in_the_class = pred_boxes[pred_boxes[:, 4] == class_index]

        gt_boxes_in_the_class_list.append(gt_boxes_in_the_class)
        pred_boxes_in_the_class_list.append(pred_boxes_in_the_class)
    return pred_boxes_in_the_class_list, gt_boxes_in_the_class_list


def _tp_and_fp(class_pred_boxes, class_gt_boxes, overlap_thresh):
    """Calculate tp and fp in the classes.

    Args:
        pred_boxes_in_the_class_list(list): predicted boxes in the class.
            pred_boxes_in_the_class_list[image_index] shape is [num_pred_boxes, 6(x, y, w, h, class, scores)]
        gt_boxes_in_the_class_list(list): ground truth boxes in the class.
            gt_boxes_in_the_class_list[image_index] shape is [num_gt_boxes, 5(x, y, w, h, class)]

    Return:
        tps(np.ndarray): prediction boxes length vector of tp sorted by score.
        fps(np.ndarray): prediction boxes length vector of fp sorted by score.
        num_gt_boxes(int): number of gt boxes.
    """
    assert len(class_pred_boxes) == len(class_gt_boxes)

    num_gt_boxes = 0

    tps = []
    fps = []
    scores = []

    for image_index in range(len(class_pred_boxes)):
        pred_boxes_in_image = class_pred_boxes[image_index]
        gt_boxes_in_image = class_gt_boxes[image_index]

        num_gt_boxes += len(gt_boxes_in_image)
        tp, fp, score = tp_fp_in_the_image(pred_boxes_in_image, gt_boxes_in_image, overlap_thresh)
        tps.append(tp)
        fps.append(fp)
        scores.append(score)

    tps = np.concatenate(tps)
    fps = np.concatenate(fps)
    scores = np.concatenate(scores)

    sort_index = np.argsort(-scores, axis=0)

    return tps[sort_index], fps[sort_index], num_gt_boxes


def tp_fp_in_the_image(pred_boxes, gt_boxes, overlap_thresh):
    """Calculate tp and fp in the image.

    Args:
        pred_boxes(numpy.ndarray): predicted boxes in the image.
            shape is [num_pred_boxes, 6(x, y, w, h, class, scores)]
        gt_boxes(numpy.ndarray): ground truth boxes in the image.
            shape is [num_gt_boxes, 5(x, y, w, h, class)]

    Return:
       tp(numpy.ndarray): prediction boxes length vector of tp.
       fp(numpy.ndarray): prediction boxes length vector of fp.
       score(numpy.ndarray): prediction boxes length vector of score.
    """
    sort_index = np.argsort(-pred_boxes[:, 5], axis=0)
    sorted_pred_boxes = pred_boxes[sort_index]

    tp = np.zeros(len(pred_boxes), dtype=np.float32)
    fp = np.zeros(len(pred_boxes), dtype=np.float32)
    score = np.empty(len(pred_boxes), dtype=np.float32)
    is_gt_boxes_used = [False] * len(gt_boxes)

    for box in range(len(pred_boxes)):
        pred_box = sorted_pred_boxes[box]
        score[box] = pred_box[5]

        # when ground truth boxes is zero, all predicted boxes mark false positive.
        if gt_boxes.size == 0:
            fp[box] = 1.
            continue

        overlaps = _calc_overlap(gt_boxes, pred_box)
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > overlap_thresh:
            if not is_gt_boxes_used[jmax]:
                tp[box] = 1.
                is_gt_boxes_used[jmax] = True
            else:
                fp[box] = 1.
        else:
            fp[box] = 1.

    return tp, fp, score


def _calc_precision_recall(tp, fp, num_gt_boxes):
    """Calculate precision and recall array.

    Args:
        tp(np.ndarray): sorted tp.
        fp(np.ndarray): sorted fp.
        num_gt_boxes(int): number of gt boxes

    Return:
        precision: detection boxes size precision array
        recall: detection boxes size recall array
        scalar_precision: precision
        scalar_recall: recall
    """
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    # when gt_boxes is zero, Recall define 100%.
    if num_gt_boxes == 0:
        recall = np.ones(tp.size)
    else:
        recall = tp / num_gt_boxes

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    scalar_precision = precision[-1] if precision.size != 0 else 0
    scalar_recall = recall[-1] if recall.size != 0 else 0

    return precision, recall, scalar_precision, scalar_recall


def _calc_average_precision(precision, recall):
    # correct AP calculation
    # first append sentinel values at the end
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recall[1:] != recall[:-1])[0]

    # and sum (\Delta recall) * prec
    mean_precision = precision[i + 1]
    diff_recall = recall[i + 1] - recall[i]
    average_precision = np.sum(diff_recall * mean_precision)

    return average_precision


def _calc_overlap(gt_boxes, pred_box):
    """Calcurate overlap
    Args:
        gt_boxes: ground truth boxes in the image. shape is [num_boxes, 5(x, y, w, h, class)]
        pred_box: a predict box in the image. shape is [6(x, y, w, h, class, prob)]
    Return:
    """
    assert gt_boxes.size != 0, "Cannot clculate if ground truth boxes is zero"

    # compute overlaps
    gt_boxes_xmin = gt_boxes[:, 0]
    gt_boxes_ymin = gt_boxes[:, 1]
    gt_boxes_xmax = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_boxes_ymax = gt_boxes[:, 1] + gt_boxes[:, 3]

    pred_box_xmin = pred_box[0]
    pred_box_ymin = pred_box[1]
    pred_box_xmax = pred_box[0] + pred_box[2]
    pred_box_ymax = pred_box[1] + pred_box[3]

    # If border pixels are supposed to expand the bounding boxes each 0.5 pixel,
    # we have to add 1 pixel to any difference `xmax - xmin` or `ymax - ymin`.
    d = 1.

    # intersection
    inter_xmin = np.maximum(gt_boxes_xmin, pred_box_xmin)
    inter_ymin = np.maximum(gt_boxes_ymin, pred_box_ymin)
    inter_xmax = np.minimum(gt_boxes_xmax, pred_box_xmax)
    inter_ymax = np.minimum(gt_boxes_ymax, pred_box_ymax)
    inter_w = np.maximum(inter_xmax - inter_xmin + d, 0.)
    inter_h = np.maximum(inter_ymax - inter_ymin + d, 0.)
    inters = inter_w * inter_h

    # union
    union = (pred_box_xmax - pred_box_xmin + d) * (pred_box_ymax - pred_box_ymin + d) \
        + (gt_boxes_xmax - gt_boxes_xmin + d) * (gt_boxes_ymax - gt_boxes_ymin + d) \
        - inters

    return inters / union
