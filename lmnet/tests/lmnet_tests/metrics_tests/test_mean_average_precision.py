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

from blueoil.nn.metrics.mean_average_precision import (
    _mean_average_precision,
    _calc_average_precision,
    _average_precision,
    tp_fp_in_the_image,
    average_precision,
)


def test_mean_average_precision():
    # case1
    classes = ["class_1", "class_2", ]
    gt_boxes = np.array([
        [[415, 185, 34, 27, 1],
         [362, 224, 53, 40, 1],
         [455, 227, 24, 21, 0],
         [417, 221, 26, 20, 1],
         [0, 0, 0, 0, -1], ],
        [[1415, 1185, 34, 27, 1],
         [162, 1124, 53, 40, 1],
         [1425, 2127, 24, 21, 0],
         [1100, 1121, 26, 20, 0],
         [0, 0, 0, 0, -1], ]
    ])

    all_boxes = [
        np.array(
            [[415, 185, 34, 27, 0, 0.2],
             [415, 185, 34, 27, 1, 0.8],
             [362, 224, 53, 40, 0, 0.8],
             [362, 224, 53, 40, 1, 0.2],
             [455, 227, 24, 21, 0, 0.7],
             [455, 227, 24, 21, 1, 0.3],
             [417, 221, 26, 20, 0, 0.1],
             [417, 221, 26, 20, 1, 0.9]]
        ),
        np.array(
            [[1415, 1185, 34, 27, 0, 0.8],
             [1415, 1185, 34, 27, 1, 0.2],
             [162, 1124, 53, 40, 0, 0.6],
             [162, 1124, 53, 40, 1, 0.4],
             [1425, 2127, 24, 21, 0, 0.5],
             [1425, 2227, 24, 21, 1, 0.5],
             [1100, 1121, 26, 20, 0, 0.9],
             [1100, 1121, 26, 20, 1, 0.1]]
        )
    ]

    expected = {
        'MeanAveragePrecision': (2./3.+117./28./5.)/2.,
        'AveragePrecision': [2./3., 117./28./5.],
        'Precision': [3./8, 5./8],
        'Recall': [1., 1.],
        'OrderedPrecision': np.array(
            [[1.0, 0.5, 1./3, 0.5, 0.4, 0.5, 3./7, 3./8], [1.0, 1.0, 2./3, 3./4., 0.6, 2./3, 5./7, 5./8]]
        ),
        'OrderedRecall': np.array(
            [[1./3, 1./3, 1./3, 2./3, 2./3, 1., 1., 1.], [0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 1., 1.]]
        ),
    }

    result = _mean_average_precision(all_boxes, gt_boxes, classes)

    assert np.allclose(result['MeanAveragePrecision'], expected['MeanAveragePrecision'])
    assert np.allclose(result['AveragePrecision'], expected['AveragePrecision'])
    assert np.allclose(result['Precision'], expected['Precision'])
    assert np.allclose(result['Recall'], expected['Recall'])
    assert np.allclose(result['OrderedPrecision'], expected['OrderedPrecision'])
    assert np.allclose(result['OrderedRecall'], expected['OrderedRecall'])

    # case:2
    classes = ["class_1", "class_2", ]
    gt_boxes = [
        np.array(
            [[-30, 0, 34, 27, 0],
             [-66, -84, 33, 40, 1],
             [5, 27, 24, 41, 0],
             [87, -1, 46, 40, 1],
             [0, 0, 0, 0, -1], ]
        ),
        np.array(
            [[-35, 3, 34, 27, 0],
             [-162, -124, 53, 40, 0],
             [0, 0, 24, 21, 1],
             [50, 1, 26, 20, 1],
             [50, -50, 30, 30, 0]]
        )
    ]

    all_boxes = [
        np.array(
            [[-34, 5, 34, 27, 0, 0.8],
             [-34, 5, 34, 27, 1, 0.2],
             [-62, -54, 30, 42, 0, 0.4],
             [-62, -54, 30, 42, 1, 0.6],
             [4, 27, 24, 44, 0, 0.7],
             [4, 27, 24, 44, 1, 0.3],
             [87, 1, 46, 40, 0, 0.05],
             [87, 1, 46, 40, 1, 0.95],
             [87, 10, 46, 40, 0, 0.05],
             [87, 10, 46, 40, 1, 0.95]]
        ),
        np.array(
            [[-30, 5, 34, 27, 0, 0.45],
             [-30, 5, 34, 27, 1, 0.55],
             [-162, -123, 50, 40, 0, 0.55],
             [-162, -123, 50, 40, 1, 0.45],
             [50, 2, 24, 21, 0, 0.2],
             [50, 2, 24, 21, 1, 0.8],
             [50, -50, 30, 30, 0, 0.9],
             [50, -50, 30, 30, 1, 0.1]]
        )
    ]

    expected = {
        'MeanAveragePrecision': (1. + 5./12)/2,
        'AveragePrecision': [1., 5./12.],
        'Precision': [5./9, 2./9],
        'Recall': [1., 0.5],
        'OrderedPrecision': np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 5./6, 5./7, 5./8, 5./9], [1.0, 0.5, 2./3, 0.5, 0.4, 1./3, 2./7, 2./8, 2./9]]
        ),
        'OrderedRecall': np.array(
            [[1./5, 2./5, 3./5, 4./5, 1., 1., 1., 1., 1.], [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
        ),
    }

    result = _mean_average_precision(all_boxes, gt_boxes, classes, 0.5)

    assert np.allclose(result['MeanAveragePrecision'], expected['MeanAveragePrecision'])
    assert np.allclose(result['AveragePrecision'], expected['AveragePrecision'])
    assert np.allclose(result['Precision'], expected['Precision'])
    assert np.allclose(result['Recall'], expected['Recall'])
    assert np.allclose(result['OrderedPrecision'], expected['OrderedPrecision'])
    assert np.allclose(result['OrderedRecall'], expected['OrderedRecall'])

    # case:3 class 1 dont have ground truth.
    classes = ["class_1", "class_2", ]
    gt_boxes = np.array([
        [[-66, -84, 33, 40, 1],
         [87, -1, 46, 40, 1]],
        [[0, 0, 24, 21, 1],
         [50, 1, 26, 20, 1]]
    ])

    all_boxes = [
        np.array(
            [[-34, 5, 34, 27, 0, 0.8],
             [-34, 5, 34, 27, 1, 0.2],
             [-62, -54, 30, 42, 0, 0.4],
             [-62, -54, 30, 42, 1, 0.6],
             [4, 27, 24, 44, 0, 0.7],
             [4, 27, 24, 44, 1, 0.3],
             [87, 1, 46, 40, 0, 0.05],
             [87, 1, 46, 40, 1, 0.95],
             [87, 10, 46, 40, 0, 0.05],
             [87, 10, 46, 40, 1, 0.95]]
        ),
        np.array(
            [[-30, 5, 34, 27, 0, 0.45],
             [-30, 5, 34, 27, 1, 0.55],
             [-162, -123, 50, 40, 0, 0.55],
             [-162, -123, 50, 40, 1, 0.45],
             [50, 2, 24, 21, 0, 0.2],
             [50, 2, 24, 21, 1, 0.8],
             [50, -50, 30, 30, 0, 0.9],
             [50, -50, 30, 30, 1, 0.1]]
        )
    ]

    expected = {
        'MeanAveragePrecision': (0. + 5./12)/2,
        'AveragePrecision': [0., 5./12.],
        'Precision': [0, 2./9],
        'Recall': [1., 0.5],
        'OrderedPrecision': np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 2./3, 0.5, 0.4, 1./3, 2./7, 2./8, 2./9]]
        ),
        'OrderedRecall': np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1.], [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
        ),
    }

    result = _mean_average_precision(all_boxes, gt_boxes, classes, 0.5)
    assert np.allclose(result['MeanAveragePrecision'], expected['MeanAveragePrecision'])
    assert np.allclose(result['AveragePrecision'], expected['AveragePrecision'])
    assert np.allclose(result['Precision'], expected['Precision'])
    assert np.allclose(result['Recall'], expected['Recall'])
    assert np.allclose(result['OrderedPrecision'], expected['OrderedPrecision'])
    assert np.allclose(result['OrderedRecall'], expected['OrderedRecall'])


def _tf_mean_average_precision(classes, gt_boxes, all_boxes, expected):
    graph = tf.Graph()
    with graph.as_default():
        overlap_thresh = 0.5
        labels = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 5])
        predict_boxes = [tf.compat.v1.placeholder(tf.float32, shape=[None, 6]) for boxes in all_boxes]

        average_precisions = []
        ordered_precisions = []
        ordered_recalls = []
        precisions = []
        recalls = []

        updates = []
        for class_id, class_name in enumerate(classes):
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
                num_gt_boxes = tf.Print(num_gt_boxes, [num_gt_boxes], message="num_gt_boxes_{}: ".format(class_name))

                tp, fp, score = tf.py_func(
                    tp_fp_in_the_image,
                    [predict_boxes_in_the_image, labels_in_the_image, overlap_thresh],
                    [tf.float32, tf.float32, tf.float32],
                    stateful=False,
                )

                tp = tf.Print(tp, [tp], message="tp_{}: ".format(class_name), summarize=200)
                fp = tf.Print(fp, [fp], message="fp_{}: ".format(class_name), summarize=200)

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

            precision_array = \
                tf.Print(precision_array, [precision_array], message="precision_{}: ".format(class_name), summarize=200)
            recall_array = \
                tf.Print(recall_array, [recall_array], message="recall_{}: ".format(class_name), summarize=200)

            updates.append(update_op)

            average_precisions.append(average_precision_value)
            ordered_precisions.append(precision_array)
            ordered_recalls.append(recall_array)
            precisions.append(precision)
            recalls.append(recall)

        mean_average_precision = tf.add_n(average_precisions) / len(classes)
        metrics_update_op = tf.group(*updates)
        init_op = tf.local_variables_initializer()

    sess = tf.Session(graph=graph)

    sess.run(init_op)

    feed_dict = {
        labels: gt_boxes,
        predict_boxes[0]: all_boxes[0],
        predict_boxes[1]: all_boxes[1],
    }
    sess.run(metrics_update_op, feed_dict=feed_dict)

    assert np.allclose(sess.run(mean_average_precision), expected['MeanAveragePrecision'])
    assert np.allclose(sess.run(average_precisions), expected['AveragePrecision'])
    assert np.allclose(sess.run(precisions), expected['Precision'])
    assert np.allclose(sess.run(recalls), expected['Recall'])
    for i, j in zip(sess.run(ordered_precisions), expected['OrderedPrecision']):
        assert np.allclose(i, j)
    for i, j in zip(sess.run(ordered_recalls), expected['OrderedRecall']):
        assert np.allclose(i, j)


def test_tf_mean_average_precision():
    # case1
    classes = ["class_1", "class_2", ]

    gt_boxes = np.array([
        [[415, 185, 34, 27, 1],
         [362, 224, 53, 40, 1],
         [455, 227, 24, 21, 0],
         [417, 221, 26, 20, 1],
         [0, 0, 0, 0, -1], ],
        [[1415, 1185, 34, 27, 1],
         [162, 1124, 53, 40, 1],
         [1425, 2127, 24, 21, 0],
         [1100, 1121, 26, 20, 0],
         [0, 0, 0, 0, -1], ]
    ])

    all_boxes = [
        np.array(
            [[415, 185, 34, 27, 0, 0.2],
             [415, 185, 34, 27, 1, 0.8],
             [362, 224, 53, 40, 0, 0.8],
             [362, 224, 53, 40, 1, 0.2],
             [455, 227, 24, 21, 0, 0.7],
             [455, 227, 24, 21, 1, 0.3],
             [417, 221, 26, 20, 0, 0.1],
             [417, 221, 26, 20, 1, 0.9]]
        ),
        np.array(
            [[1415, 1185, 34, 27, 0, 0.8],
             [1415, 1185, 34, 27, 1, 0.2],
             [162, 1124, 53, 40, 0, 0.6],
             [162, 1124, 53, 40, 1, 0.4],
             [1425, 2127, 24, 21, 0, 0.5],
             [1425, 2227, 24, 21, 1, 0.5],
             [1100, 1121, 26, 20, 0, 0.9],
             [1100, 1121, 26, 20, 1, 0.1]]
        )
    ]

    expected = {
        'MeanAveragePrecision': (2./3.+117./28./5.)/2.,
        'AveragePrecision': [2./3., 117./28./5.],
        'Precision': [3./8, 5./8],
        'Recall': [1., 1.],
        'OrderedPrecision': np.array(
            [[1.0, 0.5, 1./3, 0.5, 0.4, 0.5, 3./7, 3./8], [1.0, 1.0, 2./3, 3./4., 0.6, 2./3, 5./7, 5./8]]
        ),
        'OrderedRecall': np.array(
            [[1./3, 1./3, 1./3, 2./3, 2./3, 1., 1., 1.], [0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 1., 1.]]
        ),
    }

    _tf_mean_average_precision(classes, gt_boxes, all_boxes, expected)

    # case:2
    classes = ["class_1", "class_2", ]
    gt_boxes = [
        np.array(
            [[-30, 0, 34, 27, 0],
             [-66, -84, 33, 40, 1],
             [5, 27, 24, 41, 0],
             [87, -1, 46, 40, 1],
             [0, 0, 0, 0, -1], ]
        ),
        np.array(
            [[-35, 3, 34, 27, 0],
             [-162, -124, 53, 40, 0],
             [0, 0, 24, 21, 1],
             [50, 1, 26, 20, 1],
             [50, -50, 30, 30, 0]]
        )
    ]

    all_boxes = [
        np.array(
            [[-34, 5, 34, 27, 0, 0.8],
             [-34, 5, 34, 27, 1, 0.2],
             [-62, -54, 30, 42, 0, 0.4],
             [-62, -54, 30, 42, 1, 0.6],
             [4, 27, 24, 44, 0, 0.7],
             [4, 27, 24, 44, 1, 0.3],
             [87, 1, 46, 40, 0, 0.05],
             [87, 1, 46, 40, 1, 0.95],
             [87, 10, 46, 40, 0, 0.05],
             [87, 10, 46, 40, 1, 0.95]]
        ),
        np.array(
            [[-30, 5, 34, 27, 0, 0.45],
             [-30, 5, 34, 27, 1, 0.55],
             [-162, -123, 50, 40, 0, 0.55],
             [-162, -123, 50, 40, 1, 0.45],
             [50, 2, 24, 21, 0, 0.2],
             [50, 2, 24, 21, 1, 0.8],
             [50, -50, 30, 30, 0, 0.9],
             [50, -50, 30, 30, 1, 0.1]]
        )
    ]

    expected = {
        'MeanAveragePrecision': (1. + 5./12)/2,
        'AveragePrecision': [1., 5./12.],
        'Precision': [5./9, 2./9],
        'Recall': [1., 0.5],
        'OrderedPrecision': np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 5./6, 5./7, 5./8, 5./9], [1.0, 0.5, 2./3, 0.5, 0.4, 1./3, 2./7, 2./8, 2./9]]
        ),
        'OrderedRecall': np.array(
            [[1./5, 2./5, 3./5, 4./5, 1., 1., 1., 1., 1.], [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
        ),
    }

    _tf_mean_average_precision(classes, gt_boxes, all_boxes, expected)

    # case:3 class 1 dont have ground truth.
    classes = ["class_1", "class_2", ]
    gt_boxes = np.array([
        [[-66, -84, 33, 40, 1],
         [87, -1, 46, 40, 1]],
        [[0, 0, 24, 21, 1],
         [50, 1, 26, 20, 1]]
    ])

    all_boxes = [
        np.array(
            [[-34, 5, 34, 27, 0, 0.8],
             [-34, 5, 34, 27, 1, 0.2],
             [-62, -54, 30, 42, 0, 0.4],
             [-62, -54, 30, 42, 1, 0.6],
             [4, 27, 24, 44, 0, 0.7],
             [4, 27, 24, 44, 1, 0.3],
             [87, 1, 46, 40, 0, 0.05],
             [87, 1, 46, 40, 1, 0.95],
             [87, 10, 46, 40, 0, 0.05],
             [87, 10, 46, 40, 1, 0.95]]
        ),
        np.array(
            [[-30, 5, 34, 27, 0, 0.45],
             [-30, 5, 34, 27, 1, 0.55],
             [-162, -123, 50, 40, 0, 0.55],
             [-162, -123, 50, 40, 1, 0.45],
             [50, 2, 24, 21, 0, 0.2],
             [50, 2, 24, 21, 1, 0.8],
             [50, -50, 30, 30, 0, 0.9],
             [50, -50, 30, 30, 1, 0.1]]
        )
    ]

    expected = {
        'MeanAveragePrecision': (0. + 5./12)/2,
        'AveragePrecision': [0., 5./12.],
        'Precision': [0, 2./9],
        'Recall': [1., 0.5],
        'OrderedPrecision': np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 2./3, 0.5, 0.4, 1./3, 2./7, 2./8, 2./9]]
        ),
        'OrderedRecall': np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1.], [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
        ),
    }

    _tf_mean_average_precision(classes, gt_boxes, all_boxes, expected)

    # case:4 class 1 dont have ground truth. class 2 dont have detections.
    classes = ["class_1", "class_2", ]
    gt_boxes = np.array([
        [[-66, -84, 33, 40, 1],
         [87, -1, 46, 40, 1]],
        [[0, 0, 24, 21, 1],
         [50, 1, 26, 20, 1]]
    ])

    all_boxes = [
        np.array(
            [[-34, 5, 34, 27, 0, 0.8],
             [-62, -54, 30, 42, 0, 0.4],
             [4, 27, 24, 44, 0, 0.7],
             [87, 1, 46, 40, 0, 0.05],
             [87, 10, 46, 40, 0, 0.05], ]
        ),
        np.array(
            [[-30, 5, 34, 27, 0, 0.45],
             [-162, -123, 50, 40, 0, 0.55],
             [50, 2, 24, 21, 0, 0.2],
             [50, -50, 30, 30, 0, 0.9], ]
        )
    ]

    expected = {
        'MeanAveragePrecision': (0. + 0.)/2,
        'AveragePrecision': [0., 0.],
        'Precision': [0, 0],
        'Recall': [1., 0.],
        'OrderedPrecision': [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [],
        ],
        'OrderedRecall': np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1.], []]
        ),
    }

    _tf_mean_average_precision(classes, gt_boxes, all_boxes, expected)


def test_average_precision():
    n = 100
    precision = np.random.random(n)
    recall = np.random.random(n)

    average_precision = _calc_average_precision(precision, recall)

    tf.InteractiveSession()
    tf_average_precision = _average_precision(tf.convert_to_tensor(precision), tf.convert_to_tensor(recall))

    assert np.allclose(average_precision, tf_average_precision.eval())


if __name__ == '__main__':
    test_mean_average_precision()
    test_tf_mean_average_precision()
    test_average_precision()
