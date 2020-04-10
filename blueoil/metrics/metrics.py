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
import tensorflow as tf


def tp_tn_fp_fn_for_each(output, labels, threshold=0.5):
    """Calculate True Positive, True Negative, False Positive, False Negative.

    Args:
        output: network output sigmoided tensor. shape is [batch_size, num_class]
        labels: multi label encoded bool tensor. shape is [batch_size, num_class]
        threshold: python float

    Returns:
        shape is [4(tp, tn, fp, fn), num_class]

    """
    predicted = tf.greater_equal(output, threshold)
    gt_positive = tf.reduce_sum(tf.cast(labels, tf.int32), axis=0, keepdims=True)
    gt_negative = tf.reduce_sum(tf.cast(tf.logical_not(labels), tf.int32), axis=0, keepdims=True)
    true_positive = tf.math.logical_and(predicted, labels)
    true_positive = tf.reduce_sum(tf.cast(true_positive, tf.int32), axis=0, keepdims=True)

    true_negative = tf.math.logical_and(tf.logical_not(predicted), tf.math.logical_not(labels))
    true_negative = tf.reduce_sum(tf.cast(true_negative, tf.int32), axis=0, keepdims=True)
    false_negative = gt_positive - true_positive
    false_positive = gt_negative - true_negative

    return tf.concat(axis=0, values=[true_positive, true_negative, false_positive, false_negative])


def tp_tn_fp_fn(output, labels, threshold=0.5):
    """Calculate True Positive, True Negative, False Positive, False Negative.

    Args:
        output: network output sigmoided tensor. shape is [batch_size, num_class]
        labels: multi label encoded bool tensor. shape is [batch_size, num_class]
        threshold: python float

    """
    predicted = tf.greater_equal(output, threshold)

    gt_positive = tf.reduce_sum(tf.cast(labels, tf.int32))
    gt_negative = tf.reduce_sum(tf.cast(tf.logical_not(labels), tf.int32))

    true_positive = tf.math.logical_and(predicted, labels)
    true_positive = tf.reduce_sum(tf.cast(true_positive, tf.int32))

    true_negative = tf.math.logical_and(tf.logical_not(predicted), tf.math.logical_not(labels))
    true_negative = tf.reduce_sum(tf.cast(true_negative, tf.int32))

    false_negative = gt_positive - true_positive

    false_positive = gt_negative - true_negative

    return true_positive, true_negative, false_positive, false_negative
