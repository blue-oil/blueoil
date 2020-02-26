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
from abc import abstractmethod

import tensorflow as tf


class BaseNetwork(object):

    def __init__(self,
                 optimizer=tf.train.AdamOptimizer(),
                 *args, **kwargs):
        self.optimizer = optimizer

        self.is_training_placeholder = tf.compat.v1.placeholder(tf.bool, name="is_training_placeholder")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self._build_placeholders_dict()
        self._build_network(self.is_training_placeholder)
        self._build_post_processed()
        self._build_loss()
        self._build_train_op()
        self._build_summary()
        self._build_metrics()
        self._build_metrics_summary()

    def get_feed_dict(self, samples_dict, is_training):
        feed_dict = {self.placeholders_dict[key]: samples_dict[key]
                     for key in self.placeholders_dict.keys()}
        feed_dict[self.is_training_placeholder] = is_training
        return feed_dict

    def get_metrics(self):
        return self.metrics_ops_dict, self.metrics_update_op

    def get_metrics_summary(self):
        return self.metrics_summary_op, self.metrics_placeholders

    def get_summary(self):
        return self.summary

    def get_train_op(self):
        return self.train_op

    @abstractmethod
    def _build_placeholders_dict(self):
        self.placeholders_dict = None

    @abstractmethod
    def _build_network(self, is_training):
        self.output = None

    @abstractmethod
    def _build_post_processed(self):
        self.post_processed = None

    @abstractmethod
    def _build_loss(self, global_step):
        self.loss = None

    def _build_train_op(self):
        with tf.name_scope("train"):
            var_list = tf.compat.v1.trainable_variables()

        gradients = self.optimizer.compute_gradients(self.loss, var_list=var_list)
        self.train_op = self.optimizer.apply_gradients(gradients, global_step=self.global_step)

    @abstractmethod
    def _build_summary(self):
        self.summary = None

    @abstractmethod
    def _build_metrics(self):
        self.metrics_ops_dict = None
        self.metrics_update_op = None

    def _build_metrics_summary(self):
        with tf.name_scope("metrics"):
            metrics_placeholders = []
            metrics_summaries = []
            for (metrics_key, metrics_op) in self.metrics_ops_dict.items():
                metrics_placeholder = tf.compat.v1.placeholder(
                    tf.float32, name="{}_placeholder".format(metrics_key)
                )
                summary = tf.compat.v1.summary.scalar(metrics_key, metrics_placeholder)
                metrics_placeholders.append(metrics_placeholder)
                metrics_summaries.append(summary)

            metrics_summary_op = tf.summary.merge(metrics_summaries)

        self.metrics_summary_op = metrics_summary_op
        self.metrics_placeholders = metrics_placeholders
