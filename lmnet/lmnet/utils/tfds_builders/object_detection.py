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
import csv
import math
import os
import xml.etree.ElementTree as ET

import tensorflow as tf
import tensorflow_datasets as tfds


class ObjectDetectionBuilder(tfds.core.GeneratorBasedBuilder):
    """Generic TFDS builder for object detection dataset"""
    VERSION = tfds.core.Version("0.1.0")

    def __init__(self, dataset_name, dataset_class=None, dataset_kwargs=None, **kwargs):
        self.name = dataset_name
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Generic TFDS builder for object detection dataset",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "objects": tfds.features.SequenceDict({
                    "label": tfds.features.ClassLabel(),
                    "bbox": tfds.features.BBoxFeature(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        available_splits = {
            "train": tfds.Split.TRAIN,
            "validation": tfds.Split.VALIDATION,
            "test": tfds.Split.TEST,
        }

        splits = []
        for subset in self.dataset_class.available_subsets:
            if subset in available_splits:
                try:
                    dataset = self.dataset_class(subset=subset, **self.dataset_kwargs)
                except:
                    continue

                self.info.features["objects"]["label"].names = dataset.classes

                splits.append(
                    tfds.core.SplitGenerator(
                        name=available_splits[subset],
                        num_shards=self._num_shards(dataset),
                        gen_kwargs=dict(dataset=dataset)
                    )
                )

        return splits

    def _num_shards(self, dataset):
        total_size = 0
        max_shard_size = 256 * 1024 * 1024 # 256MiB
        for image, _ in dataset:
            total_size += image.nbytes

        return int(math.ceil(total_size / max_shard_size))

    def _generate_examples(self, dataset):
        for image, annotations in dataset:
            height, width, _ = image.shape

            objects = []
            for annotation in annotations:
                xmin, ymin, w, h, label = annotation
                if label == -1:
                    continue

                objects.append({
                    "label": label,
                    "bbox": tfds.features.BBox(
                        ymin / height,
                        xmin / width,
                        (ymin + h) / height,
                        (xmin + w) / width,
                    )
                })

            yield {
                "image": image,
                "objects": objects
            }
