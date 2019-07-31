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
import os

import tensorflow as tf
import tensorflow_datasets as tfds


class ImageFolder(tfds.core.GeneratorBasedBuilder):
    """Generic TFDS builder for classification dataset"""
    VERSION = tfds.core.Version("0.1.0")

    def __init__(self, dataset_name, raw_data_path, **kwargs):
        self.name = dataset_name
        self.raw_data_path = raw_data_path
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Generic TFDS builder for classification dataset",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(),
            }),
        )

    def _split_generators(self, dl_manager):
        data_path = self.raw_data_path
        split_names = []
        classes = []
        for split_name in tf.io.gfile.listdir(data_path):
            if not tf.io.gfile.isdir(os.path.join(data_path, split_name)):
                continue
            
            split_names.append(split_name)
            for class_name in tf.io.gfile.listdir(os.path.join(data_path, split_name)):
                if not tf.io.gfile.isdir(os.path.join(data_path, split_name, class_name)):
                    continue

                classes.append(class_name)

        classes = list(set(classes))
        self.info.features["label"].names = sorted(classes)

        available_splits = {
            "train": tfds.Split.TRAIN,
            "val": tfds.Split.VALIDATION,
            "test": tfds.Split.TEST,
        }

        splits = []
        for split_name in split_names:
            if split_name in available_splits:
                splits.append(
                    tfds.core.SplitGenerator(
                        name=available_splits[split_name],
                        num_shards=self._num_shards(data_path, split_name),
                        gen_kwargs=dict(data_path=data_path, split_name=split_name)
                    )
                )

        return splits

    def _num_shards(self, data_path, split_name):
        total_size = 0
        max_shard_size = 256 * 1024 * 1024 # 256MiB
        for image_file in self._image_files(data_path, split_name):
            total_size += tf.io.gfile.Gfile(image_file).size()

        return int(math.ceil(total_size / max_shard_size))

    def _image_files(self, data_path, split_name):
        split_dir = os.path.join(data_path, split_name)
        for parent, _, files in tf.io.gfile.walk(split_dir):
            for file in files:
                splitted = os.path.splitext(file)
                if len(splitted) == 2 and splitted[1] in ("png", "jpg"):
                    yield os.path.join(parent, file)

    def _generate_examples(self, data_path, split_name):
        for image_file in self._image_files(data_path, split_name):
            class_dir, file = os.path.split(image_file)
            label = os.path.basename(class_dir)

            yield {
                "image": image_file,
                "label": label
            }
