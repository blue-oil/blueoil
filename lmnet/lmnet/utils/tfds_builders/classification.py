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

import tensorflow_datasets as tfds


class ClassificationBuilder(tfds.core.GeneratorBasedBuilder):
    """Generic TFDS builder for classification dataset"""
    VERSION = tfds.core.Version("0.1.0")

    def __init__(self, dataset_name, dataset_class=None, dataset_kwargs=None, **kwargs):
        self.name = dataset_name
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
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
        available_splits = {
            "train": tfds.Split.TRAIN,
            "validation": tfds.Split.VALIDATION,
            "test": tfds.Split.TEST,
        }

        splits = []
        for subset in self.dataset_class.available_subsets:
            if subset in available_splits:
                dataset = self.dataset_class(subset=subset, **self.dataset_kwargs)
                self.info.features["label"].names = dataset.classes

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
        max_shard_size = 256 * 1024 * 1024  # 256MiB
        for image, _ in dataset:
            total_size += image.nbytes

        return int(math.ceil(total_size / max_shard_size))

    def _generate_examples(self, dataset):
        for image, label in dataset:
            yield {
                "image": image,
                "label": label.tolist().index(1)
            }
