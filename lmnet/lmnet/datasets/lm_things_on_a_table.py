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
import functools
import os.path

from lmnet.datasets.delta_mark import ObjectDetectionBase
from lmnet.utils.random import shuffle, train_test_split


class LmThingsOnATable(ObjectDetectionBase):
    """Leapmind things on a table dataset for object detection.

    images: images numpy array. shape is [batch_size, height, width]
    labels: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
    """
    classes = ["hand", "salad", "steak", "whiskey", "book"]
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "lm_things_on_a_table"

    def __init__(
            self,
            *args,
            **kwargs
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.single = {
            "json": os.path.join(self.data_dir, "json/single_label_bound.json"),
            "dir": os.path.join(self.data_dir, "Data_single")
        }

        self.multi = {
            "json": os.path.join(self.data_dir, "json/multi_label_bound_all.json"),
            "dir": os.path.join(self.data_dir, "Data_multi")
        }

    def _single_files_and_annotations(self):
        json_file = self.single["json"]
        image_dir = self.single["dir"]
        files, labels = self._files_and_annotations(json_file, image_dir)

        return files, labels

    def _multi_files_and_annotations(self):
        json_file = self.multi["json"]
        image_dir = self.multi["dir"]
        files, labels = self._files_and_annotations(json_file, image_dir)

        return files, labels

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and labels list."""
        single_split_rate = 0.1
        multi_split_rate = 0.1
        single_files, single_labels = self._single_files_and_annotations()
        multi_files, multi_labels = self._multi_files_and_annotations()

        train_single_files, test_single_files, train_single_labels, test_single_labels =\
            train_test_split(single_files,
                             single_labels,
                             test_size=single_split_rate,
                             seed=1)

        train_multi_files, test_multi_files, train_multi_labels, test_multi_labels =\
            train_test_split(multi_files,
                             multi_labels,
                             test_size=multi_split_rate,
                             seed=1)

        if self.subset == "train":
            files = train_multi_files + train_single_files
            labels = train_multi_labels + train_single_labels
        else:
            files = test_multi_files + test_single_files
            labels = test_multi_labels + test_single_labels

        files, labels = shuffle(files, labels, seed=1)

        print("files and annotations are ready")
        return files, labels

    @property
    def num_max_boxes(self):
        # calulated by cls.count_max_boxes()
        return 6
