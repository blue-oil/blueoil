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

import numpy as np

from lmnet.pre_processor import load_image
from lmnet.datasets.base import ObjectDetectionBase
from lmnet.datasets.pascalvoc_2007 import Pascalvoc2007
from lmnet.datasets.pascalvoc_2012 import Pascalvoc2012


class Pascalvoc20072012(ObjectDetectionBase):
    classes = default_classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    num_classes = len(classes)
    available_subsets = ["train", "validation", "test"]
    extend_dir = None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls, skip_difficult=True):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, is_shuffle=False, skip_difficult=skip_difficult)
            gt_boxes_list = obj.annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    def __init__(
            self,
            subset="train",
            is_standardize=True,
            is_shuffle=True,
            skip_difficult=True,
            *args,
            **kwargs
    ):
        super().__init__(
            subset=subset,
            *args,
            **kwargs,
        )

        self.is_standardize = is_standardize
        self.is_shuffle = is_shuffle
        self.skip_difficult = skip_difficult

        self._init_files_and_annotations(*args, **kwargs)

    def _init_files_and_annotations(self, *args, **kwargs):
        """Create files and annotations."""
        if self.subset == "train":
            subset = "train_validation"
        elif self.subset == "validation" or self.subset == "test":
            subset = "test"

        if subset == "train_validation":
            pascalvoc_2007 = Pascalvoc2007(subset=subset, skip_difficult=self.skip_difficult, *args, **kwargs)
            pascalvoc_2012 = Pascalvoc2012(subset=subset, skip_difficult=self.skip_difficult, *args, **kwargs)
            self.files = pascalvoc_2007.files + pascalvoc_2012.files
            self.annotations = pascalvoc_2007.annotations + pascalvoc_2012.annotations
        elif subset == "test":
            pascalvoc_2007 = Pascalvoc2007(subset=subset, skip_difficult=self.skip_difficult, *args, **kwargs)
            self.files = pascalvoc_2007.files
            self.annotations = pascalvoc_2007.annotations

    @property
    def num_max_boxes(self):
        # calculate by cls.count_max_boxes(self.skip_difficult)
        if self.skip_difficult:
            return 39
        else:
            return 56

    @property
    def num_per_epoch(self):
        return len(self.files)

    def __getitem__(self, i, type=None):
        target_file = self.files[i]
        image = load_image(target_file)

        gt_boxes = self.annotations[i]
        gt_boxes = np.array(gt_boxes)
        gt_boxes = gt_boxes.copy()  # is it really needed?
        gt_boxes = self._fill_dummy_boxes(gt_boxes)

        return (image, gt_boxes)

    def __len__(self):
        return self.num_per_epoch
