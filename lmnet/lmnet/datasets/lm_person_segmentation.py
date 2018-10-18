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
import glob

from lmnet.datasets.fruits_segmentation import FruitsSegmentation
from lmnet.utils.random import train_test_split


class LmPersonSegmentation(FruitsSegmentation):
    """Leapmind original person in meeting segmentation"""
    classes = [
        "__background__",
        "person",
    ]
    num_classes = len(classes)
    extend_dir = "lm_person_segmentation"

    def __init__(
            self,
            subset="train",
            batch_size=32,
            *args,
            **kwargs
    ):

        super().__init__(
            subset=subset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""
        split_rate = 0.2
        jpg_image_files = [image_path for image_path in glob.glob(self.image_dir + "/*.jpg")]
        png_image_files = [image_path for image_path in glob.glob(self.image_dir + "/*.png")]
        JPG_image_files = [image_path for image_path in glob.glob(self.image_dir + "/*.JPG")]
        jpeg_image_files = [image_path for image_path in glob.glob(self.image_dir + "/*.jpeg")]
        all_image_files = jpg_image_files + png_image_files + JPG_image_files + jpeg_image_files

        train_image_files, test_image_files = train_test_split(all_image_files, test_size=split_rate, seed=1)

        if self.subset == "train":
            image_files = train_image_files
        else:
            image_files = test_image_files

        label_files = [image_path.replace(self.image_dir, self.annotation_dir) for image_path in image_files]
        label_files = [image_path.replace("jpg", "png") for image_path in label_files]
        label_files = [image_path.replace("JPG", "png") for image_path in label_files]
        label_files = [image_path.replace("jpeg", "png") for image_path in label_files]

        print("files and annotations are ready", self.subset, len(image_files), len(label_files))
        return image_files, label_files
