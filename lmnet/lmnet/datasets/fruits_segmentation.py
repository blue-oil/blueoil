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
import os.path

import numpy as np
import PIL.Image

from lmnet.datasets.base import SegmentationBase
from lmnet.utils.random import train_test_split


class FruitsSegmentation(SegmentationBase):
    classes = [
        "__background__",
        "banana",
        "orange",
        "apple",
        "strawberry",
    ]
    num_classes = len(classes)
    extend_dir = "fruits_segmentation"

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
        self.image_dir = os.path.join(self.data_dir, "images")
        self.annotation_dir = os.path.join(self.data_dir, "annotations")

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations[0])

    @property
    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""
        split_rate = 0.1
        all_image_files = [image_path for image_path in glob.glob(self.image_dir + "/*.jpg")]
        train_image_files, test_image_files = train_test_split(all_image_files, test_size=split_rate, seed=1)

        if self.subset == "train":
            image_files = train_image_files
        else:
            image_files = test_image_files

        label_files = [image_path.replace(self.image_dir, self.annotation_dir) for image_path in image_files]
        label_files = [image_path.replace("jpg", "png") for image_path in label_files]

        print("files and annotations are ready")
        return image_files, label_files

    def get_image(self, filename, convert_rgb=True):
        """Returns numpy array of an image"""
        image = PIL.Image.open(filename)
        #  sometime image data is gray.
        if convert_rgb:
            image = image.convert("RGB")
        else:
            image = image.convert("L")

        image = np.array(image)

        return image

    def _element(self):
        """Return an image and label."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0

        image_files, label_files = self.files_and_annotations

        image = self.get_image(image_files[index])
        label = self.get_image(label_files[index], convert_rgb=False)

        samples = {'image': image, 'mask': label}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        label = samples['mask']

        return image, label

    def feed(self):
        """Returns batch size numpy array of images and binarized labels."""
        images, labels = zip(*[self._element() for _ in range(self.batch_size)])

        images, labels = np.array(images), np.array(labels)

        return images, labels
