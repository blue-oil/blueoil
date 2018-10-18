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
import json
import os.path


import numpy as np
import PIL.Image
import pandas as pd

from lmnet.datasets.base import ObjectDetectionBase
from lmnet.utils.random import shuffle, train_test_split


class LmFlower(ObjectDetectionBase):
    """Leapmind flower dataset for object detection.

    images: images numpy array. shape is [batch_size, height, width]
    labels: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
    """
    classes = ["sunflower", "calla", "poppy (Iceland poppy)", "carnation", "cosmos"]
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "lm_flower"

    @classmethod
    def count_max_boxes(cls):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset)
            gt_boxes_list = obj.annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    def __init__(
            self,
            *args,
            **kwargs
    ):

        super().__init__(
            *args,
            **kwargs,
        )
        self.json = os.path.join(self.data_dir, "project_126_1507252774.json")
        self.images_dir = os.path.join(self.data_dir, "images")

        self._init_files_and_annotations()

    @property
    def num_per_epoch(self):
        return len(self.files)

    def _element(self):
        """Return an image, gt_boxes."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

        files, gt_boxes_list = self.files, self.annotations
        target_file = files[index]
        gt_boxes = gt_boxes_list[index]
        gt_boxes = np.array(gt_boxes)

        image = PIL.Image.open(target_file)
        image = np.array(image)

        samples = {'image': image, 'gt_boxes': gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        gt_boxes = samples['gt_boxes']

        return image, gt_boxes

    def _files_and_annotations_from_json(self, json_file):
        """Return files and gt_boxes list."""
        image_ids = self._image_ids(json_file)

        image_files = [self._image_file_from_image_id(json_file, image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(json_file, image_id) for image_id in image_ids]

        return image_files, gt_boxes_list

    def _image_file_from_image_id(self, json_file, image_id):
        images = self._images_from_json(json_file)
        file_name = images[images.id == image_id].file_name.tolist()[0]

        return os.path.join(self.images_dir, file_name)

    def _gt_boxes_from_image_id(self, json_file, image_id):
        annotations = self._annotations_from_json(json_file)
        category_ids = annotations[annotations.image_id == image_id].category_id.tolist()

        categories = self._categories_from_json(json_file)
        category_names = [
            categories[categories.id == category_id].name.iloc[0]
            for category_id in category_ids
        ]

        labels = [self.classes.index(category) for category in category_names]

        bboxes = annotations[annotations.image_id == image_id].bbox.tolist()

        gt_boxes = []

        for class_id, bbox in zip(labels, bboxes):
            # ignore width 0 or height 0 box.
            if bbox[2] == 0 or bbox[3] == 0:
                continue

            gt_boxes.append(bbox + [class_id])

        return gt_boxes

    @functools.lru_cache(maxsize=None)
    def _load_json(self, json_file):
        f = open(json_file)
        data = json.load(f)
        f.close()

        return data

    def _image_ids(self, json_file):
        images = self. _images_from_json(json_file)

        return images.id.tolist()

    @functools.lru_cache(maxsize=None)
    def _annotations_from_json(self, json_file):
        data = self._load_json(json_file)
        annotations = pd.DataFrame(data["annotations"])

        return annotations

    @functools.lru_cache(maxsize=None)
    def _categories_from_json(self, json_file):
        data = self._load_json(json_file)
        categories = pd.DataFrame(data["categories"])

        return categories

    @functools.lru_cache(maxsize=None)
    def _images_from_json(self, json_file):
        data = self._load_json(json_file)
        images = pd.DataFrame(data["images"])

        return images

    @functools.lru_cache(maxsize=None)
    def _files_and_annotations(self):
        """Return all files and labels list."""
        single_split_rate = 0.1
        files, labels = self._files_and_annotations_from_json(self.json)

        train_files, test_files, train_labels, test_labels =\
            train_test_split(files,
                             labels,
                             test_size=single_split_rate,
                             seed=1)

        if self.subset == "train":
            files = train_files
            labels = train_labels
        else:
            files = test_files
            labels = test_labels

        print("files and annotations are ready")
        return files, labels

    @property
    @functools.lru_cache(maxsize=None)
    def num_max_boxes(self):
        return type(self).count_max_boxes()

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """
        images, gt_boxes_list = zip(*[self._element() for _ in range(self.batch_size)])

        images = np.array(images)

        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        return images, gt_boxes_list

    def _shuffle(self):
        """Shuffle data if train."""

        if self.subset == "train":
            self.files, self.annotations = shuffle(
                self.files, self.annotations, seed=self.seed)
            print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
            self.seed += 1

    def _init_files_and_annotations(self):
        self.files, self.annotations = self._files_and_annotations()
        self._shuffle()
