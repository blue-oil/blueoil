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
import os
import os.path

import pandas as pd
import PIL
import numpy as np
import xml.etree.ElementTree as ET

from lmnet.datasets.base import ObjectDetectionBase
from lmnet.utils.random import shuffle


class PascalvocBase(ObjectDetectionBase):
    _cache = dict()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls, skip_difficult=True):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, skip_difficult=skip_difficult)
            gt_boxes_list = obj.annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    default_classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ]
    classes = []
    num_classes = len(classes)
    available_subsets = None
    extend_dir = None

    def __init__(
            self,
            subset="train",
            batch_size=10,
            skip_difficult=True,
            *args,
            **kwargs
    ):
        self.skip_difficult = skip_difficult

        super().__init__(
            subset=subset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self.jpegimages_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.annotations_dir = os.path.join(self.data_dir, 'Annotations')
        self.imagesets_dir = os.path.join(self.data_dir, 'ImageSets', 'Main')

        self._init_files_and_annotations()

    @property
    @functools.lru_cache(maxsize=None)
    def num_max_boxes(self):
        cls = self.__class__
        return cls.count_max_boxes(self.skip_difficult)

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
        gt_boxes = np.array(gt_boxes, dtype=np.float32)

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

    def _get_boxes_from_annotation(self, xml_file):
        """Get gt boxes list from annotation object.

        Args:
            xml_file: file name for bounding box annotation xml

        Return:
           gt boxes list [[x, y, w, h, class_id]].
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bboxes = []

        for obj in root.iter('object'):
            if self.skip_difficult and int(obj.find('difficult').text) >= 1:
                continue

            class_name = obj.find('name').text
            if class_name not in self.classes:
                continue

            class_index = self.classes.index(class_name)

            for e in list(obj):
                if e.tag == "bndbox":
                    # subtract 1 to make boxes 0-based coordinate.
                    ymin = int(e.find("ymin").text) - 1
                    xmin = int(e.find("xmin").text) - 1
                    ymax = int(e.find("ymax").text) - 1
                    xmax = int(e.find("xmax").text) - 1

                    w = xmax - xmin
                    h = ymax - ymin
                    bbox = [xmin, ymin, w, h, class_index]
                    bboxes.append(bbox)

        return bboxes

    def _annotation_file_from_image_id(self, image_id):
        """Return annotation xml file path."""

        return os.path.join(self.annotations_dir, "{}.xml".format(image_id))

    def _image_file_from_image_id(self, image_id):
        """Return image file name of a image."""

        return os.path.join(self.jpegimages_dir, "{}.jpg".format(image_id))

    @functools.lru_cache(maxsize=None)
    def _gt_boxes_from_image_id(self, image_id):
        """Return gt boxes list ([[x, y, w, h, class_id]]) of a image."""

        annotation_file = self._annotation_file_from_image_id(image_id)
        return self._get_boxes_from_annotation(annotation_file)

    @functools.lru_cache(maxsize=None)
    def _image_ids(self, data_type=None):
        """Get image ids in data_type, classes."""

        all_image_ids = self._all_image_ids(data_type)
        all_gt_boxes_list = [self._gt_boxes_from_image_id(image_id) for image_id in all_image_ids]

        if self.classes == self.default_classes:
            return all_image_ids

        image_ids = [
            image_id for image_id, gt_boxes in zip(all_image_ids, all_gt_boxes_list) if len(gt_boxes) is not 0
        ]

        return image_ids

    @functools.lru_cache(maxsize=None)
    def _all_image_ids(self, data_type=None, is_debug=False):
        """Get all image ids in data_type."""

        if data_type is None:
            raise ValueError("Must provide data_type = train or val or trainval or test")

        filename = os.path.join(self.imagesets_dir, data_type + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['image_id'])

        if is_debug:
            df = df[:50]

        return df.image_id.tolist()

    def _shuffle(self):
        """Shuffle data if train."""

        if self.subset == "train" or self.subset == "train_validation":
            self.files, self.annotations = shuffle(
                self.files, self.annotations, seed=self.seed)
            print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
            self.seed += 1

    def _files_and_annotations(self):
        raise NotImplemented()

    def _init_files_and_annotations(self):
        """Init files and gt_boxes list, Cache these."""

        cache_key = self.subset + self.data_dir + str(self.classes) + str(self.skip_difficult)
        cls = self.__class__

        if cache_key in cls._cache:
            cached_obj = cls._cache[cache_key]
            self.files, self.annotations = cached_obj.files, cached_obj.annotations
        else:
            self.files, self.annotations = self._files_and_annotations()
            self._shuffle()
            cls._cache[cache_key] = self

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """
        images, gt_boxes_list = zip(*[self._element() for _ in range(self.batch_size)])

        images = np.array(images)

        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, gt_boxes_list
