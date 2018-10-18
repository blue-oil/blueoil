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

import PIL
import numpy as np

from lmnet.datasets.pascalvoc2007 import Pascalvoc2007
from lmnet import data_processor
from lmnet import data_augmentor


class Pascalvoc2007Label(Pascalvoc2007):

    @classmethod
    def count_max_boxes(cls, data_dir=None):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, data_dir=data_dir)
            files, gt_boxes_list = obj.files_and_annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    classes = [
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
    num_classes = len(classes)
    available_subsets = ['train', 'validation']

    def __init__(
            self,
            subset="train",
            batch_size=10,
            image_size=[256, 256],  # (height, width)
            is_standardize=True,
            is_augment=False,
            aug_options={
                'use_affine': 0,
                'use_change_height': 0,
                'use_flip': 0,
                'rand_flip': 0,
                'flip_side': 0,
                'max_brightness': 0,
                'use_brightness': 0,
                'max_contrast': 0,
                'use_contrast': 0,
                'use_color': 0,
                'max_color': 0,
                'use_color_filter': 0,
                'use_hue': 0,
                'change_amount_hue': 0,
                'use_superpixels': 0,
                'n_segments': 0,
                'p_replace': 0,
                'use_blur': 0,
                'max_bluring': 0,
                'use_rotate': 0,
            },
    ):

        super().__init__(
            subset=subset,
            batch_size=batch_size,
            image_size=image_size,
        )
        self.is_standardize = is_standardize
        self.is_augment = is_augment
        self.aug_options = aug_options

        self.jpegimages_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.annotations_dir = os.path.join(self.data_dir, 'Annotations')
        self.imagesets_dir = os.path.join(self.data_dir, 'ImageSets', 'Main')

    @property
    @functools.lru_cache(maxsize=None)
    def num_max_boxes(self):
        return Pascalvoc2007.count_max_boxes(self.data_dir)

    @property
    def num_per_epoch(self):
        files, _ = self.files_and_annotations
        return len(files)

    def _one_data(self):
        """Return an image, gt_boxes."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0

        files, gt_boxes_list = self.files_and_annotations
        target_file = files[index]
        gt_boxes = gt_boxes_list[index]

        image = PIL.Image.open(target_file)
        image = np.array(image)
        image = data_processor.resize(image, self.image_size)

        if self.is_augment and self.subset == "train":
            image, gt_boxes = data_augmentor.data_augmentation(image, gt_boxes, self.aug_options, bounding_box=False)

        if self.is_standardize:
            image = data_processor.per_image_standardization(image)

        return image, gt_boxes

    def _labels_from_annotation(self, annotation):
        """Get labels list from annotation object.

        Args:
            annotation: BeautifulSoup object of traget image id.

        Return:
           labels list [[class_id],[class_id]].
        """
        objs = annotation.findAll('object')

        boxes = []
        for obj in objs:

            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                difficult = obj.findChildren('difficult')[0].contents[0]
                # Exclude the samples labeled as difficult
                # It is the same py-faster-rcnn setting
                # https://github.com/rbgirshick/py-faster-rcnn/blob/96dc9f1dea3087474d6da5a98879072901ee9bf9/lib/datasets/pascal_voc.py#L47
                if difficult == 1:
                    continue

                class_name = str(name_tag.contents[0])

                # ignore category of foot, head, hand
                if class_name not in self.classes:
                    continue

                class_index = self.classes.index(class_name)

                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])

                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin

                boxes.append([x, y, w, h, class_index])

        return boxes

    @functools.lru_cache(maxsize=None)
    def _labels_from_image_id(self, image_id):
        """Return gt boxes list ([[x, y, w, h, class_id]]) of a image."""
        annotation = self._load_annotation(image_id)
        labels = self._labels_from_annotation(annotation)

        return labels

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            data_type = "train"

        if self.subset == "validation":
            data_type = "val"

        image_ids = self._image_ids(data_type)
        files = [self._image_file_from_image_id(image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(image_id) for image_id in image_ids]

        print("files and annotations are ready")

        return files, gt_boxes_list

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """
        images, gt_boxes_list = zip(*[self._one_data() for _ in range(self.batch_size)])

        images = np.array(images)

        gt_boxes_list = self.change_gt_boxes_shape(gt_boxes_list)

        return images, gt_boxes_list
