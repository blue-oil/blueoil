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
import os
import os.path

from blueoil.datasets.pascalvoc_base import PascalvocBase


class Pascalvoc2012(PascalvocBase):
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
    available_subsets = ['train', 'validation', "train_validation"]
    extend_dir = "PASCALVOC_2012/VOCdevkit/VOC2012"

    @property
    def num_max_boxes(self):
        # calculate by cls.count_max_boxes(self.skip_difficult)
        if self.skip_difficult:
            return 39
        else:
            return 56

    def _annotation_file_from_image_id(self, image_id):
        """Return annotation xml file path."""

        annotation_file = os.path.join(self.annotations_dir, "{}.xml".format(image_id))
        return annotation_file

    def _image_file_from_image_id(self, image_id):
        """Return image file name of a image."""

        return os.path.join(self.jpegimages_dir, "{}.jpg".format(image_id))

    def _files_and_annotations(self):
        """Create files and gt_boxes list."""

        if self.subset == "train":
            data_type = "train"

        if self.subset == "validation":
            data_type = "val"

        if self.subset == "train_validation":
            data_type = "trainval"

        image_ids = self._image_ids(data_type)
        files = [self._image_file_from_image_id(image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(image_id) for image_id in image_ids]

        print("{} {} files and annotations are ready".format(self.__class__.__name__, self.subset))

        return files, gt_boxes_list
