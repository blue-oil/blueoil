# Copyright 2020 The Blueoil Authors. All Rights Reserved.
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

from blueoil.datasets.pascalvoc_base import PascalvocBase


class PascalVOCCustom(PascalvocBase):
    classes = []
    available_subsets = []
    extend_dir = ""

    def __init__(self, classes, available_subsets, extend_dir, *args, **kwargs):
        PascalVOCCustom.classes = classes
        PascalVOCCustom.available_subsets = available_subsets
        PascalVOCCustom.extend_dir = extend_dir
        super().__init__(*args, **kwargs)

    @property
    def num_max_boxes(self):
        # Took from pascalvoc_2007.py
        if self.skip_difficult:
            return 37
        else:
            return 42

    def _annotation_file_from_image_id(self, image_id):
        annotation_file = os.path.join(self.annotations_dir, "{:06d}.xml".format(image_id))
        return annotation_file

    def _image_file_from_image_id(self, image_id):
        """Return image file name of a image."""
        return os.path.join(self.jpegimages_dir, "{:06d}.jpg".format(image_id))

    def _files_and_annotations(self):
        """Create files and gt_boxes list."""

        if self.subset == "train":
            data_type = "train"

        if self.subset == "validation":
            data_type = "val"

        if self.subset == "test":
            data_type = "test"

        if self.subset == "train_validation":
            data_type = "trainval"

        image_ids = self._image_ids(data_type)
        files = [self._image_file_from_image_id(image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(image_id) for image_id in image_ids]

        print("{} {} files and annotations are ready".format(self.__class__.__name__, self.subset))

        return files, gt_boxes_list
