# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
import os.path

import numpy as np
from pycocotools.coco import COCO

from lmnet.utils.image import load_image
from lmnet.datasets.base import KeypointDetectionBase


class MscocoSinglePersonKeypoints(KeypointDetectionBase):
    """
    MSCOCO_2017 dataset loader for single-person pose estimation.

    References:
        https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/mscoco/keypoints.py

    """

    classes = ['person']
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "MSCOCO_2017"

    def __init__(
        self,
        subset="train",
        batch_size=10,
        *args,
        **kwargs
    ):
        super().__init__(
            subset=subset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

        if subset == 'train':
            self.json = os.path.join(self.data_dir, "annotations/person_keypoints_train2017.json")
            self.image_dir = os.path.join(self.data_dir, "train2017")
        elif subset == 'validation':
            self.json = os.path.join(self.data_dir, "annotations/person_keypoints_val2017.json")
            self.image_dir = os.path.join(self.data_dir, "val2017")

        self._coco = None

        self.num_joints = 17

        self.files, self.box_list, self.joints_list = self._load_json()

    def _load_json(self):
        """Read items from JSON files"""
        files = []
        box_list = []
        joints_list = []

        self._coco = COCO(self.json)

        image_ids = sorted(self._coco.getImgIds())
        assert self._coco.getCatIds()[0] == 1

        for entry in self._coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self.data_dir, dirname, filename)
            valid_boxes, valid_joints = self._labels_from_entry(entry)

            for i in range(len(valid_boxes)):
                files.append(abs_path)
                box_list.append(valid_boxes[i])
                joints_list.append(valid_joints[i])

        return files, box_list, joints_list

    def _labels_from_entry(self, entry):
        """
        Extract labels from entry.
        Args:
            entry: a dict to store all labeled examples in an image.

        Returns:
            valid_boxes: a list of valid boxes which uses [x1, y1, x2, y2] format.
            valid_joints: a list of corresponding joints which is a numpy array of shape (17, 3) here.

        """

        coco = self._coco
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_boxes = []
        valid_joints = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue

            if obj['category_id'] != 1:
                continue

            if max(obj['keypoints']) == 0:
                continue

            # need accurate floating point box representation
            x1, y1, w, h = obj['bbox']
            x2, y2 = x1 + np.maximum(0, w), y1 + np.maximum(0, h)
            # clip to image boundary
            x1 = np.minimum(width, np.maximum(0, x1))
            y1 = np.minimum(height, np.maximum(0, y1))
            x2 = np.minimum(width, np.maximum(0, x2))
            y2 = np.minimum(height, np.maximum(0, y2))

            if obj['area'] <= 0 or x2 <= x1 or y2 <= y1:
                continue

            num_dimensions = 2
            joints = np.zeros((self.num_joints, num_dimensions + 1), dtype=np.float32)
            for i in range(self.num_joints):
                joints[i, 0] = obj['keypoints'][i * 3 + 0]
                joints[i, 1] = obj['keypoints'][i * 3 + 1]

                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints[i, 2] = visible

            valid_boxes.append([x1, y1, x2, y2])
            valid_joints.append(joints)

        return valid_boxes, valid_joints

    def __getitem__(self, item):
        """
        Get an item given index.
        Args:
            item: int, index.

        Returns:
            cropped_image: a numpy array of shape (height, width, 3).
            joints: a numpy array of shape (17, 3), which has local coordinates in cropped_image.
        """
        full_image = load_image(self.files[item])
        box = self.box_list[item]
        joints = self.joints_list[item]

        cropped_image, joints = self.crop_from_full_image(full_image, box, joints)

        return cropped_image, joints

    def __len__(self):
        return len(self.files)

    @property
    def num_per_epoch(self):
        return len(self.files)
