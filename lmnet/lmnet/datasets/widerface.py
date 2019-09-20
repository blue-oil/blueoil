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
import functools

import numpy as np

from lmnet.pre_processor import load_image
from lmnet.datasets.base import ObjectDetectionBase


class WiderFace(ObjectDetectionBase):
    classes = ["face"]
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "WIDER_FACE"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls, base_path=None):
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, base_path=base_path)
            gt_boxes_list = obj.bboxs

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    @property
    def num_max_boxes(self):
        return self.max_boxes

    @property
    def num_per_epoch(self):
        return len(self.paths)

    def __init__(self,
                 subset="train",
                 enable_prefetch=False,
                 max_boxes=3,
                 num_workers=8,
                 *args,
                 **kwargs):

        if enable_prefetch:
            self.use_prefetch = True
        else:
            self.use_prefetch = False

        self.max_boxes = max_boxes
        self.num_workers = num_workers

        super().__init__(subset=subset,
                         *args,
                         **kwargs)

        self.img_dirs = {
            "train": os.path.join(self.data_dir, "WIDER_train", "images"),
            "validation": os.path.join(self.data_dir, "WIDER_val", "images")
        }
        self.img_dir = self.img_dirs[subset]
        self._init_files_and_annotations()

    def _init_files_and_annotations(self):

        base_dir = os.path.join(self.data_dir, "wider_face_split")

        if self.subset == "train":
            file_name = os.path.join(base_dir, "wider_face_train_bbx_gt.txt")

        if self.subset == "validation":
            file_name = os.path.join(base_dir, "wider_face_val_bbx_gt.txt")

        paths = []
        bboxs = []
        labels = []

        with open(file_name) as f:
            lines = f.readlines()
        while True:
            if len(lines) == 0:
                break
            path = lines.pop(0)[:-1]
            num_boxes = int(lines.pop(0)[:-1])
            bbox = []
            label = {}
            skip_image = False
            if num_boxes > self.num_max_boxes:
                lines = lines[num_boxes:]
                continue
            else:
                for i in range(num_boxes):
                    line = lines.pop(0)[:-1]
                    x, y, w, h, blur, expression, illumination, invalid, occlusion, pose, _ = line.split(" ")
                    temp = [int(x), int(y), int(w), int(h), 0]

                    if self.subset == "train":
                        # w == 0 or h == 0 means the annotation is broken
                        if int(w) == 0 or int(h) == 0:
                            skip_image = True

                    bbox.append(temp)
                    label["blur"] = int(blur)
                    label["expression"] = int(expression)
                    label["illumination"] = int(illumination)
                    label["invalid"] = int(invalid)
                    label["occlusion"] = int(occlusion)
                    label["pose"] = int(pose)
                    label["event"], _ = path.split("/")

                if skip_image:
                    continue

                bbox = np.array(bbox, dtype=np.float32)
                paths.append(path)
                bboxs.append(bbox)
                labels.append(label)

        self.paths = paths
        self.bboxs = bboxs
        # Keep labels here in case of future use
        self.labels = labels

    def __getitem__(self, i, type=None):
        target_file = os.path.join(self.img_dir, self.paths[i])

        image = load_image(target_file)

        gt_boxes = self.bboxs[i]
        gt_boxes = np.array(gt_boxes)
        gt_boxes = gt_boxes.copy()  # is it really needed?
        gt_boxes = self._fill_dummy_boxes(gt_boxes)

        return (image, gt_boxes)

    def __len__(self):
        return self.num_per_epoch
