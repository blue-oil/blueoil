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
from multiprocessing import Pool

from PIL import Image
import numpy as np

from lmnet.datasets.base import ObjectDetectionBase
from lmnet.utils.random import shuffle


def fetch_one_data(args):
    path, gt_boxes, label, augmentor, pre_processor, is_train = args
    image = Image.open(path)
    image = np.array(image)
    gt_boxes = np.array(gt_boxes)
    samples = {'image': image, 'gt_boxes': gt_boxes}

    if callable(augmentor) and is_train:
        samples = augmentor(**samples)

    if callable(pre_processor):
        samples = pre_processor(**samples)

    image = samples['image']
    gt_boxes = samples['gt_boxes']

    return image, gt_boxes


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
            obj = cls(subset=subset, base_path=base_path, is_shuffle=False)
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
                 is_shuffle=True,
                 enable_prefetch=False,
                 max_boxes=3,
                 num_workers=None,
                 *args,
                 **kwargs):

        if enable_prefetch:
            self.use_prefetch = True
        else:
            self.use_prefetch = False

        self.is_shuffle = is_shuffle
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
        self._shuffle()

        if self.use_prefetch:
            self.enable_prefetch()
            print("ENABLE prefetch")
        else:
            print("DISABLE prefetch")

    def prefetch_args(self, index):
        path = self.paths[index]
        gt_boxes = self.bboxs[index]
        label = self.labels[index]
        path = os.path.join(self.img_dir,
                            path)
        return path, gt_boxes, label, self.augmentor, self.pre_processor, self.subset == "train"

    def enable_prefetch(self):
        self.pool = Pool(processes=self.num_workers)
        self.start_prefetch()

    def start_prefetch(self):
        index = self.current_element_index
        batch_size = self.batch_size
        start = index
        end = min(index + batch_size, self.num_per_epoch)
        pool = self.pool

        args = []
        for i in range(start, end):
            args.append(self.prefetch_args(i))

        self.current_element_index += batch_size
        if self.current_element_index >= self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

            rest = batch_size - len(args)
            for i in range(0, rest):
                args.append(self.prefetch_args(i))
            self.current_element_index += rest

        self.prefetch_result = pool.map_async(fetch_one_data, args)

    def _init_files_and_annotations(self):

        base_dir = os.path.join(self.data_dir,
                                "wider_face_split")

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
            if num_boxes > self.num_max_boxes:
                lines = lines[num_boxes:]
                continue
            else:
                for i in range(num_boxes):
                    line = lines.pop(0)[:-1]
                    x, y, w, h, blur, expression, illumination, invalid, occlusion, pose, _ = line.split(" ")
                    if int(w) <= 0 or int(h) <= 0:
                        continue
                    temp = [int(x), int(y), int(w), int(h), 0]
                    bbox.append(temp)
                    label["blur"] = int(blur)
                    label["expression"] = int(expression)
                    label["illumination"] = int(illumination)
                    label["invalid"] = int(invalid)
                    label["occlusion"] = int(occlusion)
                    label["pose"] = int(pose)
                    label["event"], _ = path.split("/")

                bbox = np.array(bbox, dtype=np.int32)
                paths.append(path)
                bboxs.append(bbox)
                labels.append(label)

        self.paths = paths
        self.bboxs = bboxs
        # Keep labels here in case of future use
        self.labels = labels

    def _shuffle(self):

        if not self.is_shuffle:
            return

        if self.subset == "train":
            self.paths, self.bboxs, self.labels = shuffle(
                self.paths, self.bboxs, self.labels, seed=self.seed)
            print("Shuffle {} train dataset with seed {}.".format(self.__class__.__name__, self.seed))
            self.seed = self.seed + 1

    def _one_data(self):
        """Return an image, gt_boxes."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

        paths, bboxs = self.paths, self.bboxs
        path = paths[index]
        gt_boxes = bboxs[index]
        gt_boxes = gt_boxes.copy()

        path = os.path.join(self.img_dir,
                            path)

        image = Image.open(path)
        image = np.array(image)

        samples = {"image": image, "gt_boxes": gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples["image"]
        gt_boxes = samples["gt_boxes"]

        return image, gt_boxes

    def get_data(self):
        if self.use_prefetch:
            data_list = self.prefetch_result.get(None)
            images, gt_boxes_list = zip(*data_list)
            return images, gt_boxes_list
        else:
            images, gt_boxes_list = zip(*[self._one_data() for _ in range(self.batch_size)])
            return images, gt_boxes_list

    def feed(self):

        images, gt_boxes_list = self.get_data()

        if self.use_prefetch:
            self.start_prefetch()

        images = np.array(images)
        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, gt_boxes_list
