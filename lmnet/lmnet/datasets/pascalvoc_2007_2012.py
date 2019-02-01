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
from multiprocessing import Pool

import PIL
import numpy as np

from lmnet.datasets.base import ObjectDetectionBase
from lmnet.datasets.pascalvoc_2007 import Pascalvoc2007
from lmnet.datasets.pascalvoc_2012 import Pascalvoc2012
from lmnet.utils.random import shuffle


def fetch_one_data(args):
    image_file, gt_boxes, augmentor, pre_processor, is_train = args
    image = PIL.Image.open(image_file)
    image = np.array(image)
    gt_boxes = np.array(gt_boxes)
    samples = {'image': image, 'gt_boxes': gt_boxes}

    if callable(augmentor) and is_train:
        samples = augmentor(**samples)

    if callable(pre_processor):
        samples = pre_processor(**samples)

    image = samples['image']
    gt_boxes = samples['gt_boxes']

    return (image, gt_boxes)


class Pascalvoc20072012(ObjectDetectionBase):
    classes = default_classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    num_classes = len(classes)
    available_subsets = ["train", "validation", "test"]
    extend_dir = None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls, skip_difficult=True):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, is_shuffle=False, skip_difficult=skip_difficult)
            gt_boxes_list = obj.annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    def __init__(
            self,
            subset="train",
            is_standardize=True,
            is_shuffle=True,
            skip_difficult=True,
            *args,
            **kwargs
    ):
        if "enable_prefetch" in kwargs:
            if kwargs["enable_prefetch"]:
                self.use_prefetch = True
            else:
                self.use_prefetch = False
            del kwargs["enable_prefetch"]
        else:
            self.use_prefetch = False

        super().__init__(
            subset=subset,
            *args,
            **kwargs,
        )

        self.is_standardize = is_standardize
        self.is_shuffle = is_shuffle
        self.skip_difficult = skip_difficult

        self._init_files_and_annotations(*args, **kwargs)
        self._shuffle()

        if self.use_prefetch:
            self.enable_prefetch()
            print("ENABLE prefetch")
        else:
            print("DISABLE prefetch")

    def prefetch_args(self, i):
        return (self.files[i], self.annotations[i], self.augmentor,
                self.pre_processor, self.subset == "train")

    def enable_prefetch(self):
        # TODO(tokunaga): the number of processes should be configurable
        self.pool = Pool(processes=8)
        self.start_prefetch()
        self.use_prefetch = True

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

    def _init_files_and_annotations(self, *args, **kwargs):
        """Create files and annotations."""
        if self.subset == "train":
            subset = "train_validation"
        elif self.subset == "validation" or self.subset == "test":
            subset = "test"

        if subset == "train_validation":
            pascalvoc_2007 = Pascalvoc2007(subset=subset, skip_difficult=self.skip_difficult, *args, **kwargs)
            pascalvoc_2012 = Pascalvoc2012(subset=subset, skip_difficult=self.skip_difficult, *args, **kwargs)
            self.files = pascalvoc_2007.files + pascalvoc_2012.files
            self.annotations = pascalvoc_2007.annotations + pascalvoc_2012.annotations
        elif subset == "test":
            pascalvoc_2007 = Pascalvoc2007(subset=subset, skip_difficult=self.skip_difficult, *args, **kwargs)
            self.files = pascalvoc_2007.files
            self.annotations = pascalvoc_2007.annotations

    def _shuffle(self):
        """Shuffle data if train."""
        if not self.is_shuffle:
            return

        if self.subset == "train":
            self.files, self.annotations = shuffle(
                self.files, self.annotations, seed=self.seed)
            print(
                "Shuffle {} train dataset with random state {}.".format(
                    self.__class__.__name__,
                    self.seed))
            self.seed += 1

    @property
    def num_max_boxes(self):
        # calculate by cls.count_max_boxes(self.skip_difficult)
        if self.skip_difficult:
            return 39
        else:
            return 56

    @property
    def num_per_epoch(self):
        return len(self.files)

    def _one_data(self):
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

        return (image, gt_boxes)

    def get_data(self):
        if self.use_prefetch:
            data_list = self.prefetch_result.get(None)
            images, gt_boxes_list = zip(*data_list)
            return images, gt_boxes_list
        else:
            images, gt_boxes_list = zip(
                *[self._one_data() for _ in range(self.batch_size)])
            return images, gt_boxes_list

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """

        images, gt_boxes_list = self.get_data()

        if self.use_prefetch:
            self.start_prefetch()

        images = np.array(images)
        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, gt_boxes_list


def main():
    import time

    s = time.time()
    Pascalvoc20072012(subset="train", enable_prefetch=False)
    e = time.time()
    print("elapsed:", e-s)


if __name__ == '__main__':
    main()
