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
import os.path

import numpy as np
from PIL import Image

from blueoil.common import get_color_map
from blueoil.datasets.base import SegmentationBase, StoragePathCustomizable
from blueoil.utils.random import shuffle, train_test_split


def get_image(filename, convert_rgb=True, ignore_class_idx=None):
    """Returns numpy array of an image"""
    image = Image.open(filename)
    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
        image = np.array(image)
    else:
        image = image.convert("L")
        image = np.array(image)
        if ignore_class_idx is not None:
            # Replace ignore labelled class with enough large value
            image = np.where(image == ignore_class_idx, 255, image)
            image = np.where((image > ignore_class_idx) & (image != 255), image - 1, image)

    return image


class CamvidBase(SegmentationBase):
    """Base class for CamVid and the variant dataset formats.

    http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    """
    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    extend_dir = "CamVid"
    ignore_class_idx = None

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations[0])

    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            text = "train.txt"

        if self.subset == "validation":
            text = "val.txt"

        filename = os.path.join(self.data_dir, text)

        image_files, label_files = list(), list()
        with open(filename) as f:
            for line in f:                
                items  = line.split()
                image_files.append(items[0])
                label_files.append(items[1])

        image_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in image_files]
        label_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in label_files]

        return image_files, label_files

    def __getitem__(self, i):
        image_files, label_files = self.files_and_annotations

        image = get_image(image_files[i])
        label = get_image(label_files[i], convert_rgb=False, ignore_class_idx=self.ignore_class_idx).copy()

        return (image, label)

    def __len__(self):
        return self.num_per_epoch


class Camvid(CamvidBase):
    """CamVid

    Original CamVid dataset format.
    http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    """

    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 480
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
    NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101

    classes = [
        "sky",
        "building",
        "pole",
        "road",
        "pavement",
        "tree",
        "signsymbol",
        "fence",
        "car",
        "pedestrian",
        "bicyclist",
        # "unlabelled",  # it is not use.
    ]
    num_classes = len(classes)

    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @property
    def label_colors(self):
        sky = [128, 128, 128]
        building = [128, 0, 0]
        pole = [192, 192, 128]
        road = [128, 64, 128]
        pavement = [60, 40, 222]
        tree = [128, 128, 0]
        signsymbol = [192, 128, 128]
        fence = [64, 64, 128]
        car = [64, 0, 128]
        pedestrian = [64, 64, 0]
        bicyclist = [0, 128, 192]
        unlabelled = [0, 0, 0]

        label_colors = np.array([
            sky, building, pole, road, pavement, tree, signsymbol, fence, car, pedestrian, bicyclist, unlabelled,
        ])

        return label_colors

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            text = "train.txt"

        if self.subset == "validation":
            text = "val.txt"

        filename = os.path.join(self.data_dir, text)


        image_files, label_files = list(), list()
        with open(filename) as f:
            for line in f:
                items = line.split()
                image_files.append(items[0])
                label_files.append(items[1])

        image_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in image_files]
        label_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in label_files]

        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files


class CamvidCustom(StoragePathCustomizable, CamvidBase):
    """CamvidCustom

    CamVid base custom dataset format.
    """

    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self.parse_labels()

    @property
    def label_colors(self):
        classes = self.parse_labels()
        return get_color_map(len(classes))

    @property
    def classes(self):
        classes = self.parse_labels()
        return classes

    @property
    def num_classes(self):
        classes = self.parse_labels()
        return len(classes)

    def parse_labels(self):
        with open(os.path.join(self.data_dir, "labels.txt")) as f:
            classes = f.readlines()
            classes = [cls.replace('\n', '') for cls in classes]

        if 'Ignore' in classes:
            self.ignore_class_idx = classes.index("Ignore")
            classes.remove("Ignore")

        return classes

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return image and annotation file list.
        If there is no test dataset, then split dataset to train and test lists with specific ratio.
        """
        if self.subset == "train" or self.validation_size > 0:
            text = "train.txt"
        else:
            text = "val.txt"

        filename = os.path.join(self.data_dir, text)

        image_files, label_files = list(), list()
        with open(filename) as f:
            for line in f:
                items = line.split()
                image_files.append(items[0])
                label_files.append(items[1])

        image_files = [os.path.join(self.data_dir, filename) for filename in image_files]
        label_files = [os.path.join(self.data_dir, filename) for filename in label_files]

        if self.validation_size > 0:
            train_image_files, test_image_files, train_label_files, test_label_files = \
                train_test_split(image_files, label_files, test_size=self.validation_size, seed=1)
            if self.subset == "train":
                image_files = train_image_files
                label_files = train_label_files
            else:
                image_files = test_image_files
                label_files = test_label_files

        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files

