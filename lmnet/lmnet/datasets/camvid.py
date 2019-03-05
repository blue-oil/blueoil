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
import pandas as pd
from PIL import Image

from lmnet.datasets.base import SegmentationBase, StoragePathCustomizable
from lmnet.utils.random import shuffle, train_test_split


def get_image(filename, convert_rgb=True):
    """Returns numpy array of an image"""
    image = Image.open(filename)

    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
    else:
        image = image.convert("L")

    image = np.array(image)
    return image


class CamvidBase(SegmentationBase):
    """
    Base class for CamVid and the variant dataset formats.

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

    @property
    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

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
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['image_files', 'label_files'],
        )

        image_files = df.image_files.tolist()
        label_files = df.label_files.tolist()

        image_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in image_files]
        label_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in label_files]

        return image_files, label_files

    def __getitem__(self, i, type=None):
        image_files, label_files = self.files_and_annotations

        image = get_image(image_files[i])
        label = get_image(label_files[i], convert_rgb=False).copy()

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
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['image_files', 'label_files'],
        )

        image_files = df.image_files.tolist()
        label_files = df.label_files.tolist()

        image_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in image_files]
        label_files = [filename.replace("/SegNet/CamVid", self.data_dir) for filename in label_files]

        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files


class CamvidCustom(StoragePathCustomizable, CamvidBase):
    """CamvidCustom

    CamVid base custom dataset format.
    To define your own color labels, make `labels_colors.txt` like original CamVid color label description file.
    http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt
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

    @property
    def label_colors(self):
        colors, _ = self.parse_label_colors()
        return np.array(colors)

    @property
    def classes(self):
        _, classes = self.parse_label_colors()
        return classes

    @property
    def num_classes(self):
        _, classes = self.parse_label_colors()
        return len(classes)

    def parse_label_colors(self):
        with open(os.path.join(self.data_dir, "label_colors.txt")) as f:
            lines = f.readlines()
            lines = [line.rstrip('\n').split('\t') for line in lines]

            # Split "R G B" -> ["R", "G", "B"]
            colors = [line[0].split(' ') for line in lines]
            classes = [line[-1] for line in lines]

            # Void (empty) label is not use for train
            classes.remove('Void')

            return colors, classes

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
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['image_files', 'label_files'],
        )

        image_files = df.image_files.tolist()
        label_files = df.label_files.tolist()

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
