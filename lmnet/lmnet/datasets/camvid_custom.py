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
import functools
import os.path
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import PIL.Image

from lmnet.datasets.base import SegmentationBase
from lmnet.utils.random import shuffle


def get_image(filename, convert_rgb=True):
    """Returns numpy array of an image"""
    image = PIL.Image.open(filename)

    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
    else:
        image = image.convert("L")

    image = np.array(image)
    return image


def fetch_one_data(args):
    image_file, label_file, augmentor, pre_processor, is_train = args

    image = get_image(image_file)
    mask = get_image(label_file, convert_rgb=False)

    samples = {'image': image, 'mask': mask}

    if callable(augmentor) and is_train:
        samples = augmentor(**samples)

    if callable(pre_processor):
        samples = pre_processor(**samples)

    image = samples['image']
    mask = samples['mask']

    return (image, mask)


class CamvidCustom(SegmentationBase):
    """CamvidCustom

    CamVid base custom dataset format.
    http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

    To define your own colors of annotation, make `labels_colors.txt` like CamVid color description file.
    http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt
    """

    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        self.use_prefetch = kwargs.pop("enable_prefetch", False)
        self.num_prefetch_process = kwargs.pop("num_prefetch_processes", 8)

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        if self.use_prefetch:
            self.enable_prefetch()
            print("ENABLE prefetch")
        else:
            print("DISABLE prefetch")

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

            return colors, classes

    def prefetch_args(self, i):
        return (self.image_files[i], self.label_files[i], self.augmentor, self.pre_processor, self.subset == "train")

    def enable_prefetch(self):
        self.pool = Pool(processes=self.num_prefetch_process)
        self._shuffle()
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

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations[0])

    @property
    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

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

        image_files = [os.path.join(self.data_dir, filename) for filename in image_files]
        label_files = [os.path.join(self.data_dir, filename) for filename in label_files]

        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files

    def _shuffle(self):
        image_files, label_files = self.files_and_annotations
        image_files, label_files = zip(*random.sample(list(zip(image_files, label_files)), len(image_files)))
        self.image_files = image_files
        self.label_files = label_files

    def _element(self):
        """Return an image and label."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0

        image_files, label_files = self.files_and_annotations

        image = get_image(image_files[index])
        label = get_image(label_files[index], convert_rgb=False)

        samples = {'image': image, 'mask': label}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        label = samples['mask']

        return (image, label)

    def get_data(self):
        if self.use_prefetch:
            data_list = self.prefetch_result.get(None)
            self.start_prefetch()
            images, masks = zip(*data_list)
            return images, masks
        else:
            images, masks = zip(*[self._element() for _ in range(self.batch_size)])

            return images, masks

    def feed(self):
        """Returns batch size numpy array of images and binarized labels."""
        images, labels = self.get_data()
        images, labels = np.array(images), np.array(labels)

        if self.data_format == 'NCHW':
            images = np.transpose(images, [0, 3, 1, 2])
        return images, labels
