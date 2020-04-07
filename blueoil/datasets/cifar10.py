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
import pickle

import numpy as np

from blueoil import data_processor
from blueoil.datasets.base import Base
from blueoil.utils.random import shuffle


class Cifar10(Base):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    num_classes = len(classes)
    extend_dir = "CIFAR_10/cifar-10-batches-py"
    available_subsets = ["train", "validation"]

    def __init__(
            self,
            subset="train",
            batch_size=100,
            *args,
            **kwargs
    ):
        super().__init__(
            subset=subset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self._init_images_and_labels()

    @property
    def num_per_epoch(self):
        """Returns the number of datas in the data subset."""

        images = self.images
        return len(images)

    def _get_image(self, index):
        """Returns numpy array of an image"""

        image = self.images[index, :]
        image = image.reshape((3, 32, 32))
        image = image.transpose([1, 2, 0])

        return image

    def _init_images_and_labels(self):
        self.images, self.labels = self._images_and_labels()

    @functools.lru_cache(maxsize=None)
    def _images_and_labels(self):
        if self.subset == "train":
            files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

        else:
            files = ["test_batch"]

        data = [self._load_data(filename) for filename in files]

        images = [images for images, labels in data]
        images = np.concatenate(images, axis=0)

        labels = [labels for images, labels in data]
        labels = np.concatenate(labels, axis=0)

        # randomaize
        if self.subset == "train":
            images, labels = shuffle(images, labels, seed=0)
        return images, labels

    def _unpickle(self, filename):
        filename = os.path.join(self.data_dir, filename)
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            return data

    @functools.lru_cache(maxsize=None)
    def _load_data(self, filename):
        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        labels = np.array(data[b'labels'])

        return images, labels

    def __getitem__(self, i, type=None):
        image = self._get_image(i)
        label = data_processor.binarize(self.labels[i], self.num_classes)
        label = np.reshape(label, (self.num_classes))
        return (image, label)

    def __len__(self):
        return self.num_per_epoch
