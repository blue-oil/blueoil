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


class Cifar100(Base):
    classes = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"] # NOQA

    num_classes = len(classes)
    extend_dir = "CIFAR_100/cifar-100-python"
    available_subsets = ["train", "train_validation_saving", "validation"]

    def __init__(
            self,
            subset="train",
            batch_size=100,
            train_validation_saving_size=0,
            *args,
            **kwargs
    ):
        super().__init__(
            subset=subset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self.train_validation_saving_size = train_validation_saving_size

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
        if self.subset == "train" or self.subset == "train_validation_saving":
            files = ["train"]

        else:
            files = ["test"]

        data = [self._load_data(filename) for filename in files]

        images = [images for images, labels in data]
        images = np.concatenate(images, axis=0)

        labels = [labels for images, labels in data]
        labels = np.concatenate(labels, axis=0)

        if self.train_validation_saving_size > 0:
            # split the train set into train and train_validation_saving
            if self.subset == "train":
                images, _ = np.split(images, [-self.train_validation_saving_size], axis=0)
                labels, _ = np.split(labels, [-self.train_validation_saving_size], axis=0)
            elif self.subset == "train_validation_saving":
                _, images = np.split(images, [-self.train_validation_saving_size], axis=0)
                _, labels = np.split(labels, [-self.train_validation_saving_size], axis=0)

        # randomaize
        if self.subset == "train":
            images, labels = shuffle(images, labels, seed=0)
        return images, labels

    def _unpickle(self, filename):
        filename = os.path.join(self.data_dir, filename)
        with open(filename, "rb") as file:
            data = pickle.load(file, encoding="bytes")
            return data

    @functools.lru_cache(maxsize=None)
    def _load_data(self, filename):
        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        images = data[b"data"]

        # Get the class-numbers for each image. Convert to numpy-array.
        labels = np.array(data[b"fine_labels"])

        return images, labels

    def __getitem__(self, i, type=None):
        image = self._get_image(i)
        label = data_processor.binarize(self.labels[i], self.num_classes)
        label = np.reshape(label, (self.num_classes))
        return (image, label)

    def __len__(self):
        return self.num_per_epoch
