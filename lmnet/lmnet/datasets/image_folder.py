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
from glob import glob
import imghdr
import os
import os.path

import numpy as np
import PIL.Image

from lmnet.datasets.base import Base, StoragePathCustomizable
from lmnet import data_processor
from lmnet.utils.random import shuffle, train_test_split


class ImageFolderBase(StoragePathCustomizable, Base):
    """Abstract class of dataset for loading image files stored in a folder.

    structure like
        $DATA_DIR/extend_dir/cat/0001.jpg
        $DATA_DIR/extend_dir/cat/xxxa.jpeg
        $DATA_DIR/extend_dir/cat/yyyb.png
        $DATA_DIR/extend_dir/dog/123.jpg
        $DATA_DIR/extend_dir/dog/023.jpg
        $DATA_DIR/extend_dir/dog/wwww.jpg

    When child class has `validation_extend_dir`, the `validation` subset consists from the folders.
       $DATA_DIR/validation_extend_dir/cat/0001.jpg
       $DATA_DIR/validation_extend_dir/cat/xxxa.png
    """

    def __init__(
            self,
            is_shuffle=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.is_shuffle = is_shuffle
        self.element_counter = 0

    @property
    @functools.lru_cache(maxsize=None)
    def classes(self):
        """Returns the classes list in the data set."""

        classes = os.listdir(self.data_dir)
        classes = [class_name for class_name in classes if class_name != ".DS_Store"]
        classes.sort(key=lambda item: item.lower())

        return classes

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_per_epoch(self):
        return len(self.data_files)

    def _all_files(self):
        all_image_files = []
        for image_class in self.classes:
            image_dir = os.path.join(self.data_dir, image_class)
            for image_path in glob(os.path.join(image_dir, "*")):
                if os.path.isfile(image_path) and imghdr.what(image_path) in ["jpeg", "png"]:
                    all_image_files.append(image_path)

        return all_image_files

    @property
    @functools.lru_cache(maxsize=None)
    def data_files(self):
        all_image_files = self._all_files()

        if self.validation_size > 0:
            train_image_files, test_image_files = train_test_split(
                all_image_files, test_size=self.validation_size, seed=1)

            if self.subset == "train":
                files = train_image_files
            else:
                files = test_image_files

            return files

        return all_image_files

    def get_label(self, filename):
        """Returns label."""
        class_name = os.path.basename(os.path.dirname(filename))
        label = self.classes.index(class_name)

        return label

    def get_image(self, filename):
        """Returns numpy array of an image"""
        image = PIL.Image.open(filename)

        #  sometime image data is gray.
        image = image.convert("RGB")

        image = np.array(image)

        return image

    @property
    def feed_indices(self):
        if not hasattr(self, "_feed_indices"):
            if self.subset == "train" and self.is_shuffle:
                self._feed_indices = shuffle(range(self.num_per_epoch), seed=self.seed)
            else:
                self._feed_indices = list(range(self.num_per_epoch))

        return self._feed_indices

    def _get_index(self, counter):
        return self.feed_indices[counter]

    def _shuffle(self):
        if self.subset == "train" and self.is_shuffle:
            self._feed_indices = shuffle(range(self.num_per_epoch), seed=self.seed)
            print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
            self.seed = self.seed + 1

    def _element(self):
        """Return an image and label."""
        index = self._get_index(self.element_counter)

        self.element_counter += 1
        if self.element_counter == self.num_per_epoch:
            self.element_counter = 0
            self._shuffle()

        target_file = self.data_files[index]

        image = self.get_image(target_file)
        label = self.get_label(target_file)

        samples = {'image': image}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        return image, label

    def feed(self):
        """Returns batch size numpy array of images and binarized labels."""
        images, labels = zip(*[self._element() for _ in range(self.batch_size)])

        labels = data_processor.binarize(labels, self.num_classes)

        images = np.array(images)

        if self.data_format == 'NCHW':
            images = np.transpose(images, [0, 3, 1, 2])
        return images, labels
