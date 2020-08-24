# Copyright 2020 The Blueoil Authors. All Rights Reserved.
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
import csv
from glob import glob

import numpy as np

from blueoil.datasets.base import Base
from blueoil.utils.random import shuffle


def _load_csv(path):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        lines = [row for row in reader]
    return lines[1:]  # pass the header


class FER2013(Base):
    """
    Dataset loader class for loading FER2013 format csv.

    FER2013 format csv:
        emotion,pixels,Usage
        e1,pixels1,usage1
        e2,pixels2,usage2
        ...

        where
          e      : a integer in [0, 6]
          pixels : 48 x 48 integers in [0, 255] separated with one space
          usage  : a string, one of Training or PublicTest or PrivateTest
    """

    classes = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral"
    ]
    image_size = 48
    num_classes = len(classes)
    available_subsets = ["train", "test"]
    extend_dir = "FER2013"

    def __init__(self, subset="train", batch_size=100, *args, **kwargs):
        super().__init__(subset=subset, batch_size=batch_size, *args, **kwargs)
        self.images, self.labels = self._images_and_labels()

    @property
    def num_classes(self):
        """Returns the number of classes"""
        return len(self.classes)

    @property
    def num_per_epoch(self):
        """Returns the number of datas in the data subset"""

        return len(self.images)

    def _load_data(self, path):
        # Load the label and image data from csv
        lines = _load_csv(path)
        train = []
        public_test = []
        for line in lines:
            if line[2] == "Training":
                train.append(line)
            elif line[2] == "PublicTest":
                public_test.append(line)

        train_num = len(train)
        public_test_num = len(public_test)
        image_size = self.image_size

        train_images = np.empty((train_num, image_size, image_size), dtype=np.uint8)
        train_labels = np.empty(train_num, dtype=np.uint8)
        public_test_images = np.empty((public_test_num, image_size, image_size), dtype=np.uint8)
        public_test_labels = np.empty(public_test_num, dtype=np.uint8)

        for i in range(train_num):
            train_images[i] = np.array(train[i][1].split(' '), np.uint8).reshape((image_size, image_size))
            train_labels[i] = train[i][0]
        for i in range(public_test_num):
            public_test_images[i] = np.array(public_test[i][1].split(' '), np.uint8).reshape((image_size, image_size))
            public_test_labels[i] = public_test[i][0]

        return (train_images, train_labels), (public_test_images, public_test_labels)

    def _images_and_labels(self):
        images = np.empty([0, self.image_size, self.image_size])
        labels = np.empty([0])
        for path in self._all_files():
            (train_imgs, train_lbls), (test_imgs, test_lbls) = self._load_data(path)
            if self.subset == "train":
                images = np.concatenate([images, train_imgs])
                labels = np.concatenate([labels, train_lbls])
            else:
                images = np.concatenate([images, test_imgs])
                labels = np.concatenate([labels, test_lbls])

        if self.subset == "train":
            images, labels = shuffle(images, labels, seed=0)

        return images, labels

    def _all_files(self):
        return glob(os.path.join(self.data_dir, "*.csv"))

    def __getitem__(self, i):
        return (self.images[i], self.labels[i])

    def __len__(self):
        return self.num_per_epoch
