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
import functools
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
            **kwargs
        )
        self.images, self.labels = self._images_and_labels()

    @property
    def num_classes(self):
        """Returns the number of classes"""
        return len(self.classes)

    @property
    def num_per_epoch(self):
        """Returns the number of datas in the data subset"""

        return len(self.images)

    @functools.lru_cache(maxsize=None)
    def _load_data(self, path):
        # Load the label and image data from csv
        lines = _load_csv(path)
        train = [line for line in lines if line[2] == "Training"]
        public_test = [line for line in lines if line[2] == "PublicTest"]

        train_num = len(train)
        public_test_num = len(public_test)

        train_images = [None] * train_num
        train_labels = [None] * train_num
        public_test_images = [None] * public_test_num
        public_test_labels = [None] * public_test_num

        for i in range(train_num):
            train_images[i] = train[i][1].split(' ')
            train_labels[i] = train[i][0]
        for i in range(public_test_num):
            public_test_images[i] = public_test[i][1].split(' ')
            public_test_labels[i] = public_test[i][0]
        
        return (train_images, train_labels), (public_test_images, public_test_labels)

    @functools.lru_cache(maxsize=None)
    def _images_and_labels(self):
        images = []
        labels = []
        for path in self._all_files():
            (train_imgs, train_lbls), (test_imgs, test_lbls) = self._load_data(path)
            if self.subset == "train":
                images += train_imgs
                labels += train_lbls
            else:
                images += test_imgs
                labels += test_lbls

        images = np.array(images, np.uint8).reshape((len(images), self.image_size, self.image_size))
        labels = np.array(labels, np.uint8)

        if self.subset == "train":
            images, labels = shuffle(images, labels, seed=0)

        return images, labels

    def _all_files(self):
        all_csv_files = []
        for csv_path in glob(os.path.join(self.data_dir, "*.csv")):
            all_csv_files.append(csv_path)

        return all_csv_files
    
    def __getitem__(self, i):
        image = self.images[i]
        label = self.labels[i]

        return (image, label)

    def __len__(self):
        return self.num_per_epoch