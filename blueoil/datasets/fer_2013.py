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
    available_subsets = ["train", "validation", "test"]
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

    def _load_data(self, path, data_type):
        # Load the label and image data from csv
        lines = _load_csv(path)

        target = ""
        if data_type == "train":
            target = "Training"
        elif data_type == "validation":
            target = "PublicTest"
        elif data_type == "test":
            target = "PrivateTest"
        else:
            raise ValueError("Must provide data_type = train or validation or test")

        data = [line for line in lines if line[2] == target]
        data_num = len(data)
        image_size = self.image_size

        images = np.empty((data_num, image_size, image_size, 3), dtype=np.uint8)
        labels = np.empty(data_num, dtype=np.uint8)
        for i in range(data_num):
            tmp_img = np.array(data[i][1].split(' '), np.uint8).reshape((image_size, image_size))
            # convert grayscale to RGB
            images[i] = np.stack((tmp_img, ) * 3, axis=-1)
            labels[i] = data[i][0]
        # convert to one hot
        labels = np.eye(self.num_classes)[labels]

        return (images, labels)

    def _images_and_labels(self):
        images = np.empty([0, self.image_size, self.image_size, 3])
        labels = np.empty([0, self.num_classes])
        for path in self._all_files():
            (curr_imgs, curr_lbls) = self._load_data(path, self.subset)
            images = np.concatenate([images, curr_imgs])
            labels = np.concatenate([labels, curr_lbls])

        if self.subset == "train":
            images, labels = shuffle(images, labels, seed=0)

        return images, labels

    def _all_files(self):
        return glob(os.path.join(self.data_dir, "*.csv"))

    def __getitem__(self, i):
        return (self.images[i], self.labels[i])

    def __len__(self):
        return self.num_per_epoch
