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

import numpy as np

from lmnet.datasets.cifar10 import Cifar10
from lmnet.datasets.base import DistributionInterface
from lmnet.utils.random import shuffle


class Cifar10Distribution(Cifar10, DistributionInterface):
    def __init__(
            self,
            batch_size=100,
            *args,
            **kwargs
    ):
        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self._init_images_and_labels()

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

        return images, labels

    def update_dataset(self, indices):
        """Update own dataset by indices."""
        # Re Initialize dataset
        self._init_images_and_labels()
        # Update dataset by given indices
        self.images = self.images[indices, :]
        self.labels = self.labels[indices]

        self.current_element_index = 0

    def get_shuffle_index(self):
        """Return list of shuffled index."""
        images, _ = self._images_and_labels()
        random_indices = shuffle(range(len(images)), seed=self.seed)
        print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
        self.seed += 1

        return random_indices
