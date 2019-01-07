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
import os
import os.path

import numpy as np
import PIL
import pandas as pd


from lmnet.datasets.base import Base
from lmnet import data_processor
from lmnet.utils.random import shuffle


class Ilsvrc2012(Base):

    extend_dir = "ILSVRC2012"
    # classes = [str(n) for n in range(0, 1000)]
    num_classes = 1000
    # `test` subsets don't have ground truth.
    available_subsets = ["train", "validation"]

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs,)

        self.dirs = {
            "train": os.path.join(self.data_dir, "train"),
            "validation": os.path.join(self.data_dir, "val"),
            "test": os.path.join(self.data_dir, "test"),
        }

        self.texts = {
            "train": os.path.join(self.data_dir, "train.txt"),
            "validation": os.path.join(self.data_dir, "val.txt"),
            "test": os.path.join(self.data_dir, "test.txt"),
        }
        self._init_files_and_annotations()

    @property
    def classes(self):
        # wget https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt
        df = pd.read_csv(os.path.join(self.data_dir, 'imagenet_classes.txt'), sep="\n", header=None)
        return df[0].tolist()

    @property
    def num_per_epoch(self):
        files, _ = self._files_and_annotations()
        return len(files)

    def feed(self):
        """Returns numpy array of batch size data.

        Returns:
            images: images numpy array. shape is [batch_size, height, width]
            labels: one hot labels. shape is [batch_size, num_classes]
        """

        images, labels = zip(*[self._element() for _ in range(self.batch_size)])

        labels = data_processor.binarize(labels, self.num_classes)

        images = np.array(images)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, labels

    def _element(self):
        """Return an image and label."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

        files, labels = self.files, self.annotations
        target_file = files[index]
        label = labels[index]

        image = PIL.Image.open(target_file)
        # sometime image data be gray.
        image = image.convert("RGB")

        image = np.array(image)

        samples = {'image': image}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        return image, label

    def _shuffle(self):
        """Shuffle data if train."""

        if self.subset == "train":
            self.files, self.annotations = shuffle(
                self.files, self.annotations, seed=self.seed)
            print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
            self.seed += 1

    def _init_files_and_annotations(self):
        self.files, self.annotations = self._files_and_annotations()
        self._shuffle()

    @functools.lru_cache(maxsize=None)
    def _files_and_annotations(self):
        txt_file = self.texts[self.subset]

        df = pd.read_csv(
            txt_file,
            delim_whitespace=True,
            header=None,
            names=['filename', 'class_id'])

        files = df.filename.tolist()
        files = [os.path.join(self.dirs[self.subset], filename) for filename in files]

        labels = df.class_id.tolist()

        return files, labels
