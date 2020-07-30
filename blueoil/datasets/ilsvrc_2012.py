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
import os.path

import numpy as np

from blueoil import data_processor
from blueoil.utils.image import load_image
from blueoil.datasets.base import Base


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
    @functools.lru_cache(maxsize=None)
    def classes(self):
        # wget https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt
        with open(os.path.join(self.data_dir, 'imagenet_classes.txt')) as f:
            return [line.rstrip('\n') for line in f]

    @property
    def num_per_epoch(self):
        files, _ = self._files_and_annotations()
        return len(files)

    def _init_files_and_annotations(self):
        self.files, self.annotations = self._files_and_annotations()

    @functools.lru_cache(maxsize=None)
    def _files_and_annotations(self):
        txt_file = self.texts[self.subset]

        files, labels = list(), list()
        with open(txt_file) as f:
            for line in f:
                items = line.split()
                files.append(items[0])
                labels.append(int(items[1]))

        files = [os.path.join(self.dirs[self.subset], filename) for filename in files]
        return files, labels

    def __getitem__(self, i):
        filename = self.files[i]

        image = load_image(filename)

        label = data_processor.binarize(self.annotations[i], self.num_classes)
        label = np.reshape(label, (self.num_classes))
        return (image, label)

    def __len__(self):
        return self.num_per_epoch
