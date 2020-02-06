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
import imghdr
import os
import os.path
from glob import glob

import numpy as np

from nn import data_processor
from nn.utils.image import load_image
from nn.datasets.base import Base, StoragePathCustomizable
from nn.utils.random import train_test_split


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
        return len(self.files)

    def _all_files(self):
        all_image_files = []
        for image_class in self.classes:
            image_dir = os.path.join(self.data_dir, image_class)
            for image_path in glob(os.path.join(image_dir, "*")):
                if os.path.isfile(image_path) and imghdr.what(image_path) in {"jpeg", "png"}:
                    all_image_files.append(image_path)

        return all_image_files

    @property
    @functools.lru_cache(maxsize=None)
    def files(self):
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

    def __getitem__(self, i, type=None):
        target_file = self.files[i]

        image = load_image(target_file)
        label = self.get_label(target_file)

        label = data_processor.binarize(label, self.num_classes)
        label = np.reshape(label, (self.num_classes))
        return (image, label)

    def __len__(self):
        return self.num_per_epoch
