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
from glob import glob

from lmnet.datasets.base import Base
from blueoil.utils.image import load_image


class Div2k(Base):
    classes = []
    num_classes = 0
    extend_dir = "DIV2K"
    available_subsets = ["train", "validation"]

    @property
    @functools.lru_cache(maxsize=None)
    def files(self):
        if self.subset == "train":
            images_dir = os.path.join(self.data_dir, "DIV2K_train_HR")
        else:
            images_dir = os.path.join(self.data_dir, "DIV2K_valid_HR")

        return [filepath for filepath in glob(os.path.join(images_dir, "*.png"))]

    @property
    def num_per_epoch(self):
        return len(self.files)

    def __getitem__(self, i, type=None):
        target_file = self.files[i]
        image = load_image(target_file)

        return image, None

    def __len__(self):
        return self.num_per_epoch
