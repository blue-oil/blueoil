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
import os.path

import numpy as np
import pandas as pd
from PIL import Image

from lmnet.datasets.base import SegmentationBase


class LipChip(SegmentationBase):
    """LIP multi-person dataset

    http://sysu-hcp.net/lip/overview.php
    """
    classes = [
        "unlabelled",
        "person",
    ]

    num_classes = len(classes)
    extend_dir = "LIP/CHIP/instance-level_human_parsing"

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    @property
    def label_colors(self):
        unlabelled = [0, 0, 0]
        person = [128, 0, 0]

        label_colors = np.array([
            unlabelled, person,
        ])

        return label_colors

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations[0])

    @property
    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation', 'test']

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            base_path = os.path.join(self.data_dir,  "Training")
            txt_file = os.path.join(base_path, "train_id.txt")

        if self.subset == "validation":
            base_path = os.path.join(self.data_dir,  "Validation")
            txt_file = os.path.join(base_path, "val_id.txt")

        df = pd.read_csv(
            txt_file,
            delim_whitespace=True,
            header=None,
            names=['file_ids'])

        # if self.subset == "validation":
        #     df = df[:10]

        ids = df.file_ids.tolist()

        image_files = [os.path.join(base_path, "Images", "{:07d}.jpg".format(id)) for id in ids]
        label_files = [os.path.join(base_path, "Human", "{:07d}.png".format(id)) for id in ids]

        return image_files, label_files

    def __getitem__(self, i, type=None):
        image_files, label_files = self.files_and_annotations

        image = Image.open(image_files[i]).convert("RGB")
        image = np.array(image)

        label = Image.open(label_files[i]).convert("L")
        label = np.array(label)

        # all label map to person
        label[label > 0] = 1

        return (image, label)

    def __len__(self):
        return self.num_per_epoch
