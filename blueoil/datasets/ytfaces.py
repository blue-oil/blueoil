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
import os.path

import numpy as np

from blueoil.utils.image import load_image
from blueoil.datasets.base import KeypointDetectionBase


class YoutubeFacialLandmarks(KeypointDetectionBase):
    """Youtube Facial landmarks detection dataset. This dataset is taken from:
    https://github.com/jerrychen44/face_landmark_detection_pytorch

    """

    classes = ['face']
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "ytfaces"

    def __init__(self, subset="train", batch_size=10, *args, **kwargs):
        super().__init__(subset=subset, batch_size=batch_size, *args, **kwargs)

        if subset == 'train':
            self.csv = os.path.join(self.data_dir, "training_frames_keypoints.csv")
            self.image_dir = os.path.join(self.data_dir, "training")
        elif subset == 'validation':
            self.csv = os.path.join(self.data_dir, "test_frames_keypoints.csv")
            self.image_dir = os.path.join(self.data_dir, "test")

        self.num_joints = 68

        self.files, self.joints_list = self._load_csv()

    def _load_csv(self):
        """Read items from JSON files"""
        files = []
        joints_list = []
        num_dimensions = 2

        with open(self.csv) as f:
            # skip header line
            f.readline()
            for line in f:
                filename, *landmarks = line.split(",")
                abs_path = os.path.join(self.image_dir, filename)

                joints = np.ones((self.num_joints, num_dimensions + 1), dtype=np.float32)

                for i in range(self.num_joints):
                    joints[i, 0] = landmarks[i * 2 + 0]
                    joints[i, 1] = landmarks[i * 2 + 1]

                files.append(abs_path)
                joints_list.append(joints)

        return files, joints_list

    def __getitem__(self, item):
        """Get an item given index.

        Args:
            item: int, index.

        Returns:
            image: a numpy array of shape (height, width, 3).
            joints: a numpy array of shape (68, 3), which has coordinates in image.

        """

        return load_image(self.files[item]), self.joints_list[item]

    def __len__(self):
        return len(self.files)

    @property
    def num_per_epoch(self):
        return len(self.files)
