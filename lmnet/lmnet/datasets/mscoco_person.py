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

import sklearn.utils

from lmnet.datasets.mscoco import Mscoco


# TODO(wakisaka): deprecated
class MscocoPerson(Mscoco):
    """Mscoco for segmentation."""

    extend_dir = "MSCOCO"

    def __init__(
        self,
        subset="train",
        batch_size=10,
        *args,
        **kwargs
    ):

        super().__init__(
            subset=subset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

        self.select_classes = ['person', 'dining table', ]
        self.classes = ["__background__"] + ['person', ]
        self.num_classes = len(self.classes)

    @property
    @functools.lru_cache(maxsize=None)
    def _image_ids(self):
        """Return all files and gt_boxes list."""

        target_class_ids = self.coco.getCatIds(catNms=self.select_classes)
        image_ids = self.coco.getImgIds(catIds=target_class_ids)

        new_image_ids = []
        threshold = 0.1

        # use only person area > 0.1
        for image_id in image_ids:
            label = self._label_from_image_id(image_id)
            if (label.sum() / label.size) > threshold:
                new_image_ids.append(image_id)

        image_ids = sklearn.utils.shuffle(new_image_ids)
        print("len", len(image_ids))
        return image_ids
