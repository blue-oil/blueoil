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
from abc import ABCMeta
from abc import abstractmethod
import os

import numpy as np
import PIL

from lmnet import environment


class Base(metaclass=ABCMeta):
    """Dataset base class"""

    def __init__(
            self,
            subset="train",
            batch_size=10,
            augmentor=None,
            pre_processor=None,
            data_format='NHWC',
            seed=None,
            **kwargs
    ):
        assert subset in self.available_subsets, self.available_subsets
        self.subset = subset
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_processor = pre_processor
        self.data_format = data_format
        self.seed = seed or 0

    @property
    def data_dir(self):
        extend_dir = self.__class__.extend_dir

        if extend_dir is None:
            data_dir = environment.DATA_DIR
        else:
            data_dir = os.path.join(environment.DATA_DIR, extend_dir)
        return data_dir

    @property
    @staticmethod
    @abstractmethod
    def classes():
        """Return the classes list in the data set."""
        pass

    @property
    @staticmethod
    @abstractmethod
    def num_classes():
        """Return the number of classes in the data set."""
        pass

    @property
    @staticmethod
    @abstractmethod
    def extend_dir():
        """Return the extend dir path of the data set."""
        pass

    @property
    @staticmethod
    @abstractmethod
    def available_subsets():
        """Returns the list of available subsets."""
        return ['train', 'train_validation_saving', 'validation']

    @property
    @abstractmethod
    def num_per_epoch(self):
        """Returns the number of datas in the data subset."""
        pass

    @property
    @abstractmethod
    def __getitem__(self, i, type=None):
        """Returns the i-th item of the dataset."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def __len__(self):
        """returns the number of items in the dataset."""
        raise NotImplementedError()


class SegmentationBase(Base, metaclass=ABCMeta):

    def __init__(self, *args, label_colors=None, **kwargs):
        super(SegmentationBase, self).__init__(*args, **kwargs)
        self._label_colors = label_colors

    @property
    def label_colors(self):
        if self._label_colors:
            return self._label_colors
        random_state = np.random.RandomState(seed=self.seed)
        self._label_colors = random_state.choice(256, (self.num_classes, 3))
        return self._label_colors


class ObjectDetectionBase(Base, metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def count_max_boxes(cls):
        """Count max boxes size over all subsets."""
        pass

    @property
    @abstractmethod
    def num_max_boxes(self):
        """Returns conunt max box size of available subsets."""
        pass

    def _fill_dummy_boxes(self, gt_boxes):
        dummy_gt_box = [0, 0, 0, 0, -1]
        if len(gt_boxes) == 0:
            gt_boxes = np.array(dummy_gt_box * self.num_max_boxes)
            return gt_boxes.reshape([self.num_max_boxes, 5])
        elif len(gt_boxes) < self.num_max_boxes:
            diff = self.num_max_boxes - len(gt_boxes)
            gt_boxes = np.append(gt_boxes, [dummy_gt_box] * diff, axis=0)
            return gt_boxes
        return gt_boxes

    def _change_gt_boxes_shape(self, gt_boxes_list):
        """Change gt boxes list shape from [batch_size, num_boxes, 5] to [batch_size, num_max_boxes, 5].

        fill dummy box when num boxes < num max boxes.

        Args:
          gt_boxes_list: python list of gt_boxes(np.ndarray). gt_boxes's shape is [batch_size, num_boxes, 5]

        Return:
          gt_boxes_list: numpy ndarray [batch_size, num_max_boxes, 5].
        """
        results = []

        for gt_boxes in gt_boxes_list:
            gt_boxes = self._fill_dummy_boxes(gt_boxes)
            results.append(gt_boxes)

        return np.array(results)

    def _get_image(self, target_file):
        image = PIL.Image.open(target_file)
        image = image.convert("RGB")
        image = np.array(image)
        return image


class DistributionInterface(metaclass=ABCMeta):

    @abstractmethod
    def update_dataset(self, indices):
        """Update own dataset by indices."""
        pass

    @abstractmethod
    def get_shuffle_index(self):
        """Return list of shuffled index."""
        pass


class StoragePathCustomizable():
    """Make it possible to specify train, validation path.

    class.extend_dir: specify train path.
    class.validation_extend_dir: specify validation path.

    When validation_extend_dir doesn't set, generate validation data from train set.
    You should implement the validation subset split from train data with `validation_size` in sub class.
    """

    available_subsets = ['train', 'validation']

    def __init__(
            self,
            validation_size=0.1,
            *args,
            **kwargs
    ):
        # validation subset size
        self.validation_size = validation_size
        if hasattr(self.__class__, "validation_extend_dir"):
            self.validation_size = 0

        super().__init__(*args, **kwargs)

    @property
    def _train_data_dir(self):
        extend_dir = self.__class__.extend_dir
        if extend_dir is None:
            data_dir = environment.DATA_DIR
        else:
            data_dir = os.path.join(environment.DATA_DIR, extend_dir)
        return data_dir

    @property
    def _validation_data_dir(self):
        extend_dir = self.__class__.extend_dir
        if hasattr(self.__class__, "validation_extend_dir"):
            extend_dir = self.__class__.validation_extend_dir

        if extend_dir is None:
            data_dir = environment.DATA_DIR
        else:
            data_dir = os.path.join(environment.DATA_DIR, extend_dir)
        return data_dir

    @property
    def data_dir(self):
        if self.subset is "train":
            return self._train_data_dir

        if self.subset is "validation":
            return self._validation_data_dir
