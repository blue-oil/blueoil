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
import tensorflow as tf
import tensorflow_datasets as tfds

from lmnet.datasets.base import Base


def _label_to_one_hot(record, depth):
    return {
        'image': record['image'],
        'label': tf.one_hot(record['label'], depth)
    }


class TFDSBase(Base):
    """
    Abstract dataset class for loading TensorFlow Datasets.
    Only images which has 3 channels (RGB) can be loaded for now.
    TODO(fujiwara): Convert grayscale images into RGB images.
    """
    available_subsets = ["train", "validation"]
    extend_dir = None

    def __init__(
            self,
            tfds_name,
            tfds_data_dir,
            tfds_download=False,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        builder = tfds.builder(tfds_name, data_dir=tfds_data_dir)
        if tfds_download:
            builder.download_and_prepare()

        self.info = builder.info
        self._init_available_splits()
        self._validate_feature_structure()

        self.tf_dataset = builder.as_dataset(split=self.available_splits[self.subset])
        self.tf_dataset = self.tf_dataset.map(
            lambda record: _label_to_one_hot(record, self.num_classes),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    @property
    def num_per_epoch(self):
        split = self.available_splits[self.subset]
        return self.info.splits[split].num_examples

    @property
    def classes(self):
        return self.info.features["label"].names

    @property
    def num_classes(self):
        return self.info.features["label"].num_classes

    @property
    def __getitem__(self, i):
        raise NotImplementedError()

    def __len__(self):
        return self.num_per_epoch

    def _init_available_splits(self):
        """
        Initializing available splits dictionary depending on
        what kind of splits the dataset has.
        """
        self.available_splits = {}
        if tfds.Split.TRAIN not in self.info.splits:
            raise ValueError("Datasets need to have a split \"TRAIN\".")

        if tfds.Split.VALIDATION in self.info.splits and tfds.Split.TEST in self.info.splits:
            self.available_splits["train"] = tfds.Split.TRAIN
            self.available_splits["validation"] = tfds.Split.VALIDATION
            self.available_splits["test"] = tfds.Split.TEST

        elif tfds.Split.VALIDATION in self.info.splits:
            self.available_splits["train"] = tfds.Split.TRAIN
            self.available_splits["validation"] = tfds.Split.VALIDATION

        elif tfds.Split.TEST in self.info.splits:
            self.available_splits["train"] = tfds.Split.TRAIN
            self.available_splits["validation"] = tfds.Split.TEST

        else:
            raise ValueError("Datasets need to have a split \"VALIDATION\" or \"TEST\".")

    def _validate_feature_structure(self):
        """
        Checking if the given dataset has a valid feature structure.
        This method will raise a ValueError if the structure is invalid.
        """
        raise NotImplementedError()


class TFDSClassification(TFDSBase):
    """
    A dataset class for loading TensorFlow Datasets for classification.
    TensorFlow Datasets which have "label" and "image" features can be loaded by this class.
    """
    def _validate_feature_structure(self):
        is_valid = \
            "label" in self.info.features and \
            "image" in self.info.features and \
            isinstance(self.info.features["label"], tfds.features.ClassLabel) and \
            isinstance(self.info.features["image"], tfds.features.Image)

        if not is_valid:
            raise ValueError("Datasets should have \"label\" and \"image\" features.")
