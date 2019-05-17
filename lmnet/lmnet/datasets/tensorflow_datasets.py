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
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lmnet.datasets.base import Base, ObjectDetectionBase
from lmnet import data_processor


class TensorFlowDatasetsBase(Base):
    """
    Abstract dataset class for loading TensorFlow Datasets.
    The parameter "extend_dir" should be a path to the prepared data of tfds.
    "extend_dir" can be a path either in local or in Google Cloud Storage.
    """
    available_subsets = ["train", "train_validation_saving", "validation"]

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # The last directory name of extend_dir is the dataset name
        splitted_path = self.extend_dir.rstrip('/').split('/')
        data_dir = '/'.join(splitted_path[:-1])
        name = splitted_path[-1]
        builder = tfds.builder(name, data_dir=data_dir)

        self.info = builder.info
        self._init_available_splits()

        self.dataset = builder.as_dataset(split=self.available_splits[self.subset])
        self._init_features()

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
            self.available_splits["train_validation_saving"] = tfds.Split.TEST

        elif tfds.Split.VALIDATION in self.info.splits:
            self.available_splits["train"] = tfds.Split.TRAIN
            self.available_splits["validation"] = tfds.Split.VALIDATION

        elif tfds.Split.TEST in self.info.splits:
            self.available_splits["train"] = tfds.Split.TRAIN
            self.available_splits["validation"] = tfds.Split.TEST

        else:
            train, validation = tfds.Split.TRAIN.subsplit([9, 1])
            self.available_splits["train"] = train
            self.available_splits["validation"] = validation

    def _init_features(self):
        """
        Initializing feature variables by using tf.Tensor from datasets.
        For classification, "images" and "labels" should be initialized.
        For object detection, "images" and "gt_boxes" should be initialized.
        """
        raise NotImplementedError()

    @property
    def num_per_epoch(self):
        return len(self.images)


class TensorFlowDatasetsClassification(TensorFlowDatasetsBase):
    """
    A dataset class for loading TensorFlow Datasets for classification.
    """
    @property
    def classes(self):
        return self.info.features["label"].names

    @property
    def num_classes(self):
        return self.info.features["label"].num_classes

    def _init_features(self):
        self.images = []
        self.labels = []

        iterator = self.dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # tf.Session is needed because "next_element" returns a tf.Tensor.
        with tf.Session() as sess:
            while True:
                try:
                    element = sess.run(next_element)
                    image = element["image"]
                    label = element["label"]

                    # Converting grayscale images into RGB images.
                    # This workaround is needed because the method PIL.Image.fromarray()
                    # in pre_prpcessor requires images array to have a shape of (h, w, 3).
                    if image.shape[2] == 1:
                        image = np.stack([image] * 3, 3)
                        image = image.reshape(image.shape[:2] + (3,))

                    label = data_processor.binarize(label, self.num_classes)
                    label = np.reshape(label, (self.num_classes))

                    self.images.append(image)
                    self.labels.append(label)

                except tf.errors.OutOfRangeError:
                    break;

    def __getitem__(self, i, type=None):
        return (self.images[i], self.labels[i])

    def __len__(self):
        return self.num_per_epoch
