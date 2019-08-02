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

from lmnet.datasets.base import Base, ObjectDetectionBase


def _grayscale_to_rgb(record):
    return {
        "image": tf.image.grayscale_to_rgb(record["image"]),
        "label": record["label"]
    }


def _format_classification_record(record, image_size, num_classes):
    image = tf.image.resize(record["image"], image_size)
    label = tf.one_hot(record["label"], num_classes)

    return {"image": image, "label": label}


def _format_object_detection_record(record, image_size, num_max_boxes):
    image = tf.image.resize(record["image"], image_size)

    # Convert coordinates from relative to absolute
    ymin = tf.slice(record["objects"]["bbox"], [0, 0], [-1, 1])
    xmin = tf.slice(record["objects"]["bbox"], [0, 1], [-1, 1])
    ymax = tf.slice(record["objects"]["bbox"], [0, 2], [-1, 1])
    xmax = tf.slice(record["objects"]["bbox"], [0, 3], [-1, 1])

    ymin = tf.cast(ymin * image_size[0], tf.int64)
    xmin = tf.cast(xmin * image_size[1], tf.int64)
    ymax = tf.cast(ymax * image_size[0], tf.int64)
    xmax = tf.cast(xmax * image_size[1], tf.int64)

    height = ymax - ymin
    width = xmax - xmin

    # Combine boxes and labels
    label = tf.expand_dims(record["objects"]["label"], axis=1)
    gt_boxes = tf.concat([xmin, ymin, width, height, label], axis=1)

    # Fill gt_boxes with dummy boxes
    dummy_boxes = tf.stack([tf.constant([0, 0, 0, 0, -1], tf.int64)] * num_max_boxes, axis=0)
    gt_boxes = tf.concat([gt_boxes, dummy_boxes], axis=0)
    gt_boxes = tf.slice(gt_boxes, [0, 0], [num_max_boxes, 5])

    return {"image": image, "label": gt_boxes}


class TFDSMixin:
    """
    A Mixin to compose dataset classes for TFDS.
    """
    available_subsets = ["train", "validation"]
    extend_dir = None

    def __init__(
            self,
            tfds_name,
            tfds_data_dir,
            tfds_image_size,
            tfds_download=False,
            num_max_boxes=100,
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
        self.tfds_image_size = tfds_image_size
        self._num_max_boxes = num_max_boxes
        self._format_dataset()

    @property
    def num_per_epoch(self):
        split = self.available_splits[self.subset]
        return self.info.splits[split].num_examples

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

    def _format_dataset(self):
        """
        Converting the format of loaded dataset.
        """
        raise NotImplementedError()


class TFDSClassification(TFDSMixin, Base):
    """
    A dataset class for loading TensorFlow Datasets for classification.
    TensorFlow Datasets which have "label" and "image" features can be loaded by this class.
    """
    @property
    def classes(self):
        return self.info.features["label"].names

    @property
    def num_classes(self):
        return self.info.features["label"].num_classes

    def _validate_feature_structure(self):
        is_valid = \
            "label" in self.info.features and \
            "image" in self.info.features and \
            isinstance(self.info.features["label"], tfds.features.ClassLabel) and \
            isinstance(self.info.features["image"], tfds.features.Image)

        if not is_valid:
            raise ValueError("Datasets should have \"label\" and \"image\" features.")

    def _format_dataset(self):
        if self.info.features['image'].shape[2] == 1:
            self.tf_dataset = self.tf_dataset.map(
                _grayscale_to_rgb,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        self.tf_dataset = self.tf_dataset.map(
            lambda record: _format_classification_record(record, self.tfds_image_size, self.num_classes),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )


class TFDSObjectDetection(TFDSMixin, ObjectDetectionBase):
    """
    A dataset class for loading TensorFlow Datasets for object detection.
    TensorFlow Datasets which have "objects" and "image" features can be loaded by this class.
    """
    @classmethod
    def count_max_boxes(cls):
        raise NotImplementedError()

    @property
    def classes(self):
        return self.info.features["objects"]["label"].names

    @property
    def num_classes(self):
        return self.info.features["objects"]["label"].num_classes

    @property
    def num_max_boxes(self):
        return self._num_max_boxes

    def _validate_feature_structure(self):
        is_valid = \
            "image" in self.info.features and \
            "objects" in self.info.features and \
            "label" in self.info.features["objects"] and \
            "bbox" in self.info.features["objects"] and \
            isinstance(self.info.features["image"], tfds.features.Image) and \
            isinstance(self.info.features["objects"], tfds.features.SequenceDict) and \
            isinstance(self.info.features["objects"]["label"], tfds.features.ClassLabel) and \
            isinstance(self.info.features["objects"]["bbox"], tfds.features.BBoxFeature)

        if not is_valid:
            raise ValueError("Datasets should have \"objects\" and \"image\" features and "
                             "\"objects\" should be a SequenceDict containing \"label\" and \"bbox\".")

    def _format_dataset(self):
        if self.info.features['image'].shape[2] == 1:
            self.tf_dataset = self.tf_dataset.map(
                _grayscale_to_rgb,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        self.tf_dataset = self.tf_dataset.map(
            lambda record: _format_object_detection_record(record, self.tfds_image_size, self.num_max_boxes),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
