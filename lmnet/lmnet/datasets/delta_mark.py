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
import json
import os.path

import numpy as np
import PIL.Image
import pandas as pd

from lmnet.datasets.base import Base, ObjectDetectionBase, StoragePathCustomizable, DistributionInterface
from lmnet import data_processor
from lmnet.utils.random import shuffle, train_test_split


@functools.lru_cache(maxsize=None)
def _load_json(json_file):
    f = open(json_file)
    data = json.load(f)
    f.close()

    return data


@functools.lru_cache(maxsize=None)
def _image_ids(json_file):
    images = _images_from_json(json_file)

    return images.id.tolist()


@functools.lru_cache(maxsize=None)
def _annotations_from_json(json_file):
    data = _load_json(json_file)
    annotations = pd.DataFrame(data["annotations"])

    return annotations


@functools.lru_cache(maxsize=None)
def _categories_from_json(json_file):
    data = _load_json(json_file)
    categories = pd.DataFrame(data["categories"])

    return categories


@functools.lru_cache(maxsize=None)
def _images_from_json(json_file):
    data = _load_json(json_file)
    images = pd.DataFrame(data["images"])

    return images


class DeltaMarkMixin(DistributionInterface):

    def __init__(
            self,
            is_shuffle=True,
            json_file="json/annotation.json",
            image_dir="images",
            *args,
            **kwargs
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.is_shuffle = is_shuffle
        self.element_counter = 0

        self.path = {
            "json": os.path.join(self.data_dir, json_file),
            "dir": os.path.join(self.data_dir, image_dir)
        }
        self.files, self.annotations = self.files_and_annotations

    @property
    def indices(self):
        if not hasattr(self, "_indices"):
            if self.subset == "train" and self.is_shuffle:
                self._indices = shuffle(range(self.num_per_epoch), seed=self.seed)
            else:
                self._indices = list(range(self.num_per_epoch))

        return self._indices

    def _get_index(self, counter):
        return self.indices[counter]

    def _shuffle(self):
        if self.subset == "train" and self.is_shuffle:
            self._indices = shuffle(range(self.num_per_epoch), seed=self.seed)
            print("Shuffle {} train dataset with random seed {}.".format(self.__class__.__name__, self.seed))
            self.seed = self.seed + 1

    @property
    def classes(self):
        data_frame = _categories_from_json(self.path["json"])
        return list(data_frame.name)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_per_epoch(self):
        files = self.files
        return len(files)

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and labels list."""

        all_files, all_annotations = self._files_and_annotations(self.path['json'], self.path['dir'])

        if self.validation_size > 0:
            train_files, test_files, train_annotations, test_annotations =\
                train_test_split(all_files,
                                 all_annotations,
                                 test_size=self.validation_size,
                                 seed=1)

            if self.subset == "train":
                files = train_files
                annotations = train_annotations
            else:
                files = test_files
                annotations = test_annotations
        else:
            files, annotations = all_files, all_annotations

        return files, annotations

    def _files_and_annotations(self):
        raise NotImplementedError

    # For distributed training
    def update_dataset(self, indices):
        """Update own dataset by indices."""
        # Re Initialize dataset
        files, annotations = self.files_and_annotations
        # Update dataset by given indices
        self.files = [files[i] for i in indices]
        self.annotations = [annotations[i] for i in indices]

        self.element_counter = 0

    def get_shuffle_index(self):
        """Return list of shuffled index."""
        files, _ = self.files_and_annotations
        random_indices = shuffle(range(len(files)), seed=self.seed)
        print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
        self.seed += 1

        return random_indices


class ClassificationBase(DeltaMarkMixin, StoragePathCustomizable, Base):
    """Abstract class of dataset for classification created by Delta-Mark

    When you create the extend class with `extend_path` and validation extend dir as class property,
    the training and validation data consists from following structure.
    When you only set extend_path as class property,
    training and validation dataset are generated from extend_path according to validation_size value.

    structure:
        $DATA_DIR/extend_dir/json/annotation.json
        $DATA_DIR/extend_dir/images/xxxa.jpeg
        $DATA_DIR/extend_dir/images/yyyb.png
        $DATA_DIR/extend_dir/images/123.jpg

        $DATA_DIR/validation_extend_dir/json/annotation.json
        $DATA_DIR/validation_extend_dir/images/xxxa.png
    """

    available_subsets = ["train", "validation"]

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          labels: labels numpy array. shape is [batch_size, num_classes]
        """
        images, labels_list = zip(*[self._element() for _ in range(self.batch_size)])
        images = np.array(images)

        labels_list = data_processor.binarize(labels_list, self.num_classes)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, labels_list

    def _element(self):
        """Return an image, labels."""
        index = self._get_index(self.element_counter)
        self.element_counter += 1
        if self.element_counter == self.num_per_epoch:
            self.element_counter = 0
            self._shuffle()

        target_file = self.files[index]
        labels = self.annotations[index]
        labels = np.array(labels)

        image = PIL.Image.open(target_file)
        image = image.convert("RGB")
        image = np.array(image)

        samples = {'image': image}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']

        return image, labels

    def _files_and_annotations(self, json_file, image_dir):
        """Return files and gt_boxes list."""
        image_ids = _image_ids(json_file)

        image_files = [self._image_file_from_image_id(image_dir, json_file, image_id) for image_id in image_ids]
        labels_list = [self._labels_from_image_id(json_file, image_id) for image_id in image_ids]

        return image_files, labels_list

    def _image_file_from_image_id(self, image_dir, json_file, image_id):
        images = _images_from_json(json_file)
        file_name = images[images.id == image_id].file_name.tolist()[0]

        return os.path.join(image_dir, file_name)

    def _labels_from_image_id(self, json_file, image_id):
        annotations = _annotations_from_json(json_file)
        category_ids = annotations[annotations.image_id == image_id].category_id.tolist()

        categories = _categories_from_json(json_file)
        category_names = [
            categories[categories.id == category_id].name.iloc[0]
            for category_id in category_ids
        ]

        return [self.classes.index(category) for category in category_names]


class ObjectDetectionBase(DeltaMarkMixin, StoragePathCustomizable, ObjectDetectionBase):
    """Abstract class of dataset for object detection created by Delta-Mark

    When you create the extend class with `extend_path` and validation extend dir as class property,
    the training and validation data consists from following structure.
    When you only set extend_path as class property,
    training and validation dataset are generated from extend_path according to validation_size value.

    structure:
        $DATA_DIR/extend_dir/json/annotation.json
        $DATA_DIR/extend_dir/images/xxxa.jpeg
        $DATA_DIR/extend_dir/images/yyyb.png
        $DATA_DIR/extend_dir/images/123.jpg

        $DATA_DIR/validation_extend_dir/json/annotation.json
        $DATA_DIR/validation_extend_dir/images/xxxa.png
    """

    available_subsets = ["train", "validation"]

    @classmethod
    def count_max_boxes(cls):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, is_shuffle=False)
            gt_boxes_list = obj.annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max
            del obj
        return num_max_boxes

    @property
    @functools.lru_cache(maxsize=None)
    def num_max_boxes(self):
        cls = self.__class__
        return cls.count_max_boxes()

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array.
              shape is [batch_size, height, width] if data_format is NHWC.
              shape is [height, width, batch_size] if data_format is NCHW.
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """
        images, gt_boxes_list = zip(*[self._element() for _ in range(self.batch_size)])

        images = np.array(images)

        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, gt_boxes_list

    def _element(self):
        """Return an image, gt_boxes."""

        index = self._get_index(self.element_counter)
        self.element_counter += 1
        if self.element_counter == self.num_per_epoch:
            self.element_counter = 0
            self._shuffle()

        files = self.files
        annotations = self.annotations
        target_file = files[index]
        gt_boxes = annotations[index]
        gt_boxes = np.array(gt_boxes)

        image = PIL.Image.open(target_file)
        image = image.convert("RGB")
        image = np.array(image)

        samples = {'image': image, 'gt_boxes': gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        gt_boxes = samples['gt_boxes']

        return image, gt_boxes

    def _files_and_annotations(self, json_file, image_dir):
        """Return files and gt_boxes list."""
        image_ids = _image_ids(json_file)

        image_files = [self._image_file_from_image_id(image_dir, json_file, image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(json_file, image_id) for image_id in image_ids]

        return image_files, gt_boxes_list

    def _image_file_from_image_id(self, image_dir, json_file, image_id):
        images = _images_from_json(json_file)
        file_name = images[images.id == image_id].file_name.tolist()[0]

        return os.path.join(image_dir, file_name)

    def _gt_boxes_from_image_id(self, json_file, image_id):
        annotations = _annotations_from_json(json_file)
        category_ids = annotations[annotations.image_id == image_id].category_id.tolist()

        categories = _categories_from_json(json_file)
        category_names = [
            categories[categories.id == category_id].name.iloc[0]
            for category_id in category_ids
        ]

        labels = [self.classes.index(category) for category in category_names]

        bboxes = annotations[annotations.image_id == image_id].bbox.tolist()

        gt_boxes = []

        for class_id, bbox in zip(labels, bboxes):
            # ignore width 0 or height 0 box.
            if bbox[2] == 0 or bbox[3] == 0:
                continue

            gt_boxes.append(bbox + [class_id])

        return gt_boxes
