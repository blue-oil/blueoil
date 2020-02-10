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
import csv
import functools
import json
import os
import os.path
import re
from collections import OrderedDict

import numpy as np

from lmnet import data_processor
from blueoil.nn.datasets.base import Base, ObjectDetectionBase, StoragePathCustomizable
from lmnet.utils.image import load_image
from lmnet.utils.random import train_test_split


class OpenImagesV4(Base):
    extend_dir = "open_images_v4"
    available_subsets = ["train", "validation", "test"]

    task_extend = ""

    def __init__(
            self,
            is_shuffle=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.is_shuffle = is_shuffle

    @property
    def class_descriptions_csv(self):
        return os.path.join(self.data_dir, self.task_extend, 'class-descriptions.csv')

    @property
    @functools.lru_cache(maxsize=None)
    def _classes_meta(self):
        classes_meta = OrderedDict()
        with open(self.class_descriptions_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                class_name = re.sub('\W', '', row[1])
                classes_meta[row[0]] = class_name
        return classes_meta

    @property
    def classes(self):
        return list(self._classes_meta.values())

    @property
    def images_dir(self):
        dirs = {
            "train": os.path.join(self.data_dir, "images/train"),
            "validation": os.path.join(self.data_dir, "images/val"),
            "test": os.path.join(self.data_dir, "images/test"),
        }
        return dirs[self.subset]

    @property
    def num_per_epoch(self):
        files, _ = self.files_and_annotations
        return len(files)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def files_and_annotations(self):
        raise NotImplementedError


class OpenImagesV4BoundingBox(OpenImagesV4, ObjectDetectionBase):
    task_extend = "bounding_boxes"

    available_subsets = ["train", "validation", "test"]

    def __init__(self,
                 class_level=0,
                 *args,
                 **kwargs):
        super(OpenImagesV4BoundingBox, self).__init__(*args, **kwargs)

        self._class_level = class_level

    @property
    def annotations_csv(self):
        return os.path.join(
            self.data_dir, self.task_extend, '{}-annotations-bbox.csv'.format(self.subset))

    @property
    @functools.lru_cache(maxsize=None)
    def classes(self):
        classes = [self._classes_meta[label_name] for label_name in
                   set([v for k, v in self._target_labels.items()])]

        return classes

    @property
    @functools.lru_cache(maxsize=None)
    def _target_labels(self):
        """Map of {csv raw label name: Be mapped target label name}."""

        target_labels = dict(self._make_target_labels())
        return target_labels

    def _make_target_labels(self):
        f = open(os.path.join(self.data_dir, 'bbox_labels_600_hierarchy.json'), 'r')
        json_dict = json.load(f)
        for sub in json_dict["Subcategory"]:
            yield from self._search_subcategory(json_dict["LabelName"], sub, 0)

    def _search_subcategory(self, parent_label_name, d, level):
        current_label_name = d["LabelName"]
        target_label_name = parent_label_name if level > self._class_level else current_label_name
        next_level = level + 1
        for sub in d.get("Subcategory", []):
            yield from self._search_subcategory(target_label_name, sub, next_level)

        yield current_label_name, target_label_name

    def _bboxes(self):
        bboxes = OrderedDict()
        with open(self.annotations_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            indexing = dict(
                [(c, header.index(c)) for c in ["ImageID", "LabelName", "XMin", "YMin", "XMax", "YMax"]])
            for row in reader:
                bboxes.setdefault(row[indexing["ImageID"]], []).append(
                    [float(row[indexing["XMin"]]),
                     float(row[indexing["YMin"]]),
                     float(row[indexing["XMax"]]) - float(row[indexing["XMin"]]),
                     float(row[indexing["YMax"]]) - float(row[indexing["YMin"]]),
                     row[indexing["LabelName"]]]
                )

        return bboxes

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        return self._files_and_annotations()

    def _files_and_annotations(self):
        files = []
        annotations = []
        bboxes = self._bboxes()
        for k, v in bboxes.items():
            files.append(os.path.join(self.images_dir, "{}.jpg".format(k)))
            for b in v:
                class_name = self._classes_meta[self._target_labels[b[4]]]
                b[4] = self.classes.index(class_name)
            annotations.append(v)

        return files, annotations

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, is_shuffle=False)
            _, gt_boxes_list = obj.files_and_annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    @property
    def num_max_boxes(self):
        return self.count_max_boxes()

    def __getitem__(self, i, type=None):
        files, gt_boxes_list = self.files_and_annotations
        target_file = files[i]
        gt_boxes = gt_boxes_list[i]

        image = load_image(target_file)
        height = image.shape[0]
        width = image.shape[1]

        gt_boxes = np.array(gt_boxes)

        # Change box coordinate from [0, 1] to [0, image size].
        gt_boxes = np.stack([
            gt_boxes[:, 0] * width,
            gt_boxes[:, 1] * height,
            gt_boxes[:, 2] * width,
            gt_boxes[:, 3] * height,
            gt_boxes[:, 4],
        ], axis=1)

        gt_boxes = self._fill_dummy_boxes(gt_boxes)

        return (image, gt_boxes)

    def __len__(self):
        return self.num_per_epoch


class OpenImagesV4Classification(OpenImagesV4):
    task_extend = "classification"

    available_subsets = ["train", "validation", "test"]

    def __init__(
            self,
            batch_size=100,
            *args,
            **kwargs
    ):
        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @property
    def annotations_csv(self):
        return os.path.join(
            self.data_dir, self.task_extend, '{}-annotations-human-imagelabels.csv'.format(self.subset))

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        files = []
        annotations = []
        with open(self.annotations_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            indexing = dict(
                [(c, header.index(c)) for c in ["ImageID", "Source", "LabelName", "Confidence"]])
            for row in reader:
                label_id = self._classes_meta[row[indexing["LabelName"]]]
                if int(row[indexing["Confidence"]]) == 1 and label_id in self.classes:
                    files.append(os.path.join(self.images_dir, "{}.jpg".format(row[indexing["ImageID"]])))
                    annotations.append(self.classes.index(label_id))

        return files, annotations

    def __getitem__(self, i, type=None):
        files, labels = self.files_and_annotations

        filename = files[i]

        image = load_image(filename)

        label = data_processor.binarize(labels[i], self.num_classes)
        label = np.reshape(label, (self.num_classes))
        return (image, label)

    def __len__(self):
        return self.num_per_epoch


class OpenImagesV4BoundingBoxBase(StoragePathCustomizable, OpenImagesV4BoundingBox):
    """Abstract class of dataset Open Images v4 format dataset.

    structure like
        $DATA_DIR/extend_dir/class-descriptions.csv
        $DATA_DIR/extend_dir/annotations-bbox.csv
        $DATA_DIR/extend_dir/images/xxxa.jpeg
        $DATA_DIR/extend_dir/images/yyyb.png
        $DATA_DIR/extend_dir/images/123.jpg
        $DATA_DIR/extend_dir/images/023.jpg
        $DATA_DIR/extend_dir/images/wwww.jpg

    When child class has `validation_extend_dir`, the `validation` subset consists from the folders.
        $DATA_DIR/validation_extend_dir/annotations-bbox.csv
        $DATA_DIR/validation_extend_dir/images/xxxa.jpeg
        $DATA_DIR/validation_extend_dir/images/yyyb.png
        $DATA_DIR/validation_extend_dir/images/123.jpg
        $DATA_DIR/validation_extend_dir/images/023.jpg
        $DATA_DIR/validation_extend_dir/images/wwww.jpg

    """

    task_extend = ""

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
    def class_descriptions_csv(self):
        # if train_path has 'class-descriptions.csv', use it.
        csv = os.path.join(self._train_data_dir, 'class-descriptions.csv')
        if os.path.exists(csv):
            return csv

        return os.path.join(self.data_dir, 'annotations-bbox.csv')

    @property
    def annotations_csv(self):
        return os.path.join(self.data_dir, 'annotations-bbox.csv')

    @property
    def images_dir(self):
        return os.path.join(self.data_dir, "images")

    @property
    def classes(self):
        return list(self._classes_meta.values())

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        files, annotations = self._files_and_annotations()
        if self.validation_size > 0:
            train_files, test_files, train_annotations, test_annotations = train_test_split(
                files, annotations, test_size=self.validation_size, seed=1)

            if self.subset == "train":
                files = train_files
                annotations = train_annotations
            if self.subset == "validation":
                files = test_files
                annotations = test_annotations

        files = files
        annotations = annotations

        return files, annotations

    def _files_and_annotations(self):
        files = []
        annotations = []
        bboxes = self._bboxes()
        for k, v in bboxes.items():
            files.append(os.path.join(self.images_dir, "{}.jpg".format(k)))
            for b in v:
                b[4] = self.classes.index(self._classes_meta[b[4]])
            annotations.append(v)

        return files, annotations
