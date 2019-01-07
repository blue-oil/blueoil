# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
import PIL.Image
from pycocotools.coco import COCO

from lmnet.datasets.base import SegmentationBase
from lmnet.datasets.base import ObjectDetectionBase
from lmnet.utils.random import shuffle


DEFAULT_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]


# TODO(wakisaka): shuffle
class Mscoco(SegmentationBase):
    """Mscoco for segmentation."""

    classes = ["__background__"] + DEFAULT_CLASSES
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
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

        if subset == 'train':
            self.json = os.path.join(self.data_dir, "annotations/instances_train2014.json")
            self.image_dir = os.path.join(self.data_dir, "train2014")
        elif subset == 'validation':
            self.json = os.path.join(self.data_dir, "annotations/instances_val2014.json")
            self.image_dir = os.path.join(self.data_dir, "val2014")

    @property
    def num_per_epoch(self):
        return len(self._image_ids)

    @property
    @functools.lru_cache(maxsize=None)
    def coco(self):
        return COCO(self.json)

    @property
    @functools.lru_cache(maxsize=None)
    def _image_ids(self):
        """Return all files and gt_boxes list."""

        classes = [class_name for class_name in self.classes if class_name is not "__background__"]
        target_class_ids = self.coco.getCatIds(catNms=classes)
        image_ids = []
        for target_class_id in target_class_ids:
            target_image_ids = self.coco.getImgIds(catIds=[target_class_id])
            image_ids = image_ids + target_image_ids

        # remove duplicate with order preserving
        seen = set()
        return [x for x in image_ids if x not in seen and not seen.add(x)]

    @functools.lru_cache(maxsize=None)
    def _label_from_image_id(self, image_id):

        coco_image = self.coco.loadImgs(image_id)[0]
        height = coco_image["height"]
        width = coco_image["width"]
        label = np.zeros((height, width), dtype='uint8')

        classes = [class_name for class_name in self.classes if class_name is not "__background__"]
        for target_class in classes:
            target_class_id = self.coco.getCatIds(catNms=[target_class])[0]
            annotation_ids = self.coco.getAnnIds(imgIds=[image_id], catIds=[target_class_id], iscrowd=None)
            annotations = self.coco.loadAnns(annotation_ids)

            class_label = np.zeros((height, width), dtype='uint8')
            for annotation in annotations:
                # annToMask() return same size of image ndarray. value is 0 or 1.
                class_label += self.coco.annToMask(annotation)

            # label += class_label * self.classes.index(target_class)
            label[class_label == 1] = (self.classes.index(target_class))

        return label

    def _image(self, filename, convert_rgb=True):
        """Returns numpy array of an image"""
        image = PIL.Image.open(filename)

        #  sometime image data is gray.
        if convert_rgb:
            image = image.convert("RGB")
        else:
            image = image.convert("L")

        image = np.array(image)

        return image

    def _image_file_from_image_id(self, image_id):
        image = self.coco.loadImgs(image_id)
        return os.path.join(self.image_dir, image[0]["file_name"])

    def _element(self):
        """Return an image, mask image."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0

        image_ids = self._image_ids
        target_image_id = image_ids[index]

        image_file = self._image_file_from_image_id(target_image_id)

        image = self._image(image_file)
        label = self._label_from_image_id(target_image_id)

        samples = {'image': image, 'mask': label}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        label = samples['mask']

        return image, label

    def feed(self):
        """Returns batch size numpy array of images and label images"""
        images, labels = zip(*[self._element() for _ in range(self.batch_size)])

        images, labels = np.array(images), np.array(labels)

        return images, labels


class ObjectDetection(Mscoco, ObjectDetectionBase):
    """MSCOCO for object detection.

    feed() returns images and ground truth boxes.
    images: images numpy array. shape is [batch_size, height, width]
    labels: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
    """

    _cache = dict()
    classes = DEFAULT_CLASSES
    num_classes = len(classes)

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self._init_files_and_annotations()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset)
            gt_boxes_list = obj.annotations

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    @property
    @functools.lru_cache(maxsize=None)
    def num_max_boxes(self):
        cls = type(self)
        return cls.count_max_boxes()

    @property
    def num_per_epoch(self):
        return len(self.files)

    @functools.lru_cache(maxsize=None)
    def _gt_boxes_from_image_id(self, image_id):
        """Return gt boxes list ([[x, y, w, h, class_id]]) of a image."""
        boxes = []

        classes = [class_name for class_name in self.classes if class_name is not "__background__"]

        for target_class in classes:
            target_class_id = self.coco.getCatIds(catNms=[target_class])[0]
            annotation_ids = self.coco.getAnnIds(imgIds=[image_id], catIds=[target_class_id], iscrowd=None)
            annotations = self.coco.loadAnns(annotation_ids)

            for annotation in annotations:
                class_id = self.classes.index(target_class)
                box = annotation["bbox"] + [class_id]
                boxes.append(box)

        return boxes

    def _element(self):
        """Return an image, gt_boxes."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

        files, gt_boxes_list = self.files, self.annotations
        target_file = files[index]
        gt_boxes = gt_boxes_list[index]
        gt_boxes = np.array(gt_boxes)

        image = self._image(target_file)

        samples = {'image': image, 'gt_boxes': gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        gt_boxes = samples['gt_boxes']

        return image, gt_boxes

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """
        images, gt_boxes_list = zip(*[self._element() for _ in range(self.batch_size)])

        images = np.array(images)

        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        return images, gt_boxes_list

    def _files_and_annotations(self):
        """Create files and gt_boxes list."""
        image_ids = self._image_ids

        files = [self._image_file_from_image_id(image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(image_id) for image_id in image_ids]

        return files, gt_boxes_list

    def _init_files_and_annotations(self):

        cache_key = self.subset + self.data_dir + str(self.classes)

        cls = type(self)

        if cache_key in cls._cache:
            cached_obj = cls._cache[cache_key]
            self.files, self.annotations = cached_obj.files, cached_obj.annotations
        else:
            self.files, self.annotations = self._files_and_annotations()
            self._shuffle()
            cls._cache[cache_key] = self

    def _shuffle(self):
        """Shuffle data if train."""

        if self.subset == "train":
            self.files, self.annotations = shuffle(
                self.files, self.annotations, seed=self.seed)
            print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
            self.seed += 1


class ObjectDetectionPerson(ObjectDetection):
    """"MSCOCO only person class for object detection.

    feed() returns images and ground truth boxes.
    images: images numpy array. shape is [batch_size, height, width]
    labels: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
    """
    classes = ["person"]
    num_classes = len(classes)

    def __init__(
            self,
            threshold_size=64*64,
            *args,
            **kwargs
    ):
        # use only box size > threshold_size (px * px)
        self.threshold_size = threshold_size

        super().__init__(
            *args,
            **kwargs,
        )

    @functools.lru_cache(maxsize=None)
    def _gt_boxes_from_image_id(self, image_id):
        """Return gt boxes list ([[x, y, w, h, class_id]]) of a image."""
        gt_boxes = super()._gt_boxes_from_image_id(image_id)

        boxes = []

        for box in gt_boxes:
            w = box[2]
            h = box[3]
            size = w * h
            # remove small box.
            if size < self.threshold_size:
                continue
            boxes.append(box)

        return boxes

    @property
    @functools.lru_cache(maxsize=None)
    def _image_ids(self):
        """Return all files and gt_boxes list."""
        image_ids = super()._image_ids

        new_image_ids = []

        for image_id in image_ids:
            gt_boxes = self._gt_boxes_from_image_id(image_id)

            if len(gt_boxes) > 0:
                new_image_ids.append(image_id)

        return new_image_ids
