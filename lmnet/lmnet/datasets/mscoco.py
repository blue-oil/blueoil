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
from pycocotools.coco import COCO

from lmnet.utils.image import load_image
from lmnet.datasets.base import ObjectDetectionBase, SegmentationBase

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
class MscocoSegmentation(SegmentationBase):
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

    def _image_file_from_image_id(self, image_id):
        image = self.coco.loadImgs(image_id)
        return os.path.join(self.image_dir, image[0]["file_name"])

    def __getitem__(self, i, type=None):
        image_id = self._image_ids[i]
        image_file = self._image_file_from_image_id(image_id)
        image = load_image(image_file)

        label = self._label_from_image_id(image_id)

        return (image, label)

    def __len__(self):
        return self.num_per_epoch


class MscocoObjectDetection(ObjectDetectionBase):
    """MSCOCO for object detection.

    images: images numpy array. shape is [batch_size, height, width]
    labels: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
    """

    _cache = dict()
    classes = DEFAULT_CLASSES
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "MSCOCO"

    def __init__(
            self,
            subset="train",
            *args,
            **kwargs
    ):
        super().__init__(
            subset=subset,
            *args,
            **kwargs
        )

        if subset == 'train':
            self.json = os.path.join(self.data_dir, "annotations/instances_train2014.json")
            self.image_dir = os.path.join(self.data_dir, "train2014")
        elif subset == 'validation':
            self.json = os.path.join(self.data_dir, "annotations/instances_val2014.json")
            self.image_dir = os.path.join(self.data_dir, "val2014")

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
        r = [x for x in image_ids if x not in seen and not seen.add(x)]

        r = sorted(r)
        return r

    @functools.lru_cache(maxsize=None)
    def _image_file_from_image_id(self, image_id):
        image = self.coco.loadImgs(image_id)
        return os.path.join(self.image_dir, image[0]["file_name"])

    @functools.lru_cache(maxsize=None)
    def coco_category_id_to_lmnet_class_id(self, cat_id):
        target_class = self.coco.loadCats(cat_id)[0]['name']
        class_id = self.classes.index(target_class)
        return class_id

    @functools.lru_cache(maxsize=None)
    def _gt_boxes_from_image_id(self, image_id):
        """Return gt boxes list ([[x, y, w, h, class_id]]) of a image."""
        boxes = []
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)
        for annotation in annotations:
            class_id = self.coco_category_id_to_lmnet_class_id(annotation['category_id'])
            box = annotation["bbox"] + [class_id]
            boxes.append(box)

        return boxes

    def _files_and_annotations(self):
        """Create files and gt_boxes list."""
        image_ids = self._image_ids

        files = [self._image_file_from_image_id(image_id) for image_id in image_ids]
        gt_boxes_list = [self._gt_boxes_from_image_id(image_id) for image_id in image_ids]

        return files, gt_boxes_list

    def _init_files_and_annotations(self):
        self.files, self.annotations = self._files_and_annotations()

    def __getitem__(self, i, type=None):
        target_file = self.files[i]
        image = load_image(target_file)

        gt_boxes = self.annotations[i]
        gt_boxes = np.array(gt_boxes)
        gt_boxes = self._fill_dummy_boxes(gt_boxes)

        return (image, gt_boxes)

    def __len__(self):
        return self.num_per_epoch


class MscocoObjectDetectionPerson(MscocoObjectDetection):
    """"MSCOCO only person class for object detection.

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

        person_class_id = self.coco.getCatIds(catNms=['person'])[0]

        boxes = []
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)
        for annotation in annotations:
            category_id = annotation['category_id']
            if category_id != person_class_id:
                continue

            class_id = self.coco_category_id_to_lmnet_class_id(category_id)
            box = annotation["bbox"] + [class_id]

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
        """Return all files which contains person bounding boxes."""
        image_ids = []
        target_class_ids = self.coco.getCatIds(catNms=['person'])
        for image_id in self.coco.getImgIds(catIds=[target_class_ids[0]]):
            gt_boxes = self._gt_boxes_from_image_id(image_id)
            if len(gt_boxes) > 0:
                image_ids.append(image_id)

        return image_ids


def main():
    import time

    s = time.time()
    MscocoObjectDetection(subset="train", enable_prefetch=False)
    e = time.time()
    print("elapsed:", e-s)


if __name__ == '__main__':
    main()
