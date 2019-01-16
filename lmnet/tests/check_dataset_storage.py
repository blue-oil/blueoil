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
"""
Test dataset storage.

This is the checker of dataset storage,
check the each datasets is stored corect on storage, and the number of images in each datasets.
And also test that each dataset class don't get error.
Normally developer don't needs to path all tests
because they only forcus some of dataset, not needs the all of datasets.
"""
# TODO(wakisaka): move this script to somewhere.

import time

import numpy as np
import PIL.Image
import PIL.ImageDraw

from lmnet.pre_processor import Resize, ResizeWithMask, ResizeWithGtBoxes
from lmnet.datasets.pascalvoc_2007 import Pascalvoc2007
from lmnet.datasets.pascalvoc_2012 import Pascalvoc2012
from lmnet.datasets.pascalvoc_2007_2012 import Pascalvoc20072012
from lmnet.datasets.caltech101 import Caltech101
from lmnet.datasets.cifar10 import Cifar10
from lmnet.datasets.camvid import Camvid
from lmnet.datasets.ilsvrc_2012 import Ilsvrc2012
from lmnet.datasets.lm_things_on_a_table import LmThingsOnATable
from lmnet.datasets.mscoco import (
    Mscoco,
    ObjectDetection as MscocoObjectDetection,
    ObjectDetectionPerson as MscocoObjectDetectionPerson,
)
from lmnet.datasets.widerface import WiderFace
from lmnet.datasets.bdd100k import BDD100K


IS_DEBUG = False


def _show_images(images):
    """show image for debug."""
    if not IS_DEBUG:
        return

    images_min = abs(images.min())
    images_max = (images + images_min).max()

    images = (images + images_min) * (255 / images_max)
    images = (images).astype(np.uint8)

    for image in images:
        image = PIL.Image.fromarray(image)
        image.show()
        time.sleep(0.5)


def _show_images_with_boxes(images, labels):
    """show image for debug"""
    if not IS_DEBUG:
        return

    images_min = abs(images.min())
    images_max = (images + images_min).max()

    images = (images + images_min) * (255 / images_max)
    images = (images).astype(np.uint8)

    for image, label in zip(images, labels):
        image = PIL.Image.fromarray(image)
        draw = PIL.ImageDraw.Draw(image)

        for box in label:
            xy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            draw.rectangle(xy)

        image.show()
        time.sleep(0.5)


STEP_SIZE = 5


def test_caltech101():
    batch_size = 3
    image_size = [256, 512]
    dataset = Caltech101(
        batch_size=batch_size,
        pre_processor=Resize(image_size))

    assert dataset.num_classes == 101

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == dataset.num_classes


def test_cifar10():
    batch_size = 3
    image_size = [256, 512]
    dataset = Cifar10(batch_size=batch_size, pre_processor=Resize(image_size))
    val_dataset = Cifar10(
        subset="validation",
        batch_size=batch_size,
        pre_processor=Resize(image_size))

    assert dataset.num_classes == 10
    assert dataset.num_per_epoch == 50000
    assert val_dataset.num_per_epoch == 10000

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == dataset.num_classes


def test_pascalvoc_2007():
    batch_size = 3
    image_size = [256, 512]

    num_max_boxes = 37
    num_train = 2501
    num_validation = 2510
    num_test = 4952

    assert Pascalvoc2007.count_max_boxes() == num_max_boxes

    dataset = Pascalvoc2007(batch_size=batch_size,
                            pre_processor=ResizeWithGtBoxes(image_size))
    assert dataset.num_per_epoch == num_train

    val_dataset = Pascalvoc2007(
        subset="validation",
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == num_validation

    test_dataset = Pascalvoc2007(
        subset="test",
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size))
    assert test_dataset.num_per_epoch == num_test

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = test_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_pascalvoc_2007_not_skip_difficult():
    batch_size = 3
    image_size = [256, 512]

    num_max_boxes = 42
    num_train = 2501
    num_validation = 2510
    num_test = 4952

    assert Pascalvoc2007.count_max_boxes(skip_difficult=False) == num_max_boxes

    dataset = Pascalvoc2007(batch_size=batch_size,
                            pre_processor=ResizeWithGtBoxes(image_size),
                            skip_difficult=False)
    assert dataset.num_per_epoch == num_train

    val_dataset = Pascalvoc2007(
        subset="validation",
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size),
        skip_difficult=False)
    assert val_dataset.num_per_epoch == num_validation

    test_dataset = Pascalvoc2007(
        subset="test",
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size),
        skip_difficult=False)
    assert test_dataset.num_per_epoch == num_test

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = test_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


class TargetClassesPascalvoc2007(Pascalvoc2007):
    """Target classes are 'aeroplane', 'tvmonitor'"""
    classes = ['aeroplane', 'tvmonitor']

    @property
    def num_max_boxes(self):
        cls = self.__class__
        return cls.count_max_boxes(self.skip_difficult)


def test_pascalvoc_2007_with_target_classes():
    batch_size = 3
    image_size = [256, 512]

    num_max_boxes = 12
    num_train = 240
    num_validation = 254
    num_test = 433

    assert TargetClassesPascalvoc2007.count_max_boxes() == num_max_boxes

    dataset = TargetClassesPascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size), )
    assert dataset.num_per_epoch == num_train

    val_dataset = TargetClassesPascalvoc2007(subset="validation",
                                             batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == num_validation

    test_dataset = TargetClassesPascalvoc2007(subset="test",
                                              batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size))
    assert test_dataset.num_per_epoch == num_test

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = test_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_pascalvoc_2012():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2012(batch_size=batch_size,
                            pre_processor=ResizeWithGtBoxes(image_size))
    num_max_boxes = 39

    assert dataset.num_max_boxes == num_max_boxes
    assert Pascalvoc2012.count_max_boxes() == num_max_boxes
    assert dataset.num_per_epoch == 5717

    val_dataset = Pascalvoc2012(
        subset="validation",
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == 5823

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_pascalvoc_2007_2012():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc20072012(
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size))
    num_max_boxes = 39

    num_train_val_2007 = 2501 + 2510
    num_train_val_2012 = 5717 + 5823
    num_test_2007 = 4952

    assert dataset.num_max_boxes == num_max_boxes
    assert Pascalvoc20072012.count_max_boxes() == num_max_boxes
    assert dataset.num_per_epoch == num_train_val_2007 + num_train_val_2012

    val_dataset = Pascalvoc20072012(subset="validation", batch_size=batch_size,
                                    pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == num_test_2007

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_pascalvoc_2007_2012_no_skip_difficult():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc20072012(
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size),
        skip_difficult=False,
    )
    num_max_boxes = 56

    num_train_val_2007 = 2501 + 2510
    num_train_val_2012 = 5717 + 5823
    num_test_2007 = 4952

    assert dataset.num_max_boxes == num_max_boxes
    assert Pascalvoc20072012.count_max_boxes(skip_difficult=False) == num_max_boxes
    assert dataset.num_per_epoch == num_train_val_2007 + num_train_val_2012

    val_dataset = Pascalvoc20072012(subset="validation", batch_size=batch_size,
                                    pre_processor=ResizeWithGtBoxes(image_size),
                                    skip_difficult=False)
    assert val_dataset.num_per_epoch == num_test_2007

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_camvid():
    batch_size = 3
    image_size = [256, 512]
    dataset = Camvid(
        batch_size=batch_size,
        pre_processor=ResizeWithMask(image_size))
    val_dataset = Camvid(
        subset="validation",
        batch_size=batch_size,
        pre_processor=ResizeWithMask(image_size))

    assert dataset.num_classes == 11
    assert dataset.num_per_epoch == 367
    assert val_dataset.num_per_epoch == 101

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == image_size[0]
        assert labels.shape[2] == image_size[1]


def test_lm_things_of_a_table():
    batch_size = 3
    image_size = [256, 512]
    dataset = LmThingsOnATable(batch_size=batch_size,
                               pre_processor=ResizeWithGtBoxes(image_size))

    num_max_boxes = LmThingsOnATable.count_max_boxes()

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_mscoco():
    batch_size = 3
    image_size = [256, 512]
    dataset = Mscoco(
        batch_size=batch_size,
        pre_processor=ResizeWithMask(image_size))

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == image_size[0]
        assert labels.shape[2] == image_size[1]


def test_mscoco_object_detection():
    batch_size = 3
    image_size = [256, 512]

    num_max_boxes = 93

    num_train = 82081
    num_val = 40137

    dataset = MscocoObjectDetection(
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size))

    assert MscocoObjectDetection.count_max_boxes() == num_max_boxes
    assert dataset.num_max_boxes == num_max_boxes
    assert dataset.num_per_epoch == num_train

    val_dataset = MscocoObjectDetection(subset="validation", batch_size=batch_size,
                                        pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == num_val

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_mscoco_object_detection_person():
    batch_size = 3
    image_size = [256, 512]

    num_max_boxes = 14

    num_train = 38699
    num_val = 18513

    dataset = MscocoObjectDetectionPerson(
        batch_size=batch_size,
        pre_processor=ResizeWithGtBoxes(image_size))

    assert MscocoObjectDetectionPerson.count_max_boxes() == num_max_boxes
    assert dataset.num_max_boxes == num_max_boxes
    assert dataset.num_per_epoch == num_train
    val_dataset = MscocoObjectDetectionPerson(subset="validation", batch_size=batch_size,
                                              pre_processor=ResizeWithGtBoxes(image_size))

    num_max_boxes = MscocoObjectDetectionPerson.count_max_boxes()
    assert val_dataset.num_per_epoch == num_val

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_ilsvrc_2012():
    batch_size = 3
    image_size = [256, 512]
    dataset = Ilsvrc2012(
        batch_size=batch_size,
        pre_processor=Resize(image_size))
    val_dataset = Ilsvrc2012(
        subset="validation",
        batch_size=batch_size,
        pre_processor=Resize(image_size))

    num_train = 1281167
    num_validation = 50000

    assert dataset.num_per_epoch == num_train
    assert val_dataset.num_per_epoch == num_validation

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == dataset.num_classes

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == dataset.num_classes


def test_widerface():
    batch_size = 3
    image_size = [160, 160]

    num_max_boxes = 3

    num_train = 7255
    num_val = 1758

    dataset = WiderFace(batch_size=batch_size,
                        max_boxes=num_max_boxes,
                        pre_processor=ResizeWithGtBoxes(image_size))

    assert dataset.num_max_boxes == num_max_boxes
    assert dataset.num_per_epoch == num_train

    val_dataset = WiderFace(subset="validation",
                            batch_size=batch_size,
                            max_boxes=num_max_boxes,
                            pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == num_val

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


def test_bdd100k():
    batch_size = 3
    image_size = [320, 320]

    num_max_boxes = 100

    num_train = 70000
    num_val = 10000

    dataset = BDD100K(batch_size=batch_size,
                      max_boxes=num_max_boxes,
                      pre_processor=ResizeWithGtBoxes(image_size))

    assert dataset.num_max_boxes == num_max_boxes
    assert dataset.num_per_epoch == num_train

    val_dataset = BDD100K(subset="validation",
                          batch_size=batch_size,
                          max_boxes=num_max_boxes,
                          pre_processor=ResizeWithGtBoxes(image_size))
    assert val_dataset.num_per_epoch == num_val

    for _ in range(STEP_SIZE):
        images, labels = dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(STEP_SIZE):
        images, labels = val_dataset.feed()
        # _show_images_with_boxes(images, labels)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


if __name__ == '__main__':
    test_caltech101()
    test_cifar10()
    test_camvid()
    test_pascalvoc_2007()
    test_pascalvoc_2007_not_skip_difficult()
    test_pascalvoc_2007_with_target_classes()
    test_pascalvoc_2012()
    test_pascalvoc_2007_2012()
    test_pascalvoc_2007_2012_no_skip_difficult()
    test_lm_things_of_a_table()
    test_mscoco()
    test_mscoco_object_detection()
    test_mscoco_object_detection_person()
    test_ilsvrc_2012()
    test_widerface()
    test_bdd100k()
