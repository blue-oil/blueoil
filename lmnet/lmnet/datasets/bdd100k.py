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
import os
import functools
from multiprocessing import Pool

from os.path import join, splitext, basename, isfile
import glob
import pickle

from PIL import Image
import numpy as np
import json

from lmnet.datasets.base import ObjectDetectionBase
from lmnet.utils.random import shuffle
import lmnet.environment as lmenv


def fetch_one_data(args):
    path, gt_boxes, augmentor, pre_processor, is_train = args
    image = Image.open(path)
    image = np.array(image)
    gt_boxes = np.array(gt_boxes)
    samples = {'image': image, 'gt_boxes': gt_boxes}

    if callable(augmentor) and is_train:
        samples = augmentor(**samples)

    if callable(pre_processor):
        samples = pre_processor(**samples)

    image = samples['image']
    gt_boxes = samples['gt_boxes']

    return image, gt_boxes


class BDD100K(ObjectDetectionBase):
    """BDD100K Dataset for Object Detection (Car Camera)
    https://github.com/ucbdrive/bdd-data
    """
    classes = ["bike",
               "bus",
               "car",
               "motor",
               "person",
               "rider",
               "traffic_light",
               "traffic_sign",
               "train",
               "truck"]
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = "bdd100k"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def count_max_boxes(cls, base_path=None):
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, base_path=base_path, is_shuffle=False)
            gt_boxes_list = obj.bboxs

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    @property
    def num_max_boxes(self):
        # from dataset:
        # train - max 91 boxes, val - max 66 boxes
        return self.max_boxes

    @property
    def num_per_epoch(self):
        return len(self.paths)

    def __init__(self,
                 subset="train",
                 is_shuffle=True,
                 enable_prefetch=False,
                 max_boxes=100,
                 num_workers=None,
                 *args,
                 **kwargs):

        super().__init__(subset=subset, *args, **kwargs)

        if enable_prefetch:
            self.use_prefetch = True
        else:
            self.use_prefetch = False

        self.is_shuffle = is_shuffle
        self.max_boxes = max_boxes
        self.num_workers = num_workers

        subset_dir = "train" if subset == "train" else "val"
        self.img_dir = join(self.data_dir, "images", "100k", subset_dir)
        self.anno_dir = join(self.data_dir, "labels", "100k", subset_dir)
        self.paths = []
        self.bboxs = []

        self._init_files_and_annotations()
        self._shuffle()

        if self.use_prefetch:
            self.enable_prefetch()
            print("ENABLE prefetch")
        else:
            print("DISABLE prefetch")

    def prefetch_args(self, index):
        path = self.paths[index]
        gt_boxes = self.bboxs[index]
        return path, gt_boxes, self.augmentor, self.pre_processor, self.subset == "train"

    def enable_prefetch(self):
        self.pool = Pool(processes=self.num_workers)
        self.start_prefetch()

    def start_prefetch(self):
        index = self.current_element_index
        batch_size = self.batch_size
        start = index
        end = min(index + batch_size, self.num_per_epoch)
        pool = self.pool

        args = []
        for i in range(start, end):
            args.append(self.prefetch_args(i))

        self.current_element_index += batch_size
        if self.current_element_index >= self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

            rest = batch_size - len(args)
            for i in range(0, rest):
                args.append(self.prefetch_args(i))
            self.current_element_index += rest

        self.prefetch_result = pool.map_async(fetch_one_data, args)

    def _init_files_and_annotations(self):

        os.makedirs(lmenv.TMP_DIR, exist_ok=True)
        files_pickle_path = join(lmenv.TMP_DIR,
                                 self.subset + "_bdd100k_files.pickle")
        annos_pickle_path = join(lmenv.TMP_DIR,
                                 self.subset + "_bdd100k_annos.pickle")

        if(not (isfile(files_pickle_path) and isfile(annos_pickle_path))):
            # read and process all raw files
            img_paths = glob.glob(join(self.img_dir, "*.jpg"))
            anno_paths = [join(self.anno_dir, splitext(basename(f))[0] + ".json")
                          for f in img_paths]

            self.paths = img_paths
            for f in anno_paths:
                with open(f) as fp:
                    raw = json.load(fp)
                objs = raw["frames"][0]["objects"]
                bbox = []
                for obj in objs:
                    cat = obj["category"]
                    cat = cat.replace(" ", "_")
                    if cat in self.classes:
                        cls_idx = self.classes.index(cat)
                        x1 = int(round(obj["box2d"]["x1"]))
                        x2 = int(round(obj["box2d"]["x2"]))
                        y1 = int(round(obj["box2d"]["y1"]))
                        y2 = int(round(obj["box2d"]["y2"]))
                        x = x1
                        y = y1
                        w = x2 - x1
                        h = y2 - y1
                        bbox.append([x, y, w, h, cls_idx])
                bbox = np.array(bbox, dtype=np.int32)
                self.bboxs.append(bbox)

            with open(files_pickle_path, "wb") as fp:
                pickle.dump(self.paths, fp)
            with open(annos_pickle_path, "wb") as fp:
                pickle.dump(self.bboxs, fp)
            print("done saved pickle")
        else:
            # load from pickle
            print("loading from pickle file: {}".format(files_pickle_path))
            with open(files_pickle_path, "rb") as fp:
                self.paths = pickle.load(fp)
            with open(annos_pickle_path, "rb") as fp:
                self.bboxs = pickle.load(fp)

    def _shuffle(self):

        if not self.is_shuffle:
            return

        if self.subset == "train":
            self.paths, self.bboxs = shuffle(
                self.paths, self.bboxs, seed=self.seed)
            print(
                "Shuffle {} train dataset with seed {}.".format(
                    self.__class__.__name__,
                    self.seed))
            self.seed = self.seed + 1

    def _one_data(self):
        """Return an image, gt_boxes."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

        path = self.paths[index]
        gt_boxes = self.bboxs[index].copy()

        image = Image.open(path)
        image = np.array(image)

        samples = {"image": image, "gt_boxes": gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples["image"]
        gt_boxes = samples["gt_boxes"]

        return image, gt_boxes

    def get_data(self):
        if self.use_prefetch:
            data_list = self.prefetch_result.get(None)
            images, gt_boxes_list = zip(*data_list)
            return images, gt_boxes_list
        else:
            images, gt_boxes_list = zip(
                *[self._one_data() for _ in range(self.batch_size)])
            return images, gt_boxes_list

    def feed(self):

        images, gt_boxes_list = self.get_data()

        if self.use_prefetch:
            self.start_prefetch()

        images = np.array(images)
        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, gt_boxes_list


def check_dataset():
    train = BDD100K(subset="train")
    print(len(train.paths))
    print(train.paths[0:5])
    print(train.bboxs[0:5])
    print("seems dataset was loaded correctly")


if __name__ == '__main__':
    check_dataset()
