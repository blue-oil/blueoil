# -*- coding: utf-8 -*-
import os
import math
import pymp
import time
import json
import glob
import pickle
import functools

import numpy as np
from collections import OrderedDict
from multiprocessing import cpu_count

from PIL import Image
from lmnet.utils.image import load_image
from lmnet.datasets.base import ObjectDetectionBase, SegmentationBase


class BDD100KObjDet(ObjectDetectionBase):
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
        self.img_dir = os.path.join(self.data_dir, "images", "100k", subset_dir)
        self.anno_dir = os.path.join(self.data_dir, "labels", "bdd100k_labels_images_" + subset_dir + ".json")
        self.paths = []
        self.bboxs = []

        self._init_files_and_annotations()

    def _init_files_and_annotations(self):
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

    def __getitem__(self, i, type=None):
        image_file_path = self.paths[i]

        image = load_image(image_file_path)

        gt_boxes = self.bboxs[i]
        gt_boxes = np.array(gt_boxes)
        gt_boxes = gt_boxes.copy()  # is it really needed?
        gt_boxes = self._fill_dummy_boxes(gt_boxes)

        return image, gt_boxes

    def __len__(self):
        return self.num_per_epoch


class BDD100KSeg(SegmentationBase):
    """BDD100K Dataset for Segmentation
    https://github.com/ucbdrive/bdd-data
    """
    available_subsets = ["train", "validation", "test"]
    extend_dir = "BDD100K/seg"
    image_dir = 'images'
    label_dir = 'labels'      # labels : gray scale labels
    classes = [
        "unlabeled",
        "ego vehicle",
        "rectification boarder",
        "out of roi",
        "static",
        "dynamic",
        "ground",
        "road",
        "sidewalk",
        "parking",
        "rail track",
        "building",
        "wall",
        "fence",
        "guard rail",
        "bridge",
        "tunnel",
        "pole",
        "polegroup",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "caravan",
        "trailer",
        "train",
        "motorcycle",
        "bicycle",
    ]
    num_classes = len(classes)

    def __init__(self, batch_size=10, *args, **kwargs):
        super().__init__(batch_size=batch_size, *args, **kwargs)

    @property
    def label_colors(self):
        unlabeled = [0, 0, 0]
        ego_vehicle = [0, 0, 0]
        rectification_boarder = [0, 0, 0]
        out_of_roi = [0, 0, 0]
        static = [0, 0, 0]
        dynamic = [111, 74, 0]
        ground = [81, 0, 81]
        road = [128, 64, 128]
        sidewalk = [244, 35, 232]
        parking = [250, 170, 160]
        rail_track = [230, 150, 140]
        building = [70, 70, 70]
        wall = [102, 102, 156]
        fence = [190, 153, 153]
        guard_rail = [180, 165, 180]
        bridge = [150, 100, 100]
        tunnel = [150, 120, 90]
        pole = [153, 153, 153]
        polegroup = [153, 153, 153]
        traffic_light = [250, 170, 30]
        traffic_sign = [220, 220, 0]
        vegetation = [107, 142, 35]
        terrain = [152, 251, 152]
        sky = [70, 130, 180]
        person = [220, 20, 60]
        rider = [255, 0, 0]
        car = [0, 0, 142]
        truck = [0, 0, 70]
        bus = [0, 60, 100]
        caravan = [0, 0, 90]
        trailer = [0, 0, 110]
        train = [0, 80, 100]
        motorcycle = [0, 0, 230]
        bicycle = [119, 11, 32]

        return np.array([
            unlabeled, ego_vehicle, rectification_boarder, out_of_roi, static,
            dynamic, ground, road, sidewalk, parking, rail_track, building,
            wall, fence, guard_rail, bridge, tunnel, pole, polegroup,
            traffic_light, traffic_sign, vegetation, terrain, sky, person,
            rider, car, truck, bus, caravan, trailer, train, motorcycle,
            bicycle])

    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        split = "train"
        if self.subset == "validation":
            split = "val"
        elif self.subset == "test":
            split = "test"
        file_path = os.path.join(self.data_dir, self.image_dir, split, "*.jpg")
        image_paths = glob.glob(file_path)
        image_paths.sort()

        file_path = os.path.join(self.data_dir, self.label_dir, split, "*.png")
        label_paths = glob.glob(file_path)
        label_paths.sort()

        return image_paths, label_paths

    def __getitem__(self, i):
        imgs, labels = self.files_and_annotations()
        img = Image.open(imgs[i])
        label = Image.open(labels[i])

        return np.array(img), np.array(label)

    def __len__(self):
        return len(self.files_and_annotations()[0])

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations()[0])


def check_dataset():
    train = BDD100KObjDet(subset="train")
    print(len(train.paths))
    print(train.paths[0:5])
    print(train.bboxs[0:5])
    print("seems dataset was loaded correctly")


if __name__ == '__main__':
    check_dataset()
