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
# Support for cityscapes dataset
# https://www.cityscapes-dataset.com/

import functools
import glob
import os.path

import numpy as np
from PIL import Image

from nn.datasets.base import SegmentationBase


class Cityscapes(SegmentationBase):
    available_subsets = ["train", "validation", "test"]
    extend_dir = "cityscapes"
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
        polygons_json = glob.glob(os.path.join(self.data_dir, "gtFine", split, "*", "*_gt*_polygons.json"))
        polygons_json.sort()

        labelIds = [i.replace("_polygons.json", "_labelIds.png") for i in polygons_json]
        leftImg8bit = [i.replace(
            os.path.join(self.data_dir, "gtFine"),
            os.path.join(self.data_dir, "leftImg8bit")
        ).replace("_gtFine_polygons.json", "_leftImg8bit.png") for i in polygons_json]

        return leftImg8bit, labelIds

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
