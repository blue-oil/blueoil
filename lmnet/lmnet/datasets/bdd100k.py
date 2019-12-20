# -*- coding: utf-8 -*-
import functools
import glob
import json
import os

import numpy as np

from lmnet.datasets.base import ObjectDetectionBase, SegmentationBase
from lmnet.utils.image import load_image


class BDD100KObjectDetection(ObjectDetectionBase):
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

    @property
    def label_colors(self):
        bike = [119, 11, 32]
        bus = [0, 60, 100]
        car = [0, 0, 142]
        motor = [0, 0, 230]
        person = [220, 20, 60]
        rider = [255, 0, 0]
        traffic_light = [250, 170, 30]
        traffic_sign = [220, 220, 0]
        train = [0, 80, 100]
        truck = [0, 0, 70]

        return np.array([
            bike, bus, car, motor,
            person, rider, traffic_light,
            traffic_sign, train, truck
        ])

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
        return 91

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
        img_paths = dict([(os.path.basename(path), path)
                          for path in glob.glob(os.path.join(self.img_dir, "*.jpg"))])
        img_names = set(img_paths.keys())

        anno_data = json.load(open(self.anno_dir))

        self.paths = []
        self.bboxs = []
        for item in anno_data:
            # Skip if Label not in images
            img_name = item['name']
            if img_name not in img_names:
                continue
            bbox = []
            for label in item['labels']:
                class_name = label['category'].replace(' ', '_')
                # Skip if Classname/Category not in Selected classes
                if class_name not in self.classes:
                    continue

                cls_idx = self.classes.index(class_name)
                x1 = int(round(label["box2d"]["x1"]))
                x2 = int(round(label["box2d"]["x2"]))
                y1 = int(round(label["box2d"]["y1"]))
                y2 = int(round(label["box2d"]["y2"]))

                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                bbox += [[x, y, w, h, cls_idx]]

            num_boxes = len(bbox)
            if num_boxes > 0:
                self.paths.append(img_paths[img_name])
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


class BDD100KSegmentation(SegmentationBase):
    """BDD100K Dataset for Segmentation
    https://github.com/ucbdrive/bdd-data
    """
    available_subsets = ["train", "validation", "test"]
    extend_dir = "seg"
    image_dir = 'images'
    label_dir = 'labels'  # labels : gray scale labels
    classes = [
        "unlabeled",
        "dynamic",
        "ego_vehicle",
        "ground",
        "static",
        "parking",
        "rail track",
        "road",
        "sidewalk",
        "bridge",
        "building",
        "fence",
        "garage",
        "guard rail",
        "tunnel",
        "wall",
        "banner",
        "billboard",
        "lane divider",
        "parking_sign",
        "pole",
        "polegroup",
        "street_light",
        "traffic_cone",
        "traffic_device",
        "traffic_light",
        "traffic_sign",
        "traffic_sign_frame",
        "terrain",
        "vegetation",
        "sky",
        "person",
        "rider",
        "bicycle",
        "bus",
        "car",
        "caravan",
        "motorcycle",
        "trailer",
        "train",
        "truck"
    ]
    num_classes = len(classes)

    def __init__(self, batch_size=10, *args, **kwargs):
        super().__init__(batch_size=batch_size, *args, **kwargs)

    @property
    def label_colors(self):
        unlabeled = [0, 0, 0]
        dynamic = [111, 74, 0]
        ego_vehicle = [0, 0, 0]
        ground = [81, 0, 81]
        static = [0, 0, 0]
        parking = [250, 170, 160]
        rail_track = [230, 150, 140]
        road = [128, 64, 128]
        sidewalk = [244, 35, 232]
        bridge = [150, 100, 100]
        building = [70, 70, 70]
        fence = [190, 153, 153]
        garage = [180, 100, 180]
        guard_rail = [180, 165, 180]
        tunnel = [150, 120, 90]
        wall = [102, 102, 156]
        banner = [250, 170, 100]
        billboard = [220, 220, 250]
        lane_divider = [255, 165, 0]
        parking_sign = [220, 20, 60]
        pole = [153, 153, 153]
        polegroup = [153, 153, 153]
        street_light = [220, 220, 100]
        traffic_cone = [255, 70, 0]
        traffic_device = [220, 220, 220]
        traffic_light = [250, 170, 30]
        traffic_sign = [220, 220, 0]
        traffic_sign_frame = [250, 170, 250]
        terrain = [152, 251, 152]
        vegetation = [107, 152, 35]
        sky = [70, 130, 180]
        person = [220, 20, 60]
        rider = [255, 0, 0]
        bicycle = [119, 11, 32]
        bus = [0, 60, 100]
        car = [0, 0, 142]
        caravan = [0, 0, 90]
        motorcycle = [0, 0, 230]
        trailer = [0, 0, 110]
        train = [0, 80, 100]
        truck = [0, 0, 70]

        return np.array([
            unlabeled, dynamic, ego_vehicle, ground, static, parking, rail_track, road, sidewalk, bridge,
            building, fence, garage, guard_rail, tunnel, wall, banner, billboard, lane_divider, parking_sign,
            pole, polegroup, street_light, traffic_cone, traffic_device, traffic_light, traffic_sign,
            traffic_sign_frame, terrain, vegetation, sky, person, rider, bicycle, bus, car, caravan,
            motorcycle, trailer, train, truck])

    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        subset_dir = "train"
        if self.subset == "validation":
            subset_dir = "val"
        elif self.subset == "test":
            subset_dir = "test"
        file_path = os.path.join(self.data_dir, self.image_dir, subset_dir, "*.jpg")
        image_paths = glob.glob(file_path)
        image_paths.sort()

        file_path = os.path.join(self.data_dir, self.label_dir, subset_dir, "*.png")
        label_paths = glob.glob(file_path)
        label_paths.sort()

        assert (len(image_paths) == len(label_paths)), "Number of Images and Labels does not match."

        return image_paths, label_paths

    def __getitem__(self, i):
        imgs, labels = self.files_and_annotations()
        img = load_image(imgs[i])
        label = load_image(labels[i])

        return img, label

    def __len__(self):
        return len(self.files_and_annotations()[0])

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations()[0])
