# -*- coding: utf-8 -*-
import functools
import glob
import json
import os

import numpy as np

from lmnet.datasets.base import ObjectDetectionBase
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

        counts = 0
        self.paths = []
        self.bboxs = []
        for item in anno_data:
            counts += 1
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


def check_dataset():
    train = BDD100KObjectDetection(subset="train")
    print(len(train.paths))
    print(train.paths[0:5])
    print(train.bboxs[0:5])
    print("seems dataset was loaded correctly")


if __name__ == '__main__':
    check_dataset()
