# -*- coding: utf-8 -*-
import functools
import glob
import json
from os.path import basename, join, splitext

import numpy as np

from lmnet.datasets.base import ObjectDetectionBase


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

        image = self._get_image(image_file_path)

        gt_boxes = self.bboxs[i]
        gt_boxes = np.array(gt_boxes)
        gt_boxes = gt_boxes.copy()  # is it really needed?
        gt_boxes = self._fill_dummy_boxes(gt_boxes)

        return image, gt_boxes

    def __len__(self):
        return self.num_per_epoch


def check_dataset():
    train = BDD100K(subset="train")
    print(len(train.paths))
    print(train.paths[0:5])
    print(train.bboxs[0:5])
    print("seems dataset was loaded correctly")


if __name__ == '__main__':
    check_dataset()
