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

import os
import re
import sys
import tqdm
import glob
import pickle
import functools
import numpy as np

import PIL

from lmnet import data_processor
from lmnet.datasets.base import Base

"""
references by the author
https://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
https://stackoverflow.com/questions/42483476/numpy-only-integer-scalar-arrays-can-be-converted-to-a-scalar-index-upgrading
https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py
"""


def train_valid_split(arr, validation_rate=0.25, seed=None):
    assert 0.0 <= validation_rate <= 1.0, "Invalid validation rate"
    random_state = np.random.RandomState(seed)

    num_samples = len(arr)
    num_train = int(num_samples * (1 - validation_rate))
    num_valid = num_samples - num_train

    indices = np.arange(num_samples)
    random_state.shuffle(indices)

    valid_indices = indices[:num_valid]
    train_indices = indices[num_valid:]

    assert len(valid_indices) == num_valid
    assert len(train_indices) == num_train

    return [arr[_] for _ in train_indices], [arr[_] for _ in valid_indices]


@functools.lru_cache(maxsize=None)
def _open_image_file(file_name, dtype=np.float32):
    # return imageio.imread(file_name).astype(dtype)
    return np.array(PIL.Image.open(file_name), dtype=dtype)


@functools.lru_cache(maxsize=None)
def _open_flo_file(file_name):
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert 202021.25 == magic, \
            "Magic number incorrect. Invalid .flo file"
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * width * height)
    return np.resize(data, (height, width, 2))


@functools.lru_cache(maxsize=None)
def _open_pfm_file(file_name):
    color, width, height, scale, endian = None, None, None, None, None
    with open(file_name, "rb") as f:
        # loading header information
        header = f.readline().rstrip().decode("utf-8")
        assert header == "PF" or header == "Pf", "Not a PFM file."
        color = (header == "PF")

        # loading wdth and height information
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        assert dim_match is not None, "Malformed PFM header."
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().rstrip().decode("utf-8"))
        if scale < 0:
            endian = "<"
            scale = -scale
        else:
            endian = ">"
        data = np.fromfile(f, endian + "f").astype(np.float32)
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape)[..., :2]


class OpticalFlowEstimationBase(Base):
    classes = ()
    num_classes = 0
    available_subsets = ["train", "validation"]
    _coef = 1 / 255.0

    def __init__(
            self, *args, validation_rate=0.1, validation_seed=1234, **kwargs):
        super().__init__(*args, **kwargs,)
        self.fetch_file_list()
        self.split_file_list(validation_rate, validation_seed)

    def __getitem__(self, index, type=None):
        image_a_path, image_b_path, flow_path = self.file_list[index]
        image_a = _open_image_file(image_a_path)
        image_b = _open_image_file(image_b_path)
        flow = self.flow_loader(flow_path)
        return self._coef * np.concatenate([image_a, image_b], axis=2), flow

    def __len__(self):
        return self.num_per_epoch

    @property
    def num_per_epoch(self):
        return len(self.file_list)

    def split_file_list(self, validation_rate, validation_seed):
        train_dataset, validation_dataset = train_valid_split(
            self.file_list, validation_rate, validation_seed)
        if self.subset == "train":
            self.file_list = train_dataset
        else:
            self.file_list = validation_dataset

    def flow_loader(self):
        NotImplementedError()


class FlyingChairs(OpticalFlowEstimationBase):
    extend_dir = "FlyingChairs/data"

    def fetch_file_list(self):
        self.file_list = []
        for flow_path in glob.glob("{}/*_flow.flo".format(self.data_dir)):
            image_a_path = re.sub(r"_flow.flo$", "_img1.ppm", flow_path)
            image_b_path = re.sub(r"_flow.flo$", "_img2.ppm", flow_path)
            if not os.path.exists(image_a_path):
                continue
            if not os.path.exists(image_b_path):
                continue
            self.file_list.append([image_a_path, image_b_path, flow_path])

    def flow_loader(self, *args, **kwargs):
        return _open_flo_file(*args, **kwargs)


class ChairsSDHom(OpticalFlowEstimationBase):
    extend_dir = "ChairsSDHom/data"

    def fetch_file_list(self):
        image_a_list = sorted(glob.glob(
            "{}/train/t0/*.png".format(self.data_dir)))
        image_b_list = sorted(glob.glob(
            "{}/train/t1/*.png".format(self.data_dir)))
        flow_list = sorted(glob.glob(
            "{}/train/flow/*.pfm".format(self.data_dir)))
        self.file_list = list(zip(image_a_list, image_b_list, flow_list))
        basename_list = [
            list(map(os.path.basename, _))
            for _ in self.file_list
        ]
        basename_list = [
            list(map(lambda t: os.path.splitext(t)[0], _))
            for _ in basename_list
        ]
        for args in basename_list:
            assert (args[0] == args[1]) and (args[1] == args[2]), \
                "Missing file detected!"

    def flow_loader(self, *args, **kwargs):
        return _open_pfm_file(*args, **kwargs)
