# -*- coding: utf-8 -*-
# Copyright 2020 The Blueoil Authors. All Rights Reserved.
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
import tempfile

import numpy as np
from PIL import Image
from tensorflow.io import gfile


_supported_protocol = ("gs://", "s3://", "hdfs://")


class File(gfile.GFile):

    def __init__(self, name: str, mode: str = 'r'):
        super().__init__(name=name, mode=mode)


def exists(path: str) -> bool:
    return gfile.exists(path)


def abspath(path: str) -> str:
    return path if path.startswith(_supported_protocol) else os.path.abspath(path)


def mkdir(path: str):
    gfile.mkdir(path)


def makedirs(path: str):
    gfile.makedirs(path)


def copy(src: str, dst: str, overwrite: bool = False) -> str:
    if gfile.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    gfile.copy(src, dst, overwrite=overwrite)

    return dst


def rmtree(path: str):
    gfile.rmtree(path)


def save_npy(path: str, arr: np.ndarray):
    with gfile.GFile(path, mode="w") as f:
        np.save(f, arr)


def load_image(path: str) -> Image:
    with tempfile.TemporaryDirectory() as t:
        tmp_image = os.path.join(t, os.path.basename(path))
        gfile.copy(path, tmp_image)
        image = Image.open(tmp_image)

    return image


def save_image(path: str, image: Image):
    with gfile.GFile(path, mode="w") as f:
        image.save(f)
