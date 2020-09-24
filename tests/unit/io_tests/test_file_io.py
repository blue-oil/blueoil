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
import pytest
import tensorflow as tf

from blueoil.io import file_io


def test_file_class():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.txt")
        with file_io.File(path, mode="w") as f:
            f.write("test")
        assert os.path.isfile(path)


def test_exists():
    with tempfile.NamedTemporaryFile() as f:
        assert file_io.exists(f.name)


def test_abspath_local():
    cur_dir = os.getcwd()
    with tempfile.NamedTemporaryFile(dir=cur_dir) as f:
        file_name = os.path.basename(f.name)
        assert os.path.join(cur_dir, file_name) == file_io.abspath(file_name)


def test_abspath_gcs():
    gcs_path = "gs://xxxx/xxxx"
    assert gcs_path == file_io.abspath(gcs_path)


def test_mkdir():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test")
        file_io.mkdir(path)
        assert os.path.isdir(path)


def test_makedirs():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test", "test")
        file_io.makedirs(path)
        assert os.path.isdir(path)


def test_copy():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        dst = os.path.join(d, "dst")
        src_content = "this\nis\nsource\nfile"
        with open(src, mode="w") as f:
            f.write(src_content)
        assert file_io.copy(src, dst) == dst
        assert os.path.isfile(dst)
        with open(dst, mode="r") as f:
            assert f.read() == src_content

        with pytest.raises(tf.errors.AlreadyExistsError):
            file_io.copy(src, dst)


def test_copy_overwrite():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        dst = os.path.join(d, "dst")
        src_content = "this\nis\nsource\nfile"
        dst_content = "this\nis\ndestination\nfile"
        with open(src, mode="w") as f:
            f.write(src_content)
        with open(dst, mode="w") as f:
            f.write(dst_content)
        assert file_io.copy(src, dst, overwrite=True) == dst
        assert os.path.isfile(dst)
        with open(dst, mode="r") as f:
            assert f.read() == src_content


def test_copy_dir_dst():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        src_content = "this\nis\nsource\nfile"
        with open(src, mode="w") as f:
            f.write(src_content)
        with tempfile.TemporaryDirectory() as d2:
            dst = file_io.copy(src, d2)
            assert dst == os.path.join(d2, "src")
            assert os.path.isfile(dst)
            with open(dst, mode="r") as f:
                assert f.read() == src_content


def test_rmtree():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test")
        os.makedirs(os.path.join(path, "a"))
        os.makedirs(os.path.join(path, "b"))
        file_io.rmtree(path)
        assert not os.path.exists(path)


def test_save_npy():
    with tempfile.TemporaryDirectory() as d:
        arr = np.array([1, 2])
        path = os.path.join(d, "test.npy")
        file_io.save_npy(path, arr)
        assert os.path.isfile(path)
        np.testing.assert_array_equal(np.load(path), arr)


def test_load_image():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.jpg")
        img = Image.new("RGB", (32, 32))
        img.save(path)
        got = file_io.load_image(path)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(img))


def test_save_image():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.jpg")
        img = Image.new("RGB", (32, 32))
        file_io.save_image(path, img)
        assert os.path.isfile(path)
        np.testing.assert_array_equal(np.asarray(Image.open(path)), np.asarray(img))
