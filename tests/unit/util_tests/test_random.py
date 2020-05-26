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
import random

import numpy as np
import pytest

from blueoil.utils.random import shuffle, train_test_split


def test_shuffle():
    num_samples = 40

    a_shape = (num_samples, 3, 5)
    a = np.random.random(a_shape)

    b_shape = (num_samples, 6, 9)
    b = np.random.random(b_shape)

    c = list(range(40))

    seed = random.randint(1, 100)

    result_a, result_b, result_c = shuffle(a, b, c, seed=seed)

    assert result_a.shape == a_shape
    assert result_b.shape == b_shape
    assert len(c) == len(result_c)
    assert not np.all(a == result_a)
    assert not np.all(b == result_b)
    assert not c == result_c

    same_seed_a, same_seed_b, same_seed_c = shuffle(a, b, c, seed=seed)
    assert same_seed_a.shape == a_shape
    assert same_seed_b.shape == b_shape
    assert len(same_seed_c) == len(c)
    assert np.all(same_seed_a == result_a)
    assert np.all(same_seed_b == result_b)
    assert same_seed_c == result_c

    diff_seed_a, diff_seed_b, diff_seed_c = shuffle(a, b, c, seed=None)
    assert diff_seed_a.shape == a_shape
    assert diff_seed_b.shape == b_shape
    assert len(diff_seed_c) == len(c)
    assert not np.all(diff_seed_a == result_a)
    assert not np.all(diff_seed_b == result_b)
    assert not same_seed_c == c


def test_shuffle_diff_length():

    a_shape = (40, 3, 5)
    a = np.random.random(a_shape)

    b_shape = (20, 6, 9)
    b = np.random.random(b_shape)

    with pytest.raises(Exception):
        result_a, result_b = shuffle(a, b)

    c = list(range(15))

    with pytest.raises(Exception):
        result_a, result_c = shuffle(a, c)


def test_shuffle_range():
    range_list = range(40)

    seed = random.randint(1, 100)

    result = shuffle(range_list, seed=seed)

    assert len(result) == len(range_list)
    assert type(result) == list
    assert result != range_list


def test_train_test_split():
    num_samples = 40
    seed = random.randint(1, 100)
    test_size = 0.714
    num_train = int(num_samples * (1 - test_size))
    num_test = num_samples - num_train
    a_shape = (num_samples, 3, 5)
    a = np.random.random(a_shape)

    b_shape = (num_samples, 6, 9)
    b = np.random.random(b_shape)

    train_a, test_a, train_b, test_b = train_test_split(a, b, test_size=test_size, seed=seed)

    assert train_a.shape == (num_train, 3, 5)
    assert train_b.shape == (num_train, 6, 9)
    assert len(train_a) == len(train_b)

    assert test_a.shape == (num_test, 3, 5)
    assert test_b.shape == (num_test, 6, 9)
    assert len(test_a) == len(test_b)

    same_seed_train_a, same_seed_test_a, same_seed_train_b, same_seed_test_b = \
        train_test_split(a, b, test_size=test_size, seed=seed)

    assert np.all(train_a == same_seed_train_a)
    assert np.all(train_b == same_seed_train_b)
    assert np.all(test_a == same_seed_test_a)
    assert np.all(test_b == same_seed_test_b)

    diff_seed_train_a, diff_seed_test_a, diff_seed_train_b, diff_seed_test_b = \
        train_test_split(a, b, test_size=test_size, seed=0)

    assert not np.all(train_a == diff_seed_train_a)
    assert not np.all(train_b == diff_seed_train_b)
    assert not np.all(test_a == diff_seed_test_a)
    assert not np.all(test_b == diff_seed_test_b)


if __name__ == '__main__':
    test_shuffle()
    test_shuffle_diff_length()
    test_shuffle_range()
    test_train_test_split()
