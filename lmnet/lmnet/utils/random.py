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
from itertools import chain

import numpy as np


def _indexing(array, indices):
    if isinstance(array, np.ndarray):
        return array[indices]

    if isinstance(array, (list, range)):
        return [array[index] for index in indices]

    raise ValueError("array should be instance of np.ndarray or list, but {}".format(type(array)))


def shuffle(*arrays, seed=None):
    """Shuffle arrays

    Args:
        arrays: Sequence of arrays. For each array should be instance of np.ndarray or list.
        seed (int, optional): The seed of random generator.

    Returns:
        list: List of shuffled arrays. len(shuffled) == len(arrays)

    """

    random_state = np.random.RandomState(seed)

    first = arrays[0]
    lengths = [len(array) for array in arrays]

    assert all([len(first) == length for length in lengths]), "Found input variables"
    "with inconsistent numbers of samples: {}".format(lengths)

    num_samples = len(first)

    indices = np.arange(num_samples)
    random_state.shuffle(indices)

    if len(arrays) == 1:
        return _indexing(arrays[0], indices)

    shuffled_arrays = [_indexing(array, indices) for array in arrays]

    return shuffled_arrays


def train_test_split(*arrays, test_size=0.25, seed=None):
    """Split arrays into random train and test.

    Args:
        arrays: Sequence of arrays. For each array should be instance of np.ndarray or list.
        test_size(float, optional): Represent the proportion of the arrays to include in the test split.
            should be between 0.0 and 1.0.
        seed (int): The seed of random generator.

    Returns:
        list: List of train-test splittd array. len(splitted) == len(arrays) * 2

    """

    random_state = np.random.RandomState(seed)

    first = arrays[0]
    lengths = [len(array) for array in arrays]

    assert all([len(first) == length for length in lengths]), "Found input variables"
    "with inconsistent numbers of samples: {}".format(lengths)

    num_samples = len(first)
    num_train = int(num_samples * (1 - test_size))
    num_test = num_samples - num_train

    indices = np.arange(num_samples)
    random_state.shuffle(indices)

    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    assert len(test_indices) == num_test
    assert len(train_indices) == num_train

    # make list of tuple. [(train, test), (train, test),....]
    list_of_tuple = [(_indexing(array, train_indices), _indexing(array, test_indices)) for array in arrays]

    # flatten list of tuple
    splitted = list(chain.from_iterable(list_of_tuple))

    return splitted
