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
import collections
import concurrent.futures
import itertools
import threading
import weakref

import loky
import numpy as np
import tensorflow as tf

from blueoil.datasets.base import ObjectDetectionBase, SegmentationBase, KeypointDetectionBase
from blueoil.datasets.tfds import TFDSMixin
from blueoil import environment

_dataset_dict = None


def _prefetch_setup(data_dir, output_dir, dataset_dict):
    environment.set_data_dir(data_dir)
    environment.set_output_dir(output_dir)
    global _dataset_dict
    _dataset_dict = dataset_dict


def _apply_augmentations(dataset, image, label):
    augmentor = dataset.augmentor
    pre_processor = dataset.pre_processor

    sample = {'image': image}

    if issubclass(dataset.__class__, SegmentationBase):
        sample['mask'] = label
    elif issubclass(dataset.__class__, ObjectDetectionBase):
        sample['gt_boxes'] = label
    elif issubclass(dataset.__class__, KeypointDetectionBase):
        sample['joints'] = label
    else:
        sample['label'] = label

    if callable(augmentor) and dataset.subset == 'train':
        sample = augmentor(**sample)

    if callable(pre_processor):
        sample = pre_processor(**sample)

    image = sample['image']

    if issubclass(dataset.__class__, SegmentationBase):
        label = sample['mask']
    elif issubclass(dataset.__class__, ObjectDetectionBase):
        label = sample['gt_boxes']
    elif issubclass(dataset.__class__, KeypointDetectionBase):
        label = sample['heatmap']
    else:
        label = sample['label']

    # FIXME(tokunaga): dataset should not have their own data format
    if dataset.data_format == "NCHW":
        image = np.transpose(image, [2, 0, 1])

    return (image, label)


def _process_one_data(dataset_id, data_id):
    dataset = _dataset_dict[dataset_id]
    image, label = dataset[data_id]
    return _apply_augmentations(dataset, image, label)


def _concat_data(data_list):
    images, labels = zip(*data_list)
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels)


def _xorshift32(r):
    r = r ^ (r << 13 & 0xFFFFFFFF)
    r = r ^ (r >> 17 & 0xFFFFFFFF)
    r = r ^ (r << 5 & 0xFFFFFFFF)
    return r & 0xFFFFFFFF


def _prepare_worker_env(inner_max_num_threads):
    env = {}

    MAX_NUM_THREADS_VARS = [
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    for var in MAX_NUM_THREADS_VARS:
        env[var] = str(inner_max_num_threads)

    TBB_ENABLE_IPC_VAR = "ENABLE_IPC"
    env[TBB_ENABLE_IPC_VAR] = "1"

    return env


class _MultiProcessDatasetReader:

    _dataset_weakdict_lock = threading.RLock()
    _dataset_weakdict = weakref.WeakValueDictionary()

    def __init__(self, dataset, seed):
        with self._dataset_weakdict_lock:
            self._dataset_weakdict[id(dataset)] = dataset
        # TODO(tokunaga): the number of processes should be configurable
        self.seed = seed + 1  # seed must not be 0 because using xorshift32.
        self.support_getitem = hasattr(dataset, "__getitem__")
        self.dataset = dataset
        self.data_ids = []
        self.result_generator = self.run()

    def gen_ids(self):
        if hasattr(self.dataset, "__len__"):
            length = len(self.dataset)
        else:
            length = self.dataset.num_per_epoch
        return list(range(0, length))

    def gen_task(self, task_batch_size):
        task_list = []
        for i in range(0, task_batch_size):
            if len(self.data_ids) == 0:
                self.data_ids = self.gen_ids()
                self.seed = _xorshift32(self.seed)
                random_state = np.random.RandomState(self.seed)
                random_state.shuffle(self.data_ids)
            data_id = self.data_ids.pop()
            task_list.append(data_id)
        return task_list

    def loop_body(self):
        while True:
            for data_id in self.gen_task(self.dataset.batch_size * 8):
                with self._dataset_weakdict_lock:
                    dataset_dict = dict(self._dataset_weakdict)
                yield (
                    loky.get_reusable_executor(
                        max_workers=loky.cpu_count(),
                        timeout=None,
                        initializer=_prefetch_setup,
                        initargs=(environment.DATA_DIR, environment.OUTPUT_DIR, dataset_dict),
                        env=_prepare_worker_env(inner_max_num_threads=1),
                    )
                    .submit(_process_one_data, id(self.dataset), data_id)
                )

    def run(self):
        result_future_generator = self.loop_body()
        result_future_queue = collections.deque()
        try:
            result_future_queue.extend(itertools.islice(result_future_generator, self.dataset.batch_size))
            while True:
                result_future_queue.append(next(result_future_generator))
                yield result_future_queue[0].result()
                result_future_queue.popleft()
        finally:
            for result_future in result_future_queue:
                result_future.cancel()
            concurrent.futures.wait(result_future_queue)
            print("break")

    def read(self):
        return _concat_data(itertools.islice(self.result_generator, self.dataset.batch_size))

    def close(self):
        self.result_generator.close()
        with self._dataset_weakdict_lock:
            self._dataset_weakdict.pop(id(self.dataset))


class _SimpleDatasetReader:

    def __init__(self, dataset, seed, shuffle=True):
        self.dataset = dataset
        self.seed = seed + 1  # seed must not be 0 because using xorshift32.
        self.shuffle = shuffle
        self.data_ids = []

    def _gen_ids(self, size):
        """Generate ids which length is `size`."""
        for _ in range(0, size):
            # when data_ids is empty, fill and shuffle.
            if len(self.data_ids) == 0:
                self.data_ids = list(range(0, len(self.dataset)))
                if self.shuffle:
                    self.seed = _xorshift32(self.seed)
                    random_state = np.random.RandomState(self.seed)
                    random_state.shuffle(self.data_ids)

            yield self.data_ids.pop()

    def read(self):
        """Return batch size data."""
        result = []
        for i in self._gen_ids(self.dataset.batch_size):
            image, label = self.dataset[i]
            image, label = _apply_augmentations(self.dataset, image, label)
            result.append((image, label))
        return _concat_data(result)

    def close(self):
        pass


def _generate_tfds_map_func(dataset):
    """
    Return callable object
    """
    pre_processor = dataset.tfds_pre_processor
    augmentor = dataset.tfds_augmentor

    @tf.function
    def _tfds_map_func(arg):
        """
        Arg:
            arg(dict): with 'image' and 'label' keys
        """
        image, label = arg['image'], arg['label']
        sample = {'image': image}

        if issubclass(dataset.__class__, ObjectDetectionBase):
            sample['gt_boxes'] = tf.cast(label, tf.float32)
        else:
            sample['label'] = label

        if callable(augmentor) and dataset.subset == 'train':
            sample = augmentor(**sample)

        if callable(pre_processor):
            sample = pre_processor(**sample)

        image = sample['image']

        if issubclass(dataset.__class__, ObjectDetectionBase):
            label = sample['gt_boxes']
        else:
            label = sample['label']

        return (image, label)

    return _tfds_map_func


class _TFDSReader:

    def __init__(self, dataset, local_rank):
        tf_dataset = dataset.tf_dataset.shuffle(1024).repeat()
        if dataset.tfds_pre_processor or dataset.tfds_augmentor:
            tf_dataset = tf_dataset.map(map_func=_generate_tfds_map_func(dataset),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        tf_dataset = tf_dataset.batch(dataset.batch_size) \
                               .prefetch(tf.data.experimental.AUTOTUNE)
        iterator = tf.compat.v1.data.make_initializable_iterator(tf_dataset)

        self.dataset = dataset
        if local_rank != -1:
            # For distributed training
            session_config = tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                    visible_device_list=str(local_rank)
                )
            )
        else:
            session_config = tf.compat.v1.ConfigProto()
        self.session = tf.compat.v1.Session(config=session_config)
        self.session.run(iterator.initializer)
        self.next_batch = iterator.get_next()

    def read(self):
        """Return batch size data."""
        if self.dataset.tfds_pre_processor or self.dataset.tfds_augmentor:
            return self.session.run(self.next_batch)

        # if normal pre_processor is defined, use this
        batch = self.session.run(self.next_batch)
        result = [
            _apply_augmentations(self.dataset, image, label)
            for image, label in zip(batch['image'], batch['label'])
        ]
        return _concat_data(result)

    def close(self):
        self.session.close()


class DatasetIterator:

    available_subsets = ["train", "validation"]

    """docstring for DatasetIterator."""

    def __init__(self, dataset, enable_prefetch=False, seed=0, local_rank=-1):
        self.dataset = dataset
        self.enable_prefetch = enable_prefetch
        self.seed = seed

        if issubclass(dataset.__class__, TFDSMixin):
            self.enable_prefetch = False
            self.reader = _TFDSReader(self.dataset, local_rank)
        else:
            if self.enable_prefetch:
                self.reader = _MultiProcessDatasetReader(self.dataset, seed)
                print("ENABLE prefetch")
            else:
                self.reader = _SimpleDatasetReader(self.dataset, seed)
                print("DISABLE prefetch")

    @property
    def num_per_epoch(self):
        return self.dataset.num_per_epoch

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_max_boxes(self):
        return self.dataset.num_max_boxes

    @property
    def label_colors(self):
        return self.dataset.label_colors

    def extend_dir(self):
        return self.dataset.extend_dir

    def __iter__(self):
        return self

    def __next__(self):
        return self.reader.read()

    def feed(self):
        return self.__next__()

    def __len__(self):
        return len(self.dataset)

    def update_dataset(self, indices):
        """Update own dataset by indices."""
        # do nothing so far
        return

    def get_shuffle_index(self):
        """Return list of shuffled index."""
        random_state = np.random.RandomState(self.seed)
        random_indices = list(range(self.num_per_epoch))
        random_state.shuffle(random_indices)
        print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
        print(random_indices[0:10])
        self.seed += 1
        return random_indices

    def close(self):
        self.reader.close()
