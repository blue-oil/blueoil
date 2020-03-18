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
import queue
import threading
from multiprocessing import Pool

import numpy as np
import tensorflow as tf

from blueoil.datasets.tfds import TFDSMixin

_dataset = None


def _prefetch_setup(dataset, seed, do_shuffle):
    global _dataset
    _dataset = dataset
    if do_shuffle:
        np.random.seed(os.getpid()+seed)
        _dataset.seed = os.getpid() + seed
        _dataset._shuffle()


def _apply_augmentations(dataset, sample):
    augmentor = dataset.augmentor
    pre_processor = dataset.pre_processor
    if callable(augmentor) and dataset.subset == 'train':
        sample = augmentor(**sample)
    if callable(pre_processor):
        sample = pre_processor(**sample)
    # FIXME(tokunaga): dataset should not have their own data format
    if dataset.data_format == "NCHW":
        sample['image'] = np.transpose(sample['image'], [2, 0, 1])
    return sample


def _process_one_data(i):
    sample = _dataset[i]
    return _apply_augmentations(_dataset, sample)


def _concat_data(sample_list):
    samples_dict = {}
    for key in sample_list[0]:
        samples_dict[key] = np.stack([sample[key] for sample in sample_list], axis=0)
    return samples_dict


def _xorshift32(r):
    r = r ^ (r << 13 & 0xFFFFFFFF)
    r = r ^ (r >> 17 & 0xFFFFFFFF)
    r = r ^ (r << 5 & 0xFFFFFFFF)
    return r & 0xFFFFFFFF


class _MultiProcessDatasetPrefetchThread(threading.Thread):
    def __init__(self, dataset, result_queue, seed):
        super().__init__()
        # TODO(tokunaga): the number of processes should be configurable
        self.seed = seed + 1  # seed must not be 0 because using xorshift32.
        self.support_getitem = hasattr(dataset, "__getitem__")
        self.pool = Pool(processes=8, initializer=_prefetch_setup,
                         initargs=(dataset, self.seed, not self.support_getitem))
        self.result_queue = result_queue
        self.batch_size = dataset.batch_size
        self.dataset = dataset
        self.data_ids = []
        self.terminate = False
        self.setDaemon(True)

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

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def refresh_pool(self):
        self.pool.close()
        self.seed += 1
        self.pool = Pool(processes=8, initializer=_prefetch_setup,
                         initargs=(self.dataset, self.seed, not self.support_getitem))

    def loop_body(self):
        task_list = self.gen_task(self.dataset.batch_size * 8)
        fetch_result = self.pool.map(_process_one_data, task_list)
        for fetch_result_chunk in self.chunks(fetch_result, self.batch_size):
            data_batch = _concat_data(fetch_result_chunk)
            put_ok = False
            while not put_ok:
                try:
                    self.result_queue.put(data_batch, 1)
                    put_ok = True
                except queue.Full:
                    if self.terminate:
                        break

    def run(self):
        count = 0
        try:
            while True:
                if count == 1000:
                    self.refresh_pool()
                    count = 0
                if self.terminate:
                    print("break")
                    break
                self.loop_body()
                count += 1
        finally:
            self.pool.close()
            self.pool.join()


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
            sample = self.dataset[i]
            sample = _apply_augmentations(self.dataset, sample)
            result.append(sample)
        return _concat_data(result)


class _TFDSReader:

    def __init__(self, dataset, local_rank):
        tf_dataset = dataset.tf_dataset.shuffle(1024) \
                                       .repeat() \
                                       .batch(dataset.batch_size) \
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
        batch = self.session.run(self.next_batch)
        result = []
        for i in range(batch["image"].shape[0]):
            sample = {key: batch[key][i] for key in batch}
            sample = _apply_augmentations(self.dataset, sample)
            result.append(sample)
        return _concat_data(result)


class DatasetIterator:
    available_subsets = ["train", "train_validation_saving", "validation"]
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
                self.prefetch_result_queue = queue.Queue(maxsize=200)
                self.prefetcher = _MultiProcessDatasetPrefetchThread(self.dataset, self.prefetch_result_queue, seed)
                self.prefetcher.start()
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
        if self.enable_prefetch:
            samples_dict = self.prefetch_result_queue.get()
        else:
            samples_dict = self.reader.read()
        return samples_dict

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
        if self.enable_prefetch:
            self.prefetcher.terminate = True
            self.prefetcher.pool.close()
            self.prefetcher.pool.join()
