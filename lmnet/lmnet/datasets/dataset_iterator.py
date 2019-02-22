from multiprocessing import Pool
import time
import sys
import threading
import numpy as np
import os
import queue
from lmnet.datasets.base import SegmentationBase, ObjectDetectionBase


_dataset = None


def _prefetch_setup(dataset, seed, do_shuffle):
    global _dataset
    _dataset = dataset
    if do_shuffle:
        np.random.seed(os.getpid()+seed)
        _dataset.seed = os.getpid() + seed
        _dataset._shuffle()


def _feed(i):
    return _dataset.feed()


def _apply_augmentations(dataset, image, label):
    augmentor = dataset.augmentor
    pre_processor = dataset.pre_processor

    sample = {'image': image}

    if issubclass(dataset.__class__, SegmentationBase):
        sample['mask'] = label
    elif issubclass(dataset.__class__, ObjectDetectionBase):
        sample['gt_boxes'] = label
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
    else:
        label = sample['label']

    # FIXME(tokunaga): dataset should not have thier own data format
    if dataset.data_format == "NCHW":
        image = np.transpose(image, [2, 0, 1])

    return (image, label)


def _process_one_data(i):
    image, label = _dataset[i]
    return _apply_augmentations(_dataset, image, label)


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


class _MultiProcessDatasetPrefetchThread(threading.Thread):
    def __init__(self, dataset, result_queue, seed):
        super().__init__()
        # TODO(tokunaga): the number of processes should be configurable
        self.seed = seed + 1  # seed must not be 0
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

    # for new style dataset class
    def loop_body_getitem(self):
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

    # for current dataset class
    def loop_body_feed(self):
        fetch_result = self.pool.map(_feed, list(range(0, 200)))
        for data_batch in fetch_result:
            put_ok = False
            while not put_ok:
                try:
                    self.result_queue.put(data_batch, 1)
                    put_ok = True
                except queue.Full:
                    if self.terminate:
                        break

    def loop_body(self):
        if self.support_getitem:
            self.loop_body_getitem()
        else:
            self.loop_body_feed()

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

    def run_sync(self):
        task_list = self.gen_task(self.dataset.batch_size)
        result = []
        for i in task_list:
            image, label = self.dataset[i]
            image, label = _apply_augmentations(self.dataset, image, label)
            result.append((image, label))
        return _concat_data(result)


# fallback class for the datasets that don't support multiprocess prefetching
class _SingleProcessDatasetPrefetchThread(threading.Thread):
    def __init__(self, dataset, result_queue, seed):
        super().__init__()

        self.seed = seed
        self.result_queue = result_queue
        self.batch_size = dataset.batch_size
        self.index = 0
        self.dataset = dataset
        self.terminate = False

    def run(self):
        count = 0
        while True:
            if self.terminate:
                break

            data_batch = self.dataset.feed()
            put_ok = False
            while not put_ok:
                try:
                    self.result_queue.put(data_batch, 1)
                    put_ok = True
                except queue.Full:
                    if self.terminate:
                        break
            count += 1


class DatasetIterator:

    available_subsets = ["train", "train_validation_saving", "validation"]

    """docstring for DatasetIterator."""
    def __init__(self, dataset, enable_prefetch=False, seed=0):
        self.dataset = dataset
        self.enable_prefetch = enable_prefetch
        self.seed = seed

        if self.enable_prefetch:
            self.prefetch_result_queue = queue.Queue(maxsize=200)

            if hasattr(dataset, "__getitem__"):
                self.prefetcher = _MultiProcessDatasetPrefetchThread(self.dataset, self.prefetch_result_queue, seed)
            else:
                self.prefetcher = _SingleProcessDatasetPrefetchThread(self.dataset, self.prefetch_result_queue, seed)

            self.prefetcher.start()
            print("ENABLE prefetch")
        else:
            if hasattr(dataset, "__getitem__"):
                self.prefetcher = _MultiProcessDatasetPrefetchThread(self.dataset, False, seed)
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

    def __next__(self):
        if self.enable_prefetch:
            (images, labels) = self.prefetch_result_queue.get()
            return images, labels
        elif self.prefetcher:
            (images, labels) = self.prefetcher.run_sync()
            return images, labels
        else:
            images, labels = self.dataset.feed()
            return images, labels

    def feed(self):
        return self.__next__()

    def __len__(self):
        return len(self.dataset)

    def update_dataset(self, indices):
        """Update own dataset by indices."""
        if self.enable_prefetch:
            # do nothing so far
            return
        elif hasattr(self.dataset, "update_dataset"):
            self.dataset.update_dataset(indices)
        else:
            print("this dataset does not support distrituted training yet.")
            sys.exit(1)

    def get_shuffle_index(self):
        """Return list of shuffled index."""
        random_state = np.random.RandomState(self.seed)
        random_indices = list(range(self.num_per_epoch))
        random_state.shuffle(random_indices)
        print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.seed))
        print(random_indices[0:10])
        self.seed += 1
        return random_indices


if __name__ == '__main__':

    from lmnet.datasets.cifar10 import Cifar10
    from lmnet.data_processor import Sequence
    from lmnet.data_augmentor import FlipLeftRight, Hue, Blur

    cifar10 = Cifar10()

    augmentor = Sequence([
        FlipLeftRight(0.5),
        Hue((-10, 10)),
        Blur(),
    ])

    dataset_iterator = DatasetIterator(dataset=cifar10, enable_prefetch=True, augmentor=augmentor)
    time.sleep(2)
    import time
    t0 = time.time()
    data_batch = next(dataset_iterator)
    t1 = time.time()
    print("time of prefetch: {}".format(t1 - t0))

    dataset_iterator2 = DatasetIterator(dataset=cifar10, enable_prefetch=False, augmentor=augmentor)
    t0 = time.time()
    data_batch = next(dataset_iterator2)
    t1 = time.time()
    print("time with/o prefetch: {}".format(t1 - t0))
