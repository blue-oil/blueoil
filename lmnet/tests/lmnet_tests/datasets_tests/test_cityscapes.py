import pytest
from lmnet.datasets.cityscapes import Cityscapes
from lmnet.datasets.dataset_iterator import DatasetIterator


def test_cityscapes():
    batch_size = 1
    train_dataset = Cityscapes(subset="train", batch_size=batch_size)
    train_dataset = DatasetIterator(train_dataset)

    assert train_dataset.num_classes == 34
    colors = train_dataset.label_colors
    assert len(colors) == 34

    train_image_files, train_label_files = train_dataset.feed()
    assert train_image_files.shape[1] == batch_size
    assert train_label_files.shape[0] == batch_size
