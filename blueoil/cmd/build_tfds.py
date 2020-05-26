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
import os
import tempfile

import click
import tensorflow as tf

from blueoil.datasets.base import ObjectDetectionBase, SegmentationBase
from blueoil.datasets.tfds import TFDSMixin
from blueoil.utils import config as config_util
from blueoil.utils.tfds_builders.classification import ClassificationBuilder
from blueoil.utils.tfds_builders.object_detection import ObjectDetectionBuilder
from blueoil.utils.tfds_builders.segmentation import SegmentationBuilder


def _get_tfds_settings(config_file):
    config = config_util.load(config_file)
    dataset_class = config.DATASET_CLASS
    dataset_kwargs = {key.lower(): val for key, val in config.DATASET.items()}

    if "tfds_kwargs" not in dataset_kwargs:
        raise ValueError("The given config file does not contain settings for building TFDS datasets.\n"
                         "Please see help messages (python blueoil/cmd/build_tfds.py -h) for detail.")

    tfds_kwargs = dataset_kwargs.pop("tfds_kwargs")
    dataset_name = tfds_kwargs["name"]
    data_dir = os.path.expanduser(tfds_kwargs["data_dir"])

    return dataset_class, dataset_kwargs, dataset_name, data_dir


def _get_tfds_builder_class(dataset_class):
    if issubclass(dataset_class, TFDSMixin):
        raise ValueError("You cannot use dataset classes which is already a TFDS format.")

    if issubclass(dataset_class, SegmentationBase):
        return SegmentationBuilder

    if issubclass(dataset_class, ObjectDetectionBase):
        return ObjectDetectionBuilder

    return ClassificationBuilder


def _copy_directory_recursively(src, dst):
    tf.io.gfile.makedirs(dst)
    for parent, directories, files in tf.io.gfile.walk(src):
        for directory in directories:
            src_dir = os.path.join(parent, directory)
            dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))

            if tf.io.gfile.exists(dst_dir):
                tf.io.gfile.rmtree(dst_dir)

            tf.io.gfile.mkdir(dst_dir)

        for file in files:
            src_file = os.path.join(parent, file)
            dst_file = os.path.join(dst, os.path.relpath(src_file, src))
            tf.io.gfile.copy(src_file, dst_file)


def run(config_file, overwrite):
    """Build custom TFDS datasets from config file"""
    dataset_class, dataset_kwargs, dataset_name, data_dir = _get_tfds_settings(config_file)

    if not overwrite and tf.io.gfile.exists(os.path.join(data_dir, dataset_name)):
        raise ValueError("Output path already exists: {}\n"
                         "Please use --overwrite if you want to overwrite."
                         .format(os.path.join(data_dir, dataset_name)))

    builder_class = _get_tfds_builder_class(dataset_class)

    # Generate data in tmp directory and copy it to data_dir at the end.
    # This is because generating data directly to remote storage (e.g. GCS) is sometimes very slow.
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = builder_class(dataset_name=dataset_name,
                                dataset_class=dataset_class,
                                dataset_kwargs=dataset_kwargs,
                                data_dir=tmpdir)

        builder.download_and_prepare()
        print("Dataset was built successfully.")

        print("Copying to destination...")
        _copy_directory_recursively(src=tmpdir, dst=data_dir)

        print("Done!!")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-c",
    "--config_file",
    help="A path to config file",
    required=True,
)
@click.option(
    "-o",
    "--overwrite",
    help="Overwrite if the output directory already exists.",
    is_flag=True,
    default=False,
)
def main(config_file, overwrite):
    """
    A script to build custom TFDS datasets

    \b
    This script can build custom TFDS datasets from existing dataset classes.
    You can use training config files to specify which dataset class to be used as data source.
    The following settings are required in the config file.

    \b
    ```
    DATASET_CLASS = <dataset class>
    DATASET.TFDS_KWARGS = {
        "name": "<a dataset name to be generated>",
        "data_dir": "<a directory path to output generated dataset>",
        "image_size": <image size array like [128, 128]>,
    }
    ```

    \b
    Note: Images will be resized into the specified size when TFRecords are loaded.
          Images stored in TFRecords still have original size.

    \b
    If you have a training config file with the settings above,
    you can execute this script and the training script with the same config file.
    Then the generated TFDS dataset will be used for training.

    \b
    ```
    python blueoil/cmd/build_tfds.py -c common_config_file.py
    python blueoil/cmd/train.py      -c common_config_file.py
    ```
    """

    run(os.path.expanduser(config_file), overwrite)


if __name__ == "__main__":
    main()
