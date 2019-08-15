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
import tensorflow_datasets as tfds

from lmnet.datasets.base import ObjectDetectionBase
from lmnet.datasets.tfds import TFDSMixin
from lmnet.utils import config as config_util
from lmnet.utils.tfds_builders.classification import ClassificationBuilder
from lmnet.utils.tfds_builders.object_detection import ObjectDetectionBuilder


def _copy_directory_recursively(src, dst):
    tf.io.gfile.makedirs(dst)
    for parent, directories, files in tf.io.gfile.walk(src):
        for directory in directories:
            src_dir = os.path.join(parent, directory)
            dst_dir =  os.path.relpath(src_dir, src)
            dst_dir = os.path.join(dst, dst_dir)

            if tf.io.gfile.exists(dst_dir):
                tf.io.gfile.rmtree(dst_dir)

            tf.io.gfile.mkdir(dst_dir)

        for file in files:
            src_file = os.path.join(parent, file)
            dst_file = os.path.relpath(src_file, src)
            dst_file = os.path.join(dst, dst_file)
            tf.io.gfile.copy(src_file, dst_file)


def run(config_file, overwrite):
    config = config_util.load(config_file)
    dataset_class = config.DATASET_CLASS
    dataset_kwargs = dict((key.lower(), val) for key, val in config.DATASET.items())

    tfds_kwargs = dataset_kwargs.pop("tfds_kwargs")
    name = tfds_kwargs['name']
    data_dir = os.path.expanduser(tfds_kwargs['data_dir'])

    if tf.io.gfile.exists(os.path.join(data_dir, name)):
        if not overwrite:
            raise ValueError("Output path already exists: {}\n"
                             "Please use --overwrite if you want to overwrite."
                             .format(os.path.join(data_dir, name)))

    if issubclass(dataset_class, TFDSMixin):
        raise ValueError("You cannot use dataset classes which is already a TFDS format.")

    if issubclass(dataset_class, ObjectDetectionBase):
        builder_class = ObjectDetectionBuilder
    else:
        builder_class = ClassificationBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = builder_class(dataset_name=name,
                                dataset_class=dataset_class,
                                dataset_kwargs=dataset_kwargs,
                                data_dir=tmpdir)

        builder.download_and_prepare()
        print("Dataset was built successfully.")

        print("Copying to destination...")
        _copy_directory_recursively(src=tmpdir, dst=data_dir)

        print("Done!!")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-c",
    "--config_file",
    help="Path to config file.",
    required=True,
)
@click.option(
    "-ow",
    "--overwrite",
    help="Overwrite if the output already exists.",
    is_flag=True,
    default=False,
)
def main(config_file, overwrite):
    """A script to build custom TFDS datasets"""
    run(os.path.expanduser(config_file), overwrite)


if __name__ == "__main__":
    main()
