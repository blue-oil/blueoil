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

from lmnet.utils.module_loader import load_tfds_builder_class


def _copy_directory_recursively(src, dst):
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


def run(name, input_path, output_path, builder_name, overwrite):
    if not tf.io.gfile.exists(input_path):
        raise ValueError("Input path does not exist: {}".format(input_path))

    if tf.io.gfile.exists(os.path.join(output_path, name)):
        if not overwrite:
            raise ValueError("Output path already exists: {}\n"
                             "Please use --overwrite if you want to overwrite."
                             .format(os.path.join(output_path, name)))

    try:
        builder_class = load_tfds_builder_class(builder_name)
    except:
        raise NotImplementedError("Builder class does not exist: lmnet.utils.tfds_builders.{}"
                                  .format(builder_name))

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = builder_class(dataset_name=name, raw_data_path=input_path, data_dir=tmpdir)
        builder.download_and_prepare()
        print("Dataset was built successfully.")

        print("Copying to destination...")
        _copy_directory_recursively(src=tmpdir, dst=output_path)

        print("Done!!")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-n",
    "--name",
    help="Dataset name.",
    required=True,
)
@click.option(
    "-b",
    "--builder",
    help="Builder name.",
    required=True,
)
@click.option(
    "-i",
    "--input_path",
    help="Raw dataset needs to be placed at <input_path>/<name>.",
    required=True,
)
@click.option(
    "-o",
    "--output_path",
    help="Generated datasets will be placed at <output_path>/<name>.",
    default=os.path.join(os.path.expanduser("~"), "tensorflow_datasets"),
)
@click.option(
    "-ow",
    "--overwrite",
    help="Overwrite if the output already exists.",
    is_flag=True,
    default=False,
)
def main(name, input_path, output_path, builder, overwrite):
    """A script to build custom TFDS datasets"""
    run(
        name,
        os.path.expanduser(input_path),
        os.path.expanduser(output_path),
        builder,
        overwrite
    )


if __name__ == "__main__":
    main()
