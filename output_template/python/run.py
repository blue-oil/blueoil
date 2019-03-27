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
import numpy as np
import click
import os
import sys

from PIL import Image
from lmnet.nnlib import NNLib as NNLib

from lmnet.common import Tasks
from lmnet.utils.output import JsonOutput, ImageFromJson
from lmnet.utils.config import (
    load_yaml,
    build_pre_process,
    build_post_process,
)


def _pre_process(raw_image, pre_processor, data_format):
    pre_process = build_pre_process(pre_processor)
    image = pre_process(image=raw_image)['image']
    if data_format == 'NCHW':
        image = np.transpose(image, [2, 0, 1])
    return image


def _post_process(output, post_processor):
    post_process = build_post_process(post_processor)
    output = post_process(outputs=output)['outputs']
    return output


def _save_json(output_dir, json_obj):
    output_file_name = os.path.join(output_dir, "output.json")
    dirname = os.path.dirname(output_file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output_file_name, "w") as json_file:
        json_file.write(json_obj)
    print("save json: {}".format(output_file_name))


def _save_images(output_dir, filename_images):
    for filename, image in filename_images:
        base_name = os.path.basename(filename)
        output_file_name = os.path.join(output_dir, "images", base_name)
        dirname = os.path.dirname(output_file_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        image.save(output_file_name)
        print("save image: {}".format(output_file_name))


def _run(model, input_image, config):
    filename, file_extension = os.path.splitext(model)
    supported_files = ['.so', '.pb']

    if file_extension not in supported_files:
        raise Exception("""
            Unknown file type. Got %s%s.
            Please check the model file (-m).
            Only .pb (protocol buffer) or .so (shared object) file is supported.
            """ % (filename, file_extension))

    # load the image
    img = Image.open(input_image).convert("RGB")

    # convert into numpy array
    data = np.asarray(img)
    raw_image = data

    # pre process for image
    data = _pre_process(data, config.PRE_PROCESSOR, config.DATA_FORMAT)

    # add the batch dimension
    data = np.expand_dims(data, axis=0)

    if file_extension == '.so':  # Shared library
        # load and initialize the generated shared model
        nn = NNLib()
        nn.load(model)
        nn.init()

    elif file_extension == '.pb':  # Protocol Buffer file
        # only load tensorflow if user wants to use GPU
        from lmnet.tensorflow_graph_runner import TensorflowGraphRunner
        nn = TensorflowGraphRunner(model)
        nn.init()

    # run the graph
    output = nn.run(data)

    return output, raw_image


def run_prediction(input_image, model, config_file, max_percent_incorrect_values=0.1):
    if not input_image or not model or not config_file:
        print('Please check usage with --help option')
        exit(1)

    config = load_yaml(config_file)

    # run the model
    output, raw_image = _run(model, input_image, config)

    print('Output: (before post process)')
    print(output)

    # pre process for output
    output = _post_process(output, config.POST_PROCESSOR)

    print('Output: ')
    print(output)

    # json output
    json_output = JsonOutput(
        task=Tasks(config.TASK),
        classes=config.CLASSES,
        image_size=config.IMAGE_SIZE,
        data_format=config.DATA_FORMAT,
    )

    image_from_json = ImageFromJson(
        task=Tasks(config.TASK),
        classes=config.CLASSES,
        image_size=config.IMAGE_SIZE,
    )

    output_dir = "output"
    outputs = output
    raw_images = [raw_image]
    image_files = [input_image]
    json_obj = json_output(outputs, raw_images, image_files)
    _save_json(output_dir, json_obj)
    filename_images = image_from_json(json_obj, raw_images, image_files)
    _save_images(output_dir, filename_images)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--input_image",
    type=click.Path(exists=True),
    help="Input image filename",
)
@click.option(
    "-m",
    "-l",
    "--model",
    type=click.Path(exists=True),
    help=u"""
        Inference Model filename
        (-l is deprecated please use -m instead)
    """,
    default="../models/lib/lib_fpga.so",
)
@click.option(
    "-c",
    "--config_file",
    type=click.Path(exists=True),
    help="Config file Path",
)
def main(input_image, model, config_file):
    _check_deprecated_arguments()
    run_prediction(input_image, model, config_file)


def _check_deprecated_arguments():
    argument_list = sys.argv
    if '-l' in argument_list:
        print("Deprecated warning: -l is deprecated please use -m instead")


if __name__ == "__main__":
    main()
