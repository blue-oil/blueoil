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
import logging
import os
import sys
import time

from lmnet.nnlib import NNLib

from blueoil.utils.image import load_image
from blueoil.common import Tasks
from blueoil.utils.predict_output.output import JsonOutput, ImageFromJson
from config import (
    load_yaml,
    build_pre_process,
    build_post_process,
)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _pre_process(raw_image, pre_process, data_format):
    image = pre_process(image=raw_image)['image']
    if data_format == 'NCHW':
        image = np.transpose(image, [2, 0, 1])

    # add the batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def _post_process(output, post_process):
    output = post_process(outputs=output)['outputs']
    return output


def _save_json(output_dir, json_obj):
    output_file_name = os.path.join(output_dir, "output.json")
    dirname = os.path.dirname(output_file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output_file_name, "w") as json_file:
        json_file.write(json_obj)
    logger.info("save json: {}".format(output_file_name))


def _save_images(output_dir, filename_images):
    for filename, image in filename_images:
        base_name = os.path.basename(filename)
        output_file_name = os.path.join(output_dir, "images", base_name)
        dirname = os.path.dirname(output_file_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        image.save(output_file_name)
        logger.info("save image: {}".format(output_file_name))


def _init(model, config):
    filename, file_extension = os.path.splitext(model)
    supported_files = ['.so', '.pb']

    if file_extension not in supported_files:
        raise Exception("""
            Unknown file type. Got %s%s.
            Please check the model file (-m).
            Only .pb (protocol buffer), .so (shared object) file is supported.
            """ % (filename, file_extension))

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

    return nn


def _run(nn, image_data):
    # run the graph
    return nn.run(image_data)


def _timerfunc(func, extraArgs, trial=1):
    if sys.version_info.major == 2:
        get_time = time.time
    else:
        get_time = time.perf_counter

    runtime = 0.
    for i in range(trial):
        start = get_time()
        value = func(*extraArgs)
        end = get_time()
        runtime += end - start
        msg = "Function {func} took {time} seconds to complete"
        logger.info(msg.format(func=func.__name__, time=end - start))
    return value, runtime / trial


def run_prediction(input_image, model, config_file, trial=1):
    if not input_image or not model or not config_file:
        logger.error('Please check usage with --help option')
        exit(1)

    config = load_yaml(config_file)

    # load the image
    image_data = load_image(input_image)
    raw_image = image_data

    # initialize Network
    nn = _init(model, config)

    pre_process = build_pre_process(config.PRE_PROCESSOR)
    post_process = build_post_process(config.POST_PROCESSOR)

    results_total = []
    results_pre = []
    results_run = []
    results_post = []

    funcs = [(_pre_process, image_data), _run, _post_process]

    for _ in range(trial):
        # pre process for image
        output, bench_pre = _timerfunc(_pre_process, (image_data, pre_process, config.DATA_FORMAT))

        # run the model to inference
        output, bench_run = _timerfunc(_run, (nn, output))

        # pre process for output
        output, bench_post = _timerfunc(_post_process, (output, post_process))

        results_total.append(bench_pre + bench_run + bench_post)
        results_pre.append(bench_pre)
        results_run.append(bench_run)
        results_post.append(bench_post)

    time_stat = {
        "total": {"mean": np.mean(results_total), "std": np.std(results_total)},
        "pre": {"mean": np.mean(results_pre), "std": np.std(results_pre)},
        "post": {"mean": np.mean(results_post), "std": np.std(results_post)},
        "run": {"mean":  np.mean(results_run), "std": np.std(results_run)},
    }

    # json output
    json_output = JsonOutput(
        task=Tasks(config.TASK),
        classes=config.CLASSES,
        image_size=config.IMAGE_SIZE,
        data_format=config.DATA_FORMAT,
        bench=time_stat,
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
    logger.info("Benchmark avg result(sec) for {} trials".format(trial))
    logger.info("{}".format(time_stat))


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
    default="../models/lib/libdlk_fpga.so",
)
@click.option(
    "-c",
    "--config_file",
    type=click.Path(exists=True),
    help="Config file Path",
)
@click.option(
    "--trial",
    help="# of trial for Benchmark",
    type=click.INT,
    default=1,
)
def main(input_image, model, config_file, trial):
    _check_deprecated_arguments()
    run_prediction(input_image, model, config_file, trial=trial)


def _check_deprecated_arguments():
    argument_list = sys.argv
    if '-l' in argument_list:
        logger.warn("Deprecated warning: -l is deprecated please use -m instead")


if __name__ == "__main__":
    main()
