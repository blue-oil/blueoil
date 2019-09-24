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
from pathlib import Path

import click
import numpy as np
from PIL import Image

from scripts.pylib.nnlib import NNLib as NNLib


def main_test(input_image: str, library: str, expected_output: str,
              max_percent_incorrect_values: float = 0.1, from_npy: bool = False) -> float:
    if not input_image or not library:
        print('Please check usage with --help option')
        exit(1)

    lib = Path(library)
    exp_out = Path(expected_output) if expected_output is not None else None

    # load and initialize the generated shared library
    nn = NNLib()
    nn.load(lib)
    nn.init()
    in_shape = nn.get_input_shape()

    if from_npy:
        data = np.load(input_image)

    else:  # from image
        image_file = Path(input_image)

        # load the image
        with Image.open(image_file).convert("RGB") as img:
            img_resize = img.resize((in_shape[2], in_shape[1]))  # (width, height)

            # convert into numpy array and add the batch dimension
            data = np.asarray(img_resize, dtype=np.float32)

            # apply the preprocessing
            data = data / 255.0

    data = np.expand_dims(data, axis=0)
    if data.shape != nn.get_input_shape():
        print('expected input shape: ' + str(nn.get_input_shape()) + ', provided: ' + str(data.shape))
        exit(1)

    # run the graph
    output = nn.run(data)
    print('Output: ')
    print(output)

    test_result = 'no test data specified'
    retval = 0

    if exp_out:
        eo = np.load(exp_out)
        if eo.shape != nn.get_output_shape():
            test_result = 'expected ouput shape: ' + str(nn.get_output_shape()) + ', provided: ' + str(eo.shape)
        else:
            rtol = atol = 0.0001
            n_failed = eo.size - np.count_nonzero(np.isclose(output, eo, rtol=rtol, atol=atol))
            percent_failed = (n_failed / eo.size) * 100.0
            passed = percent_failed < max_percent_incorrect_values

            string_percent_correct = f'{100.0 - percent_failed:.3f}% of the output values are correct'
            string_percent_failed = f'{percent_failed:.3f}% of the values does not match'

            test_result = string_percent_correct if passed else string_percent_failed

            print('Test: ' + test_result)
            retval = percent_failed
    else:
        print('test')

    return retval


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--input_image",
    type=click.Path(exists=True),
    help="Input image filename",
)
@click.option(
    "-l",
    "--library",
    type=click.Path(exists=True),
    help="Shared library filename",
)
@click.option(
    "-e",
    "--expected_output",
    type=click.Path(exists=True),
    required=False,
    help="Expected output .npy format filename (optional)",
)
@click.option(
    "-n",
    "--from_npy",
    is_flag=True,
    default=False,
    required=False,
    help="Expected output .npy format filename (optional)",
)
def run_test(input_image, library, expected_output, from_npy):
    main_test(input_image, library, expected_output, from_npy=from_npy)


if __name__ == "__main__":
    run_test()
