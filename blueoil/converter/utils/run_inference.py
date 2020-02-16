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
from sys import argv

import numpy as np
from PIL import Image

from blueoil.converter.nnlib import NNLib as NNLib

input_image_path = argv[1]
lib_path = argv[2]


if __name__ == '__main__':
    # load and initialize the generated shared library
    nn = NNLib()
    nn.load(lib_path)
    nn.init()

    # load an image
    img = Image.open(input_image_path).convert('RGB')
    img.load()

    data = np.asarray(img, dtype=np.float32)
    data = np.expand_dims(data, axis=0)

    # apply the preprocessing, which is DivideBy255 in this case
    data = data / 255.0

    # run the graph and show output
    output = nn.run(data)
    print(f'Output: {output}')
