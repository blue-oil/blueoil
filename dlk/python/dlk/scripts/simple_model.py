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

from core.model import Model
from core.operators import *


def make_simple_model() -> Model:
    """Create a simple model with a simple graph."""
    model = Model()
    graph = model.graph

    node_names = [
        'conv1_kernel',
        'conv1_bn_beta',
        'conv1_bn_gamma',
        'conv1_bn_mean',
        'conv1_bn_variance',

        'conv2_kernel',
        'conv2_bn_beta',
        'conv2_bn_gamma',
        'conv2_bn_mean',
        'conv2_bn_variance',
    ]

    data = {}
    for node_name in node_names:
        fname = os.path.join(os.path.dirname(__file__), 'simple_model_data/' + node_name + '.npy')
        data[node_name] = np.load(fname)

    c1_in = Input(
        'conv1_input',
        [1, 32, 32, 3],
        Float32(),

    )

    c1_ker = Constant(
        'conv1_kernel',
        Float32(),
        data['conv1_kernel'],  # .transpose([3, 0, 1, 2]),
        dimension_format='HWOI'
    )

    c1_qtz_ker = QTZ_binary_mean_scaling(
        'conv1_qtz_kernel',
        [3, 3, 3, 32],
        # [32, 3, 3, 3],
        Float32(),
        {'input': c1_ker},
        dimension_format='HWOI'
    )

    conv1 = Conv(
        'conv1',
        [1, 32, 32, 32],
        Float32(),
        {'X': c1_in, 'W': c1_qtz_ker},
        pads=[1, 1, 1, 1],
        dimension_format='NHWC'
    )

    c1_bn_beta = Constant(
        'conv1_bn_beta',
        Float32(),
        data['conv1_bn_beta'],
        dimension_format='C'
    )

    c1_bn_gamma = Constant(
        'conv1_bn_gamma',
        Float32(),
        data['conv1_bn_gamma'],
        dimension_format='C'
    )

    c1_bn_mean = Constant(
        'conv1_bn_mean',
        Float32(),
        data['conv1_bn_mean'],
        dimension_format='C'
    )

    c1_bn_variance = Constant(
        'conv1_bn_variance',
        Float32(),
        data['conv1_bn_variance'],
        dimension_format='C'
    )

    c1_bn = BatchNormalization(
        'conv1_bn',
        [1, 32, 32, 32],
        Float32(),
        {'X': conv1, 'scale': c1_bn_gamma, 'B': c1_bn_beta, 'mean': c1_bn_mean, 'var': c1_bn_variance},
        epsilon=0.001
    )

    c1_bn_qtz_param1 = Constant(
        'conv1_bn_qtz_param1',
        Int32(),
        np.array([2]),
        dimension_format='1'
    )

    c1_bn_qtz_param2 = Constant(
        'conv1_bn_qtz_param2',
        Float32(),
        np.array([2]),
        dimension_format='1'
    )

    c1_bn_qtz = QTZ_linear_mid_tread_half(
        'conv1_bn_qtz',
        [1, 32, 32, 32],
        Float32(),
        {'X': c1_bn, 'Y': c1_bn_qtz_param1, 'Z': c1_bn_qtz_param2}
    )

    c2_ker = Constant(
        'conv2_kernel',
        Float32(),
        data['conv2_kernel'],  # .transpose([3, 0, 1, 2]),
        dimension_format='HWOI'
    )

    c2_qtz_ker = QTZ_binary_mean_scaling(
        'conv2_qtz_kernel',
        # [64, 3, 3, 32],
        [3, 3, 32, 64],
        Float32(),
        {'input': c2_ker},
        dimension_format='HWOI'
    )

    conv2 = Conv(
        'conv2',
        [1, 32, 32, 64],
        Float32(),
        {'X': c1_bn_qtz, 'W': c2_qtz_ker},
        pads=[1, 1, 1, 1],
        dimension_format='NHWC'
    )

    c2_bn_beta = Constant(
        'conv2_bn_beta',
        Float32(),
        data['conv2_bn_beta'],
        dimension_format='C'
    )

    c2_bn_gamma = Constant(
        'conv2_bn_gamma',
        Float32(),
        data['conv2_bn_gamma'],
        dimension_format='C'
    )

    c2_bn_mean = Constant(
        'conv2_bn_mean',
        Float32(),
        data['conv2_bn_mean'],
        dimension_format='C'
    )

    c2_bn_variance = Constant(
        'conv2_bn_variance',
        Float32(),
        data['conv2_bn_variance'],
        dimension_format='C'
    )

    c2_bn = BatchNormalization(
        'conv2_bn',
        [1, 32, 32, 64],
        Float32(),
        {'X': conv2, 'scale': c2_bn_gamma, 'B': c2_bn_beta, 'mean': c2_bn_mean, 'var': c2_bn_variance},
        epsilon=0.001
    )

    c2_bn_qtz_param1 = Constant(
        'conv2_bn_qtz_param1',
        Int32(),
        np.array([2]),
        dimension_format='1'
    )

    c2_bn_qtz_param2 = Constant(
        'conv2_bn_qtz_param2',
        Float32(),
        np.array([2]),
        dimension_format='1'
    )

    c2_bn_qtz = QTZ_linear_mid_tread_half(
        'conv2_bn_qtz',
        [1, 32, 32, 64],
        Float32(),
        {'X': c2_bn, 'Y': c2_bn_qtz_param1, 'Z': c2_bn_qtz_param2}
    )

    # One output
    y = Output(
        'output',
        [1, 32, 32, 64],
        Float32(),
        {'input': c2_bn_qtz}
    )

    # add ops to the graph
    graph.add_op_and_inputs(y)

    return model
