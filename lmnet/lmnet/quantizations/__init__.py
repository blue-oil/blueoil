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
r"""This Package provides a set of quantizations.

How to use:

.. code-block:: python

    quantizer = binary_mean_scaling_quantizer() # initialize quantizer
    weights = tf.get_variable("kernel", shape=[1, 2, 3, 4]) # prepare variavle.
    quantized_weights = quantizer(weights) # use quantizer to quantize variable
    tf.nn.conv2d(inputs, quantized_weights)

Methods:
    linear_mid_tread_half_quantizer
    binary_mean_scaling_quantizer
    binary_channel_wise_mean_scaling_quantizer
    ttq_weight_quantizer
    twn_weight_quantizer
"""

from .linear import linear_mid_tread_half_quantizer
from .binary import binary_mean_scaling_quantizer, binary_channel_wise_mean_scaling_quantizer
from .ternary import ttq_weight_quantizer, twn_weight_quantizer

__all__ = [
    'linear_mid_tread_half_quantizer',
    'binary_mean_scaling_quantizer',
    'binary_channel_wise_mean_scaling_quantizer',
    'ttq_weight_quantizer',
    'twn_weight_quantizer',
]
