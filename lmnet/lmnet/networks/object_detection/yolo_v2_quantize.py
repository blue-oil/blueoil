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
from functools import partial

import tensorflow as tf

from lmnet.networks.object_detection.yolo_v2 import YoloV2
from lmnet.networks.base_quantize import BaseQuantize


class YoloV2Quantize(YoloV2, BaseQuantize):

    """Quantize YOLOv2 Network.

    It is based on original YOLO v2.
    """

    def __init__(
            self,
            quantize_first_convolution=True,
            quantize_last_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        """
        Args:
            quantize_first_convolution(bool): use quantization in first conv.
            quantize_last_convolution(bool): use quantization in last conv.
            weight_quantizer (callable): weight quantizer.
            weight_quantize_kwargs(dict): Initialize kwargs for weight quantizer.
            activation_quantizer (callable): activation quantizer
            activation_quantize_kwargs(dict): Initialize kwargs for activation quantizer.
        """

        YoloV2.__init__(
            self,
            *args,
            **kwargs,
        )

        BaseQuantize.__init__(
            self,
            activation_quantizer,
            activation_quantizer_kwargs,
            weight_quantizer,
            weight_quantizer_kwargs,
            quantize_first_convolution,
            quantize_last_convolution,
        )

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        if self.quantize_last_convolution:
            self.before_last_activation = self.activation
        else:
            self.before_last_activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

    def base(self, images, is_training):
        custom_getter = partial(
            self._quantized_variable_getter,
            weight_quantization=self.weight_quantization,
            first_layer_name="block_1/",
            last_layer_name="conv_23/",
            quantize_first_convolution=self.quantize_first_convolution,
            quantize_last_convolution=self.quantize_last_convolution,
            use_histogram=True
        )
        with tf.variable_scope("", custom_getter=custom_getter):
            return super().base(images, is_training)
