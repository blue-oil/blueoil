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
from lmnet.networks.quantize_param_init import QuantizeParamInit


class YoloV2Quantize(QuantizeParamInit, YoloV2):
    """Quantize YOLOv2 Network.

    It is based on original YOLO v2.
    QuantizeParamInit is a mixin class used to initialize variables for quantization and custom_getter.

    YoloV2 does not use lmnet_block, need to define a scope for custom_getter in base function.
    """

    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)
