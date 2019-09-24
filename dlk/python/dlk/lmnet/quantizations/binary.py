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

from .quantizer import Quantizer


class BinaryQuantizer(Quantizer):
    def __init__(self) -> None:
        super().__init__(1)


class binary_mean_scaling_quantizer(BinaryQuantizer):
    def __init__(self):
        super().__init__()

    def post(self, x: np.ndarray) -> np.ndarray:
        return np.mean(np.absolute(x)).astype(np.float32)

    def pre(self, x: np.ndarray) -> np.ndarray:
        y = np.sign(x)
        y[y < 0] = 0
        return y.astype(np.int32)


class binary_channel_wise_mean_scaling_quantizer(BinaryQuantizer):
    def __init__(self):
        super().__init__()

    def post(self, x: np.ndarray) -> np.ndarray:
        return np.mean(np.absolute(x), axis=(1, 2, 3)).astype(np.float32)

    def pre(self, x: np.ndarray) -> np.ndarray:
        y = np.sign(x)
        y[y < 0] = 0
        return y.astype(np.int32)
