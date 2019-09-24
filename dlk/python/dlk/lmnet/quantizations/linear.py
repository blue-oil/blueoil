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
from numpy import ndarray

from .quantizer import Quantizer


class LinearQuantizer(Quantizer):
    pass


class linear_mid_tread_half_quantizer(LinearQuantizer):
    def __init__(self, bit: int, max_value: int) -> None:
        super().__init__(bit)
        self.n = float(2 ** self.bit - 1)
        self.max = max_value
        self.min = 0
        self.range = self.max - self.min

    def post(self, x: ndarray) -> ndarray:
        scaled = x / self.n * self.range
        return scaled.astype(np.float32)

    def pre(self, x: ndarray) -> ndarray:
        x = np.clip(x, self.min, self.max)
        shifted = (x - self.min) / self.range
        return np.round(shifted * self.n).astype(np.int32)
