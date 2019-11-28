# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
from .base import Operator
from .convpool import Conv, Pool, MaxPool, AveragePool
from .math import Add, Gemm, Mul, Maximum, MatMul, Minimum
from .misc import Identity, BatchNormalization, Dropout, Gather, Unique, Cast, Prod, BatchNormalizationOptimized
from .output import Softmax, Relu, LeakyRelu
from .quantization import Quantizer, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half,\
    QTZ_binary_channel_wise_mean_scaling, Lookup
from .shape import SpaceToDepth, Transpose, Reshape, Flatten, ConcatOnDepth, DepthToSpace, ResizeNearestNeighbor, \
    Split, Pad, Shape, StridedSlice
from .variable import Variable, Input, Constant, Output
