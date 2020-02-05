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
"""Configuration module."""
import os

import env
from core.data_types import Uint32


class Config(object):
    """Class of a collection of configurations."""

    def __init__(self,
                 activate_hard_quantization=False,
                 threshold_skipping=False,
                 placeholder_dtype=float,
                 default_qword_dtype=Uint32,
                 test_dir=None,
                 optimized_pb_path=None,
                 output_pj_path=None,
                 debug: bool = False,
                 cache_dma: bool = False
                 ) -> None:
        """Init the config object."""
        self.activate_hard_quantization: bool = activate_hard_quantization
        self.threshold_skipping: bool = threshold_skipping and activate_hard_quantization
        self.default_qword_dtype = default_qword_dtype
        self.test_dir: str = test_dir
        self.optimized_pb_path: str = optimized_pb_path
        self.output_pj_path: str = output_pj_path
        self.__debug: bool = debug
        self.__cache_dma: bool = cache_dma

    @property
    def pre_processor(self) -> str:
        return 'DivideBy255'

    @property
    def cpu_count(self) -> int:
        return len(os.sched_getaffinity(0))

    @property
    def max_cpu_count(self) -> int:
        return env.MAX_CPU_COUNT

    @property
    def debug(self) -> bool:
        return self.__debug

    @property
    def cache(self) -> bool:
        return self.__cache_dma
