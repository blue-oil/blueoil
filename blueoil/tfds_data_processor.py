#!/usr/bin/env python3
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
import pprint
import tensorflow as tf


class TFDSProcessorSequence:
    """Sequence several processor together.

    Args:
        processors (List[Processor]): list of processor.

    Examples:
        | *TFDSProcessorSequence([*
        |     *TFDSFlipLeftRight(0.5),*
        |     *TFDSHue((-10, 10)),*
        | *])*
    """

    def __init__(self, processors):
        self.processors = processors

    @tf.function
    def __call__(self, **kwargs):
        for processor in self.processors:
            kwargs = processor(**kwargs)
        return kwargs

    def __repr__(self):
        return pprint.saferepr(self.processors)
