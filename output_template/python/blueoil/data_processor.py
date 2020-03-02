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
from abc import ABCMeta, abstractmethod

import numpy as np
import six


class Sequence:
    """Sequence several processor together.

    Args:
        processors (List[Processor]): list of processor.

    Examples:
        | *Sequence([*
        |     *FlipLeftRight(0.5),*
        |     *Hue((-10, 10)),*
        | *])*
    """

    def __init__(self, processors):
        self.processors = processors

    def __call__(self, **kwargs):
        for processor in self.processors:
            kwargs = processor(**kwargs)
        return kwargs

    def __repr__(self):
        return pprint.saferepr(self.processors)

    # TODO(wakisaka): Should create interface class to set image size for processor child class.
    def set_image_size(self, image_size):
        """Override processors image size

        Args:
            image_size(tuple): (height, width)
        """

        # Avoid circular import
        from blueoil.pre_processor import Resize, ResizeWithGtBoxes, ResizeWithMask, LetterBoxes
        from blueoil.post_processor import FormatYoloV2

        for process in self.processors:
            class_list = (Resize, ResizeWithGtBoxes, ResizeWithMask, LetterBoxes)
            if isinstance(process, class_list):
                process.size = image_size

            if isinstance(process, FormatYoloV2):
                process.image_size = image_size


@six.add_metaclass(ABCMeta)
class Processor():

    @abstractmethod
    def __call__(self, **kwargs):
        """Call processor method for each a element of data.

        Return image and labels etc.
        """
        return kwargs

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dict__)


# TODO(wakisaka): move to somewhere.
def binarize(labels, num_classes):
    """Return numpy array binarized labels."""
    targets = np.array(labels).reshape(-1)
    one_hot = np.eye(num_classes)[targets]
    return one_hot
