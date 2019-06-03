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
from typing import List
import numpy as np


def quantized_packed_type(t):
    return f"QuantizedPacked<{t}>"


class DataType(object):

    def __str__(self) -> str:
        return type(self).__name__

    @classmethod
    def pytype(cls):
        raise NotImplementedError

    @classmethod
    def cpptype(cls):
        raise NotImplementedError

    @classmethod
    def nptype(cls):
        raise NotImplementedError

    @classmethod
    def name(cls):
        return cls.__name__

    def __eq__(self, other):
        return type(self).__name__ == type(other).__name__


class Primitive(DataType):
    @classmethod
    def is_primitive(cls):
        True


class Special(DataType):
    @classmethod
    def is_primitive(cls):
        False


class Int(Special, int):
    @classmethod
    def pytype(cls):
        return int

    @classmethod
    def cpptype(cls):
        return 'int'


class UInt(Special, int):
    @classmethod
    def cpptype(cls):
        return 'unsigned'


class Int8(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'int8_t'

    @classmethod
    def nptype(cls):
        return np.int8


class Uint8(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'uint8_t'

    @classmethod
    def nptype(cls):
        return np.uint8


class Int16(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'int16_t'

    @classmethod
    def nptype(cls):
        return np.int16


class Uint16(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'uint16_t'

    @classmethod
    def nptype(cls):
        return np.uint8


class Int32(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'int32_t'

    @classmethod
    def nptype(cls):
        return np.int32


class Uint32(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'uint32_t'

    @classmethod
    def nptype(cls):
        return np.uint32


class PackedUint32(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'QuantizedPacked<uint32_t>'

    @classmethod
    def nptype(cls):
        return np.uint32


class Int64(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'int64_t'

    @classmethod
    def nptype(cls):
        return np.int64


class Uint64(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'uint64_t'

    @classmethod
    def nptype(cls):
        return np.uint64


class PackedUint64(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'QuantizedPacked<uint64_t>'

    @classmethod
    def nptype(cls):
        return np.uint64


class Float(Special, float):
    @classmethod
    def pytype(cls):
        return float

    @classmethod
    def cpptype(cls):
        return 'float'


class Float32(Primitive, float):
    @classmethod
    def cpptype(cls):
        return 'float'

    @classmethod
    def nptype(cls):
        return np.float32


class Float64(Primitive, float):
    @classmethod
    def cpptype(cls):
        return 'double'

    @classmethod
    def nptype(cls):
        return np.float64


class String(DataType):
    @classmethod
    def cpptype(cls):
        return 'std::string'


class Void(DataType):
    @classmethod
    def pytype(cls):
        return None

    @classmethod
    def cpptype(cls):
        return 'void'


class Any(DataType):
    @classmethod
    def cpptype(cls):
        return 'std::any'


class Bool(DataType):
    @classmethod
    def cpptype(cls):
        return 'bool'

    @classmethod
    def nptype(cls):
        return np.bool_


class Size(DataType, int):
    @classmethod
    def cpptype(cls):
        return 'size_t'


class Shape(DataType, List[Size]):
    @classmethod
    def cpptype(cls):
        return 'shape_t'


class Vec(DataType, List[Primitive]):
    @classmethod
    def cpptype(cls):
        return 'vec_t'


class QUANTIZED_NOT_PACKED(Primitive, int):
    @classmethod
    def cpptype(cls):
        return 'QUANTIZED_NOT_PACKED'

    @classmethod
    def nptype(cls):
        return np.int8
