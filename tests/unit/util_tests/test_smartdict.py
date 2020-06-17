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
import pytest

from blueoil.utils.smartdict import SmartDict


def test_init():
    d = SmartDict(a=10, b=20, c=30)
    assert isinstance(d, SmartDict)
    assert d.a == 10
    assert d.b == 20
    assert d.c == 30
    assert d["a"] == 10
    assert d["b"] == 20
    assert d["c"] == 30


def test_init_with_dict():
    d = SmartDict({"a": 10, "b": 20, "c": 30})
    assert isinstance(d, SmartDict)
    assert d.a == 10
    assert d.b == 20
    assert d.c == 30
    assert d["a"] == 10
    assert d["b"] == 20
    assert d["c"] == 30


def test_setitem():
    d = SmartDict()
    d["a"] = [
        100,
        {"a": 10, "b": 20, "c": 30},
    ]
    assert d.a[0] == 100
    assert isinstance(d.a[1], SmartDict)
    assert d.a[1].a == 10
    assert d.a[1].b == 20
    assert d.a[1].c == 30

    d["b"] = {"a": 10, "b": 20, "c": 30}
    assert isinstance(d.b, SmartDict)
    assert d.b.a == 10
    assert d.b.b == 20
    assert d.b.c == 30


def test_getattr():
    d = SmartDict(a=10, b=20, c=30)
    assert d.a == 10
    assert d.b == 20
    assert d.c == 30
    with pytest.raises(AttributeError):
        d.d


def test_setattr():
    d = SmartDict()
    d.a = 10
    d.b = 20
    d.c = 30
    assert d["a"] == 10
    assert d["b"] == 20
    assert d["c"] == 30
