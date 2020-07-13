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


class SmartDict(dict):
    def __init__(self, d=None, **kwargs):
        super(SmartDict, self).__init__()
        self.update(d, **kwargs)

    def update(self, d=None, **kwargs):
        d = d or {}
        d.update(kwargs)
        for key, value in d.items():
            self[key] = value

    def __setitem__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [
                self.__class__(x) if isinstance(x, dict) else x
                for x in value
            ]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(SmartDict, self).__setitem__(name, value)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            self.__class__.__name__, name,
        ))

    def __setattr__(self, name, value):
        self[name] = value

    def __dir__(self):
        parent = super(SmartDict, self).__dir__()
        attrs = parent + list(self.keys())
        return sorted(attrs)
