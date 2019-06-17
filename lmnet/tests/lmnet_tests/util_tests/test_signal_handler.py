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
import os
import time

from lmnet.utils.signal_handler import SignalHandler
from signal import SIGTERM


def test_signal_handler():
    signalhandler = SignalHandler()
    pid = os.getppid()

    _trigger_signal(pid)

    assert signalhandler.receivedTermSignal


def _trigger_signal(pid):
    time.sleep(1)
    os.kill(pid, SIGTERM)
