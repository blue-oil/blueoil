
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
import signal


class SignalHandler:
    """ Protect a piece of code from being killed by SIGINT or SIGTERM.
    It can still be killed by a force kill.

    Both functions will be executed even if a sigterm or sigkill has been received.
    """

    def __init__(self):
        self.receivedTermSignal = False
        signal.signal(signal.SIGTERM, self.handler)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, signum, frame):
        self.lastSignal = signum
        self.receivedTermSignal = True
