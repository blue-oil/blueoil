
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
import logging
import signal


class TerminateProtected:
    """ Protect a piece of code from being killed by SIGINT or SIGTERM.
    It can still be killed by a force kill.

    Example:
        with TerminateProtected():
            run_func_1()
            run_func_2()

    Both functions will be executed even if a sigterm or sigkill has been received.
    """
    killed = False

    def _handler(self, signum, frame):
        logging.error("Received SIGINT or SIGTERM! Finishing this block, then exiting.")
        self.killed = True

    def __enter__(self):
        self.old_sigint = signal.signal(signal.SIGINT, self._handler)
        self.old_sigterm = signal.signal(signal.SIGTERM, self._handler)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_sigint)
        signal.signal(signal.SIGTERM, self.old_sigterm)
