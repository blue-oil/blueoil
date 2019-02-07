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
import sys
from multiprocessing import Pool


class PoolInterface:
    """Multiprocessing interface
    This is an API Wrapper to support multiprocessing in python2.7 and python3.5
    """

    def __init__(self):
        """
        Args:
            is_python_three(boolean): Checks if the python version is 3.
            pool(Pool): The initialized  Pool process with 1 worker
        """

        self.is_python_three = (sys.version_info.major == 3)

        if self.is_python_three:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=1)
        else:
            self.pool = Pool(processes=1)

    def run(self, call_function, parameter):
        """Run specifc function on 1 process

        Args:
            outputs: The Process of the called function.

        Returns:
            pool_result(Pool): Pool process result class
        """

        if self.is_python_three:
            pool_result = self.pool.submit(call_function, (parameter))
        else:
            pool_result = self.pool.apply_async(call_function, (parameter, ))

        return pool_result

    def is_done(self, pool_result):
        return pool_result.done() if self.is_python_three else pool_result.ready()

    def get_results(self, pool_result):
        return pool_result.result() if self.is_python_three else pool_result.get()
