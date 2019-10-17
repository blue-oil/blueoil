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
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from glob import glob
from unittest import TestCase

from tstconf import DO_CLEANUP, DO_CLEANUP_OLDBUILD, FPGA_FILES, HOURS_ELAPSED_TO_ERASE, PROJECT_TAG
from tstutils import setup_de10nano

SECOND_PER_HOUR = 3600


def rmdir(path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


class TestCaseDLKBase(TestCase):
    """
    This is base class TestCase which have
    Setup and TearDown method which is common in dlk project.
    """
    build_dir = None
    fpga_setup = False

    @classmethod
    def setUpClass(cls):
        if not cls.fpga_setup:
            return

        # Setup the board. For now, DE10 Nano board
        output_path = '/tmp'
        hw_path = os.path.abspath(os.path.join('..', FPGA_FILES))

        board_available = setup_de10nano(hw_path, output_path)

        if not board_available:
            raise Exception('Not FPGA found: cannot test')

    def setUp(self) -> None:
        """
        Cleanup old build directory and make build directory
        """

        prefix0 = "-".join(["test", PROJECT_TAG])

        if DO_CLEANUP_OLDBUILD:
            dirnames = glob(tempfile.gettempdir() + '/' + prefix0 + '*')
            for dirname in dirnames:
                second_erapsed = time.time() - os.stat(dirname).st_mtime
                if second_erapsed > SECOND_PER_HOUR * HOURS_ELAPSED_TO_ERASE:
                    rmdir(dirname)
                    print(f'Old directory {dirname} deleted')

        datetimetag = datetime.now().strftime("%Y%m%d%H%M")
        classtag = self.__class__.__name__

        prefix = "-".join([prefix0, classtag, datetimetag]) + "-"
        self.build_dir = tempfile.mkdtemp(prefix=prefix)

    def tearDown(self) -> None:
        if DO_CLEANUP:
            rmdir(self.build_dir)
