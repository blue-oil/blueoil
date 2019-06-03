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
import unittest
import subprocess
import time
import os
from os.path import join

from tstconf import DO_CLEANUP

TEST_LEVEL_FUTURE_TARGET=512
FPGA_HOST = os.environ['FPGA_HOST']


def updated_dict(src_dict, updation) -> dict:
    dst_dict = dict(src_dict)
    dst_dict.update(updation)
    return dst_dict


def run_and_check(command, cwd, file_stdout=None, file_stderr=None, testcase=None,
                  keep_outputs=not DO_CLEANUP,
                  check_stdout_include=None,
                  check_stdout_block=None,
                  check_stderr_include=None,
                  check_stderr_block=None,
                  ignore_returncode=False,
                  **parameters) -> None:
    """
    return true if the command successfully
    asserted
    all words in check_stdout_include should be in stdout
    all words in check_stdout_block should not be in stdout
    all words in check_stderr_include should be in stderr
    all words in check_stderr_block should not be in stderr

    """
    testcase = testcase if testcase is not None else unittest.TestCase()

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        **parameters
    )

    out, err = proc.communicate()

    if keep_outputs and file_stdout:
        with open(file_stdout, 'w') as fout:
            fout.write(out)

    if keep_outputs and file_stderr:
        with open(file_stderr, 'w') as ferr:
            ferr.write(err)

    if len(err.strip()) > 0:
        print("---begining of stderr---")
        print(err)
        print("---end of stderr---")

    try:
        if not ignore_returncode:
            testcase.assertTrue(proc.returncode == 0)

        if check_stdout_include:
            for key in check_stdout_include:
                testcase.assertTrue(key in out)
        if check_stdout_block:
            for key in check_stdout_block:
                testcase.assertFalse(key in out)

        if check_stderr_include:
            for key in check_stderr_include:
                testcase.assertTrue(key in err)
        if check_stderr_block:
            for key in check_stderr_block:
                testcase.assertFalse(key in err)
    except AssertionError:
        print("---begining of stdout---")
        print(out)
        print("---end of stdout---")
        raise AssertionError


def wait_for_device(host: str, tries: int, seconds: int, log_path: str, testcase=None) -> bool:

    board_found = False
    for i in range(tries):
        try:
            print(f'Waiting for device {host}: try {i+1} of {tries}')
            time.sleep(seconds)

            run_and_check(
                ["ping", "-c5", host],
                log_path, join(log_path, "ping.out"), join(log_path, "ping.err"), testcase)

            board_found = True
            break
        except Exception as e:
            print(str(e))
            continue

    return board_found

def setup_de10nano(hw_path: str, output_path: str, testcase=None):

    host = FPGA_HOST
    available = wait_for_device(host, 15, 10, output_path, testcase)
    if not available:
        return False

    try:
        run_and_check(
            [ "ssh",
             "-o",
             "StrictHostKeyChecking no",
             f"root@{host}",
             f"mkdir -p ~/automated_testing; mkdir -p ~/boot; if grep -qs '/root/boot' /proc/mounts ;" \
             + "then echo 0 ; else mount /dev/mmcblk0p1 /root/boot ; fi"
             ],
            output_path,
            join(output_path, "mount.out"),
            join(output_path, "mount.err"),
            testcase
        )

        run_and_check(
            [ "scp",
              join(hw_path, 'soc_system.rbf'),
              join(hw_path, 'soc_system.dtb'),
              join(hw_path, 'preloader-mkpimage.bin'),
             f"root@{host}:~/boot/"
            ],
            output_path,
            join(output_path, "scp_hw.out"),
            join(output_path, "scp_hw.err"),
            testcase
        )

        run_and_check(
            [ "ssh",
              f"root@{host}",
              f"cd ~/boot && dd if=./preloader-mkpimage.bin of=/dev/mmcblk0p3 && sync && cd ~ && umount boot"
            ],
            output_path,
            join(output_path, "update_hw.out"),
            join(output_path, "update_hw.err"),
            testcase
        )

        run_and_check(
            [ "ssh", f"root@{host}", "reboot"],
            output_path,
            join(output_path, "reboot.out"),
            join(output_path, "reboot.err"),
            testcase,
            ignore_returncode=True
        )
    except:
        return False

    available = wait_for_device(host, 15, 10, output_path, testcase)
    if not available:
        return False

    return True

