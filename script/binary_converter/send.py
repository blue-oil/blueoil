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
import re
import glob
import parser
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('restore_path', type=str)
parser.add_argument('--user', type=str, default="root")
parser.add_argument('--host', type=str, default="192.168.1.42")
parser.add_argument('--home_dir', type=str,
                    default="/home/root/works/optical_flow_estimation")
parser.add_argument('-n', '--dry_run', action="store_true")
args = parser.parse_args()


def run():
    cmd_run_scp = "cd {}.prj; ".format(args.restore_path)
    cmd_run_scp += "scp lib_arm.so lib_fpga.so lm_arm.elf lm_fpga.elf "
    cmd_run_scp += "{}@{}:{}; ".format(args.user, args.host, args.home_dir)
    cmd_run_scp += "scp npy/*_images_placeholder:0.npy "
    cmd_run_scp += "{}@{}:{}/images_placeholder.npy; ".format(
        args.user, args.host, args.home_dir)
    cmd_run_scp += "scp npy/*_output:0.npy "
    cmd_run_scp += "{}@{}:{}/output.npy; ".format(
        args.user, args.host, args.home_dir)
    print(cmd_run_scp)

    cmd_run_copy = "cd lmnet/lmnet/networks/optical_flow_estimation/; "
    cmd_run_copy += "scp demo_lib.py demo_so.py demo_server_so.py "
    cmd_run_copy += "flow_to_image.py nn_lib.py "
    cmd_run_copy += "{}@{}:{}/ ".format(
        args.user, args.host, args.home_dir)
    print(cmd_run_copy)

    if not args.dry_run:
        subprocess.run(cmd_run_scp, shell=True)
        subprocess.run(cmd_run_copy, shell=True)


if __name__ == '__main__':
    run()
