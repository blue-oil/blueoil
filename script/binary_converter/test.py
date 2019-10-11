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
parser.add_argument('--host', type=str, default="lmdgx01")
parser.add_argument('--target', type=str, default="lm_x86_avx")
parser.add_argument('--home_dir', type=str, default="/home/ki42/works/blueoil")
parser.add_argument('-n', '--dry_run', action="store_true")
args = parser.parse_args()


def run():
    cmd_download_prj = "rsync -ravz -e ssh "
    # cmd_download_prj += "--update --exclude='*.o' --exclude='*.cpp' "
    cmd_download_prj += "--update --exclude='*.o' "
    cmd_download_prj += "{}:{}/{}.prj tmp/".format(
        args.host, args.home_dir, args.restore_path
    )
    print(cmd_download_prj)

    cmd_run_test = "cd {}.prj/; ".format(args.restore_path)
    cmd_run_test += "./{}.elf ".format(args.target)
    cmd_run_test += "./npy/*images_placeholder:0.npy "
    cmd_run_test += "./npy/*output:0.npy"
    print(cmd_run_test)

    if not args.dry_run:
        subprocess.run(cmd_download_prj, shell=True)
        subprocess.run(cmd_run_test, shell=True)


if __name__ == '__main__':
    run()
