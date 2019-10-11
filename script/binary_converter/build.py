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
import yaml
import parser
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('restore_path', type=str)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument(
    '--target', type=str, nargs="+",
    default=["x86_avx", "arm", "fpga"])
parser.add_argument('-n', '--dry_run', action="store_true")
args = parser.parse_args()


class YamlLoader(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


YamlLoader.add_constructor(None, YamlLoader.ignore_unknown)


def run():
    parsed_str = re.match(
        ".*saved/(.+)/checkpoints/(save.ckpt-[0-9]+)", args.restore_path)
    experiment_id, ckpt_id = parsed_str[1], parsed_str[2]

    with open("saved/{}/config.yaml".format(experiment_id)) as f:
        config = yaml.load(f, Loader=YamlLoader)
    image_size = "{}x{}".format(*config["IMAGE_SIZE"])

    cmd_to_pb = "CUDA_VISIBLE_DEVICES={} ".format(args.device_id)
    cmd_to_pb += "python lmnet/executor/export.py -i {} ".format(experiment_id)
    cmd_to_pb += "--restore_path {} ".format(args.restore_path)
    print(cmd_to_pb)

    cmd_dlk = "cd dlk; "
    cmd_dlk += "PYTHONPATH=python/dlk "
    cmd_dlk += "python python/dlk/scripts/generate_project.py "
    cmd_dlk += "-i ../saved/{}/export/{}/{}/minimal_graph_with_shape.pb ".format(
        experiment_id, ckpt_id, image_size)
    cmd_dlk += "-o ../tmp/ -p {} -hq -ts ".format(experiment_id)
    # cmd_dlk += "-o ../tmp/ -p {} -dbg ".format(experiment_id)
    print(cmd_dlk)

    cmd_make_so = "cd tmp/{}.prj/; ".format(experiment_id)
    cmd_make_so += "export OMP_NUM_THREADS=20; "
    cmd_make_so += "for target in {}; do make clear; make -j8 lib_$target; make clear; make -j8 lm_$target; done".format(
        " ".join(args.target))
    print(cmd_make_so)

    # cmd_make_profile = "cd tmp/{}.prj/; ".format(experiment_id)
    # cmd_make_profile += "mkdir build; "
    # cmd_make_profile += "cd build; "
    # cmd_make_profile += "cmake .. -DUSE_AVX=1; "
    # cmd_make_profile += "make -j8 lm "
    # print(cmd_make_profile)

    cmd_copy_npy = "rm -r ./tmp/{}.prj/npy; ".format(experiment_id)
    cmd_copy_npy += "cp -R ./saved/{}/export/{}/{}/inference_test_data ".format(
        experiment_id, ckpt_id, image_size)
    cmd_copy_npy += "./tmp/{}.prj/npy; ".format(experiment_id)
    print(cmd_copy_npy)

    cmd_copy_pb = "cp ./saved/{}/export/{}/{}/minimal_graph_with_shape.pb ".format(
        experiment_id, ckpt_id, image_size)
    cmd_copy_pb += "./tmp/{}.prj/ ".format(experiment_id)
    print(cmd_copy_pb)

    if not args.dry_run:
        subprocess.run(cmd_to_pb, shell=True)
        subprocess.run(cmd_dlk, shell=True)
        subprocess.run(cmd_make_so, shell=True)
        # subprocess.run(cmd_make_profile, shell=True)
        subprocess.run(cmd_copy_npy, shell=True)
        subprocess.run(cmd_copy_pb, shell=True)


if __name__ == '__main__':
    run()
