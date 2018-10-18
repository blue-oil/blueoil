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
import shlex
import subprocess
from pathlib import Path
import shutil
from logging import getLogger, FileHandler, StreamHandler, INFO, DEBUG

import env


class RunFailedException(Exception):
    pass


def delete_dir(dir_path): 
    shutil.rmtree(dir_path) 


def make_dirs(dir_pathes):
    if isinstance(dir_pathes, str):
        dir = Path(dir_pathes)
        Path.mkdir(dir, parents=True, exist_ok=True)

    elif isinstance(dir_pathes, list):
        for dir_path in dir_pathes:
            dir = Path(dir_path)
            Path.mkdir(dir, parents=True, exist_ok=True)


def make_logger(name, level=INFO):
    fheader = FileHandler(f'logs/{name}.log')
    fheader.setLevel(level)
    sheader = StreamHandler()
    sheader.setLevel(level)
    logger = getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fheader)
    logger.addHandler(sheader)
    return logger


def read_process_io(proc, logger):
    while True:
        output = proc.stdout.readline().decode().replace('\n', '')
        if output:
            logger.info(output)
        else:
            break


def run(cmd, logger, cwd=env.ROOT_DIR):
    logger.info(f'Call: {cmd}')
    try:
        my_env = os.environ.copy()
        command = shlex.split(cmd)
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=my_env)
        while proc.poll() is None:
            read_process_io(proc, logger)
        return proc.returncode
    except KeyboardInterrupt:
        logger.warning(f'Forcely Stop...')


def run_and_check(cmd, logger, cwd=env.ROOT_DIR):
    ret = run(cmd, logger, cwd)
    if ret == 0:
        logger.info(f'Success: "{cmd}".')
        return True
    else:
        logger.info(f'Error: "{cmd}".')
        raise RunFailedException
