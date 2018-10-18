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
import sys
import shutil
from enum import IntEnum
from os import path
import shutil
from logging import INFO, DEBUG

import env
import utils
from utils import make_logger, run, run_and_check


class Target(IntEnum):
    SIM = 1
    SYN = 2
    QSYS = 3
    QUARTUS = 4
    BSP = 5


TARGET_TO_SYMBOL = {
    Target.SIM: 'sim',
    Target.SYN: 'syn',
    Target.QSYS: 'qsys',
    Target.QUARTUS: 'quartus',
    Target.BSP: 'bsp',
}


TARGET_TO_NAME = {
    Target.SIM: 'HLS simulation',
    Target.SYN: 'HLS synthesis',
    Target.QSYS: 'Qsys HDL generation',
    Target.QUARTUS: 'Quartus Compilation',
    Target.BSP: 'BSP configuration',
}


def print_help():
    print('Build Help.')
    for tgt, name in TARGET_TO_NAME.items():
        print(f'{tgt} : {name}')


def clean_logs():
    if path.exists(env.LOGS_DIR):
        shutil.rmtree(env.LOGS_DIR)
    os.mkdir(env.LOGS_DIR)


def build(argv, build_logger):
    required_len_argv = 4

    if len(argv) != required_len_argv:
        print(f'{required_len_argv} arguments should be given, not {len(argv)}.')
        print_help()
        return False

    board_type = argv[1]
    target = {v: k for k, v in TARGET_TO_SYMBOL.items()}.get(argv[2])
    component_file_name = argv[3]
    component_name = 'intel_hls_qconv1x1_impl'

    if not target:
        print(f'Gien 2nd arg {argv[2]} is not supported.')
        print_help()

    src_prj_path = path.join(env.ROOT_DIR, 'projects', board_type)
    prj_name = f'{path.basename(src_prj_path)}.prj'
    PROJECT_DIR = path.join(env.ROOT_DIR, prj_name)

    component_prj = f'{component_file_name}.prj'
    IP_DIR = path.join(env.ROOT_DIR, component_prj)

    if target <= Target.SIM:
        symbol = TARGET_TO_SYMBOL[Target.SIM]
        name = TARGET_TO_NAME[Target.SIM]
        build_logger.info(f'Run {name}.')
        logger = make_logger(symbol, INFO)

        run_and_check('make clean', logger)

        cmd = f'make {symbol} -j{env.NUM_THREADS}'
        run_and_check(cmd, logger)

        cmd = f'./{symbol}.elf random'
        run_and_check(cmd, logger)

    if target <= Target.SYN:
        symbol = TARGET_TO_SYMBOL[Target.SYN]
        name = TARGET_TO_NAME[Target.SYN]
        build_logger.info(f'Run {name}.')
        logger = make_logger(symbol, INFO)

        run_and_check('make clean', logger)
        cmd = f'make {symbol} -j{env.NUM_THREADS}'
        run_and_check(cmd, logger)

        src_path = path.join(env.INTEL_HLS_DIR, component_prj)
        dst_path = path.join(IP_DIR)

        if path.exists(IP_DIR):
            shutil.rmtree(IP_DIR)
        shutil.move(src_path, dst_path)

        # cmd = f'./{symbol}.elf random'
        # run_and_check(cmd, logger)

    if target <= Target.QSYS:
        symbol = TARGET_TO_SYMBOL[Target.QSYS]
        name = TARGET_TO_NAME[Target.QSYS]
        build_logger.info(f'Run {name}.')
        logger = make_logger(symbol, INFO)

        if path.exists(PROJECT_DIR):
            shutil.rmtree(PROJECT_DIR)
        shutil.copytree(src_prj_path, PROJECT_DIR)

        component_dir = path.join(IP_DIR, 'components', component_name)

        qsys_tcl_file = path.join(PROJECT_DIR, 'soc_system.tcl')
        cmd = f'qsys-script --search-path={component_dir}/,$ --script={qsys_tcl_file}'
        run(cmd, logger, cwd=PROJECT_DIR)

        qsys_prj_file = path.join(PROJECT_DIR, 'soc_system.qsys')
        cmd = f'qsys-generate {qsys_prj_file} --search-path={component_dir}/,$ --synthesis=VHDL'
        run(cmd, logger, cwd=PROJECT_DIR)

    bootfiles_dir = path.join(PROJECT_DIR, 'bootfiles')

    if path.exists(bootfiles_dir):
        shutil.rmtree(bootfiles_dir)

    os.mkdir(bootfiles_dir)

    if target <= Target.QUARTUS:
        symbol = TARGET_TO_SYMBOL[Target.QUARTUS]
        name = TARGET_TO_NAME[Target.QUARTUS]
        build_logger.info(f'Run {name}.')
        logger = make_logger(symbol, INFO)

        quartus_prj_file = path.join(PROJECT_DIR, 'DE10_NANO_SoC_GHRD.qpf')
        cmd = f'quartus_sh --flow compile {quartus_prj_file}'
        run(cmd, logger, cwd=PROJECT_DIR)

        sof_file = path.join(PROJECT_DIR, 'output_files',
                             'DE10_NANO_SoC_GHRD.sof')
        rbf_file = path.join(bootfiles_dir, 'soc_system.rbf')
        cmd = f'quartus_cpf -c -o bitstream_compression=on {sof_file} {rbf_file}'
        run(cmd, logger, cwd=PROJECT_DIR)

    if target <= Target.BSP:
        symbol = TARGET_TO_SYMBOL[Target.QUARTUS]
        name = TARGET_TO_NAME[Target.QUARTUS]
        build_logger.info(f'Run {name}.')
        logger = make_logger(symbol, INFO)

        bsp_build_dir = path.join(PROJECT_DIR, 'bsp_build_dir')
        if path.exists(bsp_build_dir):
            shutil.rmtree(bsp_build_dir)
        os.mkdir(bsp_build_dir)

        hps_dir = path.join(PROJECT_DIR, 'hps_isw_handoff', 'soc_system_hps_0')
        cmd = f'bsp-create-settings --type spl --bsp-dir {bsp_build_dir} '
        '--settings settings.bsp --preloader-settings-dir {hps_dir}'
        run(cmd, logger, cwd=PROJECT_DIR)

        run(f'make -j{env.NUM_THREADS}', logger, cwd=bsp_build_dir)

        preloader_file_name = 'preloader-mkpimage.bin'
        preloader_file = path.join(bsp_build_dir, preloader_file_name)
        shutil.move(preloader_file, path.join(
            bootfiles_dir, preloader_file_name))

        uboot_file_name = 'u-boot.img'
        uboot_file = path.join(bsp_build_dir, 'uboot-socfpga', uboot_file_name)
        run('make clean', logger)
        run(f'make uboot -j{env.NUM_THREADS}', logger, cwd=bsp_build_dir)
        shutil.move(uboot_file, path.join(bootfiles_dir, uboot_file_name))

    # temporary
    return True


if __name__ == '__main__':
    clean_logs()

    build_logger = utils.make_logger('build', INFO)

    print('-------------------------------------')
    build_logger.info('Start.')
    print('-------------------------------------')

    return_flag = build(sys.argv, build_logger)

    print('-------------------------------------')
    if return_flag:
        build_logger.info('Succeeded!!!')
    else:
        build_logger.info('Failed...')
    print('-------------------------------------')
