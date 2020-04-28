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

from blueoil.cmd.export import run as run_export
from blueoil.converter.generate_project import run as run_generate_project


def create_output_directory(output_root_dir, output_template_dir=None):
    """Create output directory from template.

    Args:
        output_root_dir:
        output_template_dir:  (Default value = None)

    Returns:

    """

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_output_template_dir = os.environ.get(
        "OUTPUT_TEMPLATE_DIR",
        os.path.join(os.path.dirname(base_dir), "output_template"),
    )
    template_dir = env_output_template_dir if not output_template_dir else output_template_dir

    # Recreate output_root_dir from template
    if os.path.exists(output_root_dir):
        shutil.rmtree(output_root_dir)
    shutil.copytree(template_dir, output_root_dir, symlinks=False, copy_function=shutil.copy)
    # Create output directories
    output_directories = get_output_directories(output_root_dir)
    for _, output_dir in output_directories.items():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return output_directories


def get_output_directories(output_root_dir):
    """

    Args:
        output_root_dir:

    Returns:

    """

    model_dir = os.path.join(output_root_dir, "models")
    library_dir = os.path.join(model_dir, "lib")
    output_directories = dict(
        root_dir=output_root_dir,
        model_dir=model_dir,
        library_dir=library_dir,
    )
    return output_directories


def strip_binary(output):
    """Strip binary file.

    Args:
        output:

    """

    if output in {"lm_x86.elf", "lm_x86_avx.elf"}:
        subprocess.run(("strip", output))
    elif output in {"libdlk_x86.so", "libdlk_x86_avx.so"}:
        subprocess.run(("strip", "-x", "--strip-unneeded", output))
    elif output in {"lm_arm.elf", "lm_fpga.elf"}:
        subprocess.run(("arm-linux-gnueabihf-strip", output))
    elif output in {"libdlk_arm.so", "libdlk_fpga.so"}:
        subprocess.run(("arm-linux-gnueabihf-strip", "-x", "--strip-unneeded", output))


def make_all(project_dir, output_dir):
    """Make each target.

    Args:
        project_dir (str): Path to project directory
        output_dir (str): Path to output directory

    """

    make_list = [
        ["ARCH=x86 TYPE=executable", "lm_x86.elf"],
        ["ARCH=x86_avx TYPE=executable", "lm_x86_avx.elf"],
        ["ARCH=arm TYPE=executable", "lm_arm.elf"],
        ["ARCH=fpga TYPE=executable", "lm_fpga.elf"],
        ["ARCH=aarch64 TYPE=executable", "lm_aarch64.elf"],
        ["ARCH=x86 TYPE=dynamic", "libdlk_x86.so"],
        ["ARCH=x86_avx TYPE=dynamic", "libdlk_x86_avx.so"],
        ["ARCH=arm TYPE=dynamic", "libdlk_arm.so"],
        ["ARCH=fpga TYPE=dynamic", "libdlk_fpga.so"],
        ["ARCH=aarch64 TYPE=dynamic", "libdlk_aarch64.so"],
        ["ARCH=x86 TYPE=static", "libdlk_x86.a"],
        ["ARCH=x86_avx TYPE=static", "libdlk_x86_avx.a"],
        ["ARCH=arm TYPE=static", "libdlk_arm.a"],
        ["ARCH=fpga TYPE=static", "libdlk_fpga.a"],
        ["ARCH=aarch64 TYPE=static", "libdlk_aarch64.a"],
    ]
    output_dir = os.path.abspath(output_dir)
    running_dir = os.getcwd()
    # Change current directory to project directory
    os.chdir(project_dir)

    # Make each target and move output files
    for target, output in make_list:
        subprocess.run(("make", "clean", "--quiet"))
        subprocess.run(("make", "build", target, "-j4", "--quiet"))
        strip_binary(output)
        output_file_path = os.path.join(output_dir, output)
        os.rename(output, output_file_path)
    # Return running directory
    os.chdir(running_dir)


def run(experiment_id,
        restore_path,
        output_template_dir=None,
        image_size=(None, None),
        project_name=None,
        save_npy_for_debug=True):
    """Convert from trained model.

    Args:
        experiment_id:
        restore_path:
        output_template_dir:  (Default value = None)
        image_size: (Default value = (None)
        project_name: (Default value = None)

    Returns:
        str: Path of exported dir.
            (i.e. `(path to saved)/saved/det_20190326181434/export/save.ckpt-161/128x128/output/`)

    """

    # Export model
    if save_npy_for_debug:
        export_dir = run_export(experiment_id, restore_path=restore_path, image_size=image_size)
    else:
        export_dir = run_export(experiment_id, restore_path=restore_path, image_size=image_size, image=None)

    # Set arguments
    input_pb_path = os.path.join(export_dir, "minimal_graph_with_shape.pb")
    if not project_name:
        project_name = "project"
    activate_hard_quantization = True
    threshold_skipping = True
    cache_dma = True

    # Generate project
    run_generate_project(
        input_path=input_pb_path,
        dest_dir_path=export_dir,
        project_name=project_name,
        activate_hard_quantization=activate_hard_quantization,
        threshold_skipping=threshold_skipping,
        cache_dma=cache_dma
    )

    # Create output dir from template
    output_root_dir = os.path.join(export_dir, "output")
    output_directories = create_output_directory(output_root_dir, output_template_dir)

    # Save meta.yaml to model output dir
    shutil.copy(os.path.join(export_dir, "meta.yaml"), output_directories.get("model_dir"))

    # Save minimal_graph_with_shape.pb to model output dir for TensforflowGraphRunner
    shutil.copy(os.path.join(export_dir, "minimal_graph_with_shape.pb"), output_directories.get("model_dir"))

    # Make
    project_dir_name = "{}.prj".format(project_name)
    project_dir = os.path.join(export_dir, project_dir_name)
    make_all(project_dir, output_directories.get("library_dir"))

    return output_root_dir


def convert(
    experiment_id,
    checkpoint=None,
    template=None,
    image_size=(None, None),
    project_name=None,
    save_npy_for_debug=True
):
    output_dir = os.environ.get('OUTPUT_DIR', 'saved')

    if checkpoint is None:
        restore_path = None
    else:
        restore_path = os.path.join(output_dir, experiment_id, 'checkpoints', checkpoint)

    return run(experiment_id, restore_path, template, image_size, project_name, save_npy_for_debug)
