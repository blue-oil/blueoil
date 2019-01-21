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
import click
import subprocess
import shutil

from executor.export import run as run_export
from scripts.generate_project import run as run_generate_project

from lmnet.utils import executor, config as config_util
from lmnet import environment


def get_export_directory(experiment_id, restore_path):
    """Return output dir of export"""

    config = config_util.load_from_experiment()

    if restore_path is None:
        restore_file = executor.search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

    print("Restore from {}".format(restore_path))

    if not os.path.exists("{}.index".format(restore_path)):
        raise Exception("restore file {} dont exists.".format(restore_path))

    export_dir = os.path.join(environment.EXPERIMENT_DIR, "export")
    export_dir = os.path.join(export_dir, os.path.basename(restore_path))
    export_dir = os.path.join(export_dir, "{}x{}".format(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))

    return export_dir


def create_output_directory(output_root_dir):
    """Create output directory from template."""

    template_dir = "/home/blueoil/output_template"
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


def get_output_directories(output_roor_dir):
    """Return output directories."""

    model_dir = os.path.join(output_roor_dir, "models")
    library_dir = os.path.join(model_dir, "lib")
    output_directories = dict(
        root_dir=output_roor_dir,
        model_dir=model_dir,
        library_dir=library_dir,
    )
    return output_directories

def make_all(project_dir, output_dir):
    """Make each target."""

    make_list = [
        ["lm_x86", "lm_x86.elf"],
        ["lm_arm", "lm_arm.elf"],
        ["lm_aarch64", "lm_aarch64.elf"],
        ["lm_fpga", "lm_fpga.elf"],
        ["lib_x86", "lib_x86.so"],
        ["lib_arm", "lib_arm.so"],
        ["lib_aarch64", "lib_aarch64.so"],
        ["lib_fpga", "lib_fpga.so"],
        ["ar_x86", "libdlk_x86.a"],
        ["ar_arm", "libdlk_arm.a"],
        ["ar_aarch64", "libdlk_aarch64.a"],
        ["ar_fpga", "libdlk_fpga.a"],
    ]
    running_dir = os.getcwd()
    # Change current directory to project directory
    os.chdir(project_dir)
    os.environ["FLAGS"] = "-D__WITHOUT_TEST__"
    # Make each target and move output files
    for target, output in make_list:
        subprocess.run(("make", "clean", "--quiet"))
        subprocess.run(("make",  target, "-j4", "--quiet"))
        output_file_path = os.path.join(output_dir, output)
        os.rename(output, output_file_path)
    # Return running directory
    os.chdir(running_dir)


def run(experiment_id, restore_path):
    """Convert from trained model."""

    # Export model
    run_export(experiment_id, restore_path, image_size=(None, None), images=[], config_file=None)
    export_dir = get_export_directory(experiment_id, restore_path)

    # Set arguments
    input_pb_path = os.path.join(export_dir, "minimal_graph_with_shape.pb")
    dest_dir_path = export_dir
    project_name = "project"
    activate_hard_quantization = True
    threshold_skipping = True
    cache_dma = True

    # Generate project
    run_generate_project(
        input_path=input_pb_path,
        dest_dir_path=dest_dir_path,
        project_name=project_name,
        activate_hard_quantization=activate_hard_quantization,
        threshold_skipping=threshold_skipping,
        cache_dma=cache_dma
    )

    # Create output dir from template
    output_root_dir = os.path.join(export_dir, "output")
    output_directories = create_output_directory(output_root_dir)

    # Save meta.yaml to model output dir
    shutil.copy(os.path.join(export_dir, "meta.yaml"), output_directories.get("model_dir"))

    # Make
    project_dir_name = "{}.prj".format(project_name)
    project_dir = os.path.join(dest_dir_path, project_dir_name)
    make_all(project_dir, output_directories.get("library_dir"))


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--experiment_id",
    help="id of this experiment.",
    required=True,
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001",
    default=None,
)
def main(experiment_id, restore_path):
    run(experiment_id, restore_path)


if __name__ == '__main__':
    main()
