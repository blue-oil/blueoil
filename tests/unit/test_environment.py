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

from blueoil import environment


class EnvironmentTestBase(object):
    def setup_method(self, method):
        self.default_env = {
            "EXPERIMENT_ID": environment.EXPERIMENT_ID,
            "EXPERIMENT_DIR": environment.EXPERIMENT_DIR,
            "TENSORBOARD_DIR": environment.TENSORBOARD_DIR,
            "CHECKPOINTS_DIR": environment.CHECKPOINTS_DIR,
            "DATA_DIR": environment.DATA_DIR,
            "OUTPUT_DIR": environment.OUTPUT_DIR,
            "_EXPERIMENT_DIR": environment._EXPERIMENT_DIR,
            "_TENSORBOARD_DIR": environment._TENSORBOARD_DIR,
            "_CHECKPOINTS_DIR": environment._CHECKPOINTS_DIR,
        }
        environment.EXPERIMENT_ID = None
        environment.EXPERIMENT_DIR = None
        environment.TENSORBOARD_DIR = None
        environment.CHECKPOINTS_DIR = None
        environment.DATA_DIR = os.getenv(
            "DATA_DIR", os.path.join(os.getcwd(), "dataset"),
        )
        environment.OUTPUT_DIR = os.getenv(
            "OUTPUT_DIR", os.path.join(os.getcwd(), "saved"),
        )
        environment._EXPERIMENT_DIR = os.path.join(
            environment.OUTPUT_DIR, "{experiment_id}",
        )
        environment._TENSORBOARD_DIR = os.path.join(
            environment.OUTPUT_DIR, "{experiment_id}", "tensorboard",
        )
        environment._CHECKPOINTS_DIR = os.path.join(
            environment.OUTPUT_DIR, "{experiment_id}", "checkpoints",
        )

    def teardown_method(self, method):
        environment.EXPERIMENT_ID = self.default_env["EXPERIMENT_ID"]
        environment.EXPERIMENT_DIR = self.default_env["EXPERIMENT_DIR"]
        environment.TENSORBOARD_DIR = self.default_env["TENSORBOARD_DIR"]
        environment.CHECKPOINTS_DIR = self.default_env["CHECKPOINTS_DIR"]
        environment.DATA_DIR = self.default_env["DATA_DIR"]
        environment.OUTPUT_DIR = self.default_env["OUTPUT_DIR"]
        environment._EXPERIMENT_DIR = self.default_env["_EXPERIMENT_DIR"]
        environment._TENSORBOARD_DIR = self.default_env["_TENSORBOARD_DIR"]
        environment._CHECKPOINTS_DIR = self.default_env["_CHECKPOINTS_DIR"]


class TestInit(EnvironmentTestBase):
    def test_init(self):
        experiment_id = "experiment001"
        environment.init(experiment_id)
        assert environment.EXPERIMENT_ID == experiment_id
        assert environment.EXPERIMENT_DIR == os.path.join(
            environment.OUTPUT_DIR, experiment_id,
        )
        assert environment.TENSORBOARD_DIR == os.path.join(
            environment.OUTPUT_DIR, experiment_id, "tensorboard",
        )
        assert environment.CHECKPOINTS_DIR == os.path.join(
            environment.OUTPUT_DIR, experiment_id, "checkpoints",
        )

    def test_init_with_experiment_dir(self):
        experiment_id = "experiment001"
        experiment_dir = os.path.join(environment.OUTPUT_DIR, experiment_id)
        environment.init(experiment_dir)
        assert environment.EXPERIMENT_ID == experiment_id
        assert environment.EXPERIMENT_DIR == experiment_dir
        assert environment.TENSORBOARD_DIR == os.path.join(
            experiment_dir, "tensorboard",
        )
        assert environment.CHECKPOINTS_DIR == os.path.join(
            experiment_dir, "checkpoints",
        )


class TestSetDataDir(EnvironmentTestBase):
    def test_set_data_dir(self):
        path = "./dataset"
        environment.set_data_dir(path)

        abs_path = os.path.abspath(path)
        assert environment.DATA_DIR == abs_path
        assert os.environ["DATA_DIR"] == abs_path

    def test_set_data_dir_with_gcs_path(self):
        path = "gs://dataset"
        environment.set_data_dir(path)

        abs_path = os.path.abspath(path)
        assert environment.DATA_DIR == path
        assert os.environ["DATA_DIR"] == path


class TestSetOutputDir(EnvironmentTestBase):
    def test_set_output_dir(self):
        path = "./output"
        environment.set_output_dir(path)

        abs_path = os.path.abspath(path)
        assert environment.OUTPUT_DIR == abs_path
        assert os.environ["OUTPUT_DIR"] == abs_path

        assert environment._EXPERIMENT_DIR == os.path.join(
            abs_path, "{experiment_id}",
        )
        assert environment._TENSORBOARD_DIR == os.path.join(
            abs_path, "{experiment_id}", "tensorboard",
        )
        assert environment._CHECKPOINTS_DIR == os.path.join(
            abs_path, "{experiment_id}", "checkpoints",
        )
        assert environment.EXPERIMENT_DIR is None
        assert environment.TENSORBOARD_DIR is None
        assert environment.CHECKPOINTS_DIR is None

    def test_set_output_dir_after_init(self):
        experiment_id = "experiment001"
        environment.init(experiment_id)

        path = "./output"
        environment.set_output_dir(path)

        abs_path = os.path.abspath(path)
        assert environment.EXPERIMENT_DIR == os.path.join(
            abs_path, experiment_id,
        )
        assert environment.TENSORBOARD_DIR == os.path.join(
            abs_path, experiment_id, "tensorboard",
        )
        assert environment.CHECKPOINTS_DIR == os.path.join(
            abs_path, experiment_id, "checkpoints",
        )

    def test_set_output_dir_with_gcs_path(self):
        path = "gs://output"
        environment.set_output_dir(path)

        abs_path = os.path.abspath(path)
        assert environment.OUTPUT_DIR == path
        assert os.environ["OUTPUT_DIR"] == path
