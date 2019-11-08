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

import pytest
import tensorflow as tf

from executor.output_event import output

# Apply reset_default_graph() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("reset_default_graph")


def _create_tensorboard_event(tensorboard_dir):
    train_dir = os.path.join(tensorboard_dir, "train")
    test_dir = os.path.join(tensorboard_dir, "test")

    tf.InteractiveSession()

    train_writer = tf.summary.FileWriter(train_dir)
    test_writer = tf.summary.FileWriter(test_dir)
    tf.compat.v1.summary.scalar("scalar_1", tf.constant(1.0))
    tf.compat.v1.summary.scalar("scalar_2", tf.constant(2.0))
    merged = tf.compat.v1.summary.merge_all()

    train_writer.add_summary(merged.eval(), 0)
    train_writer.add_summary(merged.eval(), 2)

    test_writer.add_summary(merged.eval(), 0)

    train_writer.flush()
    test_writer.flush()


def test_output_event(tmpdir):
    # tmpdir is prepared by pytest.
    root_dir = str(tmpdir)
    tensorboard_dir = os.path.join(root_dir, "tensorboard")
    output_file_base = "metrics"
    metrics_keys = ()
    steps = ()

    _create_tensorboard_event(tensorboard_dir)
    output(tensorboard_dir, root_dir, metrics_keys, steps, output_file_base)

    expected_file = os.path.join(root_dir, output_file_base + ".md")
    assert os.path.exists(expected_file)


if __name__ == '__main__':
    test_output_event("./tmp/tests/")
