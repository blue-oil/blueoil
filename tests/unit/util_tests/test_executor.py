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
import pytest
import os
import json
import textwrap
import tensorflow as tf
from blueoil import environment
from blueoil.utils.executor import prepare_dirs, profile_train_step

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment.
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_profile_train_step():
    """Asserts that the saved profile matches expectation for a simple example."""
    environment.init("test_executor")
    prepare_dirs(recreate=True)

    with tf.compat.v1.Session() as sess:
        w = tf.Variable([[5.0, 3.0, 2.9, -4.0, 0.0]])
        v = tf.Variable([[0.21, -2.70, 0.94, 3.82, -3.65],
                        [5.0, 3.0, 2.9, -4.0, 0.0],
                        [1.96, -2.2, 0.42, -1.26, -1.06],
                        [-1.55, 4.56, -4.71, -2.43, 4.55],
                        [-3.11, 3.78, -3.45, 2.18, -4.45]])
        z = tf.matmul(w, v)  # z is [[27.933998, -29.119999, 33.458, 13.166, -39.524002]]
        sess.run(tf.compat.v1.global_variables_initializer())
        step = 0
        options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_meta = tf.compat.v1.RunMetadata()
        sess.run(z, options=options, run_metadata=run_meta)
        profile_train_step(step, sess, run_meta)

    expected_memory = textwrap.dedent("""\

        Doc:
        scope: The nodes in the model graph are organized by their names, which is hierarchical like filesystem.
        requested bytes: The memory requested by the operation, accumulatively.

        Profile:
        node name | requested bytes
        _TFProfRoot (--/768B)
          MatMul (256B/256B)
          Variable (256B/256B)
          Variable_1 (256B/256B)
        """)
    expected_timeline = [
        {
            "args": {
                "name": "Scope:0"
            },
            "name": "process_name",
            "ph": "M",
            "pid": 0
        },
        {
            "args": {
                "name": "_TFProfRoot",
                "op": "_TFProfRoot"
            },
            "cat": "Op",
            "dur": 291,
            "name": "_TFProfRoot",
            "ph": "X",
            "pid": 0,
            "tid": 0,
            "ts": 0
        },
        {
            "args": {
                "name": "Scope:1"
            },
            "name": "process_name",
            "ph": "M",
            "pid": 1
        },
        {
            "args": {
                "name": "MatMul",
                "op": "MatMul"
            },
            "cat": "Op",
            "dur": 267,
            "name": "MatMul",
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": 0
        },
        {
            "args": {
                "name": "Variable",
                "op": "Variable"
            },
            "cat": "Op",
            "dur": 4,
            "name": "Variable",
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": 267
        },
        {
            "args": {
                "name": "Variable_1",
                "op": "Variable_1"
            },
            "cat": "Op",
            "dur": 20,
            "name": "Variable_1",
            "ph": "X",
            "pid": 1,
            "tid": 0,
            "ts": 271
        }
    ]

    train_memory_path = os.path.join(environment.EXPERIMENT_DIR, "training_profile_memory")
    with open(train_memory_path) as train_memory_file:
        saved_data = train_memory_file.read()
        assert saved_data == expected_memory
    train_timeline_path = os.path.join(environment.EXPERIMENT_DIR,
                                       "training_profile_timeline_step")
    with open("{}_{}".format(train_timeline_path, step)) as train_timeline_file:
        saved_data = json.load(train_timeline_file)["traceEvents"]
        for op1, op2 in zip(expected_timeline, saved_data):
            assert op1["args"] == op2["args"]
            # Generally, timeline values are different each run, so just check the keys match.
            assert op1.keys() == op2.keys()


if __name__ == '__main__':
    test_profile_train_step()
