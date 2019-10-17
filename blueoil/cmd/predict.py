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

from executor.predict import run as run_predict


def predict(input, output, experiment_id, checkpoint=None, save_images=True):
    """Predict input images."""

    output_dir = os.environ.get("OUTPUT_DIR", "saved")

    if checkpoint is None:
        restore_path = None
    else:
        restore_path = os.path.join(
            output_dir, experiment_id, "checkpoints", checkpoint
        )

    run_predict(input, output, experiment_id, None, restore_path, save_images)
