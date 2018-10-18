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
from os import path

SCRIPT_DIR = path.abspath(path.dirname(__file__))
ROOT_DIR = path.abspath((path.join(SCRIPT_DIR, '..')))
PROJECTS_DIR = path.join(ROOT_DIR, 'projects')
LOGS_DIR = path.join(ROOT_DIR, 'logs')
SRC_DIR = path.join(ROOT_DIR, 'src')
INTEL_HLS_DIR = path.join(SRC_DIR, 'intel_hls')

# INTEL_FPGA_ROOT = os.environ['INTEL_FPGA_ROOTDIR']

NUM_THREADS = 8
