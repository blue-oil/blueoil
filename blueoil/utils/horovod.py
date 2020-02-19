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
import subprocess
import sys

try:
    import horovod.tensorflow as hvd
    horovod_installed = True
except ImportError:
    horovod_installed = False

horovod_initialized = False


def setup():
    if not horovod_installed:
        return False

    global horovod_initialized
    if horovod_initialized:
        return hvd

    hvd.init()
    horovod_initialized = True

    horovod_num_worker = hvd.size()
    horovod_rank = hvd.rank()
    # verify that MPI multi-threading is supported.
    assert hvd.mpi_threads_supported()
    # make sure MPI is not re-initialized.
    import mpi4py.rc
    mpi4py.rc.initialize = False
    # import mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    # check size and rank are synchronized
    assert horovod_num_worker == comm.Get_size()
    assert horovod_rank == comm.Get_rank()
    return hvd


def _get_pname(pid):
    p = subprocess.Popen(["ps -o cmd= {}".format(pid)], stdout=subprocess.PIPE, shell=True)
    return p.communicate()[0].decode('utf-8')


# return True if the parent process is mpirun or horovodrun
def is_enabled():
    if os.getenv("USE_HOROVOD"):
        return True
    ppid = os.getppid()
    if ppid <= 1:
        return False

    parent_process_name = _get_pname(ppid)
    if parent_process_name.startswith("horovodrun") or parent_process_name.startswith("mpirun"):
        if horovod_installed:
            return True
        else:
            print("you're trying to run on horovod, but importing Horovod failed. exit.")
            sys.exit(1)
    else:
        return False


# return True if horovod is not enabled, or enabled and the process is rank 0.
def is_rank0():
    if not horovod_installed:
        return True

    if not is_enabled():
        return True

    if hvd.rank() == 0:
        return True

    return False
