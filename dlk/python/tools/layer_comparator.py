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
import numpy as np
import click
import re
from pathlib import Path


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-d",
    "--debug_data_path",
    type=click.Path(exists=True),
    help="Directory containing the debug data generated in runtime",
)
@click.option(
    "-e",
    "--expected_data_path",
    type=click.Path(exists=True),
    help="Directory containing the .npy files with the expected output",
)
def main(debug_data_path, expected_data_path):
    if not debug_data_path or not expected_data_path:
        print('Please check usage with --help option')
        exit(1)

    click.echo('Checking debug data...')

    d_output = Path(debug_data_path)
    e_output = Path(expected_data_path)

    ptrn_ex = re.compile(r'(\d{3})_(.*):(\d+)')
    ptrn_dbg = re.compile(r'(.*)(\d+)')
    expected_output = [e for e in e_output.iterdir() if e.suffix == '.npy' and ptrn_ex.match(e.stem)]
    debug_output = [e for e in d_output.iterdir() if e.suffix == '.npy' and ptrn_dbg.match(e.stem)]

    if not debug_output:
        print(f"Debug path {dbg} is empty (no .npy files)")

    if not expected_output:
        print(f"Expected data path {e_output} is empty (no .npy files)")

    results = {}
    diffs = {}
    for o in debug_output:
        name_dbg = ptrn_dbg.match(o.stem).group(1)
        output_id_dbg = ptrn_dbg.match(o.stem).group(2)
        for eo in expected_output:
            name_ex = ptrn_ex.match(eo.stem).group(2)
            output_id_ex = ptrn_ex.match(eo.stem).group(3)
            if name_ex == name_dbg and output_id_ex == output_id_dbg:
                data_dbg = np.load(o)
                data_ex = np.load(eo)

                r_tol = 0.0001
                a_tol = 0.0001

                within_tolerance = np.isclose(data_ex.flatten() * data_dbg['scale'], data_dbg['data'],
                                              rtol=r_tol, atol=a_tol)

                if np.all(within_tolerance):
                    results[ptrn_ex.match(eo.stem).group(1)] = f"[OK]   {name_ex}:{output_id_ex}"
                else:
                    results[ptrn_ex.match(eo.stem).group(1)] = f"[FAIL] {name_ex}:{output_id_ex}"

                diffs[ptrn_ex.match(eo.stem).group(1)] = data_ex.flatten().size - np.count_nonzero(within_tolerance)

    sorted_results = [val for key, val in sorted(results.items())]
    sorted_diffs = [val for key, val in sorted(diffs.items())]
    for r, d in zip(sorted_results, sorted_diffs):
        print(r)


if __name__ == '__main__':
    main()
