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
import itertools

import click
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.io_wrapper import GetLogdirSubdirectories
import pandas as pd
import pytablewriter

from lmnet import environment


def _get_metrics_keys(event_accumulator):
    return event_accumulator.Tags()["scalars"]


def _value_step_list(event_accumulator, metrics_key):
    try:
        events = event_accumulator.Scalars(metrics_key)
        return [(event.value, event.step) for event in events]
    except KeyError as e:
        print("Key {} was not found in {}".format(metrics_key, event_accumulator.path))
        return []


def _column_name(event_accumulator, metrics_key):
    return "{}:{}".format(os.path.basename(event_accumulator.path), metrics_key)


def output(tensorboard_dir, output_dir, metrics_keys, steps, output_file_base="metrics"):
    """Output csv and markdown file which accumulated tensorflow event by step and metrics_keys."""
    subdirs = GetLogdirSubdirectories(tensorboard_dir)

    event_accumulators = []
    for subdir in subdirs:
        event_accumulator = EventAccumulator(subdir)
        # init event accumulator
        event_accumulator.Reload()

        event_accumulators.append(event_accumulator)

    if not metrics_keys:
        metrics_keys = {metrics_key
                        for event_accumulator in event_accumulators
                        for metrics_key in _get_metrics_keys(event_accumulator)}

    columns = [_column_name(event_accumulator, metrics_key)
               for event_accumulator, metrics_key in itertools.product(event_accumulators, metrics_keys)]
    columns.sort()
    df = pd.DataFrame([],  columns=columns)

    for event_accumulator in event_accumulators:
        for metrics_key in metrics_keys:
            value_step_list = _value_step_list(event_accumulator, metrics_key)
            for value, step in value_step_list:
                column_name = _column_name(event_accumulator, metrics_key)
                df.loc[step, column_name] = value

    if steps:
        df = df[steps, :]

    df = df.sort_index(ascending=False)

    # index to column. and re-order column.
    df["step"] = df.index
    df = df[["step"] + columns]

    output_csv = os.path.join(output_dir, "{}.csv".format(output_file_base))
    df.to_csv(output_csv, index=False)

    output_md = os.path.join(output_dir, "{}.md".format(output_file_base))
    writer = pytablewriter.MarkdownTableWriter()
    writer.char_left_side_row = "|"  # fix for github
    writer.from_dataframe(df)

    with open(output_md, "w") as file_stream:
        writer.stream = file_stream
        writer.write_table()

    message = """
output success

output csv: {}
output md: {}
""".format(output_csv, output_md)

    print(message)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--experiment_id",
    help="id of target experiment",
    required=True,
)
@click.option(
    "-k",
    "--metrics_keys",
    help="""Target metrics name of tensorboard scalar summaries for output.
    When it is empty, collect all scalar keys from tensorboard event.
    i.e. -k metrics/accuracy -k loss""",
    default=[],
    multiple=True,
)
@click.option(
    "-s",
    "--steps",
    help="Target step for output. When it is empty, target is all steps.",
    default=[],
    multiple=True,
    type=int,
)
@click.option(
    "-o",
    "--output_file_base",
    help="output file base name. default: `metrics`.",
    default=os.path.join("metrics"),
)
def main(output_file_base, metrics_keys, steps, experiment_id):
    environment.init(experiment_id)

    output(environment.TENSORBOARD_DIR, environment.EXPERIMENT_DIR, metrics_keys, steps, output_file_base="metrics",)


if __name__ == '__main__':
    main()
