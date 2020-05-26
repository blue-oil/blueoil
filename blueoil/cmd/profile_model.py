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
import collections
import json
import logging
import os

import click
import tensorflow as tf

from blueoil import environment
from blueoil.utils import config as config_util
from blueoil.utils import executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _profile(config, restore_path, bit, unquant_layers):
    output_root_dir = os.path.join(environment.EXPERIMENT_DIR, "export")
    if restore_path:
        output_root_dir = os.path.join(output_root_dir, os.path.basename(restore_path))

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    graph = tf.Graph()
    ModelClass = config.NETWORK_CLASS
    network_kwargs = dict((key.lower(), val) for key, val in config.NETWORK.items())

    with graph.as_default():

        model = ModelClass(
            classes=config.CLASSES,
            is_debug=config.IS_DEBUG,
            **network_kwargs,
        )

        is_training = tf.constant(False, name="is_training")

        images_placeholder, _ = model.placeholders()
        model.inference(images_placeholder, is_training)

        init_op = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver(max_to_keep=50)

    session_config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run(init_op)

    if restore_path:
        logger.info("Restore from {}".format(restore_path))
        saver.restore(sess, restore_path)

    main_output_dir = os.path.join(output_root_dir, "{}x{}".format(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    inference_graph_def = executor.convert_variables_to_constants(sess)

    inference_graph = tf.Graph()
    with inference_graph.as_default():
        tf.import_graph_def(inference_graph_def)

    scopes = {"total": 0}
    scope_idx = 1
    for node in inference_graph_def.node:
        names = node.name.split("/")
        scope = names[0]
        if scope not in scopes:
            scopes[scope] = scope_idx
            scope_idx += 1

    # [level, node name, total param, 32 bits size, quantized size, flops]
    res = []
    res, node_param_dict = _profile_params(graph, res, bit, unquant_layers)
    res, node_flops_dict = _profile_flops(inference_graph, res, scopes)

    name = ModelClass.__name__
    image_size = config.IMAGE_SIZE
    num_classes = len(config.CLASSES)
    _render(name, image_size, num_classes, bit, res)
    _save_json(name, image_size, num_classes, node_param_dict, node_flops_dict)


def _render(name, image_size, num_classes, bit, res):
    template_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "fixtures", "profile_template.md")

    with open(template_file_path, "r") as f:
        file_data = f.read()
    file_data = file_data.replace("{name}", name)\
                         .replace("{image_size_h}", str(image_size[0]))\
                         .replace("{image_size_w}", str(image_size[1]))\
                         .replace("{num_classes}", str(num_classes))\
                         .replace("{bit}", str(bit))

    table_rows, table_row = "", "| {} | {} | {} | {} | {} |"
    for r in res:
        if r[0] == 0:
            r[1] = "total"
        # Add indent
        r[1] = "&nbsp;&nbsp;" * (r[0] - 1) + r[1]
        # Add bold style
        if r[0] <= 1:
            r = ["**" + str(c) + "**" for c in r]
        table_rows += table_row.format(*r[1:]) + "\n"
    file_data = file_data.replace("{table}", table_rows)

    output_file = os.path.join(environment.EXPERIMENT_DIR, "{}_profile.md".format(name))
    with open(output_file, "w") as f:
        f.write(file_data)
    logger.info("Model's profile has been saved into {}".format(output_file))


def _save_json(name, image_size, num_classes, node_param_dict, node_flops_dict):
    prof_dict = {
        'model_name': name,
        'image_size_height': image_size[0],
        'image_size_width': image_size[1],
        'num_classes': num_classes,
        'flops': node_flops_dict,
        'parameters': node_param_dict,
    }

    output_file = os.path.join(environment.EXPERIMENT_DIR, "{}_profile.json".format(name))
    with open(output_file, "w") as f:
        f.write(json.dumps(prof_dict, indent=4))

    logger.info("save json: {}".format(output_file))


def _profile_flops(graph, res, scopes):
    float_prof = tf.compat.v1.profiler.profile(
        graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    float_res_dict = collections.defaultdict(int)
    float_res_dict["total"] = float_prof.total_float_ops
    for node in float_prof.children:
        scope = node.name.split("/")[1]
        float_res_dict[scope] += node.total_float_ops

    new_res = []
    res_dict = collections.defaultdict(list)
    for elem in res:
        res_dict[elem[1].split("/")[0]].append(elem)
    for scope, scope_idx in sorted(scopes.items(), key=lambda x: x[1]):
        flops = round(float_res_dict[scope] / 1000 ** 2, 5) if scope in float_res_dict else 0.
        if scope in res_dict:
            new_res.extend([elem + [flops if scope == elem[1] else "-"] for elem in res_dict[scope]])
        elif scope in float_res_dict:
            new_res.append([1, scope, "-", "-", "-", flops])

    node_flops_dict = {
        'total_flops': float_res_dict["total"],
        'children': [{"name": k, "flops": v} for k, v in float_res_dict.items() if k != "total"]
    }

    return new_res, node_flops_dict


def _profile_params(graph, res, bit, unquant_layers):
    prof = tf.compat.v1.profiler.profile(
        graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())

    # helper func to make profile res
    def helper(node, level):
        is_quant_kernel = all([layer not in node.name for layer in unquant_layers]) and "kernel" == \
                          node.name.split("/")[-1]
        node_name = "total" if level == 0 else node.name
        node_params = node.total_parameters
        node_size = node_params * 32
        node_quant_size = (node_params * bit) if is_quant_kernel else node_size
        node_children = []
        # Add node info to result list
        res.append(
            [level, node_name, node_params, node_size, node_quant_size])
        idx = len(res) - 1
        # Get children node info
        sumqs = 0
        for c in node.children:
            children = helper(c, level + 1)
            sumqs += children["quant_size"]
            node_children.append(children)
        node_quant_size = sumqs or node_quant_size
        # Update node_quant_size of result list
        res[idx][-1] = node_quant_size
        # Create node_dict
        if node_name == "total":
            node_param_dict = {
                'total_parameters': node_params,
                'total_size': node_size,
                'quant_bit': bit,
                'total_quant_size': node_quant_size,
                'children': node_children,
            }
        else:
            node_param_dict = {
                'name': node_name,
                'parameters': node_params,
                'size': node_size,
                'quant_size': node_quant_size,
                'children': node_children,
            }
        # Add is_quant_kernel flag to leaf node
        if not node.children:
            node_param_dict["is_quant_kernel"] = is_quant_kernel

        return node_param_dict

    node_param_dict = helper(prof, 0)
    for elem in res:
        elem[3] = round(elem[3] / 8 / 1024 ** 2, 5)
        elem[4] = round(elem[4] / 8 / 1024 ** 2, 5)
    return res, node_param_dict


def run(experiment_id, restore_path, config_file, bit, unquant_layers):
    if config_file is None and experiment_id is None:
        raise Exception("config_file or experiment_id are required")

    if experiment_id:
        environment.init(experiment_id)
        config = config_util.load_from_experiment()
        if config_file:
            config = config_util.merge(config, config_util.load(config_file))

        if restore_path is None:
            restore_file = executor.search_restore_filename(environment.CHECKPOINTS_DIR)
            restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

        if not os.path.exists("{}.index".format(restore_path)):
            raise Exception("restore file {} dont exists.".format(restore_path))

    else:
        experiment_id = "profile"
        environment.init(experiment_id)
        config = config_util.load(config_file)

    config.BATCH_SIZE = 1
    config.NETWORK.BATCH_SIZE = 1
    config.DATASET.BATCH_SIZE = 1

    executor.init_logging(config)
    config_util.display(config)

    _profile(config, restore_path, bit, unquant_layers)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--experiment_id",
    help="id of this experiment.",
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001",
    default=None,
)
@click.option(
    "-c",
    "--config_file",
    help="""config file path.
    When experiment_id is provided, The config override saved experiment config. When experiment_id is provided and the config is not provided, restore from saved experiment config.
    """,  # NOQA
)
@click.option(
    "-b",
    "--bit",
    default=32,
    help="quantized bit",
)
@click.option(
    "-uql",
    "--unquant_layers",
    multiple=True,
    help="unquantized layers",
)
def main(experiment_id, restore_path, config_file, bit, unquant_layers):
    """Profiling a trained model.

    If it exists unquantized layers, use `-uql` to point it out.
    """
    run(experiment_id, restore_path, config_file, bit, unquant_layers)


if __name__ == '__main__':
    main()
