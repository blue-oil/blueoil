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
import tensorflow as tf


class TFGraphLoadPb:
    """This class Loads Tensorflow Graph with Protocol Buffer File"""

    def __init__(self, model_path):
        """Initialize by setting the model path first.

        Args:
            model_path(string): The protocol buffer file location.
        """
        self.sess = None
        self.output_op = None
        self.images_placeholder = None
        self.model_path = model_path

    def init(self):
        """Load the tensor graph using protobuf file as model_path"""

        graph = tf.Graph()

        with graph.as_default():
            with open(self.model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            init_op = tf.global_variables_initializer()
            self.images_placeholder = graph.get_tensor_by_name('images_placeholder:0')
            self.output_op = graph.get_tensor_by_name('output:0')

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=session_config)

        self.sess.run(init_op)

    def run(self, data):
        """Run the data on the tf graph

        Args:
            data: returned result of the graph

        Returns:
            result(array): The result array of the graph
        """
        return self.sess.run(self.output_op, feed_dict={self.images_placeholder: data})
