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


class ProtobufLoader:
    def __init__(self, model_path):
        self.model_path = model_path

        sess, output_op, images_placeholder = self._load_protobuf_graph(model_path)

        self.sess = sess
        self.output_op = output_op
        self.images_placeholder = images_placeholder

    def _load_protobuf_graph(self, protobuf):
        graph = tf.Graph()

        with graph.as_default():
            with open(protobuf, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            init_op = tf.global_variables_initializer()
            images_placeholder = graph.get_tensor_by_name('images_placeholder:0')
            output_op = graph.get_tensor_by_name('output:0')

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=session_config)

        sess.run(init_op)

        return sess, output_op, images_placeholder

    def run(self, data):
        return self.sess.run(self.output_op, feed_dict={self.images_placeholder: data})
