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
"""Module of ONNX IO."""
from .base import BaseIO
from core.model import Model
from plugins.onnx import Importer

from os import path
import onnx
import json

from os import path
import onnx
from typing import Optional


class OnnxIO(BaseIO):
    """IO class that reads/writes a model from/to ONNX."""

    def read(self, pb_path: str, json_path: Optional[str] = None) -> Model:
        """Read ONNX file and load model.

        Parameters
        ----------
        pb_path : str
            Path to ONNX file

        Returns
        -------
        model : Model
            Loaded model

        """
        model = Model()

        # load onnx model
        onnx_model = onnx.load(path.abspath(pb_path))

        # debug print in JSON
        if json_path:
            from pip._internal import main
            main(['install', 'protobuf'])
            from google.protobuf.json_format import MessageToJson, Parse
            js_str = MessageToJson(onnx_model)
            js_obj = json.loads(js_str)
            with open(json_path, 'w') as fw:
                json.dump(js_obj, fw, indent=4)

        # ckeck if it's a valid model
        # onnx.checker.check_model(onnx_model)

        # import graph
        model.graph = Importer.make_graph(onnx_model)

        return model

    def write(self, model: Model, path: str) -> None:
        raise NotImplementedError
