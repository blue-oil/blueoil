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

"""Module for ONNX."""
import functools
import importlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
from tensorflow.core.framework import types_pb2

import core.operators as dlk_op
from core.data_types import DataType, Float32, Float64, Int8, Int16, Int32, \
    Int64, Uint8, Uint16, Uint32, Uint64, Bool, String
from core.exceptions import UnsupportedNode, UnsupportedDataType
from core.graph import Graph
from core.operators import Operator, Conv, Identity, QTZ_binary_mean_scaling, \
    BatchNormalization, QTZ_linear_mid_tread_half, Add, \
    MaxPool, AveragePool, Reshape, Softmax, Transpose, Relu, SpaceToDepth, \
    Mul, QTZ_binary_channel_wise_mean_scaling, ConcatOnDepth, Maximum, DepthToSpace, \
    Split, Pad, MatMul, LeakyRelu

DLK_DTYPE_MAP: Dict[str, Optional[DataType]] = {
    # any
    'DT_INVALID': None,
    # primitives
    'DT_FLOAT': Float32(),
    'DT_INT32': Int32(),
    'DT_UINT8': Uint8(),
    'DT_INT8': Int8(),
    'DT_UINT16': Uint16(),
    'DT_INT16': Int16(),
    'DT_INT64': Int64(),
    'f': Float32(),
    'i': Int32(),

    # primitive vector
    'FLOATS': None,
    'INTS': None,

    # custom
    'DT_BOOL': Bool(),
    'DT_STRING': String(),
    'DT_HALF': None,
    'DT_DOUBLE': Float64(),
    'DT_UINT32': Uint32(),
    'DT_UINT64': Uint64(),
    'DT_COMPLEX64': None,
    'DT_COMPLEX128': None,
    's': String(),

    # custom vector
    'STRINGS': None,

    # struct
    'TENSOR': None,
    'GRAPH': None,
    't': None,
    'g': None,

    # struct vector
    'TENSORS': None,
    'GRAPHS': None,
}

DLK_OPERATOR_MAP: Dict[str, str] = {
    'Conv2D': 'Conv',
    'FusedBatchNorm': 'BatchNormalization',
    'AvgPool': 'AveragePool',
    'BiasAdd': 'Add',
    'ConcatV2': 'ConcatOnDepth'
}


class Node(object):
    ATTRIBUTE_TYPE_MAP = {
        0: 'UNDEFINED',
        1: 'FLOAT',
        2: 'INT',
        3: 'STRING',
        4: 'TENSOR',
        5: 'GRAPH',
        6: 'FLOATS',
        7: 'INTS',
        8: 'STRINGS',
        9: 'TENSORS',
        10: 'GRAPHS',
    }

    ATTRIBUTE_VALUE_MAP = {
        1: 's',
        2: 'i',
        3: 'f',
        4: 'b',
        5: 'type',
        6: 'shape',
        7: 'tensor',
        8: 'list',
        9: 'func',
        10: 'placeholder',
    }

    def __init__(self, op_nd) -> None:  # type: ignore
        self.nd_ = op_nd
        self.attributes = []  # type: ignore
        for key in self.nd_.attr.keys():
            self.attributes.append(key)

    @property
    def name(self) -> str:
        """Return the name corresponding to the node."""
        return self.nd_.name.replace('/', '_').replace('-', '_')

    @property
    def node_def_object(self):
        """Return onnx object."""
        return self.nd_

    @property
    def op_type(self) -> str:
        """Return the op type of the node."""
        return self.nd_.op

    @property
    def inputs(self) -> List[str]:
        """Return the name of corresponding inputs to the node."""
        return [x.replace('/', '_').replace('-', '_').split(':', 1)[0] for x in self.nd_.input]

    @property
    def tensor_type(self):
        """Get tensor type info."""
        if self.nd_.op == 'QTZ_binary_mean_scaling' or \
           self.nd_.op == 'QTZ_linear_mid_tread_half' or \
           self.nd_.op == 'QTZ_binary_channel_wise_mean_scaling':
            typep = 1
        else:
            typep = self.nd_.attr["T"].type
        return typep

    def get_dtype(self):
        """Get dlk dtype of the node."""
        dtype_str = Input.DATA_TYPE_MAP[self.tensor_type]
        if DLK_DTYPE_MAP[dtype_str] is None:
            raise UnsupportedDataType(f'Type {dtype_str} is not supported.')
        return DLK_DTYPE_MAP[dtype_str]

    def get_shape(self) -> List[str]:
        """Get the output shape info."""
        out_shapes = []
        shapes = self.nd_.attr.get('_output_shapes')
        if shapes:
            for d in shapes.list.shape:
                shape = []
                for v in range(len(d.dim)):
                    shape.append(d.dim[v].size)
                out_shapes.append(shape)
        else:
            raise ValueError(f'{self.name} does not have output shapes.')

        if len(out_shapes) > 1 and self.nd_.op == 'Split':
            if not out_shapes[1:] == out_shapes[:-1]:
                raise ValueError(f'{self.name} does not have identical output(s) shape.')

        return out_shapes[0]

    def get_format(self):  # type: ignore
        """Get the output data format info."""
        if self.nd_.attr.get('data_format'):
            return self.nd_.attr.get('data_format').s.decode(encoding='utf-8')
        else:
            return None

    def list_attributes(self):
        """Return the attribute list corresponding to the node."""
        return self.attributes

    def attribute(self, attr_name: str) -> Any:
        """Return the attributes data corresponding to the node."""
        attrs = [x for x in self.attributes if x == attr_name]
        if len(attrs) != 1:
            raise ValueError(f'{self.op_type} {self.name} doesn\'t have the valid attribute.')

        # TODO: hard coded for now, looking for better extraction methods
        attrs_data = []
        if attr_name == 'padding' or attr_name == 'data_format':
            attrs_data.append(self.nd_.attr[attr_name].s)
        elif attr_name in ['strides', 'ksize']:
            attrs_data.append(self.nd_.attr[attr_name].list.i)
        elif attr_name in ['epsilon', 'alpha']:
            attrs_data.append(self.nd_.attr[attr_name].f)
        elif attr_name == 'is_training' or attr_name == 'use_cudnn_on_gpu':
            attrs_data.append(self.nd_.attr[attr_name].b)
        elif attr_name in ['block_size', 'num_split']:
            attrs_data.append(self.nd_.attr[attr_name].i)
        else:
            raise ValueError(f'{self.op_type} {self.name} doesn\'t have the supported attribute.')

        return attrs_data


class Input(object):
    # Data conversion table from binary to numpy
    _TF_TO_NP = {
        types_pb2.DT_HALF:
            np.float16,
        types_pb2.DT_FLOAT:
            np.float32,
        types_pb2.DT_DOUBLE:
            np.float64,
        types_pb2.DT_INT32:
            np.int32,
        types_pb2.DT_UINT16:
            np.uint16,
        types_pb2.DT_INT16:
            np.int16,
        types_pb2.DT_INT8:
            np.int8,
        types_pb2.DT_STRING:
            np.object,
        types_pb2.DT_COMPLEX64:
            np.complex64,
        types_pb2.DT_COMPLEX128:
            np.complex128,
        types_pb2.DT_INT64:
            np.int64,
        types_pb2.DT_BOOL:
            np.bool,
    }
    DATA_TYPE_MAP = {v: k for k, v in types_pb2.DataType.items()}

    def __init__(self, in_nd) -> None:  # type: ignore
        self.in_ = in_nd
        if not self.is_placeholder:
            self.tensor = self.in_.attr.get('value').tensor

    @property
    def name(self) -> str:
        """Return the name of the node."""
        return self.in_.name.replace('/', '_').replace('-', '_')

    @property
    def op_type(self) -> str:
        """Return the op type of the node."""
        return self.in_.op

    @property
    def nodedef_object(self):
        """Return node def object."""
        return self.in_

    @property
    def is_placeholder(self):
        """Check op is place holder or not."""
        return self.in_.op == 'Placeholder'

    def get_dtype(self):
        """Get data type info."""
        if self.is_placeholder:
            dtype_str = type(
                self).DATA_TYPE_MAP[self.in_.attr.get('dtype').type]
        else:
            dtype_str = type(self).DATA_TYPE_MAP[self.tensor.dtype]

        if DLK_DTYPE_MAP[dtype_str] is None:
            raise UnsupportedDataType(f'Type {dtype_str} is not supported.')

        return DLK_DTYPE_MAP[dtype_str]  # type: ignore

    def get_data(self):  # type: ignore
        """Get data in numpy format."""
        if self.is_placeholder:
            raise ValueError(
                f'{self.name} is a placeholder, which does\'t have no data...')

        # convert tensor content to numpy
        if self.tensor.tensor_content:
            dtype = type(self)._TF_TO_NP[self.tensor.dtype]
            return np.frombuffer(self.tensor.tensor_content, dtype=dtype).copy().reshape(self.get_shape())
        else:
            dtype = type(self)._TF_TO_NP[self.tensor.dtype]
            if self.tensor.dtype == 3:
                return np.asarray(self.tensor.int_val, dtype=dtype)
            if self.tensor.dtype == 1:
                return np.asarray(self.tensor.float_val, dtype=dtype)

    def get_shape(self) -> List[str]:
        """Get shape info."""
        if self.is_placeholder:
            return [d.size for d in self.in_.attr.get('shape').shape.dim]
        else:
            return [d.size for d in self.tensor.tensor_shape.dim] or [self.get_data().size]

    def set_shape(self, val: List[str]) -> None:
        """Set shape info."""
        raise NotImplemented


class Output(object):
    def __init__(self, out_nd) -> None:  # type: ignore
        self.out_ = out_nd

    @property
    def name(self) -> str:
        """Return the name corresponding to the node."""
        return self.out_.name.replace('/', '_').replace('-', '_')

    @property
    def op_type(self) -> str:
        """Return the name corresponding to the node."""
        return self.out_.op

    @property
    def inputs(self) -> List[str]:
        """Return the name of corresponding inputs to the node."""
        return [x.replace('/', '_').replace('-', '_') for x in self.out_.input]

    @property
    def node_def_object(self):
        """Return onnx object."""
        return self.out_

    @property
    def tensor_type(self):
        """Get shape info."""
        if self.out_.op == 'QTZ_binary_mean_scaling' or \
                self.out_.op == 'QTZ_linear_mid_tread_half' or \
                self.out_.op == 'QTZ_binary_channel_wise_mean_scaling':
            typep = 1
        else:
            typep = self.out_.attr["T"].type
        return typep

    def get_dtype(self):
        """Get data type info."""
        dtype_idx = self.tensor_type
        dtype_str = Input.DATA_TYPE_MAP[dtype_idx]
        if DLK_DTYPE_MAP[dtype_str] is None:
            raise UnsupportedDataType(f'Type {dtype_str} is not supported.')
        return DLK_DTYPE_MAP[dtype_str]  # type: ignore

    def get_shape(self) -> List[str]:
        """Get shape info."""
        shape = []
        for d in self.out_.attr.get('_output_shapes').list.shape:
            for v in range(len(d.dim)):
                shape.append(d.dim[v].size)

        return shape[:4]

    def set_shape(self, val: List[str]) -> None:
        """Set shape info."""
        raise NotImplemented


class Importer(object):

    @classmethod
    def make_graph(cls, tf_mp) -> Graph:  # type: ignore
        importer = Importer(tf_mp)
        graph = Graph()

        importer.add_all_nodes(graph)
        return graph

    def __init__(self, tf_mp) -> None:  # type: ignore
        """Init the graph.
        Prameters
        ---------
        tf_mp : GraphDef
            GraphDef object
        """
        self.tf_gp = tf_mp
        self.in_lst: List[Input] = []
        self.out_lst: List[Output] = []
        self.node_lst: List[Node] = []
        self.node_dic: Dict[str, Any] = {}
        node_obj: Any = None
        for node in self.tf_gp.node:
            # print(node)
            if node.op == 'Const' or node.op == 'Placeholder':
                node_obj = Input(node)
                self.in_lst.append(node_obj)
            else:
                if node.name == "output":
                    node_obj = Output(node)
                    self.out_lst.append(node_obj)
                else:
                    node_obj = Node(node)
                    self.node_lst.append(node_obj)
            self.node_dic[node_obj.name] = node_obj

        self.validate_tf()

        self.input_hash_table = self.construct_input_hash_table(
            self.node_lst, self.out_lst, self.in_lst)

    def validate_tf(self) -> None:
        """Validate if the GraphDef object is proper."""
        gp = self.tf_gp
        assert len(gp.node) == (len(self.in_lst) + len(self.node_lst) + len(self.out_lst))

    def convert_operator(self, op_type: str) -> str:
        """Convert Tensorflow operator type to DLK's one."""
        dlk_op_type = DLK_OPERATOR_MAP.get(op_type)
        return dlk_op_type if dlk_op_type else op_type

    def create_new_op(self, node: Any, op_dic: Dict[str, Operator], current_format: str,
                      input_format_list: List[str], nodes_to_remove) -> Operator:
        """Create new operators with Node, Input(Constant), Output."""
        new_op: Operator

        if isinstance(node, Node):  # operator nodes
            new_op = self.create_new_node(node, op_dic, current_format, input_format_list, nodes_to_remove)

        else:  # Input, Output or Constant
            shape: List[int] = list(map(int, node.get_shape()))
            dtype = node.get_dtype()

            # print(node.op_type, ' name:', node.name, ' shape:', shape)

            if isinstance(node, Input):
                if node.is_placeholder:  # op_type = 'Input'
                    shape = list(map(int, node.get_shape()))
                    new_op = dlk_op.Input(
                        node.name,
                        shape,
                        dtype,
                        dimension_format=current_format
                    )

                else:  # op_type = 'Constant'
                    data = node.get_data()
                    new_op = dlk_op.Constant(
                        node.name,
                        dtype,
                        data,
                        dimension_format=current_format
                    )

            elif isinstance(node, Output):  # op_type = 'Output'
                # get input operator
                input_ops = {k: op_dic[n.name] for n, k in zip(
                    self.find_inputs(node), dlk_op.Output.input_names)}

                new_op = dlk_op.Output(
                    node.name,
                    shape,
                    dtype,
                    input_ops,
                )

        return new_op

    def add_all_nodes(self, graph: Graph) -> None:
        visited: Set[Any] = set()
        added: Dict[str, Operator] = {}
        nodes_to_remove = []
        self.add_node_to_graph_recursive(self.out_lst[0], graph, visited, added, 'NHWC', nodes_to_remove)
        for node in nodes_to_remove:
            graph.remove_op(node)

    def _get_format(self, node: Any, output_format: str) -> Tuple[str, List[str]]:
        """Get the dimension format, like 'NCHW', 'HWCN', 'NHWC', etc."""

        _default_format = 'NHWC'  # TF standard for input
        _default_w_format = 'HWCN'  # TF standard for weight

        if isinstance(node, Node):
            if node.get_format() is None:
                out_format = output_format or _default_format
            else:
                out_format = node.get_format()

            in_format = out_format
            in_w_format = _default_w_format

            op_type = self.convert_operator(node.op_type)
            # op_type = node.op_type
            if op_type == 'Conv':
                # the TF standard for input and weight
                return out_format, [in_format, in_w_format, 'N']

            elif op_type == 'BatchNormalization':
                # the TF standard and vector values
                return out_format, [in_format, 'C', 'C', 'C', 'C']

            elif op_type in {'MaxPool', 'AveragePool'}:
                return out_format, [in_format]  # the TF standard

            elif op_type == 'Transpose':
                perm = list(node.attribute("perm"))
                inv_perm = [perm.index(i) for i in range(len(perm))]  # inverse permutation
                input_format = functools.reduce(
                    lambda x, y: x + y, [output_format[i] for i in inv_perm])
                return output_format, [input_format]

            elif op_type == 'QTZ_linear_mid_tread_half':
                return out_format, [in_format, '1', '1']  # two scalar constants

            elif op_type in ['QTZ_binary_mean_scaling', 'QTZ_binary_channel_wise_mean_scaling']:
                return _default_w_format, [_default_w_format]  # two scalar constants

            elif op_type == 'Gemm':
                return out_format, ['', '', '']  # three inputs

            elif op_type in ['Add', 'Mul', 'Maximum', 'MatMul']:
                return out_format, ['', '']  # two inputs

            elif op_type in ['Split', 'Pad']:
                return out_format, [in_format, '']

            elif op_type == 'ConcatOnDepth':
                return out_format, [in_format, in_format, in_format, in_format, in_format, '']

            else:
                return out_format, [out_format]

        else:  # Input or Output
            return output_format, [output_format]

    def add_node_to_graph_recursive(self, current: Any, graph: Graph, visited: Set[Any], added: Dict[str, Operator],
                                    data_format: str, nodes_to_remove) \
            -> Operator:
        if current in visited:
            return added[current.name]
            # return current

        added_op_dic: Dict[str, Operator] = {}

        current_format, input_formats = self._get_format(current, data_format)
        inputs = self.find_inputs(current)
        for in_put, in_format in zip(inputs, input_formats):
            in_op = self.add_node_to_graph_recursive(in_put, graph, visited, added, in_format, nodes_to_remove)
            added_op_dic[in_op.name] = in_op

        op = self.create_new_op(current, added_op_dic, current_format, input_formats, nodes_to_remove)

        graph.add_op(op)

        visited.add(current)
        added[op.name] = op
        return op

    def construct_input_hash_table(self, node_list: List[Any], out_list: List[Any], input_list: List[Any]) \
            -> Dict[str, Any]:
        hash_table: Dict[str, Any] = defaultdict(lambda: [])

        for x in input_list:
            hash_table[x.name].append(x)

        for node in node_list + out_list:
            for idx in node.inputs:
                hash_table[node.name].append(self.node_dic[idx])

        return hash_table

    def find_inputs(self, node: Any) -> List[Any]:
        inputs: List[Any] = []
        if not isinstance(node, Input):
            for idx in node.inputs:
                inputs.append(self.node_dic[idx])

        return inputs

    def create_new_node(self, node: Node, op_dic: Dict[str, Operator], current_format: str,
                        input_format_list: List[str], nodes_to_remove) -> Operator:
        """Create a new operator node. This might be tooooo long code...
        Parameters
        ----------
        node : Node
            TF node corresponding to the operator
        op_dic : Dict from str to Operator
            Dict of preceding operators
        current_format : Dict from str to str
            Dict of data format of current node
        input_format_list : Dict from str to str
            Dict of data format of corresponding inputs of current node
        Returns
        -------
        new_op : Operator
            Newly created dlk operator
        """
        op_type = self.convert_operator(node.op_type)
        try:
            module = importlib.import_module('core.operators')
            class_def = getattr(module, op_type)
        except AttributeError:
            message = f'Operator {op_type} is not supported.'
            raise UnsupportedNode(message)

        # else:
        #     print(op_type)  # debug

        # Create new op accordingly for the tf ops
        new_op: Operator

        def get_inputs(cdef: Type[Operator], current_node: Any) -> Dict[str, Operator]:
            input_names = cdef.input_names
            in_ops: Dict[str, Operator] = {}
            in_ops_order: List[int] = []
            for n, op in zip(input_names, current_node.inputs):
                in_ops[n] = op_dic[op]
                in_ops_order.append(n)
            return in_ops, in_ops_order

        input_ops, input_ops_order = get_inputs(class_def, node)

        # Here find the shape and data type for the op
        def infer_shape(attrs: Dict[str, Any]) -> List[int]:
            shape_dict = {n: input_ops[n].shape for n in class_def.input_names if input_ops.get(n)}
            return class_def.infer_shape(shape_dict, current_format, input_format_list, attrs)

        def infer_dtype() -> DataType:
            if node.get_dtype() is not None:
                return node.get_dtype()  # type: ignore
            else:
                return list(input_ops.values())[0].dtype

        shape: List[int] = list(map(int, node.get_shape()))
        dtype = infer_dtype()

        # debug msgs
        # print(node.op_type, ' name:', node.name, ' shape:', shape, ' inputs:', node.inputs)

        if op_type == 'Conv':
            strides = node.attribute('strides')[0][1:3]
            padding = node.attribute('padding')[0].decode(encoding='utf-8')
            # calculated pads size for tf
            input_format = input_format_list[0]
            kernel_format = input_format_list[1]

            in_h = input_ops['X'].shape[input_format.index('H')]
            in_w = input_ops['X'].shape[input_format.index('W')]
            filt_h = input_ops['W'].shape[kernel_format.index('H')]
            filt_w = input_ops['W'].shape[kernel_format.index('W')]
            stride_h = strides[0]
            stride_w = strides[1]

            pads: List[int] = []
            if padding == 'SAME':
                if in_h % stride_h == 0:
                    pad_along_height = max(filt_h - stride_h, 0)
                else:
                    pad_along_height = max(filt_h - (in_h % stride_h), 0)
                if in_w % stride_w == 0:
                    pad_along_width = max(filt_w - stride_w, 0)
                else:
                    pad_along_width = max(filt_w - (in_w % stride_w), 0)

                pad_top = pad_along_height // 2
                pad_bottom = pad_along_height - pad_top
                pad_left = pad_along_width // 2
                pad_right = pad_along_width - pad_left

                pads = [pad_top, pad_bottom, pad_left, pad_right]

            elif padding == 'VALID':
                pads = [0, 0, 0, 0]

            else:
                raise ValueError(f'{op_type} {node.name} doesn\'t have the supported padding.')

            if not shape:
                attributes = {'kernel_shape': [filt_h, filt_w],
                              'strides': strides,
                              'pads': pads}
                shape = infer_shape(attributes)

            new_op = Conv(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                kernel_shape=[filt_h, filt_w],
                strides=strides,
                pads=pads,
            )
        elif op_type == 'BatchNormalization':
            epsilon = node.attribute('epsilon')[0]
            is_test = not node.attribute('is_training')

            if not shape:
                attributes = {'epsilon': epsilon, 'is_test': is_test}
                shape = infer_shape(attributes)

            new_op = BatchNormalization(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                epsilon=epsilon,
                is_test=is_test,
            )
        elif op_type == 'Add':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Add(
                node.name,
                shape,
                dtype,
                input_ops,
            )
        elif op_type == 'Identity':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Identity(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'QTZ_linear_mid_tread_half':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = QTZ_linear_mid_tread_half(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'QTZ_binary_mean_scaling':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = QTZ_binary_mean_scaling(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'Reshape':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Reshape(
                node.name,
                shape,
                dtype,
                input_ops,
            )
        elif op_type == 'Softmax':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Softmax(
                node.name,
                shape,
                dtype,
                input_ops,
            )
        elif op_type == 'MaxPool':

            kernel_shape = node.attribute('ksize')[0][1:3]
            padding = node.attribute('padding')[0].decode(encoding='utf-8')
            strides = node.attribute('strides')[0][1:3]

            in_h = input_ops['X'].height
            in_w = input_ops['X'].width
            filt_h = kernel_shape[0]
            filt_w = kernel_shape[1]
            stride_h = strides[0]
            stride_w = strides[1]

            pads = []
            if padding == 'SAME':
                if in_h % stride_h == 0:
                    pad_along_height = max(filt_h - stride_h, 0)
                else:
                    pad_along_height = max(filt_h - (in_h % stride_h), 0)
                if in_w % stride_w == 0:
                    pad_along_width = max(filt_w - stride_w, 0)
                else:
                    pad_along_width = max(filt_w - (in_w % stride_w), 0)

                pad_top = pad_along_height // 2
                pad_bottom = pad_along_height - pad_top
                pad_left = pad_along_width // 2
                pad_right = pad_along_width - pad_left

                pads = [pad_top, pad_bottom, pad_left, pad_right]

            elif padding == 'VALID':
                pads = [0, 0, 0, 0]

            else:
                raise ValueError(f'{op_type} {node.name} doesn\'t have the supported padding.')

            if not shape:
                attributes = {'kernel_shape': kernel_shape, 'pads': pads, 'strides': strides}
                shape = infer_shape(attributes)

            new_op = MaxPool(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )
        elif op_type == 'AveragePool':

            kernel_shape = node.attribute('ksize')[0][1:3]
            padding = node.attribute('padding')[0].decode(encoding='utf-8')
            strides = node.attribute('strides')[0][1:3]

            in_h = input_ops['X'].height
            in_w = input_ops['X'].width
            filt_h = kernel_shape[0]
            filt_w = kernel_shape[1]
            stride_h = strides[0]
            stride_w = strides[1]

            pads = []
            if padding == 'SAME':
                if in_h % stride_h == 0:
                    pad_along_height = max(filt_h - stride_h, 0)
                else:
                    pad_along_height = max(filt_h - (in_h % stride_h), 0)
                if in_w % stride_w == 0:
                    pad_along_width = max(filt_w - stride_w, 0)
                else:
                    pad_along_width = max(filt_w - (in_w % stride_w), 0)

                pad_top = pad_along_height // 2
                pad_bottom = pad_along_height - pad_top
                pad_left = pad_along_width // 2
                pad_right = pad_along_width - pad_left

                pads = [pad_top, pad_bottom, pad_left, pad_right]

            elif padding == 'VALID':
                pads = [0, 0, 0, 0]

            else:
                raise ValueError(f'{op_type} {node.name} doesn\'t have the supported padding.')

            if not shape:
                attributes = {'kernel_shape': kernel_shape, 'pads': pads, 'strides': strides}
                shape = infer_shape(attributes)

            new_op = AveragePool(
                node.name,
                shape,
                dtype,
                input_ops,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
            )
        elif op_type == 'Transpose':

            perm = node.attribute("perm")

            if not shape:
                attributes = {'perm': perm}
                shape = infer_shape(attributes)

            new_op = Transpose(
                node.name,
                shape,
                dtype,
                input_ops,
                perm=perm,
            )
        elif op_type == 'Relu':

            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Relu(
                node.name,
                shape,
                dtype,
                input_ops
            )
        elif op_type == 'LeakyRelu':

            alpha = node.attribute("alpha")[0]

            if not shape:
                attributes = {'alpha': alpha}
                shape = infer_shape(attributes)

            new_op = LeakyRelu(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                alpha=alpha,
            )
        elif op_type == 'SpaceToDepth':
            bs = node.attribute('block_size')
            if not bs:
                raise ValueError(f'{op_type} {node.name} block size not specified')

            if not shape:
                attributes = {'block_size': bs[0]}
                shape = infer_shape(attributes)

            new_op = SpaceToDepth(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                block_size=bs[0]
            )
        elif op_type == 'Mul':

            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Mul(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'QTZ_binary_channel_wise_mean_scaling':

            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = QTZ_binary_channel_wise_mean_scaling(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'ConcatOnDepth':
            axis = input_ops[input_ops_order[-1]]
            if current_format.index('C') != axis:
                ValueError('f{op_type} {node.name} concatenation is only supported on the depth axis')

            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = ConcatOnDepth(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )

            input_axis_name = input_ops_order[-1]
            nodes_to_remove.append(new_op.input_ops[input_axis_name])
            new_op.remove_input(input_axis_name)
        elif op_type == 'Maximum':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Maximum(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'DepthToSpace':
            bs = node.attribute('block_size')
            if not bs:
                raise ValueError(f'{op_type} {node.name} block size not specified')

            if not shape:
                attributes = {'block_size': bs[0]}
                shape = infer_shape(attributes)

            new_op = DepthToSpace(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                block_size=bs[0]
            )
        elif op_type == 'Split':
            num_split = node.attribute('num_split')[0]

            if not isinstance(num_split, int):
                raise ValueError(f'{op_type} {node.name} only supports integer value')

            if not shape:
                attributes = {'split': num_split}
                shape = infer_shape(attributes)

            new_op = Split(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
                num_split=num_split
            )
            input_axis_name = input_ops_order[0]
            nodes_to_remove.append(new_op.input_ops[input_axis_name])
            new_op.remove_input(input_axis_name)
        elif op_type == 'Pad':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = Pad(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        elif op_type == 'MatMul':
            if not shape:
                attributes = {}
                shape = infer_shape(attributes)

            new_op = MatMul(
                node.name,
                shape,
                dtype,
                input_ops,
                dimension_format=current_format,
            )
        else:
            raise UnsupportedNode(
                f'TensorFlow importer cannot convert {op_type} operator node!')

        return new_op
