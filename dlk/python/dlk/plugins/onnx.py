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
from collections import defaultdict
import onnx.numpy_helper
from onnx import ModelProto, GraphProto, NodeProto, AttributeProto, \
    ValueInfoProto, TypeProto, TensorProto, TensorShapeProto
from typing import cast, Any, Callable, Dict, List, Optional, Set, Tuple, Type
from core.graph import Graph
from core.data_types import DataType, Float32, Float64, Int8, Int16, Int32, \
    Int64, Uint8, Uint16, Uint32, Uint64, Bool, String
from core.exceptions import UnsupportedNode, UnsupportedDataType
from core.operators import Operator, Conv, Identity, QTZ_binary_mean_scaling, \
    BatchNormalization, QTZ_linear_mid_tread_half, Add, \
    Pool, MaxPool, AveragePool, Reshape, Softmax, Transpose
import core.operators as dlk_op
import numpy as np
import importlib
import functools


DLK_DTYPE_MAP: Dict[str, Optional[DataType]] = {
    # any
    'UNDEFINED': None,
    # primitives
    'FLOAT': Float32(),
    'INT': Int32(),
    'UINT8': Uint8(),
    'INT8': Int8(),
    'UINT16': Uint16(),
    'INT16': Int16(),
    'INT32': Int32(),
    'INT64': Int64(),
    'f': Float32(),
    'i': Int32(),

    # primitive vector
    'FLOATS': None,
    'INTS': None,

    # custom
    'BOOL': Bool(),
    'STRING': String(),
    'FLOAT16': None,
    'DOUBLE': Float64(),
    'UINT32': Uint32(),
    'UINT64': Uint64(),
    'COMPLEX64': None,
    'COMPLEX128': None,
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

# map between op_types in ONNX and DLK.
DLK_OPERATOR_MAP: Dict[str, str] = {
    # The different names must be written here.
    # The same names does not have to.
}

# Map from op_type to its attributres
DLK_ATTRIBUTE_MAP: Dict[str, List[str]] = {
    'Identity': [],
    'QTZ_binary_mean_scaling': [],
    'Transpose': ['perm'],
    'Conv': ['dilations', 'pads', 'strides', 'kernel_shape'],
    'BatchNormalization': ['epsilon', 'is_test'],
    'QTZ_linear_mid_tread_half': [],
    'MaxPool': ['kernel_shape', 'pads', 'strides'],
    'AveragePool': ['kernel_shape', 'pads', 'strides'],
    'Relu': [],
    'Flatten': ['axis'],
    'Dropout': ['ratio'],
    'Gemm': ['alpha', 'beta', 'transA', 'transB'],
}


# functions that conveerts ONNX attributes to DLK attriburtes
def attr_conv_scalars(op: Callable[[str], Any]) ->Callable[[Dict[str, str]], Dict[str, Any]]:
    def convert(attrs: Dict[str, str]) -> Dict[str, Any]:
        return {k: op(v) for k, v in attrs.items()}

    return convert


def attr_conv_lists(op: Callable[[str], Any]) ->Callable[[Dict[str, List[str]]], Dict[str, List[Any]]]:
    def convert(attrs: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        return {k: list(map(op, v)) for k, v in attrs.items()}

    return convert


def ident(obj: Any) -> Any:
    return obj


def attr_conv_batch_norm(attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {n: attrs[n] for n in DLK_ATTRIBUTE_MAP['BatchNormalization'] if attrs.get(n)}


def attr_conv_gemm(attrs: Dict[str, Any]) -> Dict[str, Any]:
    float_attrs = {k: float(v) for k, v in attrs.items() if k in ['alpha', 'beta']}
    bool_attrs = {k: bool(v) for k, v in attrs.items() if k in ['transA', 'transB']}

    float_attrs.update(bool_attrs)
    return float_attrs


DLK_ATTRIBUTE_CONVERTER_MAP: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    'Identity': ident,
    'QTZ_binary_mean_scaling': ident,
    'Transpose': attr_conv_lists(int),
    'Conv': attr_conv_lists(int),
    'BatchNormalization': attr_conv_batch_norm,
    'QTZ_linear_mid_tread_half': ident,
    'MaxPool': attr_conv_lists(int),
    'AveragePool': attr_conv_lists(int),
    'Relu': ident,
    'Flatten': attr_conv_scalars(int),
    'Dropout': attr_conv_scalars(float),
    'Gemm': attr_conv_gemm,
}


class AttributeProtoWrapper(object):

    """
    ATTRIBUTE_TYPE_MAP = {
        0:  'UNDEFINED',
        1:  'FLOAT',
        2:  'INT',
        3:  'STRING',
        4:  'TENSOR',
        5:  'GRAPH',
        6:  'FLOATS',
        7:  'INTS',
        8:  'STRINGS',
        9:  'TENSORS',
        10: 'GRAPHS',
    }
    """
    ATTRIBUTE_TYPE_MAP = {v: k for k,
                          v in AttributeProto.AttributeType.items()}

    ATTRIBUTE_VALUE_MAP = {
        1: 'f',
        2: 'i',
        3: 's',
        4: 't',
        5: 'g',
        6: 'floats',
        7: 'ints',
        8: 'strings',
        9: 'tensors',
        10: 'graphs',
    }

    def __init__(self, ap: AttributeProto) -> None:
        self.ap_: AttributeProto = ap

    @property
    def name(self) -> str:
        """Return attribute's name info."""
        return self.ap_.name

    def get_dtype(self) -> str:
        """Get attribute's data type info."""
        return type(self).ATTRIBUTE_TYPE_MAP[self.ap_.type]

    @property
    def data(self) -> Any:
        """Return attribute's data."""
        value_name: str = type(self).ATTRIBUTE_VALUE_MAP[self.ap_.type]
        return getattr(self.ap_, value_name)


class NodeProtoWrapper(object):

    def __init__(self, np: NodeProto) -> None:
        self.np_: NodeProto = np
        self.attributes = [AttributeProtoWrapper(x) for x in np.attribute]

    @property
    def name(self) -> str:
        """Return the name corresponding to the node."""
        return self.np_.name

    @property
    def op_type(self) -> str:
        """Return the name corresponding to the node."""
        return self.np_.op_type

    @property
    def inputs(self) -> List[str]:
        """Return the name corresponding to the node."""
        return [x for x in self.np_.input]

    @property
    def outputs(self) -> List[str]:
        """Return the name corresponding to the node."""
        return [x for x in self.np_.output]

    def attribute(self, attr_name: str) -> Any:
        """Return the name corresponding to the node."""
        attrs = [x for x in self.attributes if x.name == attr_name]

        return attrs[0].data if attrs else None

    @property
    def onnx_object(self) -> TensorProto:
        """Return onnx object."""
        return self.np_


class TensorProtoWrapper(object):

    """
    DATA_TYPE_MAP = {
        0:  'UNDEFINED',
        1:  'FLOAT',
        2:  'UINT8',
        3:  'INT8',
        4:  'UINT16',
        5:  'INT16',
        6:  'INT32',
        7:  'INT64',
        8:  'STRING',
        9:  'BOOL',
        10: 'FLOAT16',
        11: 'DOUBLE',
        12: 'UINT32',
        13: 'UINT64',
        14: 'COMPLEX64',
        15: 'COMPLEX128',
    }
    """
    DATA_TYPE_MAP = {v: k for k, v in TensorProto.DataType.items()}

    def __init__(self, tp: TensorProto) -> None:
        self._tp: TensorProto = tp
        self._data = onnx.numpy_helper.to_array(tp)

    @property
    def name(self) -> str:
        """Return the name corresponding to the node."""
        return self._tp.name

    def get_dtype(self) -> str:
        """Get data type info."""
        return type(self).DATA_TYPE_MAP[self._tp.data_type]

    @property
    def onnx_object(self) -> TensorProto:
        """Return onnx object."""
        return self._tp

    def get_shape(self) -> List[str]:
        """Get shape info."""
        return [x for x in self._tp.dims]

    def set_shape(self, val: List[str]) -> None:
        """Set shape info."""
        raise NotImplemented

    def get_data(self) -> List[Any]:
        """Get data."""
        return self._data

    def set_data(self, val: List[str]) -> None:
        """Set shape info."""
        raise NotImplementedError


class ValueInfoProtoWrapper(object):
    """ValueInfo class that corresponds to ValueInfoProto in ONNX."""

    def __init__(self, vip: ValueInfoProto) -> None:
        """Init the object with ONNX ValueInfoProto (in JSON).

        Parameters
        ----------
        vip : ValueInfoProto
            ValueInfoProto in ONNX (in JSON format)

        """
        self.vip_: ValueInfoProto = vip

    @property
    def name(self) -> str:
        """Return the name corresponding to the node."""
        return self.vip_.name

    @property
    def onnx_object(self) -> ValueInfoProto:
        """Return onnx object."""
        return self.vip_

    @property
    def tensor_type(self) -> TypeProto.Tensor:
        """Get shape info."""
        typep: TypeProto = self.vip_.type
        return typep.tensor_type

    def get_dtype(self) -> DataType:
        """Get data type info."""
        dtype_idx = self.tensor_type.elem_type
        dtype_str = TensorProtoWrapper.DATA_TYPE_MAP[dtype_idx]
        if DLK_DTYPE_MAP[dtype_str] is None:
            raise UnsupportedDataType(f'Type {dtype_str} is not supported.')
        return DLK_DTYPE_MAP[dtype_str]  # type: ignore

    def get_shape(self) -> List[str]:
        """Get shape info."""
        return [x.dim_value for x in self.tensor_type.shape.dim]

    def set_shape(self, val: List[str]) -> None:
        """Set shape info."""
        raise NotImplemented


class Node(NodeProtoWrapper):
    def __init__(self, np: NodeProto, vinfo: Optional[ValueInfoProtoWrapper]) -> None:
        # Currently, onnx-tensorflow has a bug in wrighting value_info for nodes.
        # Plus, PyTorch does not provide any value_info. So we decided not to use
        # the value_info for all nodes.
        #
        # self.vinfo = vinfo
        self.vinfo = None
        super().__init__(np)

    @property
    def name(self) -> str:
        """Return the output instead of the name on its NodeProto"""
        return self.outputs[0]

    def get_dtype(self) -> Optional[DataType]:
        """Get data type info."""
        if self.vinfo:
            return self.vinfo.get_dtype()
        else:
            return None

    def get_shape(self) -> Optional[List[str]]:
        """Get shape info."""
        if self.vinfo:
            return self.vinfo.get_shape()
        else:
            return None


class Input(ValueInfoProtoWrapper):
    def __init__(self, vip: ValueInfoProto, tpw: Optional[TensorProtoWrapper]) -> None:
        super().__init__(vip)
        self.initializer = tpw

    @property
    def is_placeholder(self) -> bool:
        return self.initializer is None

    def get_data(self) -> np.ndarray:
        if not self.is_placeholder:
            shape = super().get_shape()
            np_dtype = self.get_dtype().nptype()
            if self.initializer:

                return np.array(self.initializer.get_data(), dtype=np_dtype).reshape(shape)
            else:
                raise ValueError(
                    f'something strange has happend in getting data from {self.name}...')

        else:
            raise ValueError(
                f'{self.name} is a placeholder, which does\'t have no data...')


class Output(ValueInfoProtoWrapper):
    pass


class Importer(object):

    @classmethod
    def make_graph(cls, onnx_mp: ModelProto) -> Graph:
        importer = Importer(onnx_mp)
        graph = Graph()

        importer.add_all_nodes(graph)
        return graph

    def __init__(self, onnx_mp: ModelProto) -> None:
        """Init the graph.

        Prameters
        ---------
        onnx_mp : ModelProto
            ONNX object

        """
        self.onnx_gp: GraphProto = onnx_mp.graph

        self.vinfo_dic = {v.name: ValueInfoProtoWrapper(
            v) for v in self.onnx_gp.value_info}
        self.init_dic = {v.name: TensorProtoWrapper(
            v) for v in self.onnx_gp.initializer}

        self.in_lst: List[Input] = [
            Input(v, self.init_dic.get(v.name)) for v in self.onnx_gp.input]
        self.out_lst: List[Output] = [Output(v) for v in self.onnx_gp.output]
        self.node_lst: List[Node] = []

        for v in self.onnx_gp.node:
            if self.vinfo_dic:
                vip = self.vinfo_dic.get(v.name)
                if vip is None:
                    # Sometimes node doen't have the name as it's optional.
                    # now we use the 1st output instead of their name.
                    vinfo_output_name: str = v.output[0]
                    vinfo_names: List[str] = [
                        s for s in self.vinfo_dic.keys() if vinfo_output_name.startswith(s)]
                    assert len(vinfo_names) == 1, \
                        f'vinfo should have only one name: {vinfo_names} for {vinfo_output_name}'
                    vinfo_name = vinfo_names[0]
                    vip = self.vinfo_dic.get(vinfo_name)
                self.node_lst.append(Node(v, vip))
            else:
                self.node_lst.append(Node(v, None))

        self.validate_onnx()

        self.input_hash_table = self.construct_input_hash_table(
            self.node_lst, self.in_lst)
        self.output_hash_table = self.construct_output_hash_table(
            self.node_lst, self.out_lst)

    def validate_onnx(self) -> None:
        """Validate if the ONNX object is proper."""
        gp: GraphProto = self.onnx_gp
        assert len(gp.input) == len(self.in_lst)
        assert len(gp.output) == len(self.out_lst)
        assert len(gp.node) == len(self.node_lst)
        assert len(gp.initializer) == len(self.init_dic)
        assert len(gp.value_info) == len(self.vinfo_dic)

    def convert_operator(self, op_type: str) -> str:
        """Convert ONNX operator type to DLK's one."""
        dlk_op_type = DLK_OPERATOR_MAP.get(op_type)
        return dlk_op_type if dlk_op_type else op_type

    def create_new_op(self, node: Any, op_dic: Dict[str, Operator], current_format: str,
                      input_format_list: List[str]) -> Operator:
        new_op: Operator

        if isinstance(node, Node):  # operator nodes
            new_op = self.create_new_node(node, op_dic, current_format, input_format_list)

        else:  # Input, Output or Constant
            shape: List[int] = list(map(int, node.get_shape()))
            dtype = node.get_dtype()

            if isinstance(node, Input):
                if node.is_placeholder:  # op_type = 'Input'
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
                # get input operators
                input_ops = {k: op_dic[n.name] for n, k in zip(
                    self.find_inputs(node), dlk_op.Output.input_names)}

                new_op = dlk_op.Output(
                    'output',  # node.name,
                    shape,
                    dtype,
                    input_ops,
                )

        return new_op

    def add_all_nodes(self, graph: Graph) -> None:
        # add nodes
        visited: Set[Any] = set()
        self.add_node_to_graph_recursive(self.out_lst[0], graph, visited, 'NCHW')

    def _get_format(self, node: Any, output_format: str) -> Tuple[str, List[str]]:
        """Get the dimension format, like 'NCHW', 'HWCN', 'NHWC', etc."""

        _default_format = 'NCHW'  # ONNX standard for input
        _default_w_format = 'OIHW'  # ONNX standard for weight

        if isinstance(node, Node):
            op_type = node.op_type
            if op_type == 'Conv':
                # the ONNX standard for input and weight
                return _default_format, [_default_format, _default_w_format, 'N']

            elif op_type == 'BatchNormalization':
                # the ONNX standard and vector values
                return _default_format, [_default_format, 'C', 'C', 'C', 'C']

            elif op_type in {'MaxPool', 'AveragePool'}:
                return _default_format, [_default_format]  # the ONNX standard

            elif op_type == 'Transpose':
                perm = list(node.attribute("perm"))
                inv_perm = [perm.index(i) for i in range(len(perm))]  # inverse permutation
                input_format = functools.reduce(
                    lambda x, y: x + y, [output_format[i] for i in inv_perm])
                return output_format, [input_format]

            elif op_type == 'QTZ_linear_mid_tread_half':
                return output_format, [output_format, '1', '1']  # two scalar constants

            elif op_type == 'Gemm':
                return output_format, ['', '', '']  # three inputs

            else:
                return output_format, [output_format]

        else:  # Input or Output
            return output_format, [output_format]

    def add_node_to_graph_recursive(self, current: Any, graph: Graph, visited: Set[Any], data_format: str)\
            -> Operator:
        if current in visited:
            return current

        added_op_dic: Dict[str, Operator] = {}

        current_format, input_formats = self._get_format(current, data_format)
        inputs = self.find_inputs(current)
        for input, format in zip(inputs, input_formats):
            op = self.add_node_to_graph_recursive(input, graph, visited, format)
            added_op_dic[op.name] = op

        op = self.create_new_op(current, added_op_dic, current_format, input_formats)

        graph.add_op(op)

        visited.add(current)
        return op

    def construct_input_hash_table(self, node_list: List[Any], input_list: List[Any]) -> Dict[str, Any]:
        hash_table: Dict[str, Any] = defaultdict(lambda: [])

        for node in node_list:
            for idx in node.outputs:
                hash_table[idx].append(node)

        for x in input_list:
            hash_table[x.name].append(x)

        return hash_table

    def construct_output_hash_table(self, node_lst: List[Any], out_lst: List[Any]) -> Dict[str, Any]:
        hash_table: Dict[str, Any] = defaultdict(lambda: [])

        for node in node_lst:
            for idx in node.inputs:
                hash_table[idx].append(node)

        for x in out_lst:
            hash_table[x.name].append(x)

        return hash_table

    def find_inputs(self, node: Any) -> List[Any]:
        inputs: List[Any] = []
        if isinstance(node, Node):
            for idx in node.inputs:
                for node in self.input_hash_table[idx]:
                    inputs.append(node)

        elif isinstance(node, Output):
            for node in self.input_hash_table[node.name]:
                inputs.append(node)

        return inputs

    def find_outputs(self, node: Any) -> List[Any]:
        outputs: List[Any] = []
        if isinstance(node, Node):
            for idx in node.outputs:
                for node in self.output_hash_table[idx]:
                    outputs.append(node)

        elif isinstance(node, Input):
            for node in self.output_hash_table[node.name]:
                outputs.append(node)

        return outputs

    def create_new_node(self, node: Node, op_dic: Dict[str, Operator], current_format: str,
                        input_format_list: List[str]) -> Operator:
        """Create a new operator node.

        Parameters
        ----------
        node : Node
            ONNX node corresponding to the operator

        op_dic : Dict from str to Operator
            Dict of preceding operators

        current_format : str
            Dimension format of this operator. The interpretasion depends on each operator.

        input_format_list : List of str
            List of dimension formats for inputs. The interpretasion depends on each operator.

        Returns
        -------
        new_op : Optional[Operator]
            Newly created dlk operator, or None if the node is Transpose.

        """
        op_type = self.convert_operator(node.op_type)
        # check if the operator is supported in DLK
        try:
            module = importlib.import_module('core.operators')
            class_def = getattr(module, op_type)
        except AttributeError as ae:
            message = f'Operator {op_type} is not supported.'
            raise UnsupportedNode(message)
        else:
            pass
            # print(op_type)  # debug

        # Create new op accordingly for the onnx op
        new_op: Operator

        def get_inputs(cdef: Type[Operator], node: Any, op_type: str) -> Dict[str, Operator]:
            input_names = cdef.input_names
            assert len(input_names) >= len(node.inputs), \
                f'node {node.name} has {len(node.inputs)}, while there are {len(input_names)} in {cdef.__name__}'

            return {n: op_dic[op] for n, op in zip(input_names, node.inputs)}

        input_ops = get_inputs(class_def, node, op_type)

        # Here find the shape and dtype for the op

        def infer_shape(attrs: Dict[str, Any]) -> List[int]:
            shapes = node.get_shape()
            if shapes is not None:
                return list(map(int, shapes))
            else:
                shape_dict = {
                    n: input_ops[n].shape for n in class_def.input_names if input_ops.get(n)}
                return class_def.infer_shape(shape_dict, current_format, input_format_list, attrs)

        def infer_dtype() -> DataType:
            if node.get_dtype():
                return node.get_dtype()  # type: ignore
            else:
                return list(input_ops.values())[0].dtype

        def get_dlk_attributes(node: 'Node') -> Dict[str, Any]:
            f = DLK_ATTRIBUTE_CONVERTER_MAP[node.op_type]
            attr_names = DLK_ATTRIBUTE_MAP[node.op_type]
            return f({n: node.attribute(n) for n in attr_names})

        attributes = get_dlk_attributes(node)
        shape: List[int] = infer_shape(attributes)
        dtype: DataType = infer_dtype()

        return class_def(node.name, shape, dtype, input_ops,
                         dimension_format=current_format, **attributes)
