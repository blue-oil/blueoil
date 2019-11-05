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
import copy
from textwrap import dedent

from core.data_types import *


class View(object):
    def __init__(self, op):
        self.op = op
        self.reuse_buffer_str = ''

    @property
    def rank(self):
        return len(self.node.shape)

    @property
    def shape(self):
        if self.op.dtype == QUANTIZED_PACKED():
            return '*'.join(map(lambda x: str(x), self.op.shape)) + f' / (sizeof({self.op.dtype.cpptype()}) * CHAR_BIT)'
        else:
            return '*'.join(map(lambda x: str(x), self.op.shape))

    @property
    def shape_list(self):
        return ','.join(map(lambda x: str(x), self.op.shape))

    def run(self):
        op = self.op
        input_ops = op.input_ops
        output_ops = op.output_ops
        inputs_string = self.inputs_to_string(input_ops)
        shape_string = self.shape_to_string(op.shape)

        if op.available_buffer != '':
            if len(op.output_ops.keys()) > 1:
                op_name = op.name + '_' + str(list(op.output_ops.keys())[0])
            else:
                op_name = op.name

            shape_string = self.shape_to_string(op.shape, channel_active=True)

            self.reuse_buffer_str = f"""TensorView<{op.dtype.cpptype()}, \
            MemoryLayout::{op.dimension}>::tensor_info_t<std::size_t> {op_name}_shape = {{ {shape_string} }};
            TensorView<{op.dtype.cpptype()}, \
            MemoryLayout::{op.dimension}> {op_name}({op.available_buffer}_raw, {op_name}_shape);"""

        if self.op.op_type == 'QTZ_binary_mean_scaling':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            return self.format_string(
                f"""
                func_QTZ_binary_mean_scaling({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'QTZ_linear_mid_tread_half':
            if len(input_ops) != 3:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_QTZ_linear_mid_tread_half({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'QTZ_binary_channel_wise_mean_scaling':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_QTZ_binary_channel_wise_mean_scaling({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Identity':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)
            return self.render_alias(op, input_ops, output_ops)

        elif self.op.op_type == 'Conv':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            x_op = input_ops['X']
            w_op = input_ops['W']

            ih = x_op.height
            iw = x_op.width
            oh = op.height
            ow = op.width
            b = 32
            od = op.channel
            pad = op.pads[0]
            stride = op.strides[0]
            nbit_qinput = 8 if x_op.op_type == 'Input' else 2

            if op.is_quantized and nbit_qinput == 2:
                qk_elems = w_op.data.shape[1]

                kh = self.op.kernel_height
                kw = self.op.kernel_width
                kd = x_op.channel
                k_elems = kh * kw * kd
                od = ((od + b - 1) // b) * b

                if input_ops['X'].op_type == 'Split':
                    for k, v in input_ops['X'].output_ops.items():
                        if v[0] == op:
                            inputs_string = str(input_ops['X'].name) + '_' + str(k)
                    inputs_string = inputs_string + ', ' + input_ops['W'].name
                else:
                    inputs_string = self.inputs_to_string(input_ops)

                if op.has_thresholds:
                    threshold = f'{op.name}_thresholds'
                    thresholds_addr = f'THRESHOLD_ADDR + {op.name}_thresholds_offset'
                    conv_func = 'func_QuantizedConv2DWithThreshold'
                    nbit_aqtz = self.op.a_quantizer[0].nbit
                    max_value = self.op.a_quantizer[0].max_v
                else:
                    threshold = 'nullptr'
                    thresholds_addr = '0'
                    conv_func = 'func_QuantizedConv2D'
                    nbit_aqtz = 2
                    max_value = 2.0

                # temporary: formula which derive number of qinput is not complete
                render_string = self.format_string(
                    f"""
                    Conv2D_struct.input_height = {ih};
                    Conv2D_struct.input_width = {iw};
                    Conv2D_struct.kernel_height = {kh};
                    Conv2D_struct.kernel_width = {kw};
                    Conv2D_struct.kernel_depth = {kd};
                    Conv2D_struct.kernel_elements = {k_elems};
                    Conv2D_struct.output_channels = {od};
                    Conv2D_struct.output_height = {oh};
                    Conv2D_struct.output_width = {ow};
                    Conv2D_struct.padding = {pad};
                    Conv2D_struct.stride_along_height = {stride};
                    Conv2D_struct.stride_along_width = {stride};

                    binConv2D_struct.normal_conv_params = Conv2D_struct;
                    binConv2D_struct.bin_input_extra_bits = 0;
                    binConv2D_struct.bin_input_bitwidth = {nbit_qinput};
                    binConv2D_struct.bin_kernel_ndata = {qk_elems};
                    binConv2D_struct.bin_input_nwords = {qk_elems};
                    binConv2D_struct.bin_input_ndata = {qk_elems}*{nbit_qinput};
                    binConv2D_struct.device_input_buf = device_input_buf;
                    binConv2D_struct.device_output_buf = device_output_buf;
                    binConv2D_struct.thresholds = {threshold};
                    binConv2D_struct.n_bit = {nbit_aqtz};
                    binConv2D_struct.max_value = {max_value};
                    binConv2D_struct.debug_name = "{op.name}";
#ifdef RUN_ON_FPGA
                    binConv2D_struct.device_kernel_phys_addr = KERNEL_ADDR + {op.name}_kernel_offset;
                    binConv2D_struct.device_thresholds_phys_addr = {thresholds_addr};
#endif

                    {conv_func}({inputs_string}, {op.name}, scaling_factors::{op.name}, binConv2D_struct);
                    """
                )

            else:
                # temporary
                # weight order is followed by tensorflow: "NHWC"
                kh = self.op.kernel_height
                kw = self.op.kernel_width
                kd = x_op.channel
                k_elems = kh * kw * kd

                if input_ops['X'].op_type == 'Split':
                    for k, v in input_ops['X'].output_ops.items():
                        if v[0] == op:
                            inputs_string = str(input_ops['X'].name) + '_' + str(k)
                    inputs_string = inputs_string + ', ' + input_ops['W'].name
                else:
                    inputs_string = self.inputs_to_string(input_ops)

                render_string = self.format_string(
                    f"""
                    Conv2D_struct.input_height = {ih};
                    Conv2D_struct.input_width = {iw};
                    Conv2D_struct.kernel_height = {kh};
                    Conv2D_struct.kernel_width = {kw};
                    Conv2D_struct.kernel_depth = {kd};
                    Conv2D_struct.kernel_elements = {k_elems};
                    Conv2D_struct.output_channels = {od};
                    Conv2D_struct.output_height = {oh};
                    Conv2D_struct.output_width = {ow};
                    Conv2D_struct.padding = {pad};
                    Conv2D_struct.stride_along_height = {stride};
                    Conv2D_struct.stride_along_width = {stride};

                    func_Conv2D({inputs_string}, {op.name}, Conv2D_struct);
                    """
                )

            return render_string

        elif self.op.op_type == 'Placeholder':
            return ""

        elif self.op.op_type == 'Const':
            return ""

        elif self.op.op_type == 'Minimum':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Minimum({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'MaxPool':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            x_op = input_ops['X']
            ih = x_op.height
            iw = x_op.width
            id = x_op.channel

            oh = op.height
            ow = op.width
            od = op.channel

            kh = op.kernel_height
            kw = op.kernel_width
            kd = 1

            elems = op.size
            pad = op.pads[0]
            stride = op.strides[0]
            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                MaxPool_struct.input_height = {ih};
                MaxPool_struct.input_width = {iw};
                MaxPool_struct.input_depth = {id};
                MaxPool_struct.kernel_height = {kh};
                MaxPool_struct.kernel_width = {kw};
                MaxPool_struct.kernel_depth = {kd};
                MaxPool_struct.output_elements = {elems};
                MaxPool_struct.output_channels = {od};
                MaxPool_struct.output_height = {oh};
                MaxPool_struct.output_width = {ow};
                MaxPool_struct.padding = {pad};
                MaxPool_struct.stride = {stride};

                func_MaxPool({inputs_string}, {op.name}, MaxPool_struct);
                """
            )

        elif self.op.op_type == 'MaxPoolWithArgmax':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            x_op = input_ops[0]

            ih = x_op.H
            iw = x_op.W
            kh = op.kernel_shape[1]
            kw = op.kernel_shape[2]
            kd = x_op.C
            oh = op.H
            ow = op.W
            elems = op.size
            pad = op.pads[0]
            stride = op.strides[0]
            inputs_string = self.inputs_to_string(input_ops)
            output = 'output'  # NotImplemented

            return self.format_string(
                f"""
                MaxPoolWithArgmax_struct.input.H = {ih};
                MaxPoolWithArgmax_struct.input.W = {iw};
                MaxPoolWithArgmax_struct.kernel.C = {id};
                MaxPoolWithArgmax_struct.kernel.H = {ih};
                MaxPoolWithArgmax_struct.kernel.W = {iw};
                MaxPoolWithArgmax_struct.output_elements = {elems};
                MaxPoolWithArgmax_struct.output.H = {oh};
                MaxPoolWithArgmax_struct.output.W = {ow};
                MaxPoolWithArgmax_struct.padding = {pad};
                MaxPoolWithArgmax_struct.stride = {stride};

                func_MaxPoolWithArgmax({inputs_string},
                                       {op.name},
                                       {output},
                                       MaxPoolWithArgmax_struct);
                """,
            )

        elif self.op.op_type == 'Unpooling':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            index_data_op = input_ops[1]

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Unpooling({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'RealDiv':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_RealDiv({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Abs':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)
            return self.render_alias(op, input_ops, output_ops)

        elif self.op.op_type == 'Max':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Max({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Mean':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)
            return self.render_alias(op, input_ops, output_ops)

        elif self.op.op_type == 'StopGradient':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)
            return self.render_alias(op, input_ops, output_ops)

        elif self.op.op_type == 'Sign':
            return ""

        elif self.op.op_type == 'Softmax':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Softmax({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Round':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Round({inputs_string}, {op.name});
                """,
            )

        elif self.op.op_type == 'Add':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Add({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Sub':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Sub({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Relu':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Relu({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'LeakyRelu':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            alpha = op.alpha

            return self.format_string(
                f"""
                func_LeakyRelu({inputs_string}, {op.name}, {alpha}f);
                """
            )

        elif self.op.op_type == 'Sqrt':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Sqrt({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'Quantize':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)
            shape_string = self.shape_to_string(op.shape)

            return self.format_string(
                f"""
                func_Quantize({inputs_string}, {op.name}, {shape_string});
                """
            )

        elif self.op.op_type == 'Scale':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)
            conv_scaling_factor = op.conv_scaling_factor

            return self.format_string(
                f"""
                func_Scale({inputs_string}, {conv_scaling_factor}, {op.name});
                """
            )

        elif self.op.op_type == 'AveragePool':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            x_op = input_ops['X']
            ih = x_op.height
            iw = x_op.width
            id = x_op.channel

            oh = op.height
            ow = op.width
            od = op.channel

            kh = op.kernel_height
            kw = op.kernel_width
            kd = 1

            pad = op.pads[0]
            stride = op.strides[0]

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                AveragePool_struct.input_height = {ih};
                AveragePool_struct.input_width = {iw};
                AveragePool_struct.input_depth = {id};
                AveragePool_struct.kernel_depth = {kd};
                AveragePool_struct.kernel_height = {kh};
                AveragePool_struct.kernel_width = {kw};
                AveragePool_struct.output_elements = {op.size};
                AveragePool_struct.output_channels = {od};
                AveragePool_struct.output_height = {oh};
                AveragePool_struct.output_width = {ow};
                AveragePool_struct.padding = {pad};
                AveragePool_struct.stride = {stride};

                func_AveragePool({inputs_string}, {op.name}, AveragePool_struct);
                """
            )

        elif self.op.op_type == 'BiasAdd':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Add({inputs_string}, {op.name});
                """
            )

        elif self.op.op_type == 'ExtractImagePatches':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            in_w = input_ops[0].W
            in_d = input_ops[0].C
            k_w = op.kernel_shape[1]
            stride_w = op.strides[1]

            inputs_string = self.inputs_to_string(input_ops)
            shape_string = self.shape_to_string(op.shape)

            args1 = f"{inputs_string}, {op.name}, "
            args2 = f"{k_w}, {stride_w}"
            return self.format_string(
                f"""
                func_ExtractImagePatches({args1}{args2});
                """
            )

        elif self.op.op_type == 'Reshape':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)
            in_shape = input_ops['data'].shape
            out_shape = op.shape

            shape_string = self.shape_to_string(op.shape)

            return self.format_string(
                f"""
                // Reshape from {in_shape} to {out_shape}'
                std::copy({input_ops["data"].name}.data(), {input_ops["data"].name}.data() + \
                {input_ops["data"].name}.size(), {op.name}.data());
                """
            )

        elif self.op.op_type == 'BatchNormalization':
            if len(input_ops) != 5:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_BatchNormalization({inputs_string}, {op.epsilon}, {op.name});
                """
            )

        elif self.op.op_type == 'SpaceToDepth':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)
            shape_string = self.shape_to_string(op.shape)

            bs = op.block_size
            x_op = input_ops['input']
            iw = x_op.width
            ic = x_op.channel

            args1 = f"{inputs_string}, {op.name}, "
            args2 = f"{bs}, {bs}"

            return self.format_string(
                f"""
                func_ExtractImagePatches({args1}{args2});
                """
            )
        elif self.op.op_type == 'Mul':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Mul({inputs_string}, {op.name});
                """
            )
        elif self.op.op_type == 'ConcatOnDepth':
            if len(input_ops) < 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)
            shape_string = self.shape_to_string(op.shape)

            number_of_inputs = len(input_ops)
            concat_input = {}
            for k, v in input_ops.items():
                if not v.is_variable:
                    concat_input[k] = v

            inputs_string = self.inputs_to_string(concat_input)

            return self.format_string(
                f"""
                func_ConcatOnDepth(std::make_tuple({inputs_string}), {op.name});
                """
            )
        elif self.op.op_type == 'Maximum':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Max({inputs_string}, {op.name});
                """
            )
        elif self.op.op_type == 'DepthToSpace':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            bs = op.block_size
            x_op = input_ops['input']
            iw = x_op.width
            ic = x_op.channel

            args1 = f"{inputs_string}, {op.name}, {iw}, "
            args2 = f"{ic}, {bs}, {bs}"

            return self.format_string(
                f"""
                func_DepthToSpace({args1}{args2});
                """
            )
        elif self.op.op_type == 'ResizeNearestNeighbor':

            inputs_string = self.inputs_to_string(input_ops)

            args1 = f"{inputs_string}, {op.name}"

            return self.format_string(
                f"""
                func_ResizeNearestNeighbor({args1});
                """
            )
        elif self.op.op_type == 'Split':
            if len(input_ops) != 1:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)
            outputs_string = self.outputs_to_string(op, output_ops)

            ns = op.num_splits

            depth_list_name = op.name + '_outputs_depth'
            depth_list = ', '.join(map(str, [op.channel for _ in range(ns)]))

            return self.format_string(
                f"""
                TensorView<{op.dtype.cpptype()}, MemoryLayout::{op.dimension}> {op.name}[] = {{ {outputs_string} }};
                T_UINT {depth_list_name}[] = {{ {depth_list} }};
                func_Split({inputs_string}, {op.name}, {depth_list_name}, {ns});
                """
            )
        elif self.op.op_type == 'Pad':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            a_op = input_ops['A']

            return self.format_string(
                f"""
                func_Pad({inputs_string}, {op.name});
                """
            )
        elif self.op.op_type == 'MatMul':
            if len(input_ops) != 2:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(
                f"""
                func_Matmul({inputs_string}, {op.name});
                """
            )
        elif self.op.op_type == 'Lookup':
            if len(input_ops) != 3:
                self.raise_invalid_args_exception(op, input_ops, output_ops)

            inputs_string = self.inputs_to_string(input_ops)

            return self.format_string(f"""func_Lookup({inputs_string}, {op.name});""")

        raise TypeError(f"{self.op.op_type} is not supported in View.run().")

    def render_alias(self, op, input_ops, output_ops):
        if len(input_ops) != 1:
            self.raise_invalid_args_exception(op, input_ops, output_ops)
        else:
            return f'{op.dtype.cpptype()}* {op.name} = {input_ops["input"].name};'

    def format_string(self, string):
        if self.reuse_buffer_str:
            string = self.reuse_buffer_str + '\n' + string

        return dedent(string).strip()

    def inputs_to_string(self, inputs):
        return ', '.join(map(lambda x: str(x.name), inputs.values()))

    def outputs_to_string(self, node, outputs):
        return ', '.join(map(lambda x: str(node.name + '_' + x), outputs.keys()))

    def shape_to_string(self, shape, channel_active=False):
        shape_copied = copy.copy(shape)

        # temporary
        if not channel_active and len(shape) > 1:
            del(shape_copied[0])
        return ', '.join(map(lambda x: str(x), shape_copied))

    def params_to_string(self, params):
        raise NotImplemented

    def raise_invalid_args_exception(self, op, input_ops, output_ops):
        error_message = self.format_string(
            f"""
            InvalidArgsException: name: {op.name}, op: {op.op_type},
            This op was taken {len(input_ops)} inputs and {len(input_ops)} outputs ops!!!
            """
        )

        print(error_message)
        raise Exception
