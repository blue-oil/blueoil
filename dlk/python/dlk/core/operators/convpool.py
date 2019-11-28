# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
"""Definition of operators."""
from .base import *
from .quantization import Quantizer

from typing import Any, Dict, Optional

Ops = Dict[str, 'Operator']


class Conv(Operator):
    """Convolution operator.

    The convolution operator consumes an input tensor and a weight, and computes the output.
    Currently this is only for 2-D images.

    Inputs
    ------
    X
        Input data tensor from previous layer. Note that this is for the 2D image.

    W
        The weight tensor that will be used in the convolutions.

    B (Optional)
        1D bias.

    Outputs
    -------
    Y
        Output data tensor that contains the result of the convolution.
        The output dimensions are functions of the kernel size, stride size, and pad lengths.

    Attributes (Optional constructor parameters)
    ----------
    kernel_shape : list of ints
        The shape of the convolution kernel. If not present, should be inferred from input W.

    kernel_dimensions : int
        The dimension of the input. The default value is 2, which means 2-D image.

    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    kernel_dim_format : str
        Dimension denotation, which must consists of 'H' and 'W', where 'H' and 'W' are the
        height and width of input image. The default is 'HW'.

    dilations : list of ints
        Dilation value along each axis of the filter. If not present, the dilation defaults to 1
        along each axis.

    pads : list of ints
        Padding for the beginning and ending along each axis, it can take any value greater than
        or equal to 0. The value represent the number of pixels added to the beginning and end
        part of the corresponding axis.
        `pads` format should be as follow [x1_begin, x2_begin, x1_end, x2_end], where
        xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number
        of pixels added at the end of axis `i`.
        If not present, the padding defaults to 0 along start and end of each axis.

    strides : list of ints
        Stride along each axis. If not present, the stride defaults to 1 along each axis.

    quantized : bool
        Whether it is quantized. If not present, the switch defaults to False.

    thresholds : list of floats
        Threshold values that are used in threshold skipping. If not present, this defaults to
        an empty list. Ignored if `quantized` is not true.

    """

    _input_names = ['X', 'W', 'B']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 kernel_shape: List[int] = [],
                 kernel_dimensions: int = 2,
                 dimension_format: str = 'NHWC',
                 kernel_dim_format: str = 'HW',
                 dilations: List[int] = [1, 1],
                 pads: List[int] = [0, 0, 0, 0],
                 strides: List[int] = [1, 1],
                 quantized: bool = False,
                 thresholds: List[float] = []) -> None:

        # currently, only 2-D is supported.
        if kernel_dimensions != 2:
            raise NotImplementedError(f"Convolution for {kernel_dimensions}-D is not defined!")

        self._num_dimensions = kernel_dimensions
        self._dilations = dilations
        self.kernel_index_H = kernel_dim_format.index('H') if 'H' in kernel_dim_format else None
        self.kernel_index_W = kernel_dim_format.index('W') if 'W' in kernel_dim_format else None
        if self.kernel_index_H is None or self.kernel_index_W is None:
            ValueError(f'kernel dimension format {kernel_dim_format} is not supported.')
        w = input_ops['W']
        k_list = [w.height, w.width]
        perm: List[int] = [self.kernel_index_H, self.kernel_index_W]  # type: ignore
        self.kernel_shape = kernel_shape if kernel_shape else [k_list[i] for i in perm]
        self._kernel_dim_format = kernel_dim_format
        self._pads = pads
        self._strides = strides
        self._is_quantized = quantized
        self._a_quantizer: List['Quantizer'] = []
        self._quantizer: Optional['Quantizer'] = None
        self._thresholds = thresholds
        self._original_shape = shape
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)
        # if kernel shape is not assigned, estimate kernel shape from input W's shape

    def _check_consistency(self) -> None:
        super()._check_consistency()
        self._assert(len(self.shape) == self._num_dimensions + 2,
                     f'{self.name} has illegal shape {self.shape}')
        self._assert(len(self.kernel_shape) == self._num_dimensions,
                     f'Illegal kernel shape: {self.kernel_shape} for {self._num_dimensions}-D.')
        self._assert(len(self._kernel_dim_format) == self._num_dimensions)
        self._assert(len(self.dilations) == self._num_dimensions)
        self._assert(len(self.pads) == self._num_dimensions + 2)
        self._assert(len(self.strides) == self._num_dimensions)
        # self._assert(len(self.dimension) == len(self.shape))

        # check the shape consistency
        if not self._is_quantized:
            in_H = self._input_ops['W'].height
            in_W = self._input_ops['W'].width
            mH = f'The kernel height {self.kernel_height} does not match the weight height {in_H}'
            mH += f' at operator {self.name}.'
            mH += f'\nThe kernel shape is {self.kernel_shape}, and the weight shape is {self._input_ops["W"].shape}.'
            mH += f'¥nThe weight format is {self._input_ops["W"].dimension}.'
            self._assert(in_H == self.kernel_height, mH)
            mW = f'The kernel width {self.kernel_width} does not match the weight width {in_W} at operator {self.name}'
            mW += f'\nThe kernel shape is {self.kernel_shape}, and the weight shape is {self._input_ops["W"].shape}.'
            mW += f'\nThe weight format is {self._input_ops["W"].dimension}.'
            self._assert(in_W == self.kernel_width, mW)
        if self.kernel_index_H is not None and self.index_H is not None:
            pad_H = self.pads[self.kernel_index_H] + \
                self.pads[self.kernel_index_H + self._num_dimensions]
            stride_H = self.strides[self.kernel_index_H]
            dilation_H = self.dilations[self.kernel_index_H]
            # print(self.name, ' input dimension: ', self.input_ops['X'].dimension)
            # print(self.name, ' input shape: ', self.input_ops['X'].shape)
            # print(self.name, ' input height: ', self.input_ops['X'].height)
            # print(self.name, ' weight shape: ', self.input_ops['W'].shape)
            # print(self.name, ' weight height: ', self.input_ops['W'].height)
            # print(self.name, ' kernel height: ', self.kernel_height)
            # print(self.name, ' pad_H: ', pad_H)
            # print(self.name, ' stride_H: ', stride_H)
            # print(self.name, ' output height: ', self.height, ' (', self.index_H, ' in ', self.dimension, ')')
            output_H_base = self.input_ops['X'].height + pad_H - \
                (self.kernel_height + 2 * (dilation_H - 1))
            # print(self.name, ' output_H_base ', output_H_base)
            output_H, output_H_rest = divmod(output_H_base, stride_H)
            output_H += 1
            message = f'Conv operator {self.name} does not match the height:'
            message += f' inferred as {output_H} but got {self.height}.'
            self._assert(output_H == self.height, message)
            # self._assert(output_H_rest == 0,
            #              f'Conv operator {self.name} should adjust the height pad to plus {output_H_rest}.')
            if output_H_rest > 0:
                print(f'mispadding height at {self.name}: {output_H_rest}')

        if self.kernel_index_W is not None and self.index_W is not None:
            pad_W = self.pads[self.kernel_index_W] + \
                self.pads[self.kernel_index_W + self._num_dimensions]
            stride_W = self.strides[self.kernel_index_W]
            dilation_W = self.dilations[self.kernel_index_W]
            # print(self.name, ' input shape: ', self.input_ops['X'].shape)
            # print(self.name, ' input width: ', self.input_ops['X'].width)
            # print(self.name, ' weight shape: ', self.input_ops['W'].shape)
            # print(self.name, ' weight width: ', self.input_ops['W'].width)
            # print(self.name, ' pad_W: ', pad_W)
            # print(self.name, ' stride_W: ', stride_W)
            # print(self.name, ' output width: ', self.width, ' (', self.index_W, ' in ', self.dimension, ')')
            output_W_base = self.input_ops['X'].width + pad_W - \
                (self.kernel_width + 2 * (dilation_W - 1))
            output_W, output_W_rest = divmod(output_W_base, stride_W)
            output_W += 1
            message = f'Conv operator {self.name} does not match the width:'
            message += f' inferred as {output_W} but got {self.width} in {self.dimension} format.\n'
            message += f'The shape is {self.shape}.'
            self._assert(output_W == self.width, message)
            # self._assert(output_W_rest == 0,
            #              f'Conv operator {self.name} should adjust the width pad to plus {output_W_rest}.')
            if output_W_rest > 0:
                print(f'mispadding width at {self.name}: {output_W_rest}')

    @property
    def kernel_dimensions(self) -> int:
        """Get the number of dimensions."""
        return self._num_dimensions

    @property
    def dilations(self) -> List[int]:
        """Get dilations."""
        return self._dilations

    @property
    def pads(self) -> List[int]:
        """Get pads."""
        return self._pads

    @property
    def strides(self) -> List[int]:
        """Get strides."""
        return self._strides

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def is_quantized(self) -> bool:
        """Return if this operator is quantized.

        Currently it always returns False, as quantized version is not supported yet.
        """
        return self._is_quantized

    @is_quantized.setter
    def is_quantized(self, val: bool) -> None:
        self._is_quantized = val

    @property
    def scaling_factor(self) -> float:
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, val: float) -> None:
        self._scaling_factor = val

    @property
    def a_quantizer(self) -> List[Quantizer]:
        return list(self._a_quantizer)

    @a_quantizer.setter
    def a_quantizer(self, op_lst: List[Quantizer]) -> None:
        self._a_quantizer = list(op_lst)

    @property
    def quantizer(self) -> Optional[Quantizer]:
        return self._quantizer

    @quantizer.setter
    def quantizer(self, op: Optional[Quantizer]) -> None:
        self._quantizer = op

    @property
    def kernel_height(self) -> int:
        """Return the height in the kernel shape."""
        if self.kernel_index_H is not None:
            return self.kernel_shape[self.kernel_index_H]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_height property.')

    @property
    def kernel_width(self) -> int:
        """Return the weight in the kernel shape."""
        if self.kernel_index_W is not None:
            return self.kernel_shape[self.kernel_index_W]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_width property.')

    # @property
    # def kernel_channels(self) -> int:
    #     if not self.is_quantized:
    #         return self.kernel_shape[self.index_C]
    #     else:
    #         raise NotImplementedError

    # @property
    # def kernel_batchsize(self) -> int:
    #     if not self.is_quantized:
    #         return self.kernel_shape[self.index_N]
    #     else:
    #         raise NotImplementedError

    @property
    def has_thresholds(self) -> bool:
        return self.is_quantized and len(self._thresholds) > 0

    @property
    def thresholds(self) -> List[float]:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, val: List[float]) -> None:
        self._thresholds = val

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        """Infer its shape from inputs' shapes."""
        idx_N = input_formats[0].index('N')  # from input X
        idx_C = input_formats[1].index('O')  # from weight W
        idx_H = input_formats[0].index('H')  # from input X
        idx_W = input_formats[0].index('W')  # from input X

        N = lists['X'][idx_N]
        C = lists['W'][idx_C]

        # calc H and W
        dilations = attrs['dilations']
        pads = attrs['pads']
        strides = attrs['strides']
        kernel_shape = attrs['kernel_shape']

        # H
        pads_H = pads[0] + pads[2]
        input_H = lists['X'][idx_H] + pads_H
        window_H = kernel_shape[0] + 2 * (dilations[0] - 1)
        stride_H = strides[0]
        H, rest_H = divmod((input_H - window_H), stride_H)
        H += 1
        # assert rest_H == 0, f'difference in height: {rest_H} at {cls.__name__}'

        # W
        pads_W = pads[1] + pads[3]
        input_W = lists['X'][idx_W] + pads_W
        window_W = kernel_shape[1] + 2 * (dilations[1] - 1)
        stride_W = strides[1]
        W, rest_W = divmod((input_W - window_W), stride_W)
        W += 1
        # assert rest_W == 0, f'difference in width: {rest_W} at {cls.__name__}'

        NCHW = [N, C, H, W]
        return [NCHW[i] for i in [format.index(s) for s in 'NCHW']]

    @property
    def preserve_quantization(self) -> bool:
        return True

    def restore_shape(self):
        if self.a_quantizer:
            real_ch = self._original_shape[3]
            data_per_ch = 2 ** self.a_quantizer[0].nbit
            del self._thresholds[real_ch * data_per_ch:]
        self.update_shape(self._original_shape, 'NHWC')


class Pool(Operator):
    """Pooling operator.

    This is a base class and must not be instantiated directly.

    """

    _input_names = ['X']
    _output_names = ['Y']

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: DataType,
                 input_ops: Ops,
                 dimension_format: str = 'NHWC',
                 kernel_shape: List[int] = [2, 2],
                 kernel_dim_format: str = 'HW',
                 dimensions: int = 2,
                 pads: List[int] = [0, 0, 0, 0],
                 strides: List[int] = [1, 1]) -> None:
        """Init the pooling operator."""
        if dimensions != 2:
            raise NotImplementedError

        self.kernel_dims = dimensions
        self.kernel_dim_format = kernel_dim_format
        self.kernel_shape = kernel_shape
        self._pads = pads
        self.strides = strides
        self.kernel_index_H = kernel_dim_format.index('H') if 'H' in kernel_dim_format else None
        self.kernel_index_W = kernel_dim_format.index('W') if 'W' in kernel_dim_format else None
        super().__init__(name, shape, dtype, input_ops, dimension_format=dimension_format)

    def _check_consistency(self) -> None:
        super()._check_consistency()

        self._assert(len(self.kernel_shape) == self.kernel_dims, 'Illegal kernel shape.')
        self._assert(len(self.kernel_dim_format) == self.kernel_dims,
                     'Illegal kernel dimension format.')
        self._assert(len(self._pads) == self.kernel_dims + 2, 'Illegal pad definitions.')
        self._assert(len(self.strides) == self.kernel_dims, 'Illegal stride definitions.')

        # check the shape consistency
        if self.kernel_index_H is not None and self.index_H is not None:
            pad_H = self._pads[self.kernel_index_H] + \
                self._pads[self.kernel_index_H + self.kernel_dims]
            output_H_base = self.input_ops['X'].shape[self.index_H] + pad_H - self.kernel_height
            stride_H = self.strides[self.kernel_index_H]
            output_H, output_H_rest = divmod(output_H_base, stride_H)
            output_H += 1
            message = f'Pooling operator {self.name} does not match the height: {output_H} vs {self.height}.'
            self._assert(output_H == self.height, message)
            self._assert(output_H_rest == 0,
                         f'Pooling operator {self.name} should adjust the height pad to plus {output_H_rest}.')

        if self.kernel_index_W is not None and self.index_W is not None:
            pad_W = self._pads[self.kernel_index_W] + \
                self._pads[self.kernel_index_W + self.kernel_dims]
            output_W_base = self.input_ops['X'].shape[self.index_W] + pad_W - self.kernel_width
            stride_W = self.strides[self.kernel_index_W]
            output_W, output_W_rest = divmod(output_W_base, stride_W)
            output_W += 1
            message = f'Pooling operator {self.name} does not match the width: {output_W} vs {self.width}.'
            self._assert(output_W == self.width, message)
            self._assert(output_W_rest == 0,
                         f'Pooling operator {self.name} should adjust the width pad to plus {output_W_rest}.')

    @property
    def kernel_height(self) -> int:
        """Get the height in the kernel shape."""
        if self.kernel_index_H is not None:
            return self.kernel_shape[self.kernel_index_H]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_height property.')

    @property
    def kernel_width(self) -> int:
        """Get the Width in the kernel shape."""
        if self.kernel_index_W is not None:
            return self.kernel_shape[self.kernel_index_W]
        else:
            raise ValueError(f'Operator {self.name} does not have the kernel_width property.')

    @property
    def pads(self) -> List[int]:
        return self._pads

    @classmethod
    def infer_shape(cls, lists: Dict[str, List[int]], format: str, input_formats: List[str],
                    attrs: Dict[str, Any]) -> List[int]:
        """Infer its shape from inputs' shapes."""
        # attributes
        pads = attrs['pads']
        strides = attrs['strides']
        kernel_shape = attrs['kernel_shape']

        idx_N = input_formats[0].index('N')  # from input X
        idx_C = input_formats[0].index('C')  # from weight W
        idx_H = input_formats[0].index('H')  # from input X
        idx_W = input_formats[0].index('W')  # from input X

        N = lists['X'][idx_N]
        C = lists['X'][idx_C]

        # H
        pads_H = pads[0] + pads[2]
        input_H = lists['X'][idx_H] + pads_H
        window_H = kernel_shape[0]
        stride_H = strides[0]
        H, rest_H = divmod((input_H - window_H), stride_H)
        H += 1
        assert rest_H == 0, f'difference in height: {rest_H} at {cls.__name__}'

        # W
        pads_W = pads[1] + pads[3]
        input_W = lists['X'][idx_W] + pads_W
        window_W = kernel_shape[1]
        stride_W = strides[1]
        W, rest_W = divmod((input_W - window_W), stride_W)
        W += 1
        assert rest_W == 0, f'difference in width: {rest_W} at {cls.__name__}'

        NCHW = [N, C, H, W]
        perm = [format.index(s) for s in 'NCHW']
        return [NCHW[i] for i in perm]

    @property
    def preserve_quantization(self) -> bool:
        return False


class MaxPool(Pool):
    """Max pooling operator.

    MaxPool consumes an input tensor X and applies max pooling across the the tensor according
    to kernel sizes, stride sizes, and pad lengths. max pooling consisting of computing the max
    on all values of a subset of the input tensor according to the kernel size and downsampling
    the data into the output tensor Y for further processing.

    Inputs
    ------
    X
        Input data tensor from the previous operator.

    Outputs
    -------
    Y
        Output data tensor from max pooling across the input tensor.
        Dimensions will vary based on various kernel, stride, and pad sizes.
        Floor value of the dimension is used.

    Attributes (Optional constructor parameters)
    --------------------------------------------
    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    kernel_shape : list of ints
        The size of the kernel along each axis.

    kernel_dim_format : str
        Dimension denotation, which must consists of H', and 'W', where 'H' and 'W' are the
        height and width of input image. The default is 'HW'.

    dimensions : int
        Dimensions. This defaults to 2, which means 2-D image.
        Currently only 2 is available.

    pads : list of ints
        Padding for the beginning and ending along each axis, it can take any value greater
        than or equal to 0. The value represent the number of pixels added to the beginning
        and end part of the corresponding axis. `pads` format should be as follow
        [x1_begin, x2_begin, x1_end, x2_end], where xi_begin the number of pixels added at
        the beginning of axis `i` and xi_end, the number of pixels added at the end of axis
        `i`. If not present, the padding defaults to 0 along start and end of each axis.

    strides : list of ints
        Stride along each axis. If not present, the stride defaults to 1 along each axis.

    """

    @property
    def _dispatch_name(self) -> str:
        return 'max_pool'

    @property
    def is_monotonic(self) -> bool:
        return False

    @property
    def preserve_quantization(self) -> bool:
        return True


class AveragePool(Pool):
    """Average pooling operator.

    AveragePool consumes an input tensor X and applies average pooling across the the tensor
    according to kernel sizes, stride sizes, and pad lengths. average pooling consisting of
    computing the average on all values of a subset of the input tensor according to the
    kernel size and downsampling the data into the output tensor Y for further processing.

    Inputs
    ------
    X
        Input data tensor from the previous operator.

    Outputs
    -------
    Y
        Output data tensor from average pooling across the input tensor.
        Dimensions will vary based on various kernel, stride, and pad sizes.
        Floor value of the dimension is used.

    Attributes (Optional constructor parameters)
    --------------------------------------------
    dimension_format : str
        Dimension denotation, which must consists of 'N', 'C', 'H', and 'W', where 'N' is the
        number of batch size, 'C' is the number of channels, 'H' and 'W' are the height and
        width of input image. The default is 'NHWC'.

    kernel_shape : list of ints
        The size of the kernel along each axis.

    kernel_dim_format : str
        Dimension denotation, which must consists of H', and 'W', where 'H' and 'W' are the
        height and width of input image. The default is 'HW'.

    dimensions : int
        Dimensions. This defaults to 2, which means 2-D image.
        Currently only 2 is available.

    pads : list of ints
        Padding for the beginning and ending along each axis, it can take any value greater
        than or equal to 0. The value represent the number of pixels added to the beginning
        and end part of the corresponding axis. `pads` format should be as follow
        [x1_begin, x2_begin, x1_end, x2_end], where xi_begin the number of pixels added at
        the beginning of axis `i` and xi_end, the number of pixels added at the end of axis
        `i`. If not present, the padding defaults to 0 along start and end of each axis.

    strides : list of ints
        Stride along each axis. If not present, the stride defaults to 1 along each axis.

    """

    @property
    def _dispatch_name(self) -> str:
        return 'average_pool'

    @property
    def is_monotonic(self) -> bool:
        return False



