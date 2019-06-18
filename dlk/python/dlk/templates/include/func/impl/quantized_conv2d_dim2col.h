/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef FUNC_IMPL_QUANTIZED_CONV2D_DIM2COL_H_INCLUDED
#define FUNC_IMPL_QUANTIZED_CONV2D_DIM2COL_H_INCLUDED

#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "tensor_view.h"

namespace dlk {

namespace impl {

using dim2col_input_elem_t = QUANTIZED_PACKED;
using dim2col_input_t = TensorView<dim2col_input_elem_t, MemoryLayout::Im2col>;

void im2col(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>&,
    const dim2col_input_t&,
    const binary_convolution_parameters& p);

void QuantizedConv2DIm2Col(const dim2col_input_t& input,
    const kernel_t& kernel,
    const binary_convolution_parameters& p);

} // namespace impl

} // namespace dlk

#endif // FUNC_IMPL_QUANTIZED_CONV2D_DIM2COL_H_INCLUDED
