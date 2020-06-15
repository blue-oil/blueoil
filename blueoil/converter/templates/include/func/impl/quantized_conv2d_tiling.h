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
=============================================================================*/

#ifndef DLK_FUNC_IMPL_QUANTIZED_CONV2D_TILING_H_INCLUDED
#define DLK_FUNC_IMPL_QUANTIZED_CONV2D_TILING_H_INCLUDED

#include "types.h"
#include "operators.h" // FIXME(nikolay): for binary_convolution_parameters definition, rid of it later
#include "tensor_view.h"

namespace dlk {

namespace impl {

using tiling_input_elem_base_t = uint32_t; // hardcoded, not configurable
using tiling_input_elem_t = QuantizedPacked<tiling_input_elem_base_t>;
using tiling_input_t = TensorView<tiling_input_elem_t, MemoryLayout::ChHWBCl>;
using tiling_kernel_elem_base_t = uint32_t; // hardcoded, not configurable
using tiling_kernel_elem_t = QuantizedPacked<tiling_kernel_elem_base_t>;
using tiling_kernel_t = TensorView<tiling_kernel_elem_t, MemoryLayout::OHWI>;

void pack_input_for_tiling(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& input,
    const tiling_input_t& output);

void convert_thresholds(BIN_CONV_OUTPUT *input, BIN_CONV_OUTPUT *output, std::size_t channels);

void QuantizedConv2DTiling(const tiling_input_t& input,
    const tiling_kernel_t& kernel,
    const binary_convolution_parameters &p);

} // namespace impl

} // namespace dlk

#endif // DLK_FUNC_IMPL_QUANTIZED_CONV2D_TILING_H_INCLUDED
