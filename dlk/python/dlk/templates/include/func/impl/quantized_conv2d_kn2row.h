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

#ifndef DLK_FUNC_IMPL_QUANTIZED_CONV2D_KN2ROW_H_INCLUDED
#define DLK_FUNC_IMPL_QUANTIZED_CONV2D_KN2ROW_H_INCLUDED

#include "global.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "tensor_view.h"

namespace dlk {

namespace impl {

using kn2row_input_elem_t = QUANTIZED_PACKED;

#ifndef RUN_ON_FPGA
void convert_thresholds(BIN_CONV_OUTPUT *input, BIN_CONV_OUTPUT *output, std::size_t channels);
using kn2row_input_t = TensorView<kn2row_input_elem_t, MemoryLayout::HWChBCl>;
void QuantizedConv2DKn2Row(const kn2row_input_t& input,
                                  const kernel_t& kernel,
                                  const binary_convolution_parameters &p);
#else
using kn2row_input_t = TensorView<kn2row_input_elem_t, MemoryLayout::ChHWBCl>;
void TCAConv2d(const kn2row_input_t& input,
    const kernel_t& kernel,
    const binary_convolution_parameters &p);
#endif


} // namespace impl

} // namespace dlk

#endif // DLK_FUNC_IMPL_QUANTIZED_CONV2D_KN2ROW_H_INCLUDED
