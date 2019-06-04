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

namespace dlk {

namespace impl {

void QuantizedConv2DIm2Col(QUANTIZED_NOT_PACKED input[], QUANTIZED_PACKED_KERNEL kernel[],
                                  const binary_convolution_parameters &p);

} // namespace impl

} // namespace dlk

#endif // FUNC_IMPL_QUANTIZED_CONV2D_DIM2COL_H_INCLUDED
