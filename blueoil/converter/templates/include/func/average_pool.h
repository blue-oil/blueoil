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

#ifndef DLK_FUNC_AVERAGE_POOL_H_INCLUDED
#define DLK_FUNC_AVERAGE_POOL_H_INCLUDED

#include "types.h"
#include "tensor_view.h"

struct avg_pooling_parameters {
  T_UINT input_height;
  T_UINT input_width;
  T_UINT input_depth;
  T_UINT output_channels;
  T_UINT output_height;
  T_UINT output_width;
  T_UINT output_elements;
  T_UINT kernel_depth;
  T_UINT kernel_height;
  T_UINT kernel_width;
  T_UINT stride;
  T_UINT padding;
};

void func_AveragePool(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    struct avg_pooling_parameters app);

#endif // DLK_FUNC_AVERAGE_POOL_H_INCLUDED
