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

#ifndef DLK_FUNC_CONCAT_V2_H_INCLUDED
#define DLK_FUNC_CONCAT_V2_H_INCLUDED

#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"

template<class T>
void func_ConcatV2(const TensorView<T, MemoryLayout::NHWC>& input1,
    const TensorView<T, MemoryLayout::NHWC>& input2,
    const TensorView<T, MemoryLayout::NHWC>& output,
    T_INT axis, T_UINT input1_depth, T_UINT input2_depth) {
  Measurement::Start("ConcatV2");

  const auto out_shape = output.get_shape();
  const auto out_height = out_shape[1];
  const auto out_width = out_shape[2];

  T_UINT output_index = 0;
  T_UINT input1_index = 0;
  T_UINT input2_index = 0;

  for(T_UINT i = 0; i < out_height * out_width; i++)
  {
    for(T_UINT d = 0; d < input1_depth; d++)
      output.data()[output_index++] = input1.data()[input1_index++];

    for(T_UINT d = 0; d < input2_depth; d++)
      output.data()[output_index++] = input2.data()[input2_index++];
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_CONCAT_V2_H_INCLUDED
