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

#ifndef DLK_FUNC_SPLIT_H_INCLUDED
#define DLK_FUNC_SPLIT_H_INCLUDED

#include "global.h"
#include "time_measurement.h"
#include "tensor_view.h"

template<class T>
void func_Split(const TensorView<T, MemoryLayout::NHWC>& input, const TensorView<T, MemoryLayout::NHWC> * const outputs, T_UINT num_split)
{
  Measurement::Start("func_Split");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto in_depth = in_shape[3];
  const auto out_depth = in_depth / num_split;

  T_UINT output_index[32] = {0};
  T_UINT input_index = 0;

  for(T_UINT i = 0; i < in_height * in_width; i++){
    for(T_UINT n = 0; n < num_split; n++){
      for(T_UINT d = 0; d < out_depth; d++){
        outputs[n].data()[output_index[n]++] = input.data()[input_index++];
      }
    }
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_SPLIT_H_INCLUDED
