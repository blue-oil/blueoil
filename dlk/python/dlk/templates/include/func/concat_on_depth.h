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

#ifndef DLK_FUNC_CONCAT_ON_DEPTH_H_INCLUDED
#define DLK_FUNC_CONCAT_ON_DEPTH_H_INCLUDED

#include "global.h"
#include "time_measurement.h"

template<class T>
void func_ConcatOnDepth(const TensorView<T, MemoryLayout::ChHWBCl> inputs[],
    T_UINT *depths, T_UINT n_inputs,
    const TensorView<T, MemoryLayout::ChHWBCl>& output) {
  Measurement::Start("func_ConcatOnDepth");
  const auto shape = output.get_shape();
  T_UINT out_height = shape[1];
  T_UINT out_width = shape[2];
  T_UINT bits = shape[3];

  T_UINT output_index = 0;
  T_UINT input_index[32] = {0};

  if (!std::is_same<T, typename Base<T>::type>::value) {
    // quantized and packed inputs
    for(T_UINT i = 0; i < n_inputs; i++)
      depths[i] /= 32;
  }

  T_UINT index = 0;
  for(T_UINT n = 0; n < n_inputs; n++)
    for(T_UINT d = 0; d < depths[n]; d++) {
      for(T_UINT h = 0; h < out_height; h++)
        for(T_UINT w = 0; w < out_width; w++) {
          for(T_UINT digit = 0; digit < bits; ++digit) {
            output(index, h, w, digit, 0) = inputs[n](d, h, w, digit, 0);
          }
        }
      ++index;
    }

  Measurement::Stop();
}

#endif // DLK_FUNC_CONCAT_ON_DEPTH_H_INCLUDED
