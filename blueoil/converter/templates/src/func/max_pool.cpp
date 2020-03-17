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
#include <cassert>
#include <cstring>
#include "types.h"
#include "func/max_pool.h"
#include "time_measurement.h"

namespace {

template<typename TYPE>
void max_pooling(
    const TensorView<TYPE, MemoryLayout::NHWC>& input,
    const TensorView<TYPE, MemoryLayout::NHWC>& output,
    struct max_pooling_parameters p)
{

  assert (p.kernel_depth == 1 && "kernel depth 1 is not supported.");
  assert (p.input_depth == p.kernel_depth * p.output_channels && \
          "input_depth must equal kernel_depth * output_channels.");

  int idx_out = 0;
  const T_FLOAT num_k_elems = p.kernel_height * p.kernel_width * p.kernel_depth;

  std::memset(output.data(), 0, output.size() * sizeof(TYPE));

  for(T_UINT oc = 0; oc < p.output_channels; oc++) {
    for(T_UINT wi = 0; wi < p.output_height; wi++) {
      for(T_UINT wj = 0; wj < p.output_width; wj++){
        TYPE out = 0;
        for(T_UINT ki = 0; ki < p.kernel_height; ki++) {
          for(T_UINT kj = 0; kj < p.kernel_width; kj++) {
	          T_INT row = (wi * p.stride) - p.padding + ki;
	          T_INT col = (wj * p.stride) - p.padding + kj;
	          T_INT inside = (row >= 0 && col >= 0 && row < (T_INT) p.input_height && col < (T_INT)p.input_width);
	          if (!inside) continue;
              for(T_UINT kz = 0; kz < p.kernel_depth; kz++) {
                if(ki == 0 && kj == 0){
                  out = input(0, row, col, oc * p.kernel_depth + kz);
                }else if (input(0, row, col, oc * p.kernel_depth + kz) > out){
                  out = input(0, row, col, oc * p.kernel_depth + kz);
                }
              }
          }
        }
        output(0, wi, wj, oc) += TYPE(out);
      }
    }
  }
}

} // namespace

void func_MaxPool(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    struct max_pooling_parameters mpp) {
  Measurement::Start("MaxPooling");

  max_pooling(input, output, mpp);

  Measurement::Stop();
}

void func_MaxPool(const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& input,
    const TensorView<QUANTIZED_NOT_PACKED, MemoryLayout::NHWC>& output,
    struct max_pooling_parameters mpp) {
  Measurement::Start("MaxPooling");

  max_pooling(input, output, mpp);

  Measurement::Stop();
}
