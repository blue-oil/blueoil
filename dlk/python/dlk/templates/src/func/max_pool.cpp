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
#include "global.h"
#include "func/max_pool.h"
#include "time_measurement.h"

namespace {

template<typename TYPE>
void max_pooling(
    TYPE input[],
    TYPE output[],
    struct max_pooling_parameters p)
{

  assert (p.kernel_depth == 1 && "kernel depth 1 is not supported.");
  assert (p.input_depth == p.kernel_depth * p.output_channels && \
          "input_depth must equal kernel_depth * output_channels.");

  int idx_out = 0;
  const T_FLOAT num_k_elems = p.kernel_height * p.kernel_width * p.kernel_depth;

  std::memset(output, 0.0f, p.output_channels * p.output_height * p.output_width * sizeof(TYPE));

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
                int idx_in = oc * p.kernel_depth
                         + row * (p.input_width * p.input_depth)
                         + col * (p.input_depth) + kz;
                if(ki == 0 && kj == 0){
                  out = input[idx_in];
                }else if (input[idx_in] > out){
                  out = input[idx_in];
                }
              }
          }
        }
        output[(p.output_channels * p.output_width) * wi + p.output_channels * wj + oc] += TYPE(out);
      }
    }
  }
}

template<typename TYPE>
void max_pooling_with_argmax(
    TYPE input[],
    TYPE output[],
    T_UINT indices[],
    struct MaxPoolWithArgmax_parameters p)
{
  const TYPE lowest = -10000000;

  // important to be zero (minimum value for T_UINT)
  for(T_UINT i = 0; i < p.output_elements; i++) { output[i] = lowest; }

  for(T_UINT wi = 0; wi < p.output_height; wi++)
    for(T_UINT wj = 0; wj < p.output_width; wj++)
    {
        for(T_UINT ki = 0; ki < p.kernel_height; ki++)
         for(T_UINT kj = 0; kj < p.kernel_width; kj++)
          for(T_UINT kz = 0; kz < p.kernel_depth; kz++)
          {
            T_INT height_index = (wi * p.stride) - p.padding + ki;
            T_INT width_index = (wj * p.stride) - p.padding + kj;
            T_INT inside = (height_index >= 0 && width_index >= 0 && height_index < (T_INT) p.input_height && width_index < (T_INT)p.input_width);
	    if (!inside) continue;

            T_UINT input_index = height_index * (p.input_width * p.kernel_depth) + width_index * (p.kernel_depth) + kz;
            TYPE e = input[input_index];

            // update the current maximum value found so far
            T_UINT index_current_maximum_value = wi * (p.kernel_depth * p.output_width) + wj * (p.kernel_depth) + kz;
            if(e > output[index_current_maximum_value])
            {
              output[index_current_maximum_value] = e;
              indices[index_current_maximum_value] = input_index;
            }
          }
    }
}

} // namespace

void func_MaxPool(T_FLOAT input[], T_FLOAT output[],
                  struct max_pooling_parameters mpp, T_UINT out_height,
                  T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("MaxPooling");

  max_pooling(input, output, mpp);

  Measurement::Stop();
}

void func_MaxPool(QUANTIZED_NOT_PACKED input[], QUANTIZED_NOT_PACKED output[],
                  struct max_pooling_parameters mpp, T_UINT out_height,
                  T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("MaxPooling");

  max_pooling(input, output, mpp);

  Measurement::Stop();
}

void func_MaxPoolWithArgmax(Quantized_t input[], Quantized_t output[],
                            T_UINT indices[],
                            struct MaxPoolWithArgmax_parameters mpp,
                            T_UINT out_height, T_UINT out_width,
                            T_UINT out_depth) {
  Measurement::Start("MaxPoolingWithArgmax");

  max_pooling_with_argmax(input, output, indices, mpp);

  Measurement::Stop();
}
