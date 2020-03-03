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
#include "func/average_pool.h"
#include "time_measurement.h"

void func_AveragePool(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
    struct avg_pooling_parameters app) {
  Measurement::Start("AveragePool");

  assert (app.kernel_depth == 1 && "kernel depth 1 is not supported.");
  assert (app.input_depth == app.kernel_depth * app.output_channels && \
          "input_depth must equal kernel_depth * output_channels.");

  const T_FLOAT num_k_elems_inv = 1.f / (app.kernel_height * app.kernel_width * app.kernel_depth);

  size_t area = app.output_height * app.output_width;
#pragma omp parallel for
  for(size_t wij = 0; wij < area; wij++) {
    size_t wi = wij / app.output_width;
    size_t wj = wij % app.output_width;
    float tmp[MAX_IN_C];
    for(size_t ic = 0; ic < app.input_depth; ic++) {
      tmp[ic] = 0.f;
    }
    for(size_t ki = 0; ki < app.kernel_height; ki++) {
      for(size_t kj = 0; kj < app.kernel_width; kj++) {
        T_INT row = (wi * app.stride) - app.padding + ki;
        T_INT col = (wj * app.stride) - app.padding + kj;
        if (row < 0 || col < 0 || row >= (T_INT) app.input_height || col >= (T_INT)app.input_width) continue;
        for(size_t ic = 0; ic < app.input_depth; ic++) {
          size_t idx_in = + row * (app.input_width * app.input_depth)
            + col * (app.input_depth)
            + ic;
          tmp[ic] += input.data()[idx_in];
        }
      }
    }
    for(size_t oc = 0; oc < app.output_channels; oc++) {
      size_t idx_out = wi * (app.output_width * app.output_channels)
        + wj * app.output_channels
        + oc;
      float out = 0.f;
      for(size_t kz = 0; kz < app.kernel_depth; kz++) {
        out += tmp[oc * app.kernel_depth + kz];
      }
      output.data()[idx_out] = out * num_k_elems_inv;
    }
  }

  Measurement::Stop();
}
