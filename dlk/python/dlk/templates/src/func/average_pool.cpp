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
#include "func/average_pool.h"
#include "time_measurement.h"

void func_AveragePool(T_FLOAT input[], T_FLOAT output[],
                      struct avg_pooling_parameters app, T_UINT out_height,
                      T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("AveragePool");

  assert (app.kernel_depth == 1 && "kernel depth 1 is not supported.");
  assert (app.input_depth == app.kernel_depth * app.output_channels && \
          "input_depth must equal kernel_depth * output_channels.");

  int idx_out = 0;
  const T_FLOAT num_k_elems = app.kernel_height * app.kernel_width * app.kernel_depth;

  std::memset(output, 0.0f, app.output_channels * app.output_height * app.output_width * sizeof(T_FLOAT));

  for(T_UINT oc = 0; oc < app.output_channels; oc++) {
    for(T_UINT wi = 0; wi < app.output_height; wi++) {
      for(T_UINT wj = 0; wj < app.output_width; wj++)
      {
        T_FLOAT out = 0;
        for(T_UINT ki = 0; ki < app.kernel_height; ki++) {
          for(T_UINT kj = 0; kj < app.kernel_width; kj++) {
	    T_INT row = (wi * app.stride) - app.padding + ki;
	    T_INT col = (wj * app.stride) - app.padding + kj;

	    T_INT inside = (row >= 0 && col >= 0 && row < (T_INT) app.input_height && col < (T_INT)app.input_width);
	    if (!inside) continue;
            for(T_UINT kz = 0; kz < app.kernel_depth; kz++) {
              int idx_in = oc * app.kernel_depth
                         + row * (app.input_width * app.input_depth)
                         + col * (app.input_depth) + kz;
              out += input[idx_in];
            }
          }
        }
        output[idx_out++] += T_FLOAT(out) / num_k_elems;
      }
    }
  }

  Measurement::Stop();
}
