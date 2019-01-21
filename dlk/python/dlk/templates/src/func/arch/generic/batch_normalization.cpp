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

#include <cmath>

#include "global.h"
#include "func/impl/batch_normalization.h"
#include "time_measurement.h"

void func_BatchNormalization(T_FLOAT input[], T_FLOAT gamma[], T_FLOAT beta[],
                             T_FLOAT mean[], T_FLOAT variance[],
                             T_FLOAT epsilon, T_FLOAT output[],
                             T_UINT out_height, T_UINT out_width,
                             T_UINT out_depth) {
  Measurement::Start("BatchNorm");

  // temporary fix: will be replaced by pre-allocated one
  T_UINT elements = out_height * out_width * out_depth;
  T_FLOAT *scale = new float[out_depth];
  T_FLOAT *shift = new float[out_depth];

  for (T_UINT i = 0; i < out_depth; i++)
    scale[i] = gamma[i] * (1.0 / std::sqrt(variance[i] + epsilon));

  for (T_UINT i = 0; i < out_depth; i++)
    shift[i] = beta[i] - (scale[i] * mean[i]);

  T_UINT index = 0;
  for (T_UINT f = 0; f < out_height * out_width; f++)
    for (T_UINT d = 0; d < out_depth; d++) {
      output[index] = input[index] * scale[d] + shift[d];
      index++;
    }

  delete[] scale;
  delete[] shift;

  Measurement::Stop();
}
