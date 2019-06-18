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
#include "func/batch_normalization.h"
#include "time_measurement.h"

void func_BatchNormalization(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::C>& gamma,
    const TensorView<T_FLOAT, MemoryLayout::C>& beta,
    const TensorView<T_FLOAT, MemoryLayout::C>& mean,
    const TensorView<T_FLOAT, MemoryLayout::C>& variance,
    T_FLOAT epsilon,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  Measurement::Start("BatchNorm");

  const unsigned out_height = output.get_shape()[1];
  const unsigned out_width = output.get_shape()[2];
  const unsigned out_depth = output.get_shape()[3];

  // temporary fix: will be replaced by pre-allocated one
  T_FLOAT *scale = new float[out_depth];
  T_FLOAT *shift = new float[out_depth];

  for (T_UINT i = 0; i < out_depth; i++)
    scale[i] = gamma(i) * (1.0 / std::sqrt(variance(i) + epsilon));

  for (T_UINT i = 0; i < out_depth; i++)
    shift[i] = beta(i) - (scale[i] * mean(i));

  for (T_UINT r = 0; r < out_height; r++) {
    for (T_UINT c = 0; c < out_width; c++) {
      for (T_UINT d = 0; d < out_depth; d++) {
        output(0, r, c, d) = input(0, r, c, d) * scale[d] + shift[d];
      }
    }
  }

  delete[] scale;
  delete[] shift;

  Measurement::Stop();
}
