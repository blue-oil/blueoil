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
#include <memory>

#include "types.h"
#include "func/batch_normalization.h"
#include "time_measurement.h"

#include <arm_neon.h>

void func_BatchNormalizationOptimized(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::C>& scale,
    const TensorView<T_FLOAT, MemoryLayout::C>& bias,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  Measurement::Start("BatchNorm");

  const auto out_shape = output.get_shape();
  T_UINT out_height = out_shape[1];
  T_UINT out_width = out_shape[2];
  T_UINT out_depth = out_shape[3];

  T_UINT size = out_height * out_width;

// TODO(nlpng): remove use of OpenMP library
#pragma omp parallel for
  for (T_UINT f = 0; f < size; f++) {
    T_FLOAT *in_temp = input.data() + f * out_depth;
    T_FLOAT *out_temp = output.data() + f * out_depth;

    T_UINT d = 0;
    for (; d + 3 < out_depth; d += 4) {
      const auto scale_v = vld1q_f32(scale.data() + d);
      const auto shift_v = vld1q_f32(bias.data() + d);
      const auto in_v = vld1q_f32(in_temp);
      vst1q_f32(out_temp, vmlaq_f32(shift_v, in_v, scale_v));
      in_temp += 4;
      out_temp += 4;
    }

    for (; d < out_depth; d++) {
      *out_temp++ = *in_temp++ * scale(d) + bias(d);
    }
  }

  Measurement::Stop();
}
