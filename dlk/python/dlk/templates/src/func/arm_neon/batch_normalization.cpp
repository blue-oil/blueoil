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

#include "global.h"
#include "func/batch_normalization.h"
#include "time_measurement.h"

#include <arm_neon.h>

static const auto scale = std::make_unique<float[]>(MAX_IN_C);
static const auto shift = std::make_unique<float[]>(MAX_IN_C);

void func_BatchNormalization(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::C>& gamma,
    const TensorView<T_FLOAT, MemoryLayout::C>& beta,
    const TensorView<T_FLOAT, MemoryLayout::C>& mean,
    const TensorView<T_FLOAT, MemoryLayout::C>& variance,
    T_FLOAT epsilon,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  Measurement::Start("BatchNorm");

  const auto out_shape = output.get_shape();
  T_UINT out_height = out_shape[1];
  T_UINT out_width = out_shape[2];
  T_UINT out_depth = out_shape[3];

  T_UINT size = out_height * out_width;

  float32x4_t eps_batch = vdupq_n_f32(epsilon);
  float32x4_t scale_b, shift_b;

  int i = 0;
  for (; i <= static_cast<int>(out_depth) - 4; i += 4) {
    float32x4_t gamma_batch = vld1q_f32(gamma.data() + i);
    float32x4_t var_batch = vld1q_f32(variance.data() + i);
    float32x4_t beta_batch = vld1q_f32(beta.data() + i);
    float32x4_t mu_batch = vld1q_f32(mean.data() + i);

    scale_b = vaddq_f32(var_batch, eps_batch);
    float32x4_t rsqrt_est = vrsqrteq_f32(scale_b);
    rsqrt_est = vrsqrtsq_f32(scale_b * rsqrt_est, rsqrt_est) * rsqrt_est;
    scale_b = vrsqrtsq_f32(scale_b * rsqrt_est, rsqrt_est) * rsqrt_est;

    scale_b = vmulq_f32(scale_b, gamma_batch);
    shift_b = vmlsq_f32(beta_batch, scale_b, mu_batch);
    vst1q_f32(&scale[i], scale_b);
    vst1q_f32(&shift[i], shift_b);
  }

  for (; i < static_cast<int>(out_depth); i++) {
    scale[i] = gamma(i) * (1.0 / std::sqrt(variance(i) + epsilon));
    shift[i] = beta(i) - (scale[i] * mean(i));
  }

// TODO(nlpng): remove use of OpenMP library
#pragma omp parallel for
  for (T_UINT f = 0; f < size; f++) {
    T_FLOAT *in_temp = input.data() + f * out_depth;
    T_FLOAT *out_temp = output.data() + f * out_depth;

    T_UINT d = 0;
    for (; d + 3 < out_depth; d += 4) {
      const auto scale_v = vld1q_f32(scale.get() + d);
      const auto shift_v = vld1q_f32(shift.get() + d);
      const auto in_v = vld1q_f32(in_temp);
      vst1q_f32(out_temp, vmlaq_f32(shift_v, in_v, scale_v));
      in_temp += 4;
      out_temp += 4;
    }

    for (; d < out_depth; d++) {
      *out_temp++ = *in_temp++ * scale[d] + shift[d];
    }
  }

  Measurement::Stop();
}
