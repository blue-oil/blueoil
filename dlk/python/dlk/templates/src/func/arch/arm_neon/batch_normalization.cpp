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

#include <arm_neon.h>

void func_BatchNormalization(T_FLOAT input[], T_FLOAT gamma[], T_FLOAT beta[],
                             T_FLOAT mean[], T_FLOAT variance[],
                             T_FLOAT epsilon, T_FLOAT output[],
                             T_UINT out_height, T_UINT out_width,
                             T_UINT out_depth) {
  Measurement::Start("BatchNorm");

  // temporary fix: will be replaced by pre-allocated one
  T_FLOAT *scale = new float[out_depth];
  T_FLOAT *shift = new float[out_depth];
  T_UINT size = out_height * out_width;

  float32x4_t eps_batch = vdupq_n_f32(epsilon);
  float32x4_t scale_b, shift_b;

  int i = 0;
  for (; i <= static_cast<int>(out_depth) - 4; i += 4) {
    float32x4_t gamma_batch = vld1q_f32(&gamma[i]);
    float32x4_t var_batch = vld1q_f32(&variance[i]);
    float32x4_t beta_batch = vld1q_f32(&beta[i]);
    float32x4_t mu_batch = vld1q_f32(&mean[i]);

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
    scale[i] = gamma[i] * (1.0 / std::sqrt(variance[i] + epsilon));
    shift[i] = beta[i] - (scale[i] * mean[i]);
  }

// TODO(nlpng): remove use of OpenMP library
#pragma omp parallel for
  for (T_UINT f = 0; f < size; f++) {

    T_FLOAT *in_temp = &input[f * out_depth];
    T_FLOAT *out_temp = &output[f * out_depth];

    T_UINT d = 0;
    for (; d < out_depth; d += 4) {
      asm volatile("vldmia %0, {d16,d17}    \t\n" // q8(d16,d17) scale
                   "vldmia %1, {d18,d19}    \t\n" // q9(d18,d19) shift
                   "vldmia %2, {d20,d21}    \t\n" // q10(d20,d21) input
                   "vmla.f32 q9, q10, q8    \t\n"
                   "vstmia %3, {d18,d19}    \t\n"
                   :
                   : "r"(&scale[d]), "r"(&shift[d]), "r"(in_temp), "r"(out_temp)
                   : "memory", "q8", "q9", "q10");
      in_temp += 4;
      out_temp += 4;
    }

    for (; d < out_depth; d++) {
      *out_temp++ = *in_temp++ * scale[d] + shift[d];
    }
  }

  delete[] scale;
  delete[] shift;

  Measurement::Stop();
}
