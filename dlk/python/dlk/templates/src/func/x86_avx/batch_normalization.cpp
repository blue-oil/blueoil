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

#include <x86intrin.h>

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

  const unsigned out_height = output.get_shape()[1];
  const unsigned out_width = output.get_shape()[2];
  const unsigned out_depth = output.get_shape()[3];

  for (T_UINT i = 0; i < out_depth; i++) {
    scale[i] = gamma(i) * (1.0 / std::sqrt(variance(i) + epsilon));
    shift[i] = beta(i) - (scale[i] * mean(i));
  }

  std::size_t size = out_height * out_width;
#pragma omp parallel for
  for (std::size_t f = 0; f < size; ++f) {
    std::size_t d;
    for (d = 0; d + 7 < out_depth; d += 8) {
      const auto index = f * out_depth + d;
      const auto vscale = _mm256_loadu_ps(scale.get() + d);
      const auto vshift = _mm256_loadu_ps(shift.get() + d);
      const auto vinput = _mm256_loadu_ps(input.data() + index);
      const auto res = _mm256_fmadd_ps(vinput, vscale, vshift);
      _mm256_storeu_ps(output.data() + index, res);
    }
    
    for (; d < out_depth; ++d) {
      const auto index = f * out_depth + d;
      output.data()[index] = input.data()[index] * scale[d] + shift[d];
    }
  }

  Measurement::Stop();
}
