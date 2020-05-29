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
#include <limits>

#include "types.h"
#include "func/softmax.h"
#include "time_measurement.h"

void func_Softmax(const TensorView<T_FLOAT, MemoryLayout::NC>& input,
    const TensorView<T_FLOAT, MemoryLayout::NC>& output) {
  Measurement::Start("SoftMax");

  T_UINT out_width = output.size();

  T_FLOAT max_val = std::numeric_limits<float>::lowest();
  for(T_UINT d = 0; d < out_width; d++)
    max_val = std::max(max_val, input(0, d));

  T_FLOAT sum = 0.f;
  for(T_UINT d = 0; d < out_width; d++)
    T_FLOAT temp = std::exp(input(0, d) - max_val);
    output(0, d) = temp;
    sum += temp;

  for(T_UINT d = 0; d < out_width; d++)
    output(0, d) /= sum;

  Measurement::Stop();
}
