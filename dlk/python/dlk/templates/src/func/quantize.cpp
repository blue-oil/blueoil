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
#include "func/quantize.h"
#include "time_measurement.h"

void func_Quantize(T_FLOAT input[], Quantized_t output[], T_UINT out_height,
                   T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("Quantize");

  T_UINT elements = out_height * out_width * out_depth;

  // activations use k = 2 bits
  T_FLOAT power_of_two_minus_one_f = 3.0;
  T_FLOAT min_value = 0.f;
  T_FLOAT max_value = 2.f;
  T_FLOAT value_range = max_value - min_value;

  for (T_UINT i = 0; i < elements; i++) {
    T_FLOAT shifted = (input[i] - min_value) / value_range;
    T_FLOAT out = std::round(power_of_two_minus_one_f * shifted);

    output[i] = Quantized_t(out);
  }

  Measurement::Stop();
}
