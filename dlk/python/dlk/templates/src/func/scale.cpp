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

#include <iostream>

#include "global.h"
#include "func/scale.h"
#include "time_measurement.h"

void func_Scale(T_INT input[], T_FLOAT factor, T_FLOAT output[],
                T_UINT out_height, T_UINT out_width, T_UINT out_depth,
                unsigned input_bitwidth) {
  Measurement::Start("Scale");

  T_UINT elements = out_height * out_width * out_depth;

  // e.g. 2 bits -> 4 - 1 -> 3
  T_FLOAT power_of_two_minus_one;

  if (input_bitwidth == 8) {
    power_of_two_minus_one = 255.0f;

    for (T_UINT i = 0; i < elements; i++) {
      output[i] = (input[i] / power_of_two_minus_one) * factor;
    }
  } else if (input_bitwidth == 2) {
    power_of_two_minus_one = 3.0;
    T_FLOAT s = factor / power_of_two_minus_one * 2.0;

    for (T_UINT i = 0; i < elements; i++) {
      output[i] = input[i] * s;
    }
  } else {
    std::cout << "2 nor 8 Convolution: Not Implemented Yet!!\n" << std::endl;
  }

  Measurement::Stop();
}
