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
#include "func/softmax.h"
#include "time_measurement.h"

void func_Softmax(T_FLOAT input[], T_FLOAT output[], T_UINT out_width) {
  Measurement::Start("SoftMax");

  T_FLOAT sum = 0;
  for(T_UINT d = 0; d < out_width; d++)
    sum += std::exp(input[d]);

  for(T_UINT d = 0; d < out_width; d++)
    output[d] = std::exp(input[d]) / sum;

  Measurement::Stop();
}
