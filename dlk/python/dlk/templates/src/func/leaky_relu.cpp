/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

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

#include "global.h"
#include "func/leaky_relu.h"
#include "time_measurement.h"

void func_LeakyRelu(T_FLOAT input[], T_FLOAT output[], T_FLOAT alpha, T_UINT out_height,
               T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("LeakyReLu");

  T_UINT elements = out_height * out_width * out_depth;

  for (T_UINT i = 0; i < elements; i++)
    output[i] = (input[i] * alpha > input[i] ? input[i] * alpha : input[i]);

  Measurement::Stop();
}
