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
#include "func/minimum.h"
#include "time_measurement.h"

void func_Minimum(T_FLOAT input1, T_FLOAT input2[], T_FLOAT output[],
                  T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("Minimum");

  T_UINT elements = out_height * out_width * out_depth;

  for (T_UINT i = 0; i < elements; i++)
    output[i] = (input1 < input2[i] ? input1 : input2[i]);

  Measurement::Stop();
}

void func_Minimum(T_FLOAT input1[], T_FLOAT input2, T_FLOAT output[],
                  T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  func_Minimum(input2, input1, output, out_height, out_width, out_depth);
}
