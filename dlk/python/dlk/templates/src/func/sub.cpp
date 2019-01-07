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
#include "func/sub.h"
#include "time_measurement.h"

void func_Sub(T_FLOAT input[], T_FLOAT term[], T_FLOAT output[],
              T_UINT out_depth) {
  Measurement::Start("Sub");

  for (T_UINT i = 0; i < out_depth; i++)
    output[i] = input[i] - term[i];

  Measurement::Stop();
}

void func_Sub_depthwise(T_FLOAT input[], T_FLOAT terms[], T_FLOAT output[],
                        T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("SubDepthWise");

  T_UINT index = 0;
  for (T_UINT h = 0; h < out_height; h++)
    for (T_UINT w = 0; w < out_width; w++)
      for (T_UINT kz = 0; kz < out_depth; kz++) {
        output[index] = input[index] - terms[kz];
        index++;
      }

  Measurement::Stop();
}
