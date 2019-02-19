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
#include "func/sqrt.h"
#include "time_measurement.h"

void func_Sqrt(T_FLOAT input[], T_FLOAT output[], T_UINT out_depth) {
  Measurement::Start("sqrt");

  for (T_UINT i = 0; i < out_depth; i++)
    output[i] = std::sqrt(input[i]);

  Measurement::Stop();
}
