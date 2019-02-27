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
#include "func/matmul.h"
#include "time_measurement.h"

void func_Matmul(T_FLOAT input[], T_FLOAT factor[], T_FLOAT output[],
                 T_UINT in_size, T_UINT out_depth) {
#ifndef RUN_AS_HLS
  Measurement::Start("MatMul");
#endif

  T_UINT index = 0;
  for (T_UINT d = 0; d < in_size; d++){
    for (T_UINT kz = 0; kz < out_depth; kz++){
      output[kz] += input[d] * factor[index];
      index++;
    }
  }

#ifndef RUN_AS_HLS
  Measurement::Stop();
#endif
}
