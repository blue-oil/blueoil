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
#include "func/mean.h"
#include "time_measurement.h"

void func_Mean(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<int32_t, MemoryLayout::C>& axis,
    const TensorView<T_FLOAT, MemoryLayout::NC>& output, 
    T_UINT in_h, T_UINT in_w) {
#ifndef RUN_AS_HLS
  Measurement::Start("Mean");
#endif
  T_UINT in_size = in_h * in_w;
  T_UINT out_depth = output.size();

  for (T_UINT dh = 0; dh < in_h; dh++){
    for (T_UINT dw = 0; dw < in_w; dw++){
      for (T_UINT dd = 0; dd < out_depth; dd++){
        output(0, dd) += input(0, dh, dw, dd) / in_size;
      }
    }
  }

#ifndef RUN_AS_HLS
  Measurement::Stop();
#endif
}
