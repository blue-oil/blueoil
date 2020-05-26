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
#include "types.h"
#include "func/pad.h"
#include "time_measurement.h"

void func_Pad(const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<int32_t, MemoryLayout::Padding>& padding,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
#ifndef RUN_AS_HLS
  Measurement::Start("Pad");
#endif

  T_UINT in_depth = input.get_shape()[3];
  T_UINT elements = output.size();
  T_UINT out_idx = 0;
  T_UINT in_idx = 0;

  T_UINT prep = padding(3, 0);
  T_UINT posp = padding(3, 1);

  for (T_UINT i = 0; i < elements; i+=(prep + in_depth + posp)) {
    out_idx = i;
    for (T_UINT pre = 0; pre < prep; pre++) {
      output.data()[out_idx] = 0;
      out_idx++;
    }
    for (T_UINT j = 0; j < in_depth; j++) {
      output.data()[out_idx] = input.data()[in_idx];
      out_idx++;
      in_idx++;
    }
    for (T_UINT pos = 0; pos < posp; pos++) {
      output.data()[out_idx] = 0;
      out_idx++;
    }
  }

#ifndef RUN_AS_HLS
  Measurement::Stop();
#endif
}
