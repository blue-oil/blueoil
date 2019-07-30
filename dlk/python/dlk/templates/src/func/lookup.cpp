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

#include "global.h"
#include "func/lookup.h"
#include "time_measurement.h"

void func_Lookup(const TensorView<float, MemoryLayout::NHWC>& input,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::TC>& lsb,
    const TensorView<QUANTIZED_PACKED_KERNEL, MemoryLayout::TC>& msb,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output) {
  const auto in_shape = input.get_shape();
  const auto h = in_shape[1];
  const auto w = in_shape[2];
  const auto c = in_shape[3];

  int b = 32;
  int packed_depth = 2;
  Measurement::Start("Lookup");

  const float * in_ptr = input.data();
  const QUANTIZED_PACKED_KERNEL * lsb_ptr = lsb.data();
  const QUANTIZED_PACKED_KERNEL * msb_ptr = msb.data();
  QUANTIZED_PACKED * out_ptr = output.data();
  for(int ih = 0; ih < h; ih++)
  for(int iw = 0; iw < w; iw++) {
    int r = int(*in_ptr++ * 255.0);
    int g = int(*in_ptr++ * 255.0);
    int b = int(*in_ptr++ * 255.0);

    auto r_lsb = lsb_ptr[r];
    auto g_lsb = lsb_ptr[g];
    auto b_lsb = lsb_ptr[b];
    auto r_msb = msb_ptr[r];
    auto g_msb = msb_ptr[g];
    auto b_msb = msb_ptr[b];

    *out_ptr++ = QUANTIZED_PACKED((b_lsb.Raw() << 20) | (g_lsb.Raw() << 10) | r_lsb.Raw());
    *out_ptr++ = QUANTIZED_PACKED((b_msb.Raw() << 20) | (g_msb.Raw() << 10) | r_msb.Raw());
  }

  Measurement::Stop();
}

