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

void func_Lookup(float *input, QUANTIZED_PACKED_KERNEL *lsb, QUANTIZED_PACKED_KERNEL *msb, QUANTIZED_PACKED* output, int h, int w, int c) {

  int b = 32;
  int packed_depth = 2;
  Measurement::Start("Lookup");

  int out_idx = 0;
  for(int ih = 0; ih < h; ih++)
  for(int iw = 0; iw < w; iw++) {
    int r = int(input[ih * w * 3 + iw * 3 + 0] * 255.0);
    int g = int(input[ih * w * 3 + iw * 3 + 1] * 255.0);
    int b = int(input[ih * w * 3 + iw * 3 + 2] * 255.0);

    auto r_lsb = lsb[r];
    auto g_lsb = lsb[g];
    auto b_lsb = lsb[b];
    auto r_msb = msb[r];
    auto g_msb = msb[g];
    auto b_msb = msb[b];

    output[out_idx++] = QUANTIZED_PACKED((b_lsb.Raw() << 20) | (g_lsb.Raw() << 10) | r_lsb.Raw());
    output[out_idx++] = QUANTIZED_PACKED((b_msb.Raw() << 20) | (g_msb.Raw() << 10) | r_msb.Raw());
  }

  Measurement::Stop();
}

