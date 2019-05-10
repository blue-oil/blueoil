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

#ifndef DLK_FUNC_EXTRACT_IMAGE_PATCHES
#define DLK_FUNC_EXTRACT_IMAGE_PATCHES

#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"
#include "pack_input_to_qwords.h"
#include <limits.h>

inline void func_ExtractImagePatches(
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    T_UINT kernel_size, T_UINT stride)
{
  Measurement::Start("ExtractImagePatches");
  const auto in_shape = input.get_shape();
  const T_UINT input_width = in_shape[2];
  const T_UINT input_depth = in_shape[0];
  const T_UINT bits_per_input = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[1];
  const T_UINT out_width = out_shape[2];
  const T_UINT out_depth = out_shape[0];

  T_UINT output_index = 0;

  if (out_depth < kernel_size * kernel_size) {
    int bit_shift = out_depth * QUANTIZED_PACKED::BitCount / (kernel_size * kernel_size);
    const QUANTIZED_PACKED::base_t mask((QUANTIZED_PACKED::base_t(1) << bit_shift) - 1);
    for(T_UINT wi = 0; wi < out_height; wi++)
      for(T_UINT wj = 0; wj < out_width; wj++)
        for(T_UINT ki = 0; ki < kernel_size; ki++)
          for(T_UINT kj = 0; kj < kernel_size; kj++)
          {
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;
            T_UINT ch = (ki * kernel_size + kj) * bit_shift;
            T_UINT ch_high = ch / QUANTIZED_PACKED::BitCount;
            T_UINT ch_low = ch % QUANTIZED_PACKED::BitCount;
            for(T_UINT digit = 0; digit < bits_per_input; ++digit) {
              output(ch_high, wi, wj, digit, 0) |= T((mask & input(0, row, col, digit, 0).Raw()) << ch_low);
            }
          }
  } else {
    for(T_UINT ih = 0; ih < input_depth; ++ih)
      for(T_UINT wi = 0; wi < out_height; wi++)
        for(T_UINT wj = 0; wj < out_width; wj++)
          for(T_UINT ki = 0; ki < kernel_size; ki++)
            for(T_UINT kj = 0; kj < kernel_size; kj++)
            {
              T_INT row = (wi * stride) + ki;
              T_INT col = (wj * stride) + kj;
              for(T_UINT digit = 0; digit < bits_per_input; ++digit) {
                output(ih * kernel_size * kernel_size
                  + ki * kernel_size + kj, wi, wj,
                  digit, 0) = input(ih, row, col, digit, 0);
              }
            }
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_EXTRACT_IMAGE_PATCHES
