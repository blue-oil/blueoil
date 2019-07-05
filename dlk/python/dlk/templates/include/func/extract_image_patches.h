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

#include <algorithm>
#include "global.h"
#include "tensor_view.h"
#include "time_measurement.h"
#include "pack_input_to_qwords.h"
#include <limits.h>

template <typename T>
void func_ExtractImagePatches(
    const TensorView<T, MemoryLayout::NHWC>& input,
    const TensorView<T, MemoryLayout::NHWC>& output,
    T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("ExtractImagePatches");

  const auto in_shape = input.get_shape();
  const T_UINT input_width = in_shape[2];
  const T_UINT input_depth = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[1];
  const T_UINT out_width = out_shape[2];
  const T_UINT out_depth = out_shape[3];

  for(T_UINT kz = 0; kz < input_depth; ++kz)
    for(T_UINT wi = 0; wi < out_height; wi++)
      for(T_UINT wj = 0; wj < out_width; wj++)
        for(T_UINT ki = 0; ki < kernel_size; ki++)
          for(T_UINT kj = 0; kj < kernel_size; kj++)
          {
            T_INT row = (wi * stride) + ki;
            T_INT col = (wj * stride) + kj;
            const auto ch = kz + (ki * kernel_size + kj) * input_depth;
            const auto out_idx = wi * out_width * out_depth
              + wj * out_depth
              + ch;
            const auto in_idx = row * input_width * input_depth
              + col * input_depth
              + kz;
            output.data()[out_idx]
                = input.data()[in_idx];
          }

  Measurement::Stop();
}

inline void func_ExtractImagePatches(
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& output,
    T_UINT kernel_size, T_UINT stride) {
  Measurement::Start("ExtractImagePatches");

  const auto in_shape = input.get_shape();
  const T_UINT input_width = in_shape[1];
  const T_UINT input_depth = in_shape[2];
  const T_UINT bits_per_input = in_shape[3];
  const auto out_shape = output.get_shape();
  const T_UINT out_height = out_shape[0];
  const T_UINT out_width = out_shape[1];
  const T_UINT out_depth = out_shape[2];

  T_UINT output_index = 0;

  if (out_depth < kernel_size * kernel_size) {
    int bit_shift = out_depth * QUANTIZED_PACKED::BitCount / (kernel_size * kernel_size);
    const QUANTIZED_PACKED::base_t mask((QUANTIZED_PACKED::base_t(1) << bit_shift) - 1);
    std::fill(output.data(), output.data() + output.size(), QUANTIZED_PACKED(0));
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
              const auto out_idx = wi * out_width * out_depth * bits_per_input
                + wj * out_depth * bits_per_input
                + ch_high * bits_per_input
                + digit;
              const auto in_idx = row * input_width * input_depth * bits_per_input
                + col * input_depth * bits_per_input
                + digit;
              output.data()[out_idx] |= QUANTIZED_PACKED((mask & input.data()[in_idx].Raw()) << ch_low);
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
                const auto ch_high = ih + (ki * kernel_size + kj) * input_depth;
                const auto out_idx = wi * out_width * out_depth * bits_per_input
                  + wj * out_depth * bits_per_input
                  + ch_high * bits_per_input
                  + digit;
                const auto in_idx = row * input_width * input_depth * bits_per_input
                  + col * input_depth * bits_per_input
                  + ih * bits_per_input
                  + digit;
                output.data()[out_idx]
                  = input.data()[in_idx];
              }
            }
  }

  Measurement::Stop();
}

inline void func_ExtractImagePatches(
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    T_UINT kernel_size, T_UINT stride)
{
  Measurement::Start("ExtractImagePatches");
  const auto in_shape = input.get_shape();
  const T_UINT input_height = in_shape[1];
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
    std::fill(output.data(), output.data() + output.size(), QUANTIZED_PACKED(0));
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
              const auto out_idx = ch_high * out_height * out_width * bits_per_input
                + wi * out_width * bits_per_input
                + wj * bits_per_input
                + digit;
              const auto in_idx = row * input_width * bits_per_input
                + col * bits_per_input
                + digit;
              output.data()[out_idx] |= QUANTIZED_PACKED((mask & input.data()[in_idx].Raw()) << ch_low);
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
                const auto ch_high = ih + (ki * kernel_size + kj) * input_depth;
                const auto out_idx = ch_high * out_height * out_width * bits_per_input
                  + wi * out_width * bits_per_input
                  + wj * bits_per_input
                  + digit;
                const auto in_idx = ih * input_height * input_width * bits_per_input
                  + row * input_width * bits_per_input
                  + col * bits_per_input
                  + digit;
                output.data()[out_idx]
                  = input.data()[in_idx];
              }
            }
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_EXTRACT_IMAGE_PATCHES
