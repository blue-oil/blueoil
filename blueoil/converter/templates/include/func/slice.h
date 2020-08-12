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

#ifndef DLK_FUNC_SLICE_H_INCLUDED
#define DLK_FUNC_SLICE_H_INCLUDED

#include "global.h"
#include "time_measurement.h"
#include "tensor_view.h"

template<class T>
void func_Slice(const TensorView<T, MemoryLayout::NHWC>& input,
    const TensorView<T, MemoryLayout::NHWC>& output, T_UINT begin, T_UINT size)
{
  Measurement::Start("func_Slice");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto in_channels = in_shape[3];

  T_UINT output_index = 0;

  for(T_UINT i = 0; i < in_height * in_width; i++){
    T_UINT input_index = i * in_channels + begin;
    for(T_UINT d = 0; d < size; d++){
      output.data()[output_index++] = input.data()[input_index++];
    }
  }

  Measurement::Stop();
}

inline void func_Slice(const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& output, T_UINT begin, T_UINT size)
{
  Measurement::Start("func_Slice");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[0];
  const auto in_width = in_shape[1];
  const auto in_channels_high = in_shape[2];
  const auto bits = in_shape[3];

  T_UINT output_index = 0;

  for(T_UINT i = 0; i < in_height * in_width; i++){
    T_UINT input_index = i * in_channels_high * bits + begin / QUANTIZED_PACKED::BitCount * bits;
    for(T_UINT d = 0; d < size; d += size){
      for(T_UINT digit = 0; digit < bits; ++digit) {
        output.data()[output_index++] = input.data()[input_index++];
      }
    }
  }

  Measurement::Stop();
}

inline void func_Slice(const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& input,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output, T_UINT begin, T_UINT size)
{
  Measurement::Start("func_Slice");

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto bits = in_shape[3];

  std::size_t offset = in_height * in_width * bits * (begin / QUANTIZED_PACKED::BitCount);
  if (input.data() + offset != output.data()) {
    const auto bytes = output.size() * sizeof(QUANTIZED_PACKED);
    std::memmove(output.data(), input.data() + offset, bytes);
  } else {
    // nothing to do
  }

  Measurement::Stop();
}

#endif // DLK_FUNC_SLICE_H_INCLUDED
