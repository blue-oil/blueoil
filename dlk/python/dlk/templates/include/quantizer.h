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

#ifndef LEAPMIND_H_INCLUDED
#define LEAPMIND_H_INCLUDED


/***************************************
 leapmind original
***************************************/
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>

#include "global.h"
#include "tensor_view.h"
#include "pack_input_to_qwords.h"
#ifdef USE_NEON
  #include <arm_neon.h>
#endif



/***************************************
 wrappers
***************************************/
inline void func_QTZ_binary_channel_wise_mean_scaling(
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  const auto shape = input.get_shape();
  T_UINT in_height = shape[1];
  T_UINT in_width = shape[2];
  T_UINT in_depth = shape[3];
  T_UINT in_channel = shape[0];
  unsigned num_elems_in_channel = in_height * in_width * in_depth;
  T_FLOAT sum, mean;

  for(unsigned i = 0; i < in_channel; i++) {
    sum = 0;
    for(unsigned j = 0; j < num_elems_in_channel; j++) {
      sum += std::abs(input.data()[i * num_elems_in_channel + j]);
    }
    mean = sum / num_elems_in_channel;
    for(unsigned j = 0; j < num_elems_in_channel; j++) {
      unsigned in_index = i * num_elems_in_channel + j;
      output.data()[in_index] = (input.data()[in_index] >= 0) ? mean : -1 * mean;
    }
  }
}

template <MemoryLayout layout>
void func_QTZ_binary_mean_scaling(
    const TensorView<T_FLOAT, layout>& input,
    const TensorView<T_FLOAT, layout>& output) {
  T_FLOAT sum = 0.f;
  unsigned num_elems = input.size();

  for(unsigned i = 0; i < num_elems; i++)
  {
    sum += std::abs(input[i]);
  }

  T_FLOAT mean = sum / num_elems;
  T_FLOAT mean_minus = -1 * mean;

  for(unsigned i = 0; i < num_elems; i++)
  {
    output.data()[i] = (input.data()[i] >= 0) ? mean : mean_minus;
  }
}


inline void func_QTZ_linear_mid_tread_half_body(
  T_FLOAT input[],
  T_INT nbit,
  T_FLOAT max_value,
  QUANTIZED_NOT_PACKED output[],
  T_UINT begin,
  T_UINT end)

{
  T_FLOAT max_value_r = 1.0f / max_value;

  T_FLOAT min_value = 0.f;
  T_FLOAT n = (1 << nbit) - 1.f;
  int i = begin;

#ifdef USE_NEON
  float32x4_t max_value_x4 = vdupq_n_f32(max_value);
  float32x4_t min_value_x4 = vdupq_n_f32(min_value);
  float32x4_t round_offset = vdupq_n_f32(0.5);
  float32x4_t max_value_rn = vdupq_n_f32(max_value_r * n);

  for (; i <= static_cast<int>(end) - 4; i += 4)
  {
    float32x4_t tmp = vld1q_f32(&input[i]);
    tmp = vmaxq_f32(tmp, min_value_x4);
    tmp = vminq_f32(tmp, max_value_x4);
    //    tmp = vmlaq_f32(tmp, max_value_rn, round_offset);
    tmp = vmulq_f32(tmp, max_value_rn);
    tmp = vaddq_f32(tmp, round_offset);
    int32x4_t r = vcvtq_s32_f32(tmp);
    int r_tmp[4];
    vst1q_s32(r_tmp, r);
    output[i] = r_tmp[0];
    output[i+1] = r_tmp[1];
    output[i+2] = r_tmp[2];
    output[i+3] = r_tmp[3];
  }
#endif

  for (; i < static_cast<int>(end); ++i)
  {
    T_FLOAT tmp = std::max(input[i], (T_FLOAT)min_value);
    tmp = std::min(tmp, max_value);
    output[i] = (T_INT)roundf(tmp * (max_value_r * n));
  }
}

inline void func_QTZ_linear_mid_tread_half(
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_INT, MemoryLayout::Atom>& nbit,
    const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& output) {
  Measurement::Start("QTZ_linear_mid_tread_half");

  unsigned num_elems = input.size();
  QUANTIZED_NOT_PACKED* output_not_packed = new QUANTIZED_NOT_PACKED[num_elems];

  unsigned int chunk_size = num_elems / std::thread::hardware_concurrency();
  if (chunk_size == 0) {
    chunk_size += 1;
  }

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_elems; i += chunk_size) {
    threads.emplace_back(std::thread([&input, &nbit, &max_value, &output_not_packed, i, chunk_size, num_elems] {
          func_QTZ_linear_mid_tread_half_body(input.data(), nbit(), max_value(), output_not_packed, i,
                                              std::min(i + chunk_size, static_cast<unsigned int>(num_elems)));
    }));
  }

  for (auto& th: threads) {
    th.join();
  }

  //static T_UINT counter = 0;
  //write_to_file("out/qconv_input_quantized_not_packed", counter++, output_not_packed, in_height * in_width * in_depth);

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto in_depth = in_shape[3];
  pack_input(output_not_packed, in_height, in_width, in_depth, nbit(), output.data());
  delete [] output_not_packed;

  Measurement::Stop();
}


inline void func_QTZ_linear_mid_tread_half(
  const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
  const TensorView<T_INT, MemoryLayout::Atom>& nbit,
  const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
  const TensorView<T_FLOAT, MemoryLayout::NHWC>& output) {
  Measurement::Start("func_QTZ_linear_mid_tread_half");

  T_FLOAT min_value = 0.f;
  T_FLOAT n = (1 << nbit()) - 1.f;
  unsigned num_elems = input.size();

  for (unsigned i = 0; i < num_elems; i++)
  {
    T_FLOAT tmp = std::max(input.data()[i], min_value);
    tmp = std::min(tmp, max_value());
    tmp = tmp / max_value();
    tmp = roundf(tmp * n) / n;
    output.data()[i] = tmp * max_value();
  }

  Measurement::Stop();
}



#endif
