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

#ifdef USE_NEON
  #include <arm_neon.h>
#endif



/***************************************
 wrappers
***************************************/
void func_QTZ_binary_channel_wise_mean_scaling(
  T_FLOAT input[],
  T_FLOAT output[],
  T_UINT in_height,
  T_UINT in_width,
  T_UINT in_depth,
  T_UINT in_channel)
{
  unsigned num_elems_in_channel = in_height * in_width * in_depth;
  T_FLOAT sum[in_channel];
  T_FLOAT mean[in_channel];

  for (unsigned i = 0; i < in_channel; i++) {
    sum[i] = 0;
    mean[i] = 0;
  }

  for(unsigned i = 0; i < in_channel; i++)
  for(unsigned j = 0; j < num_elems_in_channel; j++)
  {
    sum[i] += std::abs(input[i * num_elems_in_channel + j]);
  }

  for(unsigned i = 0; i < in_channel; i++)
  {
    mean[i] = sum[i] / num_elems_in_channel;
  }

  for(unsigned i = 0; i < in_channel; i++)
  for(unsigned j = 0; j < num_elems_in_channel; j++)
  {
    unsigned in_index = i * num_elems_in_channel + j;
    output[in_index] = (input[in_index] >= 0) ? mean[i] : -1 * mean[i];
  }
}


void func_QTZ_binary_mean_scaling(
  T_FLOAT input[],
  T_FLOAT output[],
  T_UINT in_height,
  T_UINT in_width,
  T_UINT in_depth,
  T_UINT in_channel)
{
  T_FLOAT sum = 0;
  unsigned num_elems = in_height * in_width * in_depth * in_channel;

  for(unsigned i = 0; i < num_elems; i++)
  {
    sum += std::abs(input[i]);
  }

  T_FLOAT mean = sum / num_elems;
  T_FLOAT mean_minus = -1 * mean;

  for(unsigned i = 0; i < num_elems; i++)
  {
    output[i] = (input[i] >= 0) ? mean : mean_minus;
  }
}


void func_QTZ_linear_mid_tread_half_body(
  T_FLOAT input[],
  T_INT nbit,
  T_FLOAT max_value,
  QUANTIZED_NOT_PACKED output[],
  T_UINT in_height,
  T_UINT in_width,
  T_UINT in_depth,
  T_UINT in_channel,
  T_UINT begin,
  T_UINT end)

{
  const T_FLOAT max_value_r = 1.0 / max_value;
  const T_FLOAT min_value = 0.0f;
  const T_FLOAT n = (1 << nbit) - 1;
  int i = begin;

#ifdef USE_NEON
  const float32x4_t max_value_x4 = vdupq_n_f32(max_value);
  const float32x4_t min_value_x4 = {0.0f, 0.0f, 0.0f, 0.0f};
  const float32x4_t round_offset = {0.5f, 0.5f, 0.5f, 0.5f};
  const float32x4_t max_value_rn = vdupq_n_f32(max_value_r * n);

  if ((end - begin) % 8 == 0) {
    float32x4_t tmp = vld1q_f32(&input[i]);
    int32x4_t r;
    for (; i + 7 < end; i += 8) {
      float32x4_t tmp2 = vld1q_f32(&input[i+4]);
      tmp = vmaxq_f32(tmp, min_value_x4);
      tmp = vminq_f32(tmp, max_value_x4);
      tmp = vmlaq_f32(round_offset, tmp, max_value_rn);
      r = vcvtq_s32_f32(tmp);
      output[i]   = r[0];
      output[i+1] = r[1];
      output[i+2] = r[2];
      output[i+3] = r[3];
      tmp  = vld1q_f32(&input[i+8]);
      tmp2 = vmaxq_f32(tmp2, min_value_x4);
      tmp2 = vminq_f32(tmp2, max_value_x4);
      tmp2 = vmlaq_f32(round_offset, tmp2, max_value_rn);
      r = vcvtq_s32_f32(tmp2);
      output[i+4]   = r[0];
      output[i+1+4] = r[1];
      output[i+2+4] = r[2];
      output[i+3+4] = r[3];
    }
  } else {
    for (; i + 3 < end; i += 4) {
      float32x4_t tmp = vld1q_f32(&input[i]);
      tmp = vmaxq_f32(tmp, min_value_x4);
      tmp = vminq_f32(tmp, max_value_x4);
      tmp = vmlaq_f32(round_offset, tmp, max_value_rn);
      int32x4_t r = vcvtq_s32_f32(tmp);
      output[i]   = r[0];
      output[i+1] = r[1];
      output[i+2] = r[2];
      output[i+3] = r[3];
    }
  }
#endif

  for (; i < static_cast<int>(end); ++i)
  {
    T_FLOAT tmp = std::max(input[i], (T_FLOAT)min_value);
    tmp = std::min(tmp, max_value);
    output[i] = (T_INT)roundf(tmp * (max_value_r * n));
  }
}

void func_QTZ_linear_mid_tread_half(
  T_FLOAT input[],
  T_INT nbit,
  T_FLOAT max_value,
  QUANTIZED_NOT_PACKED output[],
  T_UINT in_height,
  T_UINT in_width,
  T_UINT in_depth,
  T_UINT in_channel=1)
{
  Measurement::Start("QTZ_linear_mid_tread_half");

  unsigned num_elems = in_height * in_width * in_depth * in_channel;

  unsigned int chunk_size = num_elems / std::thread::hardware_concurrency();
  if (chunk_size == 0) {
    chunk_size += 1;
  }

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_elems; i += chunk_size) {
    threads.emplace_back(std::thread([input, nbit, max_value, &output, in_height, in_width, in_depth, in_channel, i, chunk_size, num_elems] {
          func_QTZ_linear_mid_tread_half_body(input, nbit, max_value, output, in_height, in_width, in_depth, in_channel, i,
                                              std::min(i + chunk_size, static_cast<unsigned int>(num_elems)));
    }));
  }

  for (auto& th: threads) {
    th.join();
  }

  Measurement::Stop();
}


void func_QTZ_linear_mid_tread_half(
  T_FLOAT input[],
  T_INT nbit,
  T_FLOAT max_value,
  T_FLOAT output[],
  T_UINT in_height,
  T_UINT in_width,
  T_UINT in_depth,
  T_UINT in_channel=1)
{
  Measurement::Start("func_QTZ_linear_mid_tread_half");

  T_FLOAT min_value = 0;
  T_FLOAT n = (1 << nbit) - 1;
  unsigned num_elems = in_height * in_width * in_depth * in_channel;

  for (unsigned i = 0; i < num_elems; i++)
  {
    T_FLOAT tmp = std::max(input[i], min_value);
    tmp = std::min(tmp, max_value);
    tmp = tmp / max_value;
    tmp = roundf(tmp * n) / n;
    output[i] = tmp * max_value;
  }

  Measurement::Stop();
}



#endif
