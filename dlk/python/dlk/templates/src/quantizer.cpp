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
#include <memory>

#include "quantizer.h"
#include "pack_input_to_qwords.h"
#include "time_measurement.h"
#ifdef USE_NEON
  #include <arm_neon.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif
#ifdef USE_AVX
  #include <x86intrin.h>
#endif



/***************************************
 wrappers
***************************************/
void func_QTZ_binary_channel_wise_mean_scaling(
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

void func_QTZ_linear_mid_tread_half_body(
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

  for (; i <= static_cast<int>(end) - 8; i += 8)
  {
    const auto in0 = vld1q_f32(&input[i]);
    const auto in1 = vld1q_f32(&input[i + 4]);
    const auto mx0 = vmaxq_f32(in0, min_value_x4);
    const auto mx1 = vmaxq_f32(in1, min_value_x4);
    const auto mn0 = vminq_f32(mx0, max_value_x4);
    const auto mn1 = vminq_f32(mx1, max_value_x4);
    //const auto mad0 = vmlaq_f32(round_offset, mn0, max_value_rn);
    //const auto mad1 = vmlaq_f32(round_offset, mn1, max_value_rn);
    const auto mul0 = vmulq_f32(mn0, max_value_rn);
    const auto mul1 = vmulq_f32(mn1, max_value_rn);
    const auto mad0 = vaddq_f32(mul0, round_offset);
    const auto mad1 = vaddq_f32(mul1, round_offset);
    const auto round0 = vcvtq_u32_f32(mad0);
    const auto round1 = vcvtq_u32_f32(mad1);
    const auto narrow10 = vmovn_u32(round0);
    const auto narrow11 = vmovn_u32(round1);
    const auto narrow2 = vmovn_u16(vcombine_u16(narrow10, narrow11));
    vst1_u8(output + i, narrow2);
  }
#elif defined USE_AVX
  const auto max_value_v = _mm256_set1_ps(max_value);
  const auto min_value_v = _mm256_set1_ps(min_value);
  const auto round_offset = _mm256_set1_ps(0.5f);
  const auto max_value_rn = _mm256_set1_ps(max_value_r * n);

  for (; i <= static_cast<int>(end) - 32; i += 32) {
    const auto in0 = _mm256_loadu_ps(input + i +  0);
    const auto in1 = _mm256_loadu_ps(input + i +  8);
    const auto in2 = _mm256_loadu_ps(input + i + 16);
    const auto in3 = _mm256_loadu_ps(input + i + 24);
    const auto mx0 = _mm256_max_ps(in0, min_value_v);
    const auto mx1 = _mm256_max_ps(in1, min_value_v);
    const auto mx2 = _mm256_max_ps(in2, min_value_v);
    const auto mx3 = _mm256_max_ps(in3, min_value_v);
    const auto mn0 = _mm256_min_ps(mx0, max_value_v);
    const auto mn1 = _mm256_min_ps(mx1, max_value_v);
    const auto mn2 = _mm256_min_ps(mx2, max_value_v);
    const auto mn3 = _mm256_min_ps(mx3, max_value_v);
    const auto mul0 = _mm256_mul_ps(mn0, max_value_rn);
    const auto mul1 = _mm256_mul_ps(mn1, max_value_rn);
    const auto mul2 = _mm256_mul_ps(mn2, max_value_rn);
    const auto mul3 = _mm256_mul_ps(mn3, max_value_rn);
    const auto round0 = _mm256_cvtps_epi32(mul0);
    const auto round1 = _mm256_cvtps_epi32(mul1);
    const auto round2 = _mm256_cvtps_epi32(mul2);
    const auto round3 = _mm256_cvtps_epi32(mul3);
    const auto pack02 = _mm256_packs_epi32(round0, round2);
    const auto pack13 = _mm256_packs_epi32(round1, round3);
    const auto perm02 = _mm256_permute4x64_epi64(pack02, 0xD8);
    const auto perm13 = _mm256_permute4x64_epi64(pack13, 0xD8);
    const auto pack = _mm256_packs_epi16(perm02, perm13);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), pack);
  }
#endif

  for (; i < static_cast<int>(end); ++i)
  {
    T_FLOAT tmp = std::max(input[i], (T_FLOAT)min_value);
    tmp = std::min(tmp, max_value);
    output[i] = (T_INT)roundf(tmp * (max_value_r * n));
  }
}

static const auto output_not_packed = std::make_unique<QUANTIZED_NOT_PACKED[]>(MAX_SIZE_INPUTS_PER_LAYER);

void func_QTZ_linear_mid_tread_half(
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_INT, MemoryLayout::Atom>& nbit,
    const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& output) {
  Measurement::Start("QTZ_linear_mid_tread_half");

  unsigned num_elems = input.size();

#ifdef _OPENMP
  const unsigned threads = omp_get_max_threads();
#else
  const unsigned threads = 1;
#endif
  unsigned int chunk_size = (num_elems + threads - 1) / threads;

#pragma omp parallel for
  for (unsigned int i = 0; i < num_elems; i += chunk_size) {
    func_QTZ_linear_mid_tread_half_body(input.data(), nbit(), max_value(), output_not_packed.get(), i,
                                              std::min(i + chunk_size, static_cast<unsigned int>(num_elems)));
  }

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto in_depth = in_shape[3];
  pack_input(output_not_packed.get(), in_height, in_width, in_depth, nbit(), output.data());

  Measurement::Stop();
}

void func_QTZ_linear_mid_tread_half(
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
