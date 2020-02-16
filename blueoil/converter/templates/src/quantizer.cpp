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
#include <cmath>
#include <algorithm>

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

void func_QTZ_linear_mid_tread_half(
    const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
    const TensorView<T_INT, MemoryLayout::Atom>& nbit,
    const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
    const TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& output,
    BYTE *temporary_buf) {
  Measurement::Start("QTZ_linear_mid_tread_half");

  unsigned num_elems = input.size();

#ifdef _OPENMP
  const unsigned threads = omp_get_max_threads();
#else
  const unsigned threads = 1;
#endif
  unsigned int chunk_size = (num_elems + threads - 1) / threads;

  QUANTIZED_NOT_PACKED *buf = reinterpret_cast<QUANTIZED_NOT_PACKED*>(temporary_buf);

#pragma omp parallel for
  for (unsigned int i = 0; i < num_elems; i += chunk_size) {
    func_QTZ_linear_mid_tread_half_body(input.data(), nbit(), max_value(), buf, i,
                                              std::min(i + chunk_size, static_cast<unsigned int>(num_elems)));
  }

  const auto in_shape = input.get_shape();
  const auto in_height = in_shape[1];
  const auto in_width = in_shape[2];
  const auto in_depth = in_shape[3];
  pack_input(buf, in_height, in_width, in_depth, nbit(), output.data());

  Measurement::Stop();
}

void func_QTZ_linear_mid_tread_half(
  const TensorView<T_FLOAT, MemoryLayout::NHWC>& input,
  const TensorView<T_INT, MemoryLayout::Atom>& nbit,
  const TensorView<T_FLOAT, MemoryLayout::Atom>& max_value,
  const TensorView<T_FLOAT, MemoryLayout::NHWC>& output,
  BYTE *temporary_buf) {
  Measurement::Start("func_QTZ_linear_mid_tread_half");

  T_FLOAT min_value = 0.f;
  T_FLOAT n = (1 << nbit()) - 1.f;
  unsigned num_elems = input.size();
  const auto coeff = n / max_value();
  const auto inv_coeff = max_value() / n;

  unsigned i = 0;
#ifdef USE_NEON
  constexpr std::size_t SIMD_WIDTH = 4;
  const auto num_elems_floor = num_elems - num_elems % SIMD_WIDTH;
  const auto max_value_v = vdupq_n_f32(max_value());
  const auto min_value_v = vdupq_n_f32(min_value);
  const auto coeff_v = vdupq_n_f32(coeff);
  const auto inv_coeff_v = vdupq_n_f32(inv_coeff);
#pragma omp parallel for
  for (unsigned i = 0; i < num_elems_floor; i += SIMD_WIDTH)
  {
    const auto in = vld1q_f32(input.data() + i);
    const auto lbounded = vmaxq_f32(in, min_value_v);
    const auto ubounded = vminq_f32(lbounded, max_value_v);
    const auto normed = vmulq_f32(ubounded, coeff_v);
#ifdef AARCH32
    const auto biased = vaddq_f32(normed, vdupq_n_f32(0.5f));
    const auto rounded_i = vcvtq_s32_f32(biased);
    const auto rounded = vcvtq_f32_s32(rounded_i);
#else
    const auto rounded = vrndnq_f32(normed);
#endif
    const auto result = vmulq_f32(rounded, inv_coeff_v);
    vst1q_f32(output.data() + i, result);
  }
  i = num_elems_floor;
#elif defined USE_AVX
  constexpr std::size_t SIMD_WIDTH = 8;
  const auto num_elems_floor = num_elems - num_elems % SIMD_WIDTH;
  const auto max_value_v = _mm256_set1_ps(max_value());
  const auto min_value_v = _mm256_set1_ps(min_value);
  const auto coeff_v = _mm256_set1_ps(coeff);
  const auto inv_coeff_v = _mm256_set1_ps(inv_coeff);
#pragma omp parallel for
  for (unsigned i = 0; i < num_elems_floor; i += SIMD_WIDTH)
  {
    const auto in = _mm256_loadu_ps(input.data() + i);
    const auto lbounded = _mm256_max_ps(in, min_value_v);
    const auto ubounded = _mm256_min_ps(lbounded, max_value_v);
    const auto normed = _mm256_mul_ps(ubounded, coeff_v);
    const auto rounded = _mm256_round_ps(normed, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const auto result = _mm256_mul_ps(rounded, inv_coeff_v);
    _mm256_storeu_ps(output.data() + i, result);
  }
  i = num_elems_floor;
#endif
  for (; i < num_elems; i++)
  {
    T_FLOAT tmp = std::max(input.data()[i], min_value);
    tmp = std::min(tmp, max_value());
    tmp = tmp / max_value();
    tmp = roundf(tmp * n) / n;
    output.data()[i] = tmp * max_value();
  }

  Measurement::Stop();
}
