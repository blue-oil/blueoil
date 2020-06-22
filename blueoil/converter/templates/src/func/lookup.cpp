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
=============================================================================*/

#include "global.h"
#include "func/lookup.h"
#include "time_measurement.h"
#ifdef USE_AVX
#include <x86intrin.h>
#endif

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
#ifdef USE_AVX
  const auto count = h * w;
  const auto count_floor = count - (count % 8);
  const auto coeff = _mm256_set1_ps(255.0f);
  for (std::size_t i = 0; i < count_floor; i += 8) {
    const auto vl0 = _mm256_castps128_ps256(_mm_loadu_ps(in_ptr + 3 * i +  0));
    const auto vl1 = _mm256_castps128_ps256(_mm_loadu_ps(in_ptr + 3 * i +  4));
    const auto vl2 = _mm256_castps128_ps256(_mm_loadu_ps(in_ptr + 3 * i +  8));
    const auto vh0 = _mm_loadu_ps(in_ptr + 3 * i + 12);
    const auto vh1 = _mm_loadu_ps(in_ptr + 3 * i + 16);
    const auto vh2 = _mm_loadu_ps(in_ptr + 3 * i + 20);
    const auto v0 = _mm256_insertf128_ps(vl0, vh0, 1);
    const auto v1 = _mm256_insertf128_ps(vl1, vh1, 1);
    const auto v2 = _mm256_insertf128_ps(vl2, vh2, 1);
    const auto tmp0 = _mm256_shuffle_ps(v1, v2, _MM_SHUFFLE(2, 1, 3, 2));
    const auto tmp1 = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(1, 0, 2, 1));
    const auto r = _mm256_shuffle_ps(v0, tmp0, _MM_SHUFFLE(2, 0, 3, 0));
    const auto g = _mm256_shuffle_ps(tmp1, tmp0, _MM_SHUFFLE(3, 1, 2, 0));
    const auto b = _mm256_shuffle_ps(tmp1, v2, _MM_SHUFFLE(3, 0, 3, 1));
    const auto ri = _mm256_cvtps_epi32(_mm256_mul_ps(r, coeff));
    const auto gi = _mm256_cvtps_epi32(_mm256_mul_ps(g, coeff));
    const auto bi = _mm256_cvtps_epi32(_mm256_mul_ps(b, coeff));
    const auto lr = _mm256_i32gather_epi32(reinterpret_cast<const int32_t*>(lsb_ptr), ri, 4);
    const auto lg = _mm256_i32gather_epi32(reinterpret_cast<const int32_t*>(lsb_ptr), gi, 4);
    const auto lb = _mm256_i32gather_epi32(reinterpret_cast<const int32_t*>(lsb_ptr), bi, 4);
    const auto mr = _mm256_i32gather_epi32(reinterpret_cast<const int32_t*>(msb_ptr), ri, 4);
    const auto mg = _mm256_i32gather_epi32(reinterpret_cast<const int32_t*>(msb_ptr), gi, 4);
    const auto mb = _mm256_i32gather_epi32(reinterpret_cast<const int32_t*>(msb_ptr), bi, 4);
    const auto shifted_lg = _mm256_slli_epi32(lg, 10);
    const auto shifted_lb = _mm256_slli_epi32(lb, 20);
    const auto l = lr | shifted_lg | shifted_lb;
    const auto shifted_mg = _mm256_slli_epi32(mg, 10);
    const auto shifted_mb = _mm256_slli_epi32(mb, 20);
    const auto m = mr | shifted_mg | shifted_mb;
    const auto lo0 = _mm256_unpacklo_epi32(l, m);
    const auto hi0 = _mm256_unpackhi_epi32(l, m);
    const auto lo1 = _mm256_permute2x128_si256(lo0, hi0, 0x20);
    const auto hi1 = _mm256_permute2x128_si256(lo0, hi0, 0x31);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + 2 * i + 0), lo1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + 2 * i + 8), hi1);
  }
  in_ptr += count_floor * 3;
  out_ptr += count_floor * 2;
  for (std::size_t i = count_floor; i < count; ++i) {
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
#else
  int len = h * w;
#pragma omp parallel for
  for(int i = 0; i < len; i++) {
    int r = int(in_ptr[i * 3 + 0] * 255.0f);
    int g = int(in_ptr[i * 3 + 1] * 255.0f);
    int b = int(in_ptr[i * 3 + 2] * 255.0f);

    auto r_lsb = lsb_ptr[r];
    auto g_lsb = lsb_ptr[g];
    auto b_lsb = lsb_ptr[b];
    auto r_msb = msb_ptr[r];
    auto g_msb = msb_ptr[g];
    auto b_msb = msb_ptr[b];

    out_ptr[i * 2 + 0] = QUANTIZED_PACKED((b_lsb.Raw() << 20) | (g_lsb.Raw() << 10) | r_lsb.Raw());
    out_ptr[i * 2 + 1] = QUANTIZED_PACKED((b_msb.Raw() << 20) | (g_msb.Raw() << 10) | r_msb.Raw());
  }
#endif

  Measurement::Stop();
}

