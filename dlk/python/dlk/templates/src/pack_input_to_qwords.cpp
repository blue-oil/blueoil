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
#include <limits.h>
#include "global.h"
#include "pack_input_to_qwords.h"
#include "operators.h" // FIXME(nikolay): for convolution_parameters definition, rid of it later
#include "time_measurement.h"
#ifdef USE_NEON
#include <arm_neon.h>
#endif
#ifdef USE_AVX
#include <x86intrin.h>
#endif

int pack_input(QUANTIZED_NOT_PACKED input[], size_t input_height, size_t input_width, size_t input_depth,
  size_t bits_per_input, QUANTIZED_PACKED output[]) {

  Measurement::Start("pack_input");
  const int bits_per_word = sizeof(QUANTIZED_PACKED) * CHAR_BIT;
  int full_words_in_depth = input_depth / bits_per_word;
  int remainder_bits_in_depth = input_depth % bits_per_word;
  int input_index = 0;
  int current_word = 0;

  auto len = input_height * input_width * input_depth;
#ifdef USE_NEON
  if (input_depth % 32 == 0) {
    const uint8_t coeff_ary[16] = {
      1, 2, 4, 8, 16, 32, 64, 128,
      1, 2, 4, 8, 16, 32, 64, 128,
    };
    const auto coeff = vld1q_u8(coeff_ary);
    const auto vone = vdupq_n_u8(1);
    constexpr int b = 32;
    constexpr int n_bits = 2;
    const auto blocks = len / b;
#pragma omp parallel for
    for (int i = 0; i < blocks; ++i) {
      const auto v0 = vld1q_u8(input + i * b +  0);
      const auto v1 = vld1q_u8(input + i * b + 16);
      const auto l0 = vandq_u8(v0, vone);
      const auto l1 = vandq_u8(v1, vone);
      const auto m0 = vshrq_n_u8(v0, 1);
      const auto m1 = vshrq_n_u8(v1, 1);
      const auto ml0 = vmulq_u8(l0, coeff);
      const auto ml1 = vmulq_u8(l1, coeff);
      const auto mm0 = vmulq_u8(m0, coeff);
      const auto mm1 = vmulq_u8(m1, coeff);
      const auto al0 = vpadd_u8(vget_low_u8(ml0), vget_high_u8(ml0));
      const auto al1 = vpadd_u8(vget_low_u8(ml1), vget_high_u8(ml1));
      const auto am0 = vpadd_u8(vget_low_u8(mm0), vget_high_u8(mm0));
      const auto am1 = vpadd_u8(vget_low_u8(mm1), vget_high_u8(mm1));
      const auto bl = vpadd_u8(al0, al1);
      const auto bm = vpadd_u8(am0, am1);
      const auto c = vpadd_u8(bl, bm);
      vst1_u8(reinterpret_cast<uint8_t*>(output + i * n_bits), c);
    }
    Measurement::Stop();
    return 0;
  }
#endif

#ifdef USE_AVX
  constexpr std::size_t SIMD_WIDTH = 32;
  if ((input_depth % SIMD_WIDTH) == 0) {
    const auto blocks = len / SIMD_WIDTH;
    for (int i = 0; i < blocks; ++i) {
      const auto a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(input + i * SIMD_WIDTH));
      const auto l = _mm256_movemask_epi8(_mm256_slli_epi16(a, 7));
      const auto m = _mm256_movemask_epi8(_mm256_slli_epi16(a, 6));
      output[i*2 + 0] = QUANTIZED_PACKED(l);
      output[i*2 + 1] = QUANTIZED_PACKED(m);
    }
    Measurement::Stop();
    return 0;
  }
#endif

  for (int h = 0; h < input_height; ++h)
      for (int w = 0; w < input_width; ++w) {
          for (int d = 0; d < full_words_in_depth; ++d) {
              output[current_word] = QUANTIZED_PACKED(0);
              output[current_word + 1] = QUANTIZED_PACKED(0);
              for (int d = 0; d < bits_per_word; ++d) {
                for(int b = 0; b < bits_per_input; ++b)
                  output[current_word + b] = QUANTIZED_PACKED(output[current_word + b].Raw() | ((input[input_index] >> b) & 1) << d);
                input_index++;
              }
              current_word += bits_per_input;
          }

          if(!remainder_bits_in_depth)
              continue;

          output[current_word] = QUANTIZED_PACKED(0);
          output[current_word + 1] = QUANTIZED_PACKED(0);
          for (int d = 0; d < remainder_bits_in_depth; ++d) {
             for(int b = 0; b < bits_per_input; ++b)
               output[current_word + b] = QUANTIZED_PACKED(output[current_word + b].Raw() | ((input[input_index] >> b) & 1) << d);
             input_index++;
          }
          current_word += bits_per_input;
      }

  Measurement::Stop();

  return current_word;
}

void pack_input_to_qwords(
  QUANTIZED_NOT_PACKED input[],
  QUANTIZED_PACKED output[],
  struct binary_convolution_parameters bcp)
{
  struct convolution_parameters& p = bcp.normal_conv_params;
  unsigned kernel_elems = p.kernel_height * p.kernel_width * p.kernel_depth;
  unsigned im2col_input_elems = p.output_height * p.output_width * kernel_elems;

  pack_input(input, bcp.normal_conv_params.input_height, bcp.normal_conv_params.input_width,
    bcp.normal_conv_params.kernel_depth, bcp.bin_input_bitwidth, output);
}
